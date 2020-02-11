import contextlib
import json
import logging
import pathlib
from typing import Any, Callable, cast, Dict, Generator, Optional

import filelock
import google.api_core.exceptions as gcp_exceptions
import google.cloud.storage as storage

import yogadl.constants as constants
import yogadl.core as core
import yogadl.dataref.local_lmdb_dataref as dataref
import yogadl.rw_coordinator as rw_coordinator
import yogadl.tensorflow_util as tensorflow_util


class GCSConfigurations:
    """
    Configurations for GCSStorage.
    """

    def __init__(
        self, bucket: str, bucket_directory_path: str, url: str, local_cache_dir: str = "/tmp/",
    ) -> None:
        self.bucket = bucket
        self.bucket_directory_path = pathlib.Path(bucket_directory_path)
        self.url = url
        self.local_cache_dir = pathlib.Path(local_cache_dir)
        self.cache_backend = "LMDB"


class GCSStorage(core.Storage):
    """
    Stores dataset cache in Google Cloud Storage (GCS).

    GCSStorage creates a local cache from a dataset and then uploads
    it to the specified GCS bucket. When fetching from GCS, the creation
    time of the local cache (recorded in metadata), is compared to the
    creation time of the GCS cache, if they are not equivalent, the
    local cache is overwritten.

    The GCS cache, and the local cache are potentially shared across a
    number of concurrent processes. `cacheable()` provides synchronization
    guarantees. Users should not call `submit()` and `fetch()` if they
    anticipate concurrent data accesses.

    Authentication is currently only supported via the "Application
    Default Credentials" method in GCP. Typical configuration:
    ensure your VM runs in a service account that has sufficient
    permissions to read/write/delete from the GCS bucket where
    checkpoints will be stored (this only works when running in GCE).
    """

    def __init__(self, configurations: GCSConfigurations):
        self._configurations = configurations
        self._supported_cache_backends = ["LMDB"]

        self._gcs_client = storage.Client()
        self._bucket = self._gcs_client.bucket(self._configurations.bucket)

        self._rw_client = rw_coordinator.RwCoordinatorClient(url=configurations.url)

        self._check_configurations()

    def _check_configurations(self) -> None:
        assert self._configurations.local_cache_dir.is_dir()
        assert self._configurations.cache_backend in self._supported_cache_backends
        assert self._bucket.exists()

    def submit(self, data: core.Submittable, dataset_id: str, dataset_version: str) -> None:
        """
        Stores dataset to GCS by creating a local cache and uploading it to GCS.

        If a cache with a matching filepath already exists in GCS, it will be overwritten.

        `submit()` is not safe for concurrent accesses. For concurrent accesses use
        `cacheable()`.
        """
        local_cache_filepath = self._get_local_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        local_cache_filepath.parent.mkdir(parents=True, exist_ok=True)

        if local_cache_filepath.exists():
            logging.debug(f"Removing old local cache: {local_cache_filepath}.")
            local_cache_filepath.unlink()

        # TODO: remove TF hardcoding.
        tensorflow_util.serialize_tf_dataset_to_lmdb(
            dataset=data, checkpoint_path=local_cache_filepath,
        )
        logging.debug(
            f"Serialized dataset {dataset_id}:{dataset_version} "
            f"to local cache: {local_cache_filepath}."
        )

        # Upload to GCS.
        # TODO: Need to investigate what happens when bucket has
        # non-default retention policy.
        gcs_cache_filepath = self._get_gcs_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        blob = self._bucket.blob(str(gcs_cache_filepath))

        try:
            blob.upload_from_filename(str(local_cache_filepath))
        except gcp_exceptions.GoogleAPICallError as e:
            raise AssertionError(
                f"Upload from {local_cache_filepath} to {self._configurations.bucket}"
                f"/{gcs_cache_filepath} failed with exception {e}."
            )

        # Do not need to `reload()` to get latest blob metadata after upload.
        assert blob.time_created

        # Update metadata with new upload time.
        local_metadata = self._get_local_metadata(
            dataset_id=dataset_id, dataset_version=dataset_version
        )

        local_metadata["time_created"] = blob.time_created.timestamp()
        self._save_local_metadata(
            dataset_id=dataset_id, dataset_version=dataset_version, metadata=local_metadata,
        )

    def fetch(self, dataset_id: str, dataset_version: str) -> dataref.LMDBDataRef:
        """
        Fetch a dataset from GCS and provide a DataRef for streaming it.

        The creation time of the cache in GCS is compared to the creation
        time of the local cache, if they are not identical, the local cache
        is overwritten.

        `fetch()` is not safe for concurrent accesses. For concurrent accesses use
        `cacheable()`.
        """
        gcs_cache_filepath = self._get_gcs_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        blob = self._bucket.blob(str(gcs_cache_filepath))
        assert (
            blob.exists()
        ), f"Fetching a non-existent blob: {self._configurations.bucket}/{gcs_cache_filepath}."

        local_metadata = self._get_local_metadata(
            dataset_id=dataset_id, dataset_version=dataset_version
        )
        local_cache_filepath = self._get_local_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        try:
            blob.reload()
        except gcp_exceptions.GoogleAPICallError as e:
            raise AssertionError(
                f"Getting metadata of {self._configurations.bucket}"
                f"/{gcs_cache_filepath} failed with exception {e}."
            )

        if local_metadata.get("time_created") == blob.time_created.timestamp():
            logging.info("Local cache matches GCS cache.")
        else:
            try:
                blob.download_to_filename(str(local_cache_filepath))
            except gcp_exceptions.GoogleAPICallError as e:
                raise AssertionError(
                    f"Downloading blob {self._configurations.bucket}"
                    f"/{gcs_cache_filepath} failed with exception {e}."
                )

            local_metadata["time_created"] = blob.time_created.timestamp()
            self._save_local_metadata(
                dataset_id=dataset_id, dataset_version=dataset_version, metadata=local_metadata,
            )

        assert local_cache_filepath.exists()

        return dataref.LMDBDataRef(cache_filepath=local_cache_filepath)

    def cacheable(self, dataset_id: str, dataset_version: str) -> Callable:
        """
        A decorator that calls submit and fetch and is responsible for coordinating
        amongst instatiations of Storage in different processes.

        Initially requests a read lock, if cache is not present in GCS, will request
        a write lock and submit to GCS. Once file is present in GCS, will request read
        lock and fetch.
        """

        def wrap(f: Callable) -> Callable:
            def create_dataref(*args: Any, **kwargs: Any) -> dataref.LMDBDataRef:
                local_lmdb_dataref = self._try_reading_from_gcs(
                    dataset_id=dataset_id, dataset_version=dataset_version
                )

                if not local_lmdb_dataref:
                    self._try_writing_to_gcs(
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        f=f,
                        args=args,
                        kwargs=kwargs,
                    )

                    local_lmdb_dataref = self._try_reading_from_gcs(
                        dataset_id=dataset_id, dataset_version=dataset_version
                    )

                assert local_lmdb_dataref

                return local_lmdb_dataref

            return create_dataref

        return wrap

    def _try_reading_from_gcs(
        self, dataset_id: str, dataset_version: str
    ) -> Optional[dataref.LMDBDataRef]:
        remote_cache_path = self._get_gcs_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version
        )
        local_lmdb_dataref = None  # type: Optional[dataref.LMDBDataRef]
        with self._rw_client.read_lock(
            storage_type=constants.GCS_STORAGE,
            bucket=self._configurations.bucket,
            cache_path=remote_cache_path,
        ):
            blob = self._bucket.blob(str(remote_cache_path))
            if blob.exists():
                with self._lock_local_cache(
                    dataset_id=dataset_id, dataset_version=dataset_version,
                ):
                    local_lmdb_dataref = self.fetch(
                        dataset_id=dataset_id, dataset_version=dataset_version
                    )

        return local_lmdb_dataref

    def _try_writing_to_gcs(
        self, dataset_id: str, dataset_version: str, f: Callable, args: Any, kwargs: Any
    ) -> None:
        remote_cache_path = self._get_gcs_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version
        )

        with self._rw_client.write_lock(
            storage_type=constants.GCS_STORAGE,
            bucket=self._configurations.bucket,
            cache_path=remote_cache_path,
        ):
            # It is possible that the GCS is cache was created while
            # the write lock was being acquired.
            blob = self._bucket.blob(str(remote_cache_path))
            if not blob.exists():
                with self._lock_local_cache(
                    dataset_id=dataset_id, dataset_version=dataset_version,
                ):
                    self.submit(
                        data=f(*args, **kwargs),
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                    )

    @contextlib.contextmanager
    def _lock_local_cache(
        self, dataset_id: str, dataset_version: str
    ) -> Generator[None, None, None]:
        lock_filepath = (
            self._configurations.local_cache_dir.joinpath("yogadl_local_cache")
            .joinpath(dataset_id)
            .joinpath(dataset_version)
            .joinpath("yogadl.lock")
        )
        lock_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Blocks until access is granted.
        access_lock = filelock.FileLock(str(lock_filepath))
        with access_lock.acquire():
            yield

    def _get_gcs_cache_filepath(self, dataset_id: str, dataset_version: str) -> pathlib.Path:
        assert dataset_id, "`dataset_id` must be a non-empty string."
        assert dataset_version, "`dataset_version` must be a non-empty string."
        return (
            self._configurations.bucket_directory_path.joinpath(dataset_id)
            .joinpath(dataset_version)
            .joinpath("cache.mdb")
        )

    def _get_local_cache_filepath(self, dataset_id: str, dataset_version: str) -> pathlib.Path:
        assert dataset_id, "`dataset_id` must be a non-empty string."
        assert dataset_version, "`dataset_version` must be a non-empty string."
        return (
            self._configurations.local_cache_dir.joinpath("yogadl_local_cache")
            .joinpath(dataset_id)
            .joinpath(dataset_version)
            .joinpath("cache.mdb")
        )

    def _get_local_metadata_filepath(self, dataset_id: str, dataset_version: str) -> pathlib.Path:
        return (
            self._configurations.local_cache_dir.joinpath("yogadl_local_cache")
            .joinpath(dataset_id)
            .joinpath(dataset_version)
            .joinpath("local_metadata.json")
        )

    def _get_local_metadata(self, dataset_id: str, dataset_version: str) -> Dict[str, Any]:
        metadata_filepath = self._get_local_metadata_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        if not metadata_filepath.exists():
            return {}

        with open(str(metadata_filepath), "r") as metadata_file:
            return cast(Dict[str, Any], json.load(metadata_file))

    def _save_local_metadata(
        self, dataset_id: str, dataset_version: str, metadata: Dict[str, Any]
    ) -> None:
        metadata_filepath = self._get_local_metadata_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        with open(str(metadata_filepath), "w") as metadata_file:
            json.dump(metadata, metadata_file)

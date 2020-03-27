# Copyright 2020 Determined AI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import abc
import datetime
import contextlib
import json
import logging
import pathlib
from typing import Any, Callable, cast, Dict, Generator, Optional

import filelock

import yogadl
from yogadl import dataref, rw_coordinator, tensorflow


class BaseCloudConfigurations(metaclass=abc.ABCMeta):
    """
    Configurations for BaseCloudStorage.
    """

    def __init__(
        self, bucket: str, bucket_directory_path: str, url: str, local_cache_dir: str,
    ) -> None:
        self.bucket = bucket
        self.bucket_directory_path = pathlib.Path(bucket_directory_path)
        self.url = url
        self.local_cache_dir = pathlib.Path(local_cache_dir)
        self.cache_format = "LMDB"


class BaseCloudStorage(yogadl.Storage):
    """
    Base class for using cloud storage.

    This class should never be used directly. Instead users should use
    S3Storage or GCSStorage.
    """

    def __init__(self, configurations: BaseCloudConfigurations) -> None:
        self._configurations = configurations
        self._rw_client = rw_coordinator.RwCoordinatorClient(url=self._configurations.url)
        self._supported_cache_formats = ["LMDB"]

    @property
    @abc.abstractmethod
    def _storage_type(self) -> str:
        pass

    @abc.abstractmethod
    def _is_cloud_cache_present(self, dataset_id: str, dataset_version: str) -> bool:
        pass

    @abc.abstractmethod
    def _download_from_cloud_storage(
        self, dataset_id: str, dataset_version: str, local_cache_filepath: pathlib.Path
    ) -> datetime.datetime:
        pass

    @abc.abstractmethod
    def _upload_to_cloud_storage(
        self, dataset_id: str, dataset_version: str, local_cache_filepath: pathlib.Path
    ) -> datetime.datetime:
        pass

    @abc.abstractmethod
    def _get_remote_cache_timestamp(
        self, dataset_id: str, dataset_version: str
    ) -> datetime.datetime:
        pass

    def submit(self, data: yogadl.Submittable, dataset_id: str, dataset_version: str) -> None:
        """
        Stores dataset by creating a local cache and uploading it to cloud storage.

        If a cache with a matching filepath already exists in cloud storage, it will be overwritten.

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
        tensorflow.serialize_tf_dataset_to_lmdb(
            dataset=data, checkpoint_path=local_cache_filepath,
        )
        logging.info(
            f"Serialized dataset {dataset_id}:{dataset_version} to local cache: "
            f"{local_cache_filepath} and uploading to remote storage."
        )

        timestamp = self._upload_to_cloud_storage(
            dataset_id=dataset_id,
            dataset_version=dataset_version,
            local_cache_filepath=local_cache_filepath,
        ).timestamp()
        logging.info("Cache upload to remote storage finished.")

        # Update metadata with new upload time.
        local_metadata = self._get_local_metadata(
            dataset_id=dataset_id, dataset_version=dataset_version
        )

        local_metadata["time_created"] = timestamp
        self._save_local_metadata(
            dataset_id=dataset_id, dataset_version=dataset_version, metadata=local_metadata,
        )

    def fetch(self, dataset_id: str, dataset_version: str) -> dataref.LMDBDataRef:
        """
        Fetch a dataset from cloud storage and provide a DataRef for streaming it.

        The timestamp of the cache in cloud storage is compared to the creation
        time of the local cache, if they are not identical, the local cache
        is overwritten.

        `fetch()` is not safe for concurrent accesses. For concurrent accesses use
        `cacheable()`.
        """

        local_metadata = self._get_local_metadata(
            dataset_id=dataset_id, dataset_version=dataset_version
        )
        local_cache_filepath = self._get_local_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        remote_cache_timestamp = self._get_remote_cache_timestamp(
            dataset_id=dataset_id, dataset_version=dataset_version
        ).timestamp()

        if local_metadata.get("time_created") == remote_cache_timestamp:
            logging.info("Local cache matches remote cache.")
        else:
            logging.info(f"Downloading remote cache to {local_cache_filepath}.")
            local_metadata["time_created"] = self._download_from_cloud_storage(
                dataset_id=dataset_id,
                dataset_version=dataset_version,
                local_cache_filepath=local_cache_filepath,
            ).timestamp()
            logging.info("Cache download finished.")

            self._save_local_metadata(
                dataset_id=dataset_id, dataset_version=dataset_version, metadata=local_metadata,
            )

        assert local_cache_filepath.exists()

        return dataref.LMDBDataRef(cache_filepath=local_cache_filepath)

    def cacheable(self, dataset_id: str, dataset_version: str) -> Callable:
        """
        A decorator that calls submit and fetch and is responsible for coordinating
        amongst instatiations of Storage in different processes.

        Initially requests a read lock, if cache is not present in cloud storage, will request
        a write lock and submit to cloud storage. Once file is present in cloud storage, will
        request a read lock and fetch.
        """

        def wrap(f: Callable) -> Callable:
            def create_dataref(*args: Any, **kwargs: Any) -> dataref.LMDBDataRef:
                local_lmdb_dataref = self._try_reading_from_cloud_storage(
                    dataset_id=dataset_id, dataset_version=dataset_version
                )

                if not local_lmdb_dataref:
                    self._try_writing_to_cloud_storage(
                        dataset_id=dataset_id,
                        dataset_version=dataset_version,
                        f=f,
                        args=args,
                        kwargs=kwargs,
                    )

                    local_lmdb_dataref = self._try_reading_from_cloud_storage(
                        dataset_id=dataset_id, dataset_version=dataset_version
                    )

                    assert local_lmdb_dataref, "Unable to create dataref from cloud cache."

                return local_lmdb_dataref

            return create_dataref

        return wrap

    def _try_reading_from_cloud_storage(
        self, dataset_id: str, dataset_version: str
    ) -> Optional[dataref.LMDBDataRef]:
        remote_cache_path = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version
        )
        local_lmdb_dataref = None  # type: Optional[dataref.LMDBDataRef]
        with self._rw_client.read_lock(
            storage_type=self._storage_type,
            bucket=self._configurations.bucket,
            cache_path=remote_cache_path,
        ):
            if self._is_cloud_cache_present(dataset_id=dataset_id, dataset_version=dataset_version):
                with self._lock_local_cache(
                    dataset_id=dataset_id, dataset_version=dataset_version,
                ):
                    local_lmdb_dataref = self.fetch(
                        dataset_id=dataset_id, dataset_version=dataset_version
                    )

        return local_lmdb_dataref

    def _try_writing_to_cloud_storage(
        self, dataset_id: str, dataset_version: str, f: Callable, args: Any, kwargs: Any
    ) -> None:
        remote_cache_path = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version
        )
        with self._rw_client.write_lock(
            storage_type=self._storage_type,
            bucket=self._configurations.bucket,
            cache_path=remote_cache_path,
        ):
            # It is possible that the cache was created while
            # the write lock was being acquired.
            if not self._is_cloud_cache_present(
                dataset_id=dataset_id, dataset_version=dataset_version
            ):
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

    def _get_remote_cache_filepath(self, dataset_id: str, dataset_version: str) -> pathlib.Path:
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

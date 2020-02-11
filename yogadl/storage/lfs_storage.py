import contextlib
import logging
import pathlib
from typing import Any, Callable, Generator

import filelock

import yogadl.core as core
import yogadl.dataref.local_lmdb_dataref as dataref
import yogadl.tensorflow_util as tensorflow_util


class LFSConfigurations:
    """
    Configurations for LFSStorage.
    """

    def __init__(self, storage_dir_path: str):
        self.storage_dir_path = pathlib.Path(storage_dir_path)
        self.cache_backend = "LMDB"


class LFSStorage(core.Storage):
    """
    Storage for local file system (not NFS).
    """

    def __init__(self, configurations: LFSConfigurations):
        self._configurations = configurations
        self._supported_cache_backends = ["LMDB"]
        self._check_configurations()

    def _check_configurations(self) -> None:
        assert self._configurations.storage_dir_path.is_dir()
        assert self._configurations.cache_backend in self._supported_cache_backends

    def submit(self, data: core.Submittable, dataset_id: str, dataset_version: str) -> None:
        """
        Stores dataset to a cache and updates metadata file with information.

        If a cache with a matching filepath already exists, it will be overwritten.
        """
        cache_filepath = self._get_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        cache_filepath.parent.mkdir(parents=True, exist_ok=True)

        if cache_filepath.exists():
            logging.info(f"Removing old cache: {cache_filepath}.")
            cache_filepath.unlink()

        # TODO: remove TF hardcoding.
        tensorflow_util.serialize_tf_dataset_to_lmdb(
            dataset=data, checkpoint_path=cache_filepath,
        )
        logging.info(f"Serialized dataset {dataset_id}:{dataset_version} to: {cache_filepath}.")

    def fetch(self, dataset_id: str, dataset_version: str) -> dataref.LMDBDataRef:
        """
        Fetch a dataset from storage and provide a DataRef
        for streaming it.
        """
        cache_filepath = self._get_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version
        )
        assert cache_filepath.exists()

        return dataref.LMDBDataRef(cache_filepath=cache_filepath)

    def cacheable(self, dataset_id: str, dataset_version: str) -> Callable:
        """
        A decorator that calls submit and fetch and is responsible
        for coordinating amongst instatiations of Storage in different
        processes.
        """

        def wrap(f: Callable) -> Callable:
            def create_dataref(*args: Any, **kwargs: Any) -> dataref.LMDBDataRef:
                with self._lock_this_dataset_version(dataset_id, dataset_version):
                    cache_filepath = self._get_cache_filepath(
                        dataset_id=dataset_id, dataset_version=dataset_version
                    )

                    if not cache_filepath.exists():
                        self.submit(
                            data=f(*args, **kwargs),
                            dataset_id=dataset_id,
                            dataset_version=dataset_version,
                        )

                return self.fetch(dataset_id=dataset_id, dataset_version=dataset_version)

            return create_dataref

        return wrap

    @contextlib.contextmanager
    def _lock_this_dataset_version(
        self, dataset_id: str, dataset_version: str
    ) -> Generator[None, None, None]:
        lock_filepath = (
            self._configurations.storage_dir_path.joinpath(dataset_id)
            .joinpath(dataset_version)
            .joinpath("yogadl.lock")
        )
        lock_filepath.parent.mkdir(parents=True, exist_ok=True)

        # Blocks until access is granted.
        access_lock = filelock.FileLock(str(lock_filepath))
        with access_lock.acquire():
            yield

    def _get_cache_filepath(self, dataset_id: str, dataset_version: str) -> pathlib.Path:
        assert dataset_id, "`dataset_id` must be a non-empty string."
        assert dataset_version, "`dataset_version` must be a non-empty string."
        return (
            self._configurations.storage_dir_path.joinpath(dataset_id)
            .joinpath(dataset_version)
            .joinpath("cache.mdb")
        )

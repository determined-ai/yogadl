import asyncio
import threading
from typing import Optional

import google.cloud.storage as google_storage
import tensorflow as tf
import tensorflow_datasets as tfds

import yogadl.rw_coordinator as rw_coordinator
import yogadl.storage.gcs_storage as gcs_storage
import yogadl.storage.lfs_storage as lfs_storage


def make_mnist_train_dataset() -> tf.data.Dataset:
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    # We use test because for tfds version < 1.3 the
    # train split is automatically shuffled, breaking
    # the test.
    mnist_train = mnist_builder.as_dataset(split="test")
    return mnist_train


def cleanup_lfs_storage(
    configurations: lfs_storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    cache_filepath = (
        configurations.storage_dir_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )
    if cache_filepath.exists():
        cache_filepath.unlink()


def cleanup_gcs_storage(
    configurations: gcs_storage.GCSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    gcs_cache_filepath = (
        configurations.bucket_directory_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )

    client = google_storage.Client()
    bucket = client.bucket(configurations.bucket)
    blob = bucket.blob(str(gcs_cache_filepath))
    if blob.exists():
        blob.delete()


class AccessServerHandler:
    def __init__(self, hostname: str, port: int) -> None:
        self._access_server = rw_coordinator.RwCoordinatorServer(hostname=hostname, port=port)

        self._thread_running_server = None  # type: Optional[threading.Thread]

    def run_server_in_thread(self) -> None:
        asyncio.get_event_loop().run_until_complete(self._access_server.run_server())
        self._thread_running_server = threading.Thread(target=asyncio.get_event_loop().run_forever)
        self._thread_running_server.start()

    def stop_server(self) -> None:
        self._access_server.stop_server()

        assert self._thread_running_server
        self._thread_running_server.join()

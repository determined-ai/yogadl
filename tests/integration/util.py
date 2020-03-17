import asyncio
import threading
from typing import Optional

import boto3
import google.cloud.storage as google_storage
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from yogadl import rw_coordinator, storage


def make_mnist_test_dataset() -> tf.data.Dataset:
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    # We use test because for tfds version < 1.3 the
    # train split is automatically shuffled, breaking
    # the test.
    mnist_test = mnist_builder.as_dataset(split="test")
    return mnist_test


def cleanup_lfs_storage(
    configurations: storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    cache_filepath = (
        configurations.storage_dir_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )
    if cache_filepath.exists():
        cache_filepath.unlink()


def cleanup_gcs_storage(
    configurations: storage.GCSConfigurations, dataset_id: str, dataset_version: str
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


def cleanup_s3_storage(
    configurations: storage.S3Configurations, dataset_id: str, dataset_version: str
) -> None:
    s3_cache_filepath = (
        configurations.bucket_directory_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )

    client = boto3.client("s3")
    client.delete_object(Bucket=configurations.bucket, Key=str(s3_cache_filepath))


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


def compare_datasets_graph_mode(
    original_dataset: tf.data.Dataset, dataset_from_stream: tf.data.Dataset
) -> int:
    next_element_from_stream = dataset_from_stream.make_one_shot_iterator().get_next()
    next_element_from_orig = original_dataset.make_one_shot_iterator().get_next()
    data_samples = 0

    with tf.Session() as sess:
        while True:
            try:
                element_from_stream = sess.run(next_element_from_stream)
                element_from_dataset = sess.run(next_element_from_orig)
                assert element_from_stream["label"] == element_from_dataset["label"]
                assert np.array_equal(element_from_stream["image"], element_from_dataset["image"])
                data_samples += 1
            except tf.errors.OutOfRangeError:
                break

    return data_samples


def compare_datasets_eager_mode(
    original_dataset: tf.data.Dataset, dataset_from_stream: tf.data.Dataset
) -> int:
    next_element_from_stream = dataset_from_stream.as_numpy_iterator()
    next_element_from_orig = original_dataset.as_numpy_iterator()
    data_samples = 0

    for orig_dict, from_stream_dict in zip(next_element_from_orig, next_element_from_stream):
        for orig_data, from_stream_data in zip(orig_dict, from_stream_dict):
            assert np.array_equal(orig_data, from_stream_data)
        data_samples += 1

    return data_samples


def compare_datasets(
    original_dataset: tf.data.Dataset, dataset_from_stream: tf.data.Dataset
) -> int:
    if tf.executing_eagerly():
        return compare_datasets_eager_mode(original_dataset, dataset_from_stream)
    else:
        return compare_datasets_graph_mode(original_dataset, dataset_from_stream)

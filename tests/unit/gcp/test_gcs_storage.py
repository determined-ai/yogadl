import json
import pathlib

import google.cloud.storage as google_storage
import tensorflow as tf
from tl.testing import thread

import tests.unit.util as test_util

from yogadl import dataref, storage


def create_gcs_configuration(access_server_port: int) -> storage.GCSConfigurations:
    return storage.GCSConfigurations(
        bucket="yogadl-test",
        bucket_directory_path="unit-tests",
        url=f"ws://localhost:{access_server_port}",
        local_cache_dir="/tmp/",
    )


def get_local_cache_filepath(
    configurations: storage.GCSConfigurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.local_cache_dir.joinpath("yogadl_local_cache")
        .joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )


def get_local_metadata_filepath(
    configurations: storage.GCSConfigurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.local_cache_dir.joinpath("yogadl_local_cache")
        .joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("local_metadata.json")
    )


def get_gcs_filepath(
    configurations: storage.GCSConfigurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.bucket_directory_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )


def test_gcs_storage_submit() -> None:
    range_size = 10
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_gcs_configuration(access_server_port=15032)

    client = google_storage.Client()
    bucket = client.bucket(configurations.bucket)
    gcs_cache_filepath = get_gcs_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
    )
    blob = bucket.blob(str(gcs_cache_filepath))

    previous_creation_time = None
    if blob.exists():
        blob.reload()
        previous_creation_time = blob.time_created

    gcs_storage = storage.GCSStorage(configurations=configurations)
    gcs_storage.submit(
        data=dataset, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    blob = bucket.blob(str(gcs_cache_filepath))
    blob.reload()
    assert blob.exists()
    assert blob.time_created is not None
    assert previous_creation_time != blob.time_created

    if previous_creation_time is not None:
        assert previous_creation_time < blob.time_created


def test_gcs_storage_local_metadata() -> None:
    range_size = 10
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_gcs_configuration(access_server_port=15032)

    client = google_storage.Client()
    bucket = client.bucket(configurations.bucket)
    gcs_cache_filepath = get_gcs_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    gcs_storage = storage.GCSStorage(configurations=configurations)
    gcs_storage.submit(
        data=dataset, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    local_metadata_filepath = get_local_metadata_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version
    )
    with open(str(local_metadata_filepath), "r") as metadata_file:
        metadata = json.load(metadata_file)

    blob = bucket.blob(str(gcs_cache_filepath))
    blob.reload()

    assert metadata.get("time_created")
    assert blob.time_created.timestamp() == metadata["time_created"]

    local_metadata_filepath.unlink()
    _ = gcs_storage.fetch(dataset_id=dataset_id, dataset_version=dataset_version)
    with open(str(local_metadata_filepath), "r") as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata.get("time_created")
    assert blob.time_created.timestamp() == metadata["time_created"]


def test_gcs_storage_submit_and_fetch() -> None:
    range_size = 20
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_gcs_configuration(access_server_port=15032)

    gcs_storage = storage.GCSStorage(configurations=configurations)
    gcs_storage.submit(
        data=dataset, dataset_id=dataset_id, dataset_version=dataset_version,
    )
    dataref = gcs_storage.fetch(dataset_id=dataset_id, dataset_version=dataset_version)
    stream = dataref.stream()

    assert stream.length == range_size
    data_generator = stream.iterator_fn()
    generator_length = 0
    for idx, data in enumerate(data_generator):
        assert idx == data
        generator_length += 1
    assert generator_length == range_size


def test_gcs_storage_cacheable_single_threaded() -> None:
    original_range_size = 120
    updated_range_size = 55
    dataset_id = "range-dataset"
    dataset_version = "0"
    configurations = create_gcs_configuration(access_server_port=15032)

    access_server_handler = test_util.AccessServerHandler(hostname="localhost", port=15032)
    access_server_handler.run_server_in_thread()

    gcs_cache_filepath = get_gcs_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
    )
    client = google_storage.Client()
    bucket = client.bucket(configurations.bucket)
    blob = bucket.blob(str(gcs_cache_filepath))
    if blob.exists():
        blob.delete()

    gcs_storage = storage.GCSStorage(configurations=configurations)

    @gcs_storage.cacheable(dataset_id, dataset_version)
    def make_dataref(range_size: int) -> dataref.LMDBDataRef:
        return tf.data.Dataset.range(range_size)  # type: ignore

    original_data_stream = make_dataref(range_size=original_range_size).stream()
    assert original_data_stream.length == original_range_size
    data_generator = original_data_stream.iterator_fn()
    generator_length = 0
    for idx, data in enumerate(data_generator):
        assert idx == data
        generator_length += 1
    assert generator_length == original_range_size

    updated_data_stream = make_dataref(range_size=updated_range_size).stream()
    assert updated_data_stream.length == original_range_size

    access_server_handler.stop_server()


def worker(
    configurations: storage.GCSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    range_size = 120
    gcs_storage = storage.GCSStorage(configurations=configurations)

    @gcs_storage.cacheable(dataset_id, dataset_version)
    def make_dataref(input_range_size: int) -> dataref.LMDBDataRef:
        return tf.data.Dataset.range(input_range_size)  # type: ignore

    stream = make_dataref(input_range_size=range_size).stream()
    assert stream.length == range_size

    data_generator = stream.iterator_fn()
    generator_length = 0
    for idx, data in enumerate(data_generator):
        assert idx == data
        generator_length += 1
    assert generator_length == range_size


class MultiThreadedTests(thread.ThreadAwareTestCase):  # type: ignore
    def test_gcs_storage_cacheable_multi_threaded(self) -> None:
        dataset_id = "range-dataset"
        dataset_version = "0"
        num_threads = 20
        configurations = create_gcs_configuration(access_server_port=15032)

        access_server_handler = test_util.AccessServerHandler(hostname="localhost", port=15032)
        access_server_handler.run_server_in_thread()

        gcs_cache_filepath = get_gcs_filepath(
            configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
        )
        client = google_storage.Client()
        bucket = client.bucket(configurations.bucket)
        blob = bucket.blob(str(gcs_cache_filepath))
        if blob.exists():
            blob.delete()

        try:
            with thread.ThreadJoiner(10):
                for _ in range(num_threads):
                    self.run_in_thread(lambda: worker(configurations, dataset_id, dataset_version))
        finally:
            access_server_handler.stop_server()

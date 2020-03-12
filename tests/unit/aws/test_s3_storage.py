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
import json
import pathlib

import boto3
import botocore.client as boto_client
import tensorflow as tf
from tl.testing import thread

import tests.unit.util as test_util

from yogadl import dataref, storage


def create_s3_configuration(access_server_port: int) -> storage.S3Configurations:
    return storage.S3Configurations(
        bucket="yogadl-test",
        bucket_directory_path="unit-tests",
        url=f"ws://localhost:{access_server_port}",
        local_cache_dir="/tmp/",
    )


def get_local_cache_filepath(
    configurations: storage.S3Configurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.local_cache_dir.joinpath("yogadl_local_cache")
        .joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )


def get_local_metadata_filepath(
    configurations: storage.S3Configurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.local_cache_dir.joinpath("yogadl_local_cache")
        .joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("local_metadata.json")
    )


def get_s3_filepath(
    configurations: storage.S3Configurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.bucket_directory_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )


def test_s3_storage_submit() -> None:
    range_size = 10
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_s3_configuration(access_server_port=15032)

    client = boto3.client("s3")
    aws_cache_filepath = get_s3_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    try:
        blob_info = client.head_object(Bucket=configurations.bucket, Key=str(aws_cache_filepath))
        previous_creation_time = blob_info.get("LastModified")
    except boto_client.ClientError:
        previous_creation_time = None

    s3_storage = storage.S3Storage(configurations=configurations)
    s3_storage.submit(
        data=dataset, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    blob_info = client.head_object(Bucket=configurations.bucket, Key=str(aws_cache_filepath))
    assert blob_info.get("LastModified") is not None
    assert previous_creation_time != blob_info.get("LastModified")

    if previous_creation_time is not None:
        assert previous_creation_time < blob_info.get("LastModified")


def test_s3_storage_local_metadata() -> None:
    range_size = 10
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_s3_configuration(access_server_port=15032)

    client = boto3.client("s3")
    aws_cache_filepath = get_s3_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    s3_storage = storage.S3Storage(configurations=configurations)
    s3_storage.submit(
        data=dataset, dataset_id=dataset_id, dataset_version=dataset_version,
    )

    local_metadata_filepath = get_local_metadata_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version
    )
    with open(str(local_metadata_filepath), "r") as metadata_file:
        metadata = json.load(metadata_file)

    blob_info = client.head_object(Bucket=configurations.bucket, Key=str(aws_cache_filepath))
    creation_time = blob_info.get("LastModified")

    assert metadata.get("time_created")
    assert creation_time.timestamp() == metadata["time_created"]

    local_metadata_filepath.unlink()
    _ = s3_storage.fetch(dataset_id=dataset_id, dataset_version=dataset_version)
    with open(str(local_metadata_filepath), "r") as metadata_file:
        metadata = json.load(metadata_file)

    assert metadata.get("time_created")
    assert creation_time.timestamp() == metadata["time_created"]


def test_s3_storage_submit_and_fetch() -> None:
    range_size = 20
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_s3_configuration(access_server_port=15032)

    s3_storage = storage.S3Storage(configurations=configurations)
    s3_storage.submit(
        data=dataset, dataset_id=dataset_id, dataset_version=dataset_version,
    )
    dataref = s3_storage.fetch(dataset_id=dataset_id, dataset_version=dataset_version)
    stream = dataref.stream()

    assert stream.length == range_size
    data_generator = stream.iterator_fn()
    generator_length = 0
    for idx, data in enumerate(data_generator):
        assert idx == data
        generator_length += 1
    assert generator_length == range_size


def test_s3_storage_cacheable_single_threaded() -> None:
    original_range_size = 120
    updated_range_size = 55
    dataset_id = "range-dataset"
    dataset_version = "0"
    configurations = create_s3_configuration(access_server_port=15032)

    access_server_handler = test_util.AccessServerHandler(hostname="localhost", port=15032)
    access_server_handler.run_server_in_thread()

    s3_cache_filepath = get_s3_filepath(
        configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
    )
    client = boto3.client("s3")
    client.delete_object(Bucket=configurations.bucket, Key=str(s3_cache_filepath))

    s3_storage = storage.S3Storage(configurations=configurations)

    @s3_storage.cacheable(dataset_id, dataset_version)
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


def worker(configurations: storage.S3Configurations, dataset_id: str, dataset_version: str) -> None:
    range_size = 120
    s3_storage = storage.S3Storage(configurations=configurations)

    @s3_storage.cacheable(dataset_id, dataset_version)
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
        configurations = create_s3_configuration(access_server_port=15032)

        access_server_handler = test_util.AccessServerHandler(hostname="localhost", port=15032)
        access_server_handler.run_server_in_thread()

        s3_cache_filepath = get_s3_filepath(
            configurations=configurations, dataset_id=dataset_id, dataset_version=dataset_version,
        )
        client = boto3.client("s3")
        client.delete_object(Bucket=configurations.bucket, Key=str(s3_cache_filepath))

        try:
            with thread.ThreadJoiner(10):
                for _ in range(num_threads):
                    self.run_in_thread(lambda: worker(configurations, dataset_id, dataset_version))
        finally:
            access_server_handler.stop_server()

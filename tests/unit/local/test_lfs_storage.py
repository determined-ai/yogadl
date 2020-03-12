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
import pathlib

import tensorflow as tf

from yogadl import dataref, storage


def create_configurations() -> storage.LFSConfigurations:
    return storage.LFSConfigurations(storage_dir_path="/tmp/")


def get_cache_filepath(
    configurations: storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return (
        configurations.storage_dir_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )


def test_storage_submit() -> None:
    range_size = 10
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = tf.data.Dataset.range(range_size)
    configurations = create_configurations()
    if get_cache_filepath(configurations, dataset_id, dataset_version).exists():
        get_cache_filepath(configurations, dataset_id, dataset_version).unlink()

    lfs_storage = storage.LFSStorage(configurations=configurations)
    lfs_storage.submit(data=dataset, dataset_id=dataset_id, dataset_version=dataset_version)

    assert get_cache_filepath(configurations, dataset_id, dataset_version).is_file()


def test_storage_cacheable_single_threaded() -> None:
    original_range_size = 120
    updated_range_size = 126
    dataset_id = "range-dataset"
    dataset_version = "1"
    configurations = create_configurations()
    if get_cache_filepath(configurations, dataset_id, dataset_version).exists():
        get_cache_filepath(configurations, dataset_id, dataset_version).unlink()

    lfs_storage = storage.LFSStorage(configurations=configurations)

    @lfs_storage.cacheable(dataset_id, dataset_version)
    def make_dataref(range_size: int) -> dataref.LMDBDataRef:
        return tf.data.Dataset.range(range_size)  # type: ignore

    original_data_stream = make_dataref(range_size=original_range_size).stream()
    assert original_data_stream.length == original_range_size
    data_generator = original_data_stream.iterator_fn()
    for idx in range(original_range_size):
        assert idx == next(data_generator)

    updated_data_stream = make_dataref(range_size=updated_range_size).stream()
    assert updated_data_stream.length == original_range_size

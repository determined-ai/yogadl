import json
import os
import pathlib

import yogadl.storage.lfs_storage as storage
import yogadl.dataref.lfs_dataref as dataref
import tests.unit.util as util


def create_configurations() -> storage.LFSConfigurations:
    return storage.LFSConfigurations(storage_dir_path="/tmp/",)


def get_cache_filepath(
    configurations: storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> pathlib.Path:
    return configurations.storage_dir_path.joinpath(f"{dataset_id}-{dataset_version}.mdb")


def test_storage_submit() -> None:
    range_size = 10
    dataset_id = "range-dataset"
    dataset_version = "0"
    dataset = util.create_tf_dataset_from_range(range_size=range_size)
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
    def make_dataref(range_size: int) -> dataref.LFSDataRef:
        return util.create_tf_dataset_from_range(range_size=range_size)

    original_data_stream = make_dataref(range_size=original_range_size).stream()
    assert original_data_stream.length == original_range_size
    data_generator = original_data_stream.iterator_fn()
    for idx in range(original_range_size):
        assert idx == next(data_generator)

    updated_data_stream = make_dataref(range_size=updated_range_size).stream()
    assert updated_data_stream.length == original_range_size

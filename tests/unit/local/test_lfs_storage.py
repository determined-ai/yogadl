import pathlib

import tensorflow as tf

import yogadl.dataref.local_lmdb_dataref as dataref
import yogadl.storage.lfs_storage as storage


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

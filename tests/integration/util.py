import yogadl.storage.lfs_storage as lfs_storage


def cleanup_lfs_storage(
    configurations: lfs_storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    cache_filepath = (
        configurations.storage_dir_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )
    cache_filepath.unlink()

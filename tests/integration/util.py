import yogadl.storage.lfs_storage as lfs_storage


def cleanup_lfs_storage(
    configurations: lfs_storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    cache_filepath = configurations.storage_dir_path.joinpath(f"{dataset_id}-{dataset_version}.mdb")
    cache_filepath.unlink()

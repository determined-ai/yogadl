import tests.unit.util as util

from yogadl import dataref


def test_lfs_dataref_from_checkpoint() -> None:
    range_size = 10
    checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    lfs_dataref = dataref.LMDBDataRef(cache_filepath=checkpoint_path)
    stream = lfs_dataref.stream()

    for _ in range(3):
        idx = 0
        data_generator = stream.iterator_fn()
        for data in data_generator:
            assert data == idx
            idx += 1
        assert idx == range_size


def test_lfs_dataref_with_offset() -> None:
    range_size = 10
    offset = 5
    checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    lfs_dataref = dataref.LMDBDataRef(cache_filepath=checkpoint_path)
    stream = lfs_dataref.stream(start_offset=offset)

    for epoch in range(3):
        idx = 5 if epoch == 0 else 0
        data_generator = stream.iterator_fn()
        for data in data_generator:
            assert data == idx
            idx += 1
        assert idx == range_size

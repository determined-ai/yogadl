import yogadl.dataref.lfs_dataref as dataref
import tests.unit.util as util


def test_lfs_dataref_from_checkpoint() -> None:
    range_size = 10
    checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    lfs_dataref = dataref.LfsDataRef(cache_filepath=checkpoint_path)
    stream = lfs_dataref.stream()

    for _ in range(3):
        idx = 0
        data_generator = stream.iterator_fn()
        while True:
            try:
                assert idx == next(data_generator)
            except StopIteration:
                assert idx == range_size
                break
            finally:
                idx += 1


def test_lfs_dataref_with_offset() -> None:
    range_size = 10
    offset = 5
    checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    lfs_dataref = dataref.LfsDataRef(cache_filepath=checkpoint_path)
    stream = lfs_dataref.stream(start_offset=offset)

    for epoch in range(3):
        idx = 5 if epoch == 0 else 0
        data_generator = stream.iterator_fn()
        while True:
            try:
                assert idx == next(data_generator)
            except StopIteration:
                assert idx == range_size
                break
            finally:
                idx += 1

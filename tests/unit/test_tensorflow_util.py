import os
import pathlib

import yogadl.tensorflow_util as tensorflow_util
import tests.unit.util as util


def test_read_tf_dataset() -> None:
    range_size = 10
    dataset = util.create_tf_dataset_from_range(range_size=range_size)
    yield_output = list(tensorflow_util.read_tf_dataset(dataset=dataset))
    original_dataset = range(range_size)
    assert len(original_dataset) == len(yield_output)
    for original_data, yielded_data in zip(original_dataset, yield_output):
        assert original_data == yielded_data


def test_serialize_tf_dataset_to_lmdb_metadata() -> None:
    range_size = 10
    dataset = util.create_tf_dataset_from_range(range_size=range_size)
    checkpoint_path = pathlib.Path("/tmp/test_lmdb_checkpoint.mdb")
    if checkpoint_path.exists():
        os.unlink(str(checkpoint_path))
    assert not checkpoint_path.exists()

    dataset_entries = tensorflow_util.serialize_tf_dataset_to_lmdb(
        dataset=dataset, checkpoint_path=checkpoint_path,
    )
    assert dataset_entries == range_size
    assert checkpoint_path.exists()

from typing import List

import tensorflow as tf

import tests.unit.util as util  # noqa: I202, I100

import yogadl


def shard_and_get_keys(
    lmdb_reader: yogadl.LmdbAccess, shard_index: int, world_size: int, sequential: bool
) -> List[bytes]:
    keys = lmdb_reader.get_keys()
    keys = yogadl.shard_keys(
        keys=keys, shard_index=shard_index, world_size=world_size, sequential=sequential,
    )
    return keys


def convert_int_to_byte_string(input_int: int) -> bytes:
    return u"{:08}".format(input_int).encode("ascii")


def test_lmdb_access_keys() -> None:
    range_size = 10
    lmdb_reader = yogadl.LmdbAccess(
        lmdb_path=util.create_lmdb_checkpoint_using_range(range_size=range_size)
    )
    keys = lmdb_reader.get_keys()
    assert len(keys) == range_size
    for idx, key in enumerate(keys):
        assert convert_int_to_byte_string(idx) == key


def test_lmdb_access_keys_sequential_shard() -> None:
    range_size = 10
    world_size = 3
    lmdb_checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    key_shards = []
    for shard_id in range(world_size):
        lmdb_reader = yogadl.LmdbAccess(lmdb_path=lmdb_checkpoint_path)
        key_shards.append(
            shard_and_get_keys(
                lmdb_reader=lmdb_reader,
                shard_index=shard_id,
                world_size=world_size,
                sequential=True,
            )
        )

    merged_keys = []
    for key_shard in key_shards:
        merged_keys.extend(key_shard)

    assert len(merged_keys) == range_size
    for idx, key in enumerate(merged_keys):
        assert convert_int_to_byte_string(idx) == key


def test_lmdb_access_keys_non_sequential_shard() -> None:
    range_size = 10
    world_size = 3
    lmdb_checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    key_shards = []
    for shard_id in range(world_size):
        lmdb_reader = yogadl.LmdbAccess(lmdb_path=lmdb_checkpoint_path)
        key_shards.append(
            shard_and_get_keys(
                lmdb_reader=lmdb_reader,
                shard_index=shard_id,
                world_size=world_size,
                sequential=False,
            )
        )

    merged_keys = []
    for idx in range(len(key_shards[0])):
        for key_shard in key_shards:
            if idx < len(key_shard):
                merged_keys.append(key_shard[idx])

    assert len(merged_keys) == range_size
    for idx, key in enumerate(merged_keys):
        assert convert_int_to_byte_string(idx) == key


def test_lmdb_access_shuffle() -> None:
    range_size = 10
    seed_one = 41
    seed_two = 421
    lmdb_checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)

    lmdb_reader_one = yogadl.LmdbAccess(lmdb_path=lmdb_checkpoint_path)
    keys_one = lmdb_reader_one.get_keys()
    keys_one = yogadl.shuffle_keys(keys=keys_one, seed=seed_one)

    lmdb_reader_two = yogadl.LmdbAccess(lmdb_path=lmdb_checkpoint_path)
    keys_two = lmdb_reader_two.get_keys()
    keys_two = yogadl.shuffle_keys(keys=keys_two, seed=seed_one)

    lmdb_reader_three = yogadl.LmdbAccess(lmdb_path=lmdb_checkpoint_path)
    keys_three = lmdb_reader_three.get_keys()
    keys_three = yogadl.shuffle_keys(keys=keys_three, seed=seed_two)

    assert keys_one == keys_two
    assert keys_one != keys_three


def test_lmdb_access_read_values() -> None:
    range_size = 10
    lmdb_checkpoint_path = util.create_lmdb_checkpoint_using_range(range_size=range_size)
    lmdb_reader = yogadl.LmdbAccess(lmdb_path=lmdb_checkpoint_path)
    keys = lmdb_reader.get_keys()

    for idx, key in enumerate(keys):
        assert lmdb_reader.read_value_by_key(key=key) == idx


def test_lmdb_access_shapes_and_types() -> None:
    range_size = 10
    lmdb_reader = yogadl.LmdbAccess(
        lmdb_path=util.create_lmdb_checkpoint_using_range(range_size=range_size)
    )
    matching_dataset = tf.data.Dataset.range(range_size)
    assert lmdb_reader.get_shapes() == tf.data.get_output_shapes(matching_dataset)
    assert lmdb_reader.get_types() == tf.data.get_output_types(matching_dataset)

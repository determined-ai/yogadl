import pathlib
from typing import List, Optional

import yogadl.core as core
import yogadl.keys_operator as keys_operator
import yogadl.lmdb_handler as lmdb_handler


class LMDBDataRef(core.DataRef):
    def __init__(self, cache_filepath: pathlib.Path):
        self._lmdb_access = lmdb_handler.LmdbAccess(lmdb_path=cache_filepath)
        self._keys = self._lmdb_access.get_keys()

    def stream(
        self,
        start_offset: int = 0,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        shard_rank: int = 0,
        number_of_shards: int = 1,
    ) -> core.Stream:
        """
        Create a stream from a cache.
        """
        generated_keys = self._shard_and_shuffle_keys(
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            shard_rank=shard_rank,
            number_of_shards=number_of_shards,
        )

        generator_from_keys = keys_operator.GeneratorFromKeys(
            keys=generated_keys,
            initial_offset=start_offset,
            read_val_from_key_fn=self._lmdb_access.read_value_by_key,
        )

        return core.Stream(
            iterator_fn=generator_from_keys.instantiate_generator,
            length=len(generated_keys),
            output_types=self._lmdb_access.get_types(),
            output_shapes=self._lmdb_access.get_shapes(),
        )

    def __len__(self) -> int:
        return len(self._keys)

    def _shard_and_shuffle_keys(
        self, shuffle: bool, shuffle_seed: Optional[int], shard_rank: int, number_of_shards: int,
    ) -> List[bytes]:
        generated_keys = keys_operator.shard_keys(
            keys=self._keys, shard_index=shard_rank, world_size=number_of_shards, sequential=False,
        )
        if shuffle:
            generated_keys = keys_operator.shuffle_keys(keys=generated_keys, seed=shuffle_seed)
        return generated_keys

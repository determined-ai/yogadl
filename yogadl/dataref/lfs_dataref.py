import pathlib
from typing import Any, cast, Generator, Optional

import yogadl.core as core
import yogadl.keys as key_util
import yogadl.lmdb_handler as lmdb_handler


class LfsDataRef(core.DataRef):
    def __init__(self, cache_filepath: pathlib.Path):
        self._lmdb_access = lmdb_handler.LmdbAccess(lmdb_path=cache_filepath)
        self._keys_initialized = False
        self._current_key_index = None  # type: Optional[int]

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
        self._initialize_keys(
            start_offset=start_offset,
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            shard_rank=shard_rank,
            number_of_shards=number_of_shards,
        )

        return core.Stream(
            iterator_fn=self._instantiate_generator,
            length=self.__len__(),
            output_types=self._lmdb_access.get_types(),
            output_shapes=self._lmdb_access.get_shapes(),
        )

    def __len__(self) -> int:
        assert self._keys_initialized
        return len(self._keys)

    def _initialize_keys(
        self,
        start_offset: int,
        shuffle: bool,
        shuffle_seed: Optional[int],
        shard_rank: int,
        number_of_shards: int,
    ) -> None:
        assert start_offset >= 0
        self._keys = self._lmdb_access.get_keys()
        self._keys = key_util.shard_keys(
            keys=self._keys, shard_index=shard_rank, world_size=number_of_shards, sequential=False,
        )
        if shuffle:
            self._keys = key_util.shuffle_keys(keys=self._keys, seed=shuffle_seed)
        self._current_key_index = start_offset % len(self._lmdb_access.get_keys())
        self._keys_initialized = True

    def _instantiate_generator(self) -> Generator[Any, None, None]:
        self._current_key_index = cast(int, self._current_key_index)
        while self._current_key_index < len(self._keys):
            yield self._lmdb_access.read_value_by_key(key=self._keys[self._current_key_index])
            self._current_key_index += 1
        self._current_key_index = 0

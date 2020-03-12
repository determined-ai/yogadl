# Copyright 2020 Determined AI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import pathlib
from typing import List, Optional

import yogadl


class LMDBDataRef(yogadl.DataRef):
    def __init__(self, cache_filepath: pathlib.Path):
        self._lmdb_access = yogadl.LmdbAccess(lmdb_path=cache_filepath)
        self._keys = self._lmdb_access.get_keys()

    def stream(
        self,
        start_offset: int = 0,
        shuffle: bool = False,
        shuffle_seed: Optional[int] = None,
        shard_rank: int = 0,
        number_of_shards: int = 1,
    ) -> yogadl.Stream:
        """
        Create a stream from a cache.
        """
        generated_keys = self._shard_and_shuffle_keys(
            shuffle=shuffle,
            shuffle_seed=shuffle_seed,
            shard_rank=shard_rank,
            number_of_shards=number_of_shards,
        )

        generator_from_keys = yogadl.GeneratorFromKeys(
            keys=generated_keys,
            initial_offset=start_offset,
            read_val_from_key_fn=self._lmdb_access.read_value_by_key,
        )

        return yogadl.Stream(
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
        generated_keys = yogadl.shard_keys(
            keys=self._keys, shard_index=shard_rank, world_size=number_of_shards, sequential=False,
        )
        if shuffle:
            generated_keys = yogadl.shuffle_keys(keys=generated_keys, seed=shuffle_seed)
        return generated_keys

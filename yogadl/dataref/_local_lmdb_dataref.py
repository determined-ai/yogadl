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
        skip_shuffle_at_epoch_end: bool = False,
        shuffle_seed: Optional[int] = None,
        shard_rank: int = 0,
        num_shards: int = 1,
        drop_shard_remainder: bool = False,
    ) -> yogadl.Stream:
        """
        Create a stream from a cache.
        """
        if shuffle and not skip_shuffle_at_epoch_end:
            assert shuffle_seed is not None, (
                "Please set `shuffle_seed` if enabling `shuffle` and not enabling "
                "`skip_shuffle_at_epoch_end`."
            )

        generated_keys = self._shard_keys(
            shard_rank=shard_rank, num_shards=num_shards, drop_shard_remainder=drop_shard_remainder,
        )

        generator_from_keys = yogadl.GeneratorFromKeys(
            keys=generated_keys,
            initial_offset=start_offset,
            read_val_from_key_fn=self._lmdb_access.read_value_by_key,
            shuffle_at_start=shuffle,
            shuffle_after_epoch=shuffle and not skip_shuffle_at_epoch_end,
            shuffle_seed=shuffle_seed,
        )

        return yogadl.Stream(
            iterator_fn=generator_from_keys.instantiate_generator,
            length=len(generated_keys),
            output_types=self._lmdb_access.get_types(),
            output_shapes=self._lmdb_access.get_shapes(),
        )

    def __len__(self) -> int:
        return len(self._keys)

    def _shard_keys(
        self, shard_rank: int, num_shards: int, drop_shard_remainder: bool
    ) -> List[bytes]:
        generated_keys = yogadl.shard_keys(
            keys=self._keys,
            shard_index=shard_rank,
            num_shards=num_shards,
            sequential=False,
            drop_shard_remainder=drop_shard_remainder,
        )

        return generated_keys

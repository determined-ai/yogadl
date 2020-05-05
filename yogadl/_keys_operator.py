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
import copy
from typing import Any, Callable, Generator, List, Optional

import numpy as np


def sequential_shard(keys: List[bytes], shard_index: int, num_shards: int) -> List[bytes]:
    num_keys = len(keys) // num_shards
    if shard_index < len(keys) % num_shards:
        num_keys += 1
    start_index = num_keys * shard_index + min(len(keys) % num_shards, shard_index)
    return keys[start_index : start_index + num_keys]


def non_sequential_shard(keys: List[bytes], shard_index: int, num_shards: int) -> List[bytes]:
    key_indexes = list(range(shard_index, len(keys), num_shards))
    return [keys[idx] for idx in key_indexes]


def shard_keys(
    keys: List[bytes],
    shard_index: int,
    num_shards: int,
    sequential: bool = False,
    drop_shard_remainder: bool = False,
) -> List[bytes]:
    assert shard_index >= 0, "Shard index must be greater or equal to zero."
    assert shard_index < num_shards, "Shard index must be less than num_shards."

    if drop_shard_remainder:
        assert len(keys) >= num_shards, f"Too few keys to shard across {num_shards} ranks."
        keys = keys[: len(keys) - (len(keys) % num_shards)]

    if sequential:
        return sequential_shard(keys=keys, shard_index=shard_index, num_shards=num_shards)
    else:
        return non_sequential_shard(keys=keys, shard_index=shard_index, num_shards=num_shards)


def shuffle_keys(keys: List[bytes], seed: Optional[int] = None) -> List[bytes]:
    shuffler = np.random.RandomState(seed)
    shuffler.shuffle(keys)
    return keys


class GeneratorFromKeys:
    def __init__(
        self,
        keys: List[bytes],
        initial_offset: int,
        read_val_from_key_fn: Callable,
        shuffle_at_start: bool,
        shuffle_after_epoch: bool,
        shuffle_seed: Optional[int],
    ) -> None:
        assert initial_offset >= 0
        self._keys = keys
        self._initial_offset = initial_offset % len(self._keys)
        self._current_epoch = initial_offset // len(self._keys)
        self._read_val_from_key_fn = read_val_from_key_fn
        self._shuffle_enabled = shuffle_at_start
        self._shuffle_after_epoch = shuffle_after_epoch
        self._shuffle_seed = shuffle_seed
        self._initial_epoch = True

        self._validate_args()

    def _validate_args(self) -> None:
        if self._shuffle_after_epoch:
            assert self._shuffle_enabled, "`shuffle` must be enabled to use `shuffle_after_epoch`."
            assert (
                self._shuffle_seed is not None
            ), "`shuffle_seed` must be set to use `shuffle_after_epoch`."

    def instantiate_generator(self) -> Generator[Any, None, None]:
        keys = self._shuffle_keys() if self._shuffle_enabled else self._keys
        self._current_epoch += 1

        key_index = self._initial_offset if self._initial_epoch else 0
        self._initial_epoch = False

        while key_index < len(keys):
            yield self._read_val_from_key_fn(keys[key_index])
            key_index += 1

    def _shuffle_keys(self) -> List[bytes]:
        shuffle_seed = self._shuffle_seed
        if self._current_epoch > 0 and self._shuffle_after_epoch:
            assert shuffle_seed is not None
            shuffle_seed += self._current_epoch

        return shuffle_keys(keys=copy.deepcopy(self._keys), seed=shuffle_seed)

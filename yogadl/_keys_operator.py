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
import random
from typing import Any, Callable, Generator, List, Optional


def sequential_shard(keys: List[bytes], shard_index: int, world_size: int) -> List[bytes]:
    num_keys = len(keys) // world_size
    if shard_index < len(keys) % world_size:
        num_keys += 1
    start_index = num_keys * shard_index + min(len(keys) % world_size, shard_index)
    return keys[start_index : start_index + num_keys]


def non_sequential_shard(keys: List[bytes], shard_index: int, world_size: int) -> List[bytes]:
    key_indexes = list(range(shard_index, len(keys), world_size))
    return [keys[idx] for idx in key_indexes]


def shard_keys(
    keys: List[bytes], shard_index: int, world_size: int, sequential: bool = False
) -> List[bytes]:
    assert shard_index >= 0, "Shard index must be greater or equal to zero."
    assert shard_index < world_size, "Shard index must be less than world_size."
    if sequential:
        return sequential_shard(keys=keys, shard_index=shard_index, world_size=world_size)
    else:
        return non_sequential_shard(keys=keys, shard_index=shard_index, world_size=world_size)


def shuffle_keys(keys: List[bytes], seed: Optional[int] = None) -> List[bytes]:
    if seed:
        random.seed(seed)
    random.shuffle(keys)
    return keys


class GeneratorFromKeys:
    def __init__(
        self, keys: List[bytes], initial_offset: int, read_val_from_key_fn: Callable
    ) -> None:
        assert initial_offset >= 0
        self._keys = keys
        self._initial_offset = initial_offset % len(self._keys)
        self._read_val_from_key_fn = read_val_from_key_fn
        self._initial_epoch = True

    def instantiate_generator(self) -> Generator[Any, None, None]:
        key_index = self._initial_offset if self._initial_epoch else 0
        self._initial_epoch = False
        while key_index < len(self._keys):
            yield self._read_val_from_key_fn(self._keys[key_index])
            key_index += 1

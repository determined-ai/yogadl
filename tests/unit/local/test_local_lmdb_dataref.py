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

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
import os
import pathlib

import tensorflow as tf

from yogadl import tensorflow


def test_read_tf_dataset() -> None:
    range_size = 10
    dataset = tf.data.Dataset.range(range_size)
    yield_output = list(tensorflow.read_tf_dataset(dataset=dataset, tf_config=None))
    original_dataset = range(range_size)
    assert len(original_dataset) == len(yield_output)
    for original_data, yielded_data in zip(original_dataset, yield_output):
        assert original_data == yielded_data  # type: ignore


def test_serialize_tf_dataset_to_lmdb_metadata() -> None:
    range_size = 10
    dataset = tf.data.Dataset.range(range_size)
    checkpoint_path = pathlib.Path("/tmp/test_lmdb_checkpoint.mdb")
    if checkpoint_path.exists():
        os.unlink(str(checkpoint_path))
    assert not checkpoint_path.exists()

    dataset_entries = tensorflow.serialize_tf_dataset_to_lmdb(
        dataset=dataset, checkpoint_path=checkpoint_path, tf_config=None
    )
    assert dataset_entries == range_size
    assert checkpoint_path.exists()

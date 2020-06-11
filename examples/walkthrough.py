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
"""
This file contains the code snippets which are used in the Examples section of
the documentation. It is compiled in one place here for testing purposes.

The START and END comments denote sections of this file which will become code
snippets in the documentation.
"""

# START creating a yogadl.Storage
import os
import yogadl
import yogadl.storage

# Create a yogadl.Storage object backed by the local filesystem.
storage_path = "/tmp/yogadl_cache"
os.makedirs(storage_path, exist_ok=True)
lfs_config = yogadl.storage.LFSConfigurations(storage_path)
storage = yogadl.storage.LFSStorage(lfs_config)
# END creating a yogadl.Storage


# START storing a dataset
import tensorflow as tf

# Create a dataset we can store.
records = tf.data.Dataset.range(10)

# Store this dataset as "range" version "1.0".
storage.submit(records, "range", "1.0")
# END storing a dataset


# START fetching a dataset
import yogadl.tensorflow

# Get the DataRef.
dataref = storage.fetch("range", "1.0")

# Tell the DataRef how to stream the dataset.
stream = dataref.stream(start_offset=5, shuffle=True, shuffle_seed=777)

# Interpret the stream as a tensorflow dataset
records = yogadl.tensorflow.make_tf_dataset(stream)

# It's a real tf.data.Dataset; you can use normal tf.data operations on it.
batches = records.repeat(3).batch(5)

# (this part requires TensorFlow >= 2.0)
for batch in batches:
    print(batch)
# END fetching a dataset


# START can I get the same features in fewer steps?
@storage.cacheable("range", "2.0")
def make_records():
    print("Cache not found, making range v2 dataset...")
    records = tf.data.Dataset.range(10).map(lambda x: 2*x)
    return records

# Follow the same steps as before.
dataref = make_records()
stream = dataref.stream()
records = yogadl.tensorflow.make_tf_dataset(stream)
batches = records.repeat(3).batch(5)

for batch in batches:
    print(batch)
# END can I get the same features in fewer steps?

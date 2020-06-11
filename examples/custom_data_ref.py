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
If you have an advanced use case, like generating data on an external machine
and streaming it to another machine for training or something, and you would
like to integrate with a platform that allows you to submit your dataset as a
``yogadl.DataRef``, you can implement a custom ``yogadl.DataRef``. By
implementing the ``yogadl.DataRef`` interface, you can fully customize the
behavior of how the platform interacts with your dataset. Here is a toy example
of what that might look like:
"""

# INCLUDE IN DOCS
import os
import yogadl
import yogadl.tensorflow
import tensorflow as tf

class RandomDataRef(yogadl.DataRef):
    """
    A DataRef to a a non-reproducible dataset that just produces random
    int32 values.
    """

    def __len__(self):
        return 10

    def stream(
        self,
        start_offset = 0,
        shuffle = False,
        skip_shuffle_at_epoch_end = False,
        shuffle_seed = None,
        shard_rank = 0,
        num_shards = 1,
        drop_shard_remainder = False,
    ) -> yogadl.Stream:
        """
        For custom DataRefs, .stream() will often be a pretty beefy
        function. This example simplifies it by assuming that the dataset
        is non-reproducible, meaning that shuffle and shuffle_seed
        arguments are meaningless, and the shard_rank is only used to
        determine how many records will be yielded during each epoch.
        """

        first_epoch = True

        def iterator_fn():
            nonlocal first_epoch
            if first_epoch:
                first_epoch = False
                start = start_offset + shard_rank
            else:
                start = shard_rank

            if drop_shard_remainder:
                end = len(self) - (len(self) % num_shards)
            else:
                end = len(self)

            for _ in range(start, end, num_shards):
                # Make a uint32 out of 4 random bytes
                r = os.urandom(4)
                yield r[0] + (r[1] << 8) + (r[2] << 16) + (r[3] << 24)

        # Since we will later convert to tf.data.Dataset,
        # we will supply output_types and shapes.
        return yogadl.Stream(
            iterator_fn,
            len(self),
            output_types=tf.uint32,
            output_shapes=tf.TensorShape([])
        )

dataref = RandomDataRef()
stream = dataref.stream()
records = yogadl.tensorflow.make_tf_dataset(stream)
batches = records.batch(5)
for batch in batches:
    print(batch)

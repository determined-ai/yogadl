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
#
# `normalize_img()` and the model definition are derived from the TensorFlow
# documentation: https://www.tensorflow.org/datasets/keras_example
#
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
End-to-end training example:

Here is an example of how you might use YogaDL to train on the second half of
an MNIST dataset. This illustrates the ability to continue training mid-dataset
that is simply not natively possible with tf.keras. Without YogaDL, you could
imitate this behavior using tf.data.Dataset.skip(N), but that is
prohibitively expensive for large values of N.
"""

# INCLUDE IN DOCS
import math
import os
import tensorflow as tf
import tensorflow_datasets as tfds
import yogadl
import yogadl.tensorflow
import yogadl.storage

BATCH_SIZE = 32

# Configure the yogadl storage.
storage_path = "/tmp/yogadl_cache"
os.makedirs(storage_path, exist_ok=True)
lfs_config = yogadl.storage.LFSConfigurations(storage_path)
storage = yogadl.storage.LFSStorage(lfs_config)

@storage.cacheable("mnist", "1.0")
def make_data():
    mnist = tfds.image.MNIST()
    mnist.download_and_prepare()
    dataset = mnist.as_dataset(as_supervised=True)["train"]

    # Apply dataset transformations from the TensorFlow docs:
    # (https://www.tensorflow.org/datasets/keras_example)

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    return dataset.map(normalize_img)

# Get the DataRef from the storage via the decorated function.
dataref = make_data()

# Stream the dataset starting halfway through it.
num_batches = math.ceil(len(dataref) / BATCH_SIZE)
batches_to_skip = num_batches // 2
records_to_skip = batches_to_skip * BATCH_SIZE
stream = dataref.stream(
    start_offset=records_to_skip, shuffle=True, shuffle_seed=777
)

# Convert the stream to a tf.data.Dataset object.
dataset = yogadl.tensorflow.make_tf_dataset(stream)

# Apply normal data augmentation and prefetch steps.
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

# Model is straight from the TensorFlow docs:
# https://www.tensorflow.org/datasets/keras_example
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)
model.fit(dataset)

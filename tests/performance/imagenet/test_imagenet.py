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
# `dataset_parser()` and `make_dataset_from_tf_records()` are derived from:
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
import os
import pathlib
import time
from typing import Any, Tuple

import tensorflow as tf

from tests.performance.imagenet import resnet_preprocessing

from yogadl import dataref, storage, tensorflow


def cleanup_lfs_storage(
    configurations: storage.LFSConfigurations, dataset_id: str, dataset_version: str
) -> None:
    cache_filepath = (
        configurations.storage_dir_path.joinpath(dataset_id)
        .joinpath(dataset_version)
        .joinpath("cache.mdb")
    )
    if cache_filepath.exists():
        cache_filepath.unlink()


def dataset_parser(value: Any) -> Any:
    """
    Based on [1].

    [1] - https://github.com/tensorflow/tpu/blob/master/models/
            experimental/resnet50_keras/imagenet_input.py
    """
    keys_to_features = {
        "image/encoded": tf.FixedLenFeature((), tf.string, ""),
        "image/format": tf.FixedLenFeature((), tf.string, "jpeg"),
        "image/class/label": tf.FixedLenFeature([], tf.int64, -1),
        "image/class/text": tf.FixedLenFeature([], tf.string, ""),
        "image/object/bbox/xmin": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymin": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/xmax": tf.VarLenFeature(dtype=tf.float32),
        "image/object/bbox/ymax": tf.VarLenFeature(dtype=tf.float32),
        "image/object/class/label": tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed["image/encoded"], shape=[])

    image = resnet_preprocessing.preprocess_image(  # type: ignore
        image_bytes=image_bytes, is_training=True, use_bfloat16=False
    )

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    label = tf.cast(
        tf.cast(tf.reshape(parsed["image/class/label"], shape=[1]), dtype=tf.int32) - 1,
        dtype=tf.float32,
    )

    return image, label


def make_dataset_from_tf_records(data_dir: pathlib.Path, training: bool) -> tf.data.Dataset:
    """
    Based on [1].

    [1] - https://github.com/tensorflow/tpu/blob/master/models/
            experimental/resnet50_keras/imagenet_input.py
    """
    # Process 100 out of 1024 record files.
    file_pattern = os.path.join(
        str(data_dir), "train/train-003*" if training else "validation/validation-*"
    )
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=False)

    def fetch_tf_record_file(filename: str) -> tf.data.TFRecordDataset:
        buffer_size = 8 * 1024 * 1024  # 8 MiB per file
        tf_record_dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
        return tf_record_dataset

    dataset = dataset.interleave(
        fetch_tf_record_file, cycle_length=16, num_parallel_calls=tf.data.experimental.AUTOTUNE
    )

    return dataset


def read_dataset(dataset: tf.data.Dataset) -> Tuple[float, int]:
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(
            dataset_parser, batch_size=1, num_parallel_batches=2, drop_remainder=True
        )
    )
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    next_element_from_dataset = dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        data_samples = 0
        dataset_read_start_time = time.time()

        while True:
            try:
                sess.run(next_element_from_dataset)
                data_samples += 1
            except tf.errors.OutOfRangeError:
                break

        dataset_read_time = time.time() - dataset_read_start_time

    return dataset_read_time, data_samples


def compare_performance_tf_record_dataset(data_dir: pathlib.Path) -> None:
    config = storage.LFSConfigurations(storage_dir_path="/tmp/")
    lfs_storage = storage.LFSStorage(configurations=config)

    dataset_id = "imagenet-train"
    dataset_version = "0"
    training = True

    cleanup_lfs_storage(
        configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
    )

    @lfs_storage.cacheable(dataset_id=dataset_id, dataset_version=dataset_version)
    def make_dataset() -> dataref.LMDBDataRef:
        return make_dataset_from_tf_records(data_dir=data_dir, training=training)  # type: ignore

    cache_creation_start_time = time.time()
    stream_from_cache = make_dataset().stream()
    cache_creation_time = time.time() - cache_creation_start_time
    print(f"Cache creation took: {cache_creation_time} seconds.")

    dataset_from_stream = tensorflow.make_tf_dataset(stream_from_cache)
    cache_read_time, cache_data_items = read_dataset(dataset=dataset_from_stream)
    print(f"Cache read took: {cache_read_time} seconds.")

    original_dataset_read_time, original_data_items = read_dataset(
        dataset=make_dataset_from_tf_records(data_dir=data_dir, training=training)
    )
    print(f"Original read took: {original_dataset_read_time} seconds.")

    assert cache_data_items == original_data_items


def test_lfs_imagenet() -> None:
    # This test requires that the imagenet dataset be present in TFRecords format.
    error_message = (
        "Please set `IMAGENET_DIRECTORY` environment variable to "
        "be the directory containing the TFRecords."
    )

    imagenet_directory_path = os.environ.get("IMAGENET_DIRECTORY")
    assert imagenet_directory_path, error_message
    imagenet_directory = pathlib.Path(imagenet_directory_path)
    assert imagenet_directory.is_dir(), error_message

    compare_performance_tf_record_dataset(data_dir=imagenet_directory)

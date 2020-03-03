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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow.compat.v1 as tf

IMAGE_SIZE = 224
CROP_PADDING = 32


def distorted_bounding_box_crop(
    image_bytes,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
    scope=None,
):
    with tf.name_scope(scope, "distorted_bounding_box_crop", [image_bytes, bbox]):
        shape = tf.image.extract_jpeg_shape(image_bytes)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True,
        )
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

        return image


def _at_least_x_are_equal(a, b, x):
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
        area_range=(0.08, 1.0),
        max_attempts=10,
        scope=None,
    )
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes),
        lambda: tf.image.resize_bicubic(
            [image], [IMAGE_SIZE, IMAGE_SIZE]  # pylint: disable=g-long-lambda
        )[0],
    )

    return image


def _decode_and_center_crop(image_bytes):
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (
            (IMAGE_SIZE / (IMAGE_SIZE + CROP_PADDING))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize_bicubic([image], [IMAGE_SIZE, IMAGE_SIZE])[0]

    return image


def _flip(image):
    image = tf.image.random_flip_left_right(image)
    return image


def preprocess_for_train(image_bytes, use_bfloat16):
    image = _decode_and_random_crop(image_bytes)
    image = _flip(image)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image


def preprocess_for_eval(image_bytes, use_bfloat16):
    image = _decode_and_center_crop(image_bytes)
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    return image


def preprocess_image(image_bytes, is_training=False, use_bfloat16=False):
    if is_training:
        return preprocess_for_train(image_bytes, use_bfloat16)
    else:
        return preprocess_for_eval(image_bytes, use_bfloat16)

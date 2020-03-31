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
from typing import Any, Generator, Optional, Tuple

import tensorflow as tf

import yogadl


def read_tf_dataset_eager_mode(dataset: tf.data.Dataset) -> Generator[Tuple[Any, bool], None, None]:
    # TODO: If repeat() has been applied we will hit an infinite
    # loop here. Probably best approach is to include log message
    # specifying how many data items we have read and this should
    # alert the user if we are stuck in an infinite loop.
    for next_element in dataset.as_numpy_iterator():
        yield next_element


def read_tf_dataset_graph_mode(
    dataset: tf.data.Dataset, tf_config: Optional[tf.compat.v1.ConfigProto]
) -> Generator[Tuple[Any, bool], None, None]:
    get_next_element = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session(config=tf_config) as sess:
        while True:
            try:
                # TODO: If repeat() has been applied we will hit an infinite
                # loop here. Probably best approach is to include log message
                # specifying how many data items we have read and this should
                # alert the user if we are stuck in an infinite loop.
                yield sess.run(get_next_element)
            except tf.errors.OutOfRangeError:
                break


def read_tf_dataset(
    dataset: tf.data.Dataset, tf_config: Optional[tf.compat.v1.ConfigProto]
) -> Generator[Tuple[Any, bool], None, None]:
    if tf.executing_eagerly():
        return read_tf_dataset_eager_mode(dataset)
    else:
        return read_tf_dataset_graph_mode(dataset, tf_config=tf_config)


def serialize_tf_dataset_to_lmdb(
    dataset: tf.data.Dataset,
    checkpoint_path: pathlib.Path,
    tf_config: Optional[tf.compat.v1.ConfigProto],
    write_frequency: int = 5000,
) -> int:
    assert isinstance(dataset, tf.data.Dataset)
    return yogadl.serialize_generator_to_lmdb(
        dataset_generator=read_tf_dataset(dataset=dataset, tf_config=tf_config),
        data_shapes=tf.compat.v1.data.get_output_shapes(dataset),
        data_types=tf.compat.v1.data.get_output_types(dataset),
        lmdb_path=checkpoint_path,
        write_frequency=write_frequency,
    )


def make_tf_dataset(stream: yogadl.Stream) -> tf.data.Dataset:
    """
    Produce a tf.data.Dataset from a yogadl.Stream.
    """
    return tf.data.Dataset.from_generator(
        stream.iterator_fn, output_types=stream.output_types, output_shapes=stream.output_shapes
    )

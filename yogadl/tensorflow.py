import pathlib
from typing import Any, Generator, Tuple

import tensorflow as tf

import yogadl


def read_tf_dataset_eager_mode(dataset: tf.data.Dataset) -> Generator[Tuple[Any, bool], None, None]:
    # TODO: If repeat() has been applied we will hit an infinite
    # loop here. Probably best approach is to include log message
    # specifying how many data items we have read and this should
    # alert the user if we are stuck in an infinite loop.
    for next_element in dataset.as_numpy_iterator():
        yield next_element


def read_tf_dataset_graph_mode(dataset: tf.data.Dataset) -> Generator[Tuple[Any, bool], None, None]:
    get_next_element = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    with tf.compat.v1.Session() as sess:
        while True:
            try:
                # TODO: If repeat() has been applied we will hit an infinite
                # loop here. Probably best approach is to include log message
                # specifying how many data items we have read and this should
                # alert the user if we are stuck in an infinite loop.
                yield sess.run(get_next_element)
            except tf.errors.OutOfRangeError:
                break


def read_tf_dataset(dataset: tf.data.Dataset) -> Generator[Tuple[Any, bool], None, None]:
    if tf.executing_eagerly():
        return read_tf_dataset_eager_mode(dataset)
    else:
        return read_tf_dataset_graph_mode(dataset)


def serialize_tf_dataset_to_lmdb(
    dataset: tf.data.Dataset, checkpoint_path: pathlib.Path, write_frequency: int = 5000
) -> int:
    assert isinstance(dataset, tf.data.Dataset)
    return yogadl.serialize_generator_to_lmdb(
        dataset_generator=read_tf_dataset(dataset=dataset),
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

import logging
import pathlib
from typing import Any, Generator, Tuple

import tensorflow as tf

import yogadl.check as check
import yogadl.core as core
import yogadl.lmdb_handler as lmdb_handler


def read_tf_dataset(
    dataset: tf.data.Dataset,
) -> Generator[Tuple[Any, bool], None, None]:
    # TODO: Make this TF2.0 Compatible.
    one_shot_iterator = dataset.make_one_shot_iterator()
    data_exists = True
    with tf.Session() as sess:
        while data_exists:
            try:
                # TODO: If repeat() has been applied we will hit an infinite
                # loop here. Probably best approach is to include log message
                # specifying how many data items we have read and this should
                # alert the user if we are stuck in an infinite loop.
                next_data_item = sess.run(one_shot_iterator.get_next())
            except tf.errors.OutOfRangeError:
                logging.info(f"Reached end of the dataset.")
                next_data_item = None
                data_exists = False
            finally:
                yield next_data_item, data_exists


def get_dataset_types(dataset: tf.data.Dataset) -> Any:
    return tf.data.get_output_types(dataset)


def get_dataset_shapes(dataset: tf.data.Dataset) -> Any:
    tf.data.get_output_shapes(dataset)


def serialize_tf_dataset_to_lmdb(
    dataset: tf.data.Dataset, checkpoint_path: pathlib.Path, write_frequency: int = 5000
) -> int:
    check.is_instance(dataset, tf.data.Dataset)
    return lmdb_handler.serialize_generator_to_lmdb(
        dataset_generator=read_tf_dataset(dataset=dataset),
        data_shapes=get_dataset_shapes(dataset=dataset),
        data_types=get_dataset_types(dataset=dataset),
        checkpoint_path=checkpoint_path,
        write_frequency=write_frequency,
    )


def make_tf_dataset(stream: core.Stream) -> tf.data.Dataset:
    """
    Produce a tf.data.Dataset from a yogadl.Stream.
    """
    return tf.data.Dataset.from_generator(
        stream.generator, output_types=stream.output_types, output_shapes=stream.output_shapes,
    )

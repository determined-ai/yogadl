import tensorflow as tf

import yogadl


def make_tf_dataset(stream: yogadl.Stream) -> tf.data.Dataset:
    """
    Produce a tf.data.Dataset from a yogadl.Stream.
    """
    types = stream.get_output_types()
    shapes = stream.get_output_shapes()
    return tf.data.Dataset.from_generator(
        lambda: iter(stream), output_types=types, output_shapes=shapes
    )

import pathlib
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

import tests.integration.util as util
import yogadl.tensorflow_util as tf_utils
import yogadl.storage.lfs_storage as lfs_storage
import yogadl.dataref.lfs_dataref as lfs_dataref


def make_mnist_train_dataset() -> tf.data.Dataset:
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    mnist_train = mnist_builder.as_dataset(split="train")
    return mnist_train


def test_mnist_single_threaded() -> None:
    config = lfs_storage.LFSConfigurations(storage_dir_path=pathlib.Path("/tmp/"),)
    storage = lfs_storage.LFSStorage(configurations=config)

    dataset_id = "mnist"
    dataset_version = "1"

    @storage.cacheable(dataset_id=dataset_id, dataset_version=dataset_version)
    def make_dataset() -> lfs_dataref.LFSDataRef:
        return make_mnist_train_dataset()

    stream_from_cache = make_dataset().stream()
    dataset_from_stream = tf_utils.make_tf_dataset(stream_from_cache)
    original_dataset = make_mnist_train_dataset()

    next_element_from_stream = dataset_from_stream.make_one_shot_iterator().get_next()
    next_element_from_orig = original_dataset.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        data_samples = 0
        while True:
            try:
                element_from_stream = sess.run(next_element_from_stream)
                element_from_dataset = sess.run(next_element_from_orig)
                assert element_from_stream["label"] == element_from_dataset["label"]
                assert np.array_equal(element_from_stream["image"], element_from_dataset["image"])
                data_samples += 1
            except tf.errors.OutOfRangeError:
                break

    assert data_samples == 60000
    assert stream_from_cache.length == data_samples
    util.cleanup_lfs_storage(
        configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
    )

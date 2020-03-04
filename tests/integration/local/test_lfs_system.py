import numpy as np
import tensorflow as tf

import tests.integration.util as util  # noqa: I202, I100

from yogadl import dataref, storage, tensorflow


def test_mnist_single_threaded() -> None:
    config = storage.LFSConfigurations(storage_dir_path="/tmp/")
    lfs_storage = storage.LFSStorage(configurations=config)

    dataset_id = "mnist"
    dataset_version = "1"

    util.cleanup_lfs_storage(
        configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
    )

    @lfs_storage.cacheable(dataset_id=dataset_id, dataset_version=dataset_version)
    def make_dataset() -> dataref.LMDBDataRef:
        return util.make_mnist_train_dataset()  # type: ignore

    stream_from_cache = make_dataset().stream()
    dataset_from_stream = tensorflow.make_tf_dataset(stream_from_cache)
    original_dataset = util.make_mnist_train_dataset()

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

    assert data_samples == 10000
    assert stream_from_cache.length == data_samples
    util.cleanup_lfs_storage(
        configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
    )

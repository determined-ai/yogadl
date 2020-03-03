import numpy as np
import pytest
import tensorflow as tf
from tl.testing.thread import ThreadAwareTestCase, ThreadJoiner

import tests.integration.util as util  # noqa: I202, I100

import yogadl.dataref.local_lmdb_dataref as lfs_dataref
import yogadl.storage.s3_storage as s3_storage
import yogadl.tensorflow_util as tf_utils


def create_s3_configuration(access_server_port: int) -> s3_storage.S3Configurations:
    return s3_storage.S3Configurations(
        bucket="yogadl-test",
        bucket_directory_path="integration-tests",
        url=f"ws://localhost:{access_server_port}",
        local_cache_dir="/tmp/",
    )


def worker_using_cacheable(
    config: s3_storage.S3Configurations, dataset_id: str, dataset_version: str
) -> None:
    storage = s3_storage.S3Storage(configurations=config)

    @storage.cacheable(dataset_id=dataset_id, dataset_version=dataset_version)
    def make_dataset() -> lfs_dataref.LMDBDataRef:
        return util.make_mnist_train_dataset()  # type: ignore

    stream_from_cache = make_dataset().stream()
    dataset_from_stream = tf_utils.make_tf_dataset(stream_from_cache)
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


@pytest.mark.gcp  # type: ignore
def test_mnist_single_threaded() -> None:
    dataset_id = "mnist"
    dataset_version = "1"
    config = create_s3_configuration(access_server_port=29243)

    util.cleanup_s3_storage(
        configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
    )

    access_server_handler = util.AccessServerHandler(hostname="localhost", port=29243)
    access_server_handler.run_server_in_thread()

    try:
        worker_using_cacheable(
            config=config, dataset_id=dataset_id, dataset_version=dataset_version
        )
    finally:
        access_server_handler.stop_server()
        util.cleanup_s3_storage(
            configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
        )


class MultiThreadedTests(ThreadAwareTestCase):  # type: ignore
    @pytest.mark.gcp  # type: ignore
    def test_mnist_multi_threaded(self) -> None:
        dataset_id = "mnist"
        dataset_version = "1"
        num_threads = 4

        config = create_s3_configuration(access_server_port=29243)

        util.cleanup_s3_storage(
            configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
        )

        access_server_handler = util.AccessServerHandler(hostname="localhost", port=29243)
        access_server_handler.run_server_in_thread()

        try:
            with ThreadJoiner(60):
                for _ in range(num_threads):
                    self.run_in_thread(
                        lambda: worker_using_cacheable(
                            config=config, dataset_id=dataset_id, dataset_version=dataset_version
                        )
                    )
        finally:
            access_server_handler.stop_server()
            util.cleanup_s3_storage(
                configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
            )

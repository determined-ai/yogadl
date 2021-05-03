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
import pytest
from tl.testing import thread

import tests.integration.util as util  # noqa: I202, I100

from yogadl import dataref, storage, tensorflow


def create_s3_configuration(access_server_port: int) -> storage.S3Configurations:
    return storage.S3Configurations(
        bucket="yogadl-test",
        bucket_directory_path="integration-tests",
        url=f"ws://localhost:{access_server_port}",
        local_cache_dir="/tmp/",
    )


def worker_using_cacheable(
    config: storage.S3Configurations, dataset_id: str, dataset_version: str
) -> None:
    s3_storage = storage.S3Storage(configurations=config)

    @s3_storage.cacheable(dataset_id=dataset_id, dataset_version=dataset_version)
    def make_dataset() -> dataref.LMDBDataRef:
        return util.make_mnist_test_dataset()  # type: ignore

    stream_from_cache = make_dataset().stream()
    dataset_from_stream = tensorflow.make_tf_dataset(stream_from_cache)
    original_dataset = util.make_mnist_test_dataset()

    data_samples = util.compare_datasets(original_dataset, dataset_from_stream)
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


class MultiThreadedTests(thread.ThreadAwareTestCase):  # type: ignore
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
            with thread.ThreadJoiner(60):
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

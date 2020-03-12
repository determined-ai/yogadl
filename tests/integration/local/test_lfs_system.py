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
        return util.make_mnist_test_dataset()  # type: ignore

    stream_from_cache = make_dataset().stream()
    dataset_from_stream = tensorflow.make_tf_dataset(stream_from_cache)
    original_dataset = util.make_mnist_test_dataset()

    data_samples = util.compare_datasets(original_dataset, dataset_from_stream)
    assert data_samples == 10000
    assert stream_from_cache.length == data_samples
    util.cleanup_lfs_storage(
        configurations=config, dataset_id=dataset_id, dataset_version=dataset_version
    )

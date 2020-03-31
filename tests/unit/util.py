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
import asyncio
import pathlib
import threading
from typing import Optional

import tensorflow as tf

from yogadl import rw_coordinator, tensorflow


def create_lmdb_checkpoint_using_range(range_size: int) -> pathlib.Path:
    dataset = tf.data.Dataset.range(range_size)
    checkpoint_path = pathlib.Path("/tmp/test_lmdb_checkpoint.mdb")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    tensorflow.serialize_tf_dataset_to_lmdb(
        dataset=dataset, checkpoint_path=checkpoint_path, tf_config=None
    )

    return checkpoint_path


class AccessServerHandler:
    def __init__(self, hostname: str, port: int) -> None:
        self._access_server = rw_coordinator.RwCoordinatorServer(hostname=hostname, port=port)

        self._thread_running_server = None  # type: Optional[threading.Thread]

    def run_server_in_thread(self) -> None:
        asyncio.get_event_loop().run_until_complete(self._access_server.run_server())
        self._thread_running_server = threading.Thread(target=asyncio.get_event_loop().run_forever)
        self._thread_running_server.start()

    def stop_server(self) -> None:
        self._access_server.stop_server()

        assert self._thread_running_server
        self._thread_running_server.join()

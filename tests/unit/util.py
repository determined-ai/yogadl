import asyncio
import pathlib
import threading
from typing import Optional

import tensorflow as tf

import yogadl.rw_coordinator as rw_coordinator
import yogadl.tensorflow_util as tensorflow_util


def create_lmdb_checkpoint_using_range(range_size: int) -> pathlib.Path:
    dataset = tf.data.Dataset.range(range_size)
    checkpoint_path = pathlib.Path("/tmp/test_lmdb_checkpoint.mdb")
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    tensorflow_util.serialize_tf_dataset_to_lmdb(
        dataset=dataset, checkpoint_path=checkpoint_path,
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

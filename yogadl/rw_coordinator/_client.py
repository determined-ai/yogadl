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
import contextlib
import logging
import pathlib
import urllib.parse
from typing import Generator

import lomond

from yogadl.rw_coordinator import communication_protocol


class RwCoordinatorClient:
    """
    RwCoordinatorClient acquires locks from RwCoordinatorServer.

    RwCoordinatorClient provides read and write locks. An instance of
    AccessServer must be running during lock request.
    """

    def __init__(self, url: str):
        self._url = url

    def _construct_url_request(
        self, storage_type: str, bucket: str, cache_path: pathlib.Path, read_lock: bool
    ) -> str:
        lock_request_url = (
            f"{self._url}/{storage_type}/{bucket}/{str(cache_path)}"
            f"?{urllib.parse.urlencode({'read_lock': read_lock})}"
        )

        logging.debug(f"Generated lock request url: {lock_request_url}.")

        return lock_request_url

    @contextlib.contextmanager
    def _request_lock(
        self, lock_request_url: str, expected_response: str
    ) -> Generator[None, None, None]:
        with lomond.WebSocket(lock_request_url) as socket:
            for event in socket.connect():
                if isinstance(event, lomond.events.Text):
                    assert event.text == expected_response
                    yield
                    socket.close()

    @contextlib.contextmanager
    def read_lock(
        self, storage_type: str, bucket: str, cache_path: pathlib.Path
    ) -> Generator[None, None, None]:
        lock_request_url = self._construct_url_request(
            storage_type=storage_type, bucket=bucket, cache_path=cache_path, read_lock=True,
        )

        try:
            with self._request_lock(
                lock_request_url=lock_request_url,
                expected_response=communication_protocol.READ_LOCK_GRANTED,
            ):
                yield
        except RuntimeError:
            logging.warning(f"Can not reach access server at: {self._url}.")
            raise

    @contextlib.contextmanager
    def write_lock(
        self, storage_type: str, bucket: str, cache_path: pathlib.Path
    ) -> Generator[None, None, None]:
        lock_request_url = self._construct_url_request(
            storage_type=storage_type, bucket=bucket, cache_path=cache_path, read_lock=False,
        )

        try:
            with self._request_lock(
                lock_request_url=lock_request_url,
                expected_response=communication_protocol.WRITE_LOCK_GRANTED,
            ):
                yield
        except RuntimeError:
            logging.warning(f"Can not reach access server at: {self._url}.")
            raise

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
import socket
import ssl
import urllib.parse
from typing import Generator, Optional

import lomond

from yogadl.rw_coordinator import communication_protocol


class CustomSSLWebsocketSession(lomond.session.WebsocketSession):  # type: ignore
    """
    A session class that allows for the TLS verification mode of a WebSocket connection to be
    configured.
    """

    def __init__(
        self,
        socket: lomond.WebSocket,
        skip_verify: bool,
        coordinator_cert_file: Optional[str],
        coordinator_cert_name: Optional[str],
    ) -> None:
        super().__init__(socket)
        self._coordinator_cert_name = coordinator_cert_name

        self.ctx = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)

        if skip_verify:
            return
        self.ctx.verify_mode = ssl.CERT_REQUIRED
        self.ctx.check_hostname = True
        self.ctx.load_default_certs()
        if coordinator_cert_file is not None:
            self.ctx.load_verify_locations(cafile=coordinator_cert_file)

    def _wrap_socket(self, sock: socket.SocketType, host: str) -> socket.SocketType:
        return self.ctx.wrap_socket(sock, server_hostname=self._coordinator_cert_name or host)


class RwCoordinatorClient:
    """
    RwCoordinatorClient acquires locks from RwCoordinatorServer.

    RwCoordinatorClient provides read and write locks. An instance of
    AccessServer must be running during lock request.
    """

    def __init__(
        self,
        url: str,
        skip_verify: bool = False,
        coordinator_cert_file: Optional[str] = None,
        coordinator_cert_name: Optional[str] = None,
    ):
        self._url = url
        self._skip_verify = skip_verify
        self._coordinator_cert_file = coordinator_cert_file
        self._coordinator_cert_name = coordinator_cert_name

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
            for event in socket.connect(
                session_class=lambda socket: CustomSSLWebsocketSession(
                    socket,
                    self._skip_verify,
                    self._coordinator_cert_file,
                    self._coordinator_cert_name,
                )
            ):
                if isinstance(event, lomond.events.ConnectFail):
                    raise ConnectionError(f"connect({self._url}): {event}")
                elif isinstance(event, lomond.events.Text):
                    assert event.text == expected_response
                    yield
                    socket.close()

    @contextlib.contextmanager
    def read_lock(
        self, storage_type: str, bucket: str, cache_path: pathlib.Path
    ) -> Generator[None, None, None]:
        lock_request_url = self._construct_url_request(
            storage_type=storage_type,
            bucket=bucket,
            cache_path=cache_path,
            read_lock=True,
        )

        with self._request_lock(
            lock_request_url=lock_request_url,
            expected_response=communication_protocol.READ_LOCK_GRANTED,
        ):
            yield

    @contextlib.contextmanager
    def write_lock(
        self, storage_type: str, bucket: str, cache_path: pathlib.Path
    ) -> Generator[None, None, None]:
        lock_request_url = self._construct_url_request(
            storage_type=storage_type,
            bucket=bucket,
            cache_path=cache_path,
            read_lock=False,
        )

        with self._request_lock(
            lock_request_url=lock_request_url,
            expected_response=communication_protocol.WRITE_LOCK_GRANTED,
        ):
            yield

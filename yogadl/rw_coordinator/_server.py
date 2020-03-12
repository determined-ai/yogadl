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
import async_generator
import logging
import time
import urllib.parse
from typing import Dict, Generator, Optional

import websockets

from yogadl.rw_coordinator import communication_protocol


class RWLock:
    def __init__(self) -> None:
        self.rw_cond = asyncio.Condition()
        self.writers_waiting = 0
        self.active_readers = 0
        self.active_writer = False

    @async_generator.asynccontextmanager  # type: ignore
    async def read_lock(self) -> Generator[str, None, None]:
        async with self.rw_cond:
            while self.writers_waiting > 0 or self.active_writer:
                await self.rw_cond.wait()
            self.active_readers += 1

        try:
            yield communication_protocol.READ_LOCK_GRANTED
        finally:
            async with self.rw_cond:
                self.active_readers -= 1
                self.rw_cond.notify_all()

    @async_generator.asynccontextmanager  # type: ignore
    async def write_lock(self) -> Generator[str, None, None]:
        async with self.rw_cond:
            self.writers_waiting += 1
            while self.active_readers > 0 or self.active_writer:
                await self.rw_cond.wait()
            self.active_writer = True
            self.writers_waiting -= 1

        try:
            yield communication_protocol.WRITE_LOCK_GRANTED
        finally:
            async with self.rw_cond:
                self.active_writer = False
                self.rw_cond.notify_all()


class RwCoordinatorServer:
    """
    RwCoordinatorServer provides RWlocks for clients that connect to it.

    RwCoordinatorServer provides unique RWLock for each instance of dataset id
    and dataset version. In cases of connection loss, the lock issued
    to the client is revoked.

    RwCoordinatorServer does not support synchronization across multiple instances
    of RwCoordinatorServer. Users should avoid running more than one instance of
    RwCoordinatorServer concurrently.
    """

    def __init__(self, hostname: Optional[str] = None, port: Optional[int] = None) -> None:
        self._hostname = hostname
        self._port = port

        # Used to access rw_locks dictionary.
        self._global_lock = asyncio.Lock()

        # Unique RWLock per cache.
        self._rw_locks = {}  # type: Dict[str, RWLock]

    async def run_server(self) -> None:
        self._server = await websockets.serve(
            self._process_lock_request, self._hostname, self._port
        )

    def stop_server(self) -> None:
        asyncio.get_event_loop().call_soon_threadsafe(asyncio.get_event_loop().stop)

        while asyncio.get_event_loop().is_running():
            time.sleep(1)

        self._server.close()

    @async_generator.asynccontextmanager  # type: ignore
    async def _get_lock(self, rw_lock: RWLock, read_lock: bool) -> Generator[str, None, None]:
        if read_lock:
            async with rw_lock.read_lock() as response:
                yield response
        else:
            async with rw_lock.write_lock() as response:
                yield response

    async def _process_lock_request(
        self, websocket: websockets.server.WebSocketServerProtocol, path: str
    ) -> None:
        parsed_url = urllib.parse.urlparse(path)
        resource = parsed_url.path
        parsed_query = urllib.parse.parse_qs(parsed_url.query)
        assert "read_lock" in parsed_query.keys()
        read_lock = parsed_query["read_lock"][0] == "True"

        async with self._global_lock:
            rw_lock = self._rw_locks.setdefault(resource, RWLock())

        try:
            async with self._get_lock(rw_lock=rw_lock, read_lock=read_lock) as response:
                await websocket.send(response)

                async for _ in websocket:
                    pass

        except websockets.exceptions.ConnectionClosedError:
            logging.warning("Client connection closed unexpectedly.")
            pass

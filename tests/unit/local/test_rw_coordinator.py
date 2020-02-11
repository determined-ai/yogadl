import pathlib
import time
import urllib.parse
from typing import List

import lomond
from tl.testing.thread import ThreadAwareTestCase, ThreadJoiner

import tests.unit.util as test_util

import yogadl.constants as constants
import yogadl.rw_coordinator as rw_coordinator


def read_and_sleep(
    access_client: rw_coordinator.RwCoordinatorClient,
    sleep_time: int,
    bucket: str,
    cache_path: pathlib.Path,
) -> None:
    with access_client.read_lock(
        storage_type=constants.GCS_STORAGE, bucket=bucket, cache_path=cache_path
    ):
        time.sleep(sleep_time)


def write_and_sleep(
    shared_data: List[int],
    access_client: rw_coordinator.RwCoordinatorClient,
    sleep_time: int,
    bucket: str,
    cache_path: pathlib.Path,
) -> None:
    with access_client.write_lock(
        storage_type=constants.GCS_STORAGE, bucket=bucket, cache_path=cache_path
    ):
        shared_data[0] += 1
        time.sleep(sleep_time)


def send_and_die(lock_request_url: str) -> None:
    with lomond.WebSocket(lock_request_url) as socket:
        for event in socket.connect():
            if isinstance(event, lomond.events.Text):
                return


def read_and_die(bucket: str, cache_path: pathlib.Path, ip_address: str, port: int) -> None:
    lock_request_url = (
        f"ws://{ip_address}:{port}/{constants.GCS_STORAGE}/{bucket}/"
        f"{str(cache_path)}?{urllib.parse.urlencode({'read_lock': True})}"
    )

    send_and_die(lock_request_url=lock_request_url)


def write_and_die(bucket: str, cache_path: pathlib.Path, ip_address: str, port: int) -> None:
    lock_request_url = (
        f"ws://{ip_address}:{port}/{constants.GCS_STORAGE}/{bucket}/"
        f"{str(cache_path)}?{urllib.parse.urlencode({'read_lock': False})}"
    )

    send_and_die(lock_request_url=lock_request_url)


class MultiThreadedTests(ThreadAwareTestCase):  # type: ignore
    def test_rw_coordinator(self) -> None:
        ip_address = "localhost"
        port = 10245
        bucket = "my_bucket"
        cache_path = pathlib.Path("/tmp.mdb")
        num_threads = 5
        shared_data = [0]

        access_server_handler = test_util.AccessServerHandler(hostname=ip_address, port=port)
        access_server_handler.run_server_in_thread()
        access_client = rw_coordinator.RwCoordinatorClient(url=f"ws://{ip_address}:{port}")

        try:
            with ThreadJoiner(45):
                for i in range(num_threads):
                    self.run_in_thread(
                        lambda: read_and_sleep(
                            access_client=access_client,
                            sleep_time=i + 1,
                            bucket=bucket,
                            cache_path=cache_path,
                        )
                    )
                    self.run_in_thread(
                        lambda: write_and_sleep(
                            shared_data=shared_data,
                            access_client=access_client,
                            sleep_time=i,
                            bucket=bucket,
                            cache_path=cache_path,
                        )
                    )
        finally:
            access_server_handler.stop_server()

        assert shared_data[0] == num_threads

    def test_rw_coordinator_connections_die(self) -> None:
        ip_address = "localhost"
        port = 10245
        bucket = "my_bucket"
        cache_path = pathlib.Path("/tmp.mdb")
        num_threads = 5
        shared_data = [0]
        threads_to_die = [2, 3]

        access_server_handler = test_util.AccessServerHandler(hostname=ip_address, port=port)
        access_server_handler.run_server_in_thread()
        access_client = rw_coordinator.RwCoordinatorClient(url=f"ws://{ip_address}:{port}")

        try:
            with ThreadJoiner(45):
                for i in range(num_threads):
                    if i in threads_to_die:
                        self.run_in_thread(
                            lambda: read_and_die(
                                bucket=bucket,
                                cache_path=cache_path,
                                ip_address=ip_address,
                                port=port,
                            )
                        )
                        self.run_in_thread(
                            lambda: write_and_die(
                                bucket=bucket,
                                cache_path=cache_path,
                                ip_address=ip_address,
                                port=port,
                            )
                        )
                    else:
                        self.run_in_thread(
                            lambda: read_and_sleep(
                                access_client=access_client,
                                sleep_time=i + 1,
                                bucket=bucket,
                                cache_path=cache_path,
                            )
                        )
                        self.run_in_thread(
                            lambda: write_and_sleep(
                                shared_data=shared_data,
                                access_client=access_client,
                                sleep_time=i,
                                bucket=bucket,
                                cache_path=cache_path,
                            )
                        )
        finally:
            access_server_handler.stop_server()

        assert shared_data[0] == num_threads - len(threads_to_die)

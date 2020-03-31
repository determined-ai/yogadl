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
import datetime
import pathlib
from typing import Optional

import google.api_core.exceptions as gcp_exceptions
import google.cloud.storage as google_storage
import tensorflow as tf

import yogadl.constants as constants
from yogadl import storage


class GCSConfigurations(storage.BaseCloudConfigurations):
    def __init__(
        self, bucket: str, bucket_directory_path: str, url: str, local_cache_dir: str = "/tmp/",
    ) -> None:
        super().__init__(
            bucket=bucket,
            bucket_directory_path=bucket_directory_path,
            url=url,
            local_cache_dir=local_cache_dir,
        )


class GCSStorage(storage.BaseCloudStorage):
    """
    Stores dataset cache in Google Cloud Storage (GCS).

    GCSStorage creates a local cache from a dataset and then uploads
    it to the specified GCS bucket. When fetching from GCS, the creation
    time of the local cache (recorded in metadata), is compared to the
    creation time of the GCS cache, if they are not equivalent, the
    local cache is overwritten.

    The GCS cache, and the local cache are potentially shared across a
    number of concurrent processes. `cacheable()` provides synchronization
    guarantees. Users should not call `submit()` and `fetch()` if they
    anticipate concurrent data accesses.

    Authentication is currently only supported via the "Application
    Default Credentials" method in GCP. Typical configuration:
    ensure your VM runs in a service account that has sufficient
    permissions to read/write/delete from the GCS bucket where
    checkpoints will be stored (this only works when running in GCE).
    """

    def __init__(
        self,
        configurations: GCSConfigurations,
        tensorflow_config: Optional[tf.compat.v1.ConfigProto] = None,
    ):
        super().__init__(configurations=configurations, tensorflow_config=tensorflow_config)

        self._gcs_client = google_storage.Client()
        self._bucket = self._gcs_client.bucket(self._configurations.bucket)

        self._check_configurations()

    def _check_configurations(self) -> None:
        assert self._configurations.local_cache_dir.is_dir()
        assert self._configurations.cache_format in self._supported_cache_formats
        assert self._bucket.exists()

    @property
    def _storage_type(self) -> str:
        return constants.GCS_STORAGE

    def _is_cloud_cache_present(self, dataset_id: str, dataset_version: str) -> bool:

        gcs_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        blob = self._bucket.blob(str(gcs_cache_filepath))

        blob_exists = blob.exists()
        assert isinstance(blob_exists, bool)

        return blob_exists

    def _download_from_cloud_storage(
        self, dataset_id: str, dataset_version: str, local_cache_filepath: pathlib.Path
    ) -> datetime.datetime:

        gcs_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        blob = self._bucket.blob(str(gcs_cache_filepath))

        assert (
            blob.exists()
        ), f"Downloading non-existent blob {self._configurations.bucket}/{gcs_cache_filepath}."

        try:
            blob.download_to_filename(str(local_cache_filepath))
        except gcp_exceptions.GoogleAPICallError as e:
            raise AssertionError(
                f"Downloading blob {self._configurations.bucket}"
                f"/{gcs_cache_filepath} failed with exception {e}."
            )

        return self._get_remote_cache_timestamp(
            dataset_id=dataset_id, dataset_version=dataset_version
        )

    def _upload_to_cloud_storage(
        self, dataset_id: str, dataset_version: str, local_cache_filepath: pathlib.Path
    ) -> datetime.datetime:

        gcs_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        blob = self._bucket.blob(str(gcs_cache_filepath))

        try:
            blob.upload_from_filename(str(local_cache_filepath))
        except gcp_exceptions.GoogleAPICallError as e:
            raise AssertionError(
                f"Upload from {local_cache_filepath} to {self._configurations.bucket}"
                f"/{gcs_cache_filepath} failed with exception {e}."
            )

        # Do not need to `reload()` to get latest blob metadata after upload.
        assert blob.time_created
        assert isinstance(blob.time_created, datetime.datetime)

        return blob.time_created

    def _get_remote_cache_timestamp(
        self, dataset_id: str, dataset_version: str
    ) -> datetime.datetime:

        gcs_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )
        blob = self._bucket.blob(str(gcs_cache_filepath))

        try:
            blob.reload()
        except gcp_exceptions.GoogleAPICallError as e:
            raise AssertionError(
                f"Getting metadata of {self._configurations.bucket}"
                f"/{gcs_cache_filepath} failed with exception {e}."
            )

        assert isinstance(blob.time_created, datetime.datetime)

        return blob.time_created

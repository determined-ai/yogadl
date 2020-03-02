import datetime
import pathlib
from typing import Optional

import boto3
import botocore.client as boto_client

import yogadl.constants as constants
import yogadl.storage.cloud_storage as cloud_storage


class S3Configurations(cloud_storage.BaseCloudConfigurations):
    def __init__(
        self,
        bucket: str,
        bucket_directory_path: str,
        url: str,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        endpoint_url: Optional[str] = None,
        local_cache_dir: str = "/tmp/",
    ) -> None:
        super().__init__(
            bucket=bucket,
            bucket_directory_path=bucket_directory_path,
            url=url,
            local_cache_dir=local_cache_dir,
        )
        self.access_key = access_key
        self.secret_key = secret_key
        self.endpoint_url = endpoint_url


class S3Storage(cloud_storage.BaseCloudStorage):
    """
    Stores dataset cache in AWS S3.

    S3Storage creates a local cache from a dataset and then uploads
    it to the specified S3 bucket. When fetching from S3, the creation
    time of the local cache (recorded in metadata), is compared to the
    creation time of the S3 cache, if they are not equivalent, the
    local cache is overwritten.

    The S3 cache, and the local cache are potentially shared across a
    number of concurrent processes. `cacheable()` provides synchronization
    guarantees. Users should not call `submit()` and `fetch()` if they
    anticipate concurrent data accesses.
    """

    def __init__(self, configurations: S3Configurations):
        super().__init__(configurations=configurations)

        assert isinstance(self._configurations, S3Configurations)
        self._client = boto3.client(
            "s3",
            aws_access_key_id=self._configurations.access_key,
            aws_secret_access_key=self._configurations.secret_key,
            endpoint_url=self._configurations.endpoint_url,
        )

        self._check_configurations()

    def _check_configurations(self) -> None:
        assert self._configurations.local_cache_dir.is_dir()
        assert self._configurations.cache_backend in self._supported_cache_backends

        try:
            self._client.head_bucket(Bucket=self._configurations.bucket)
        except boto_client.ClientError as error:
            raise AssertionError(
                f"Unable to access bucket {self._configurations.bucket}. "
                f"Failed with exception: {error}."
            )

    @property
    def _storage_type(self) -> str:
        return constants.AWS_STORAGE

    def _is_cloud_cache_present(self, dataset_id: str, dataset_version: str) -> bool:

        s3_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        try:
            self._client.head_object(Bucket=self._configurations.bucket, Key=str(s3_cache_filepath))
            cloud_cache_exists = True
        except boto_client.ClientError:
            cloud_cache_exists = False

        return cloud_cache_exists

    def _download_from_cloud_storage(
        self, dataset_id: str, dataset_version: str, local_cache_filepath: pathlib.Path
    ) -> datetime.datetime:

        s3_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        try:
            self._client.download_file(
                Bucket=self._configurations.bucket,
                Key=str(s3_cache_filepath),
                Filename=str(local_cache_filepath),
            )
        except boto_client.ClientError as error:
            raise AssertionError(
                f"Downloading blob {self._configurations.bucket}"
                f"/{s3_cache_filepath}. Failed with exception {error}."
            )

        return self._get_remote_cache_timestamp(
            dataset_id=dataset_id, dataset_version=dataset_version
        )

    def _upload_to_cloud_storage(
        self, dataset_id: str, dataset_version: str, local_cache_filepath: pathlib.Path
    ) -> datetime.datetime:

        s3_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        try:
            self._client.upload_file(
                Filename=str(local_cache_filepath),
                Bucket=self._configurations.bucket,
                Key=str(s3_cache_filepath),
            )
        except boto3.exceptions.S3UploadFailedError as error:
            raise AssertionError(f"Failed to upload file to S3 with exception: {error}.")

        return self._get_remote_cache_timestamp(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

    def _get_remote_cache_timestamp(
        self, dataset_id: str, dataset_version: str
    ) -> datetime.datetime:

        s3_cache_filepath = self._get_remote_cache_filepath(
            dataset_id=dataset_id, dataset_version=dataset_version,
        )

        try:
            s3_object_info = self._client.head_object(
                Bucket=self._configurations.bucket, Key=str(s3_cache_filepath)
            )
        except boto_client.ClientError as error:
            raise AssertionError(
                f"Unable to look up metadata for {self._configurations.bucket}/"
                f"{s3_cache_filepath}. Failed with exception: {error}."
            )

        timestamp = s3_object_info.get("LastModified")
        assert isinstance(timestamp, datetime.datetime)

        return timestamp

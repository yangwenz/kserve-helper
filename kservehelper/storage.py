import os
import abc
import sys
import boto3
import logging
import threading
from boto3.s3.transfer import TransferConfig


class Storage:

    @abc.abstractmethod
    def upload(self, filename: str, key: str, **kwargs):
        pass

    @abc.abstractmethod
    def download(self, key: str, filename: str, **kwargs):
        pass


class S3Storage(Storage):

    def __init__(self, bucket, region_name, aws_access_key_id, aws_secret_access_key):
        self.bucket = bucket
        self.s3 = boto3.client(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.s3_resource = boto3.resource(
            "s3",
            region_name=region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
        )
        self.config = TransferConfig(
            multipart_threshold=16 * 1024 * 1024,
            max_concurrency=10,
            multipart_chunksize=16 * 1024 * 1024,
            use_threads=True
        )

    def __del__(self):
        self.s3.close()

    def _upload_simple(self, filename: str, key: str) -> bool:
        try:
            self.s3.upload_file(
                filename=filename,
                bucket=self.bucket,
                key=key
            )
        except Exception as e:
            logging.error(e)
            return False
        return True

    def _upload_multipart(self, filename: str, key: str) -> bool:
        try:
            size = os.path.getsize(filename)
            self.s3_resource.Object(self.bucket, key).upload_file(
                filename,
                Config=self.config,
                Callback=ProgressPercentage(size)
            )
        except Exception as e:
            logging.error(e)
            return False
        return True

    def _download_simple(self, key: str, filename: str) -> bool:
        try:
            self.s3.download_file(
                bucket=self.bucket,
                key=key,
                filename=filename
            )
        except Exception as e:
            logging.error(e)
            return False
        return True

    def _download_multipart(self, key: str, filename: str) -> bool:
        try:
            size = self.s3_resource.Object(self.bucket, key).content_length
            self.s3_resource.Object(self.bucket, key).download_file(
                filename,
                Config=self.config,
                Callback=ProgressPercentage(size)
            )
        except Exception as e:
            logging.error(e)
            return False
        return True

    def upload(self, filename: str, key: str, **kwargs) -> bool:
        mode = kwargs.get("method", "multipart")
        if mode == "multipart":
            return self._upload_multipart(filename=filename, key=key)
        else:
            return self._upload_simple(filename=filename, key=key)

    def download(self, key: str, filename: str, **kwargs):
        mode = kwargs.get("method", "multipart")
        if mode == "multipart":
            return self._download_multipart(key=key, filename=filename)
        else:
            return self._download_simple(key=key, filename=filename)


class ProgressPercentage:
    def __init__(self, file_size):
        self._size = file_size
        self._seen_so_far = 0
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            sys.stdout.write(
                "\r%s / %s  (%.2f%%)" % (
                    self._seen_so_far, self._size, percentage))
            sys.stdout.flush()

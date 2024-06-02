import time
import pytest
import unittest
from kservehelper.storage import S3Storage


class TestS3Storage(unittest.TestCase):

    @pytest.mark.skip
    def test(self):
        storage = S3Storage(
            bucket="hypergai-upload-models",
            region_name="ap-southeast-1",
            aws_access_key_id="",
            aws_secret_access_key=""
        )
        start_time = time.time()
        flag = storage.upload(filename="/home/ywz/Downloads/google-chrome-stable_current_amd64.deb", key="test")
        print(flag)
        print(f"Upload time: {time.time() - start_time}")
        start_time = time.time()
        flag = storage.download(key="test", filename="/home/ywz/Downloads/test.deb")
        print(flag)
        print(f"Download time: {time.time() - start_time}")


if __name__ == "__main__":
    unittest.main()

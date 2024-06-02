import pytest
import unittest
from kservehelper.cache import ModelCache


class TestModelCache(unittest.TestCase):

    def setUp(self) -> None:
        self.aws_access_key_id = ""
        self.aws_secret_access_key = ""

    @pytest.mark.skip
    def test_set(self):
        cache = ModelCache(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        flag = cache.set(key="test_test", filepath="/Users/ywz/Downloads/Q1.txt")
        self.assertEqual(flag, True)

    @pytest.mark.skip
    def test_get(self):
        cache = ModelCache(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
        )
        path = cache.get(key="test_test")
        print(path)


if __name__ == "__main__":
    unittest.main()

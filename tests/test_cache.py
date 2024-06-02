import os
import pytest
import unittest
import shutil
import tempfile
from kservehelper.cache import \
    MemoryCache, DiskLRUCache, DiskCache


class TestMemoryCache(unittest.TestCase):

    def test_get(self):
        cache = MemoryCache(
            folder="",
            num_cached_objects=2,
            models={"a": "1", "b": "2"},
            load_func=lambda path: path
        )
        value = cache["a"]
        self.assertEqual(value, "1")
        value = cache["b"]
        self.assertEqual(value, "2")

        folder = os.path.dirname(os.path.abspath(__file__))
        folder = os.path.join(folder, "data")
        cache = MemoryCache(
            folder=folder,
            num_cached_objects=2,
            load_func=lambda path: path
        )
        value = cache["a"]
        self.assertEqual(value, os.path.join(folder, "1"))
        value = cache["b"]
        self.assertEqual(value, os.path.join(folder, "2"))
        value = cache["c"]
        self.assertEqual(value, os.path.join(folder, "3"))
        self.assertDictEqual(
            dict(cache.cache.cache),
            {"b": os.path.join(folder, "2"), "c": os.path.join(folder, "3")}
        )


class TestDiskLRUCache(unittest.TestCase):

    @staticmethod
    def _make_file(path, size):
        assert size > 1
        with open(path, "wb") as f:
            f.seek(size - 1)
            f.write(b"\0")

    def test_get(self):
        tmp_dir = tempfile.gettempdir()
        cache_dir = os.path.join(tmp_dir, "cache")
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        cache = DiskLRUCache(
            capacity=32,
            cache_dir=cache_dir
        )

        filepath = os.path.join(tmp_dir, "tmp1")
        self._make_file(filepath, 8)
        cache["file_1"] = filepath
        self.assertEqual(cache.total_size, 8)
        # Set again
        cache["file_1"] = filepath
        self.assertEqual(cache.total_size, 8)

        path = cache["file_1"]
        self.assertEqual(path, os.path.join(cache.cache_dir, "file_1"))
        path = cache["none"]
        self.assertEqual(path, None)

        filepath = os.path.join(tmp_dir, "tmp2")
        self._make_file(filepath, 16)
        cache["file_2"] = filepath
        self.assertEqual(cache.total_size, 24)

        filepath = os.path.join(tmp_dir, "tmp3")
        self._make_file(filepath, 30)
        cache["file_3"] = filepath
        self.assertEqual(cache.total_size, 54)
        self.assertEqual(len(cache.cache), 3)

        filepath = os.path.join(tmp_dir, "tmp4")
        self._make_file(filepath, 16)
        item = cache["file_1"]
        cache["file_4"] = filepath
        self.assertEqual(cache.total_size, 24)
        self.assertEqual(len(cache.cache), 2)
        self.assertListEqual(sorted(cache.cache.keys()), ["file_1", "file_4"])


class TestDiskCache(unittest.TestCase):

    def test_shard_index(self):
        cache = DiskCache()
        count = {}
        for i in range(10):
            b = cache._shard_index(i)
            count[b] = count.get(b, 0) + 1

    @pytest.mark.skip
    def test_set(self):
        os.environ["AWS_ACCESS_KEY_ID"] = ""
        os.environ["AWS_SECRET_ACCESS_KEY"] = ""
        cache = DiskCache()
        cache.set(key="test_test", filepath="/home/ywz/Downloads/google-chrome-stable_current_amd64.deb")

    @pytest.mark.skip
    def test_get(self):
        os.environ["AWS_ACCESS_KEY_ID"] = ""
        os.environ["AWS_SECRET_ACCESS_KEY"] = ""
        cache = DiskCache()
        path = cache.get("test_test")
        print(path)


if __name__ == "__main__":
    unittest.main()

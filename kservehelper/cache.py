import os
import json
import mmh3
import pickle
import shutil
import logging
import tempfile
import threading
from pathlib import Path
from typing import Dict, Callable, Any, Union
from collections import OrderedDict
from .utils import flock
from .storage import S3Storage


###################################################################
# Memory cache designed for caching models in memory
###################################################################
class MemoryLRUCache:

    def __init__(self, num_cached_objects):
        self.cache = OrderedDict()
        self.capacity = num_cached_objects
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key not in self.cache:
                return None
            else:
                self.cache.move_to_end(key)
                return self.cache[key]

    def set(self, key, value):
        with self.lock:
            self.cache[key] = value
            self.cache.move_to_end(key)
            if len(self.cache) > self.capacity:
                _, val = self.cache.popitem(last=False)
                del val
            else:
                return None


class MemoryCache:
    CONFIG_FILE = "models.json"

    def __init__(
            self,
            folder: str,
            num_cached_objects: int,
            models: Dict = None,
            load_func: Callable = None
    ):
        """
        :param folder: The folder for storing models, which can also be empty.
        :param num_cached_objects: The cache capacity (maximum number of cached objects).
        :param models: The maps from model names to model filenames, e.g., {"model_a": "model_a_file.pth"}.
        :param load_func: The function to load a model given the model filepath.
        """
        assert load_func is not None, "`load_func` for loading models is not set"
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.folder = folder
        self.cache = MemoryLRUCache(num_cached_objects)
        self.load_func = load_func

        self.models = {}
        if models is None:
            self._load_model_config()
        else:
            self.models = models

    def _load_model_config(self):
        path = os.path.join(self.folder, MemoryCache.CONFIG_FILE)
        if os.path.isfile(path):
            self.logger.info("loading model info from config file ...")
            with open(path, "r") as f:
                self.models = json.load(f)

    def __getitem__(self, key: str):
        """
        Gets the loaded model from the cache given a key.

        :param key: It can be a model name if `models` is set, or a model filepath.
        """
        model = self.cache.get(key)
        if model is not None:
            return model
        if key not in self.models:
            self._load_model_config()

        try:
            filename = self.models.get(key, key)
            model = self.load_func(os.path.join(self.folder, filename))
            if model is not None:
                self.cache.set(key, model)
        except Exception as e:
            self.logger.error(str(e))
        return model


###################################################################
# Disk cache designed for caching models loaded from S3 on disk
###################################################################
class DiskLRUCache:

    def __init__(self, capacity: int = None, cache_dir: str = None):
        if not capacity:
            capacity = 10 * 10 ** 9
        if not cache_dir:
            cache_dir = tempfile.gettempdir()

        self.capacity = capacity
        self.cache_dir = cache_dir
        self.cache = OrderedDict()
        self.total_size = 0
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.lock_path = os.path.join(self.cache_dir, "lock")
        self.index_file = os.path.join(self.cache_dir, "index")
        with flock(self.lock_path):
            if os.path.exists(self.index_file):
                self._load()
            else:
                self._save()

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _load(self):
        with open(self.index_file, "rb") as f:
            self.total_size, self.cache = pickle.load(f)

    def _save(self):
        with open(self.index_file, "wb") as f:
            pickle.dump((self.total_size, self.cache), f)

    def __getitem__(self, key: str):
        with flock(self.lock_path):
            if os.path.exists(self.index_file):
                self._load()
            if key not in self.cache:
                return None
            self.cache.move_to_end(key)
            self._save()
            item = self.cache[key]
            path = os.path.join(self.cache_dir, item["filename"])
            if os.path.isfile(path):
                return path
            else:
                return None

    def __setitem__(self, key: str, filepath: str):
        with flock(self.lock_path):
            if os.path.exists(self.index_file):
                self._load()

            # Check if the cache is full or the item exists
            while self.total_size >= self.capacity:
                self.logger.info(f"cache hit capacity {self.capacity}")
                cache_key, item = self.cache.popitem(last=False)
                self.total_size -= item["size"]
                path = os.path.join(self.cache_dir, item["filename"])
                if os.path.isfile(path):
                    os.remove(path)
                self.logger.info("evicted {cache_key} from cache")

            if key in self.cache:
                item = self.cache[key]
                self.total_size -= item["size"]
                del self.cache[key]
                path = os.path.join(self.cache_dir, item["filename"])
                if os.path.isfile(path):
                    os.remove(path)
            self._save()

            # Copy the file and update the cache
            file_stats = os.stat(filepath)
            item = {"filename": key, "size": file_stats.st_size}
            path = os.path.join(self.cache_dir, item["filename"])
            shutil.copyfile(filepath, path)
            self.cache[key] = item
            self.cache.move_to_end(key)
            self.total_size += item["size"]
            self._save()


class DiskCache:

    def __init__(
            self,
            num_shards: int = 10,
            capacity: int = 10 * 10 ** 9,
            cache_dir: str = tempfile.gettempdir(),
            aws_bucket: str = os.getenv("BUCKET", "hypergai-upload-models"),
            aws_region_name: str = os.getenv("REGION_NAME", "ap-southeast-1"),
            aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    ):
        """
        :param num_shards: The number of cache shards.
        :param capacity: The total capacity (in Bytes) of the disk cache.
        :param cache_dir: The cache directory for storing objects.
        :param aws_bucket: AWS S3 bucket name.
        :param aws_region_name: AWS S3 bucket region.
        :param aws_access_key_id: AWS S3 access key ID.
        :param aws_secret_access_key: AWS S3 secret access key.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.num_shards = num_shards
        self.cache_dir = cache_dir
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        self.caches = []
        for i in range(num_shards):
            cache = DiskLRUCache(capacity // num_shards, os.path.join(cache_dir, f"{i}"))
            self.caches.append(cache)

        self.bucket = aws_bucket
        self.region_name = aws_region_name
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key

        self.storage = None
        if self.aws_access_key_id != "" and self.aws_secret_access_key != "":
            self.storage = S3Storage(
                bucket=self.bucket,
                region_name=self.region_name,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
        else:
            self.logger.warning("S3 storage is not set")

    # https://stats.stackexchange.com/questions/26344/how-to-uniformly-project-a-hash-to-a-fixed-number-of-buckets
    def _shard_index(self, key: Any):
        b = self.num_shards
        i = mmh3.hash128(str(key), signed=False)
        p = i / float(2 ** 128)
        for j in range(b):
            if j / float(b) <= p and (j + 1) / float(b) > p:
                return j
        return b - 1

    def get(self, key: str) -> Union[str, None]:
        """
        Gets the filepath given a key (filename). If the file is not in the cache, it will
        try to download it from s3 and update the cache if it is downloaded successfully.

        :param key: A unique filename/key.
        :return: The filepath if file exists or None.
        """
        cache_index = self._shard_index(key)
        cache = self.caches[cache_index]
        try:
            path = cache[key]
            if path is not None:
                return path
        except Exception as e:
            self.logger.error(str(e))
            return None

        with flock(os.path.join(self.cache_dir, f"{key}.lock")):
            # Try again if acquired the file lock (other process might download the file)
            try:
                path = cache[key]
                if path is not None:
                    return path
            except Exception as e:
                self.logger.error(str(e))
                return None

            if self.storage is not None:
                filepath = "/tmp/tmp_file"
                if os.path.isfile(filepath):
                    os.remove(filepath)
                if not self.storage.download(key=key, filename=filepath):
                    self.logger.error(f"failed to download file: {key}")
                    return None
                try:
                    cache[key] = filepath
                    return cache[key]
                except Exception as e:
                    self.logger.error(str(e))
                    return None
            else:
                return None

    def set(self, key: str, filepath: str) -> bool:
        """
        Uploads the file (specified by `filepath`) to s3 bucket and also copies it into the disk cache.
        The filename of the copy will be `key`.

        :param key: A unique filename/key.
        :param filepath: The filepath of a local file.
        """
        cache_index = self._shard_index(key)
        cache = self.caches[cache_index]
        with flock(os.path.join(self.cache_dir, f"{key}.lock")):
            try:
                if self.storage is not None and \
                        not self.storage.upload(filename=filepath, key=key):
                    self.logger.error(f"failed to upload file: {filepath}")
                    return False
                cache[key] = filepath
                return True
            except Exception as e:
                self.logger.error(str(e))
                return False


class ModelCache:

    def __init__(
            self,
            num_shards: int = 10,
            capacity: int = 10 * 10 ** 9,
            cache_dir: str = tempfile.gettempdir(),
            num_mem_objects: int = 10,
            model_load_func: Callable = None,
            aws_bucket: str = os.getenv("BUCKET", "hypergai-upload-models"),
            aws_region_name: str = os.getenv("REGION_NAME", "ap-southeast-1"),
            aws_access_key_id: str = os.getenv("AWS_ACCESS_KEY_ID", ""),
            aws_secret_access_key: str = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    ):
        """
        :param num_shards: The number of cache shards.
        :param capacity: The total capacity (in Bytes) of the disk cache.
        :param cache_dir: The cache directory for storing objects.
        :param num_mem_objects: The maximum number of objects cached in the memory.
        :param model_load_func: The function for loading a model from a file.
        :param aws_bucket: AWS S3 bucket name.
        :param aws_region_name: AWS S3 bucket region.
        :param aws_access_key_id: AWS S3 access key ID.
        :param aws_secret_access_key: AWS S3 secret access key.
        """
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.disk_cache = DiskCache(
            num_shards=num_shards,
            capacity=capacity,
            cache_dir=cache_dir,
            aws_bucket=aws_bucket,
            aws_region_name=aws_region_name,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        self.mem_cache = MemoryLRUCache(
            num_cached_objects=num_mem_objects
        )
        self.load_func = model_load_func

    def get(self, key: str) -> Union[Any, None]:
        model = self.mem_cache.get(key)
        # Hit the memory cache
        if model is not None:
            return model
        # Try to load from the disk cache
        path = self.disk_cache.get(key)
        if path is None:
            self.logger.error(f"model with key {key} doesn't exist in disk cache")
            return None
        try:
            # Load the model
            model = self.load_func(path) if self.load_func is not None else path
            if model is None:
                return None
            self.mem_cache.set(key, model)
            return model
        except Exception as e:
            self.logger.error(str(e))
            return None

    def set(self, key: str, filepath: str) -> bool:
        # Set the disk cache first
        if not self.disk_cache.set(key, filepath):
            return False
        # Set the memory cache
        model = self.get(key)
        return model is not None

import os
import json
import time
import aiohttp
import asyncio
import aiofiles
import requests
import concurrent.futures
from typing import List
from contextlib import contextmanager
from kservehelper.types import Path


def upload_files(webhook_url: str, paths: List[Path], timeout=60):
    # outputs = asyncio.run(_upload_v2(webhook_url, paths, timeout))
    # return outputs
    if webhook_url.endswith("upload"):
        outputs = _upload_multithread(webhook_url, paths, timeout)
    elif webhook_url.endswith("upload_batch"):
        outputs = _upload_batch(webhook_url, paths, timeout)
    else:
        raise RuntimeError(f"Invalid webhook URL: {webhook_url}")
    return {"output": outputs}


def _upload(webhook_url: str, paths: List[Path], timeout):
    outputs = []
    for path in paths:
        filepath = str(path)
        with open(filepath, "rb") as f:
            files = {"file": (filepath, f)}
            response = requests.post(webhook_url, files=files, timeout=timeout)
        if response.status_code != 200:
            raise RuntimeError("failed to call upload webhook")
        r = json.loads(response.text)
        outputs.append(r["url"])
    return outputs


def _upload_batch(webhook_url: str, paths: List[Path], timeout, retires=3):
    error = None
    result = {}

    for retry in range(retires):
        # Open a list of files
        files, file_list = {}, []
        for i, path in enumerate(paths):
            filepath = str(path)
            f = open(filepath, "rb")
            files[f"file_{i}"] = (filepath, f)
            file_list.append(f)

        # Send a batch of files
        try:
            response = requests.post(
                webhook_url,
                headers={"NUM_FILES": str(len(file_list))},
                files=files,
                timeout=timeout
            )
            if response.status_code != 200:
                raise RuntimeError(f"response status code is {response.status_code}")
            error = None
            result = json.loads(response.text)
        except Exception as e:
            print(f"UPLOAD ERROR: {str(e)}")
            error = e
            time.sleep(2 * (retry + 1))

        # Close the files
        for f in file_list:
            f.close()
        if error is None:
            break

    if error is not None:
        raise error
    return result["urls"]


def _upload_multithread(webhook_url: str, paths: List[Path], timeout, retires=3):
    def _make_request(filepath):
        for i in range(retires):
            with open(filepath, "rb") as f:
                files = {"file": (filepath, f)}
                try:
                    response = requests.post(webhook_url, files=files, timeout=timeout)
                    if response.status_code != 200:
                        raise RuntimeError(f"response status code is {response.status_code}")
                    r = json.loads(response.text)
                    return filepath, r["url"]
                except Exception as e:
                    print(f"ERROR: {e}")
                    print(f"Retrying {i + 1}...")
                    time.sleep((i + 1) * 2)
        return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(_make_request, str(path)) for path in paths]
        outputs = [job.result() for job in concurrent.futures.as_completed(jobs)]
        for path in paths:
            file = str(path)
            if os.path.exists(file):
                os.remove(file)
        if None in outputs:
            raise RuntimeError("failed to upload files")

    filename2url = dict(outputs)
    return [filename2url[str(path)] for path in paths]


async def _async_upload_v1(webhook_url: str, paths: List[Path], timeout):
    outputs = []
    async with aiohttp.ClientSession() as session:
        for path in paths:
            filepath = str(path)
            with open(filepath, "rb") as f:
                data = aiohttp.FormData()
                data.add_field("file", f, filename=filepath)
                async with session.post(webhook_url, data=data, timeout=timeout) as response:
                    if not response.ok:
                        raise RuntimeError("failed to call upload webhook")
                    text = await response.text()
                    r = json.loads(text)
                    outputs.append(r["url"])
    return outputs


async def _async_upload_v2(webhook_url, paths, timeout):
    async def _upload_file(session, local_path):
        async with aiofiles.open(local_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=local_path)
            async with session.post(webhook_url, data=data, timeout=timeout) as response:
                if not response.ok:
                    raise RuntimeError("failed to call upload webhook")
                text = await response.text()
                r = json.loads(text)
                return local_path, r["url"]

    async with aiohttp.ClientSession() as sess:
        urls = await asyncio.gather(*[_upload_file(sess, str(path)) for path in paths])
        filename2url = dict(urls)
        return [filename2url[str(path)] for path in paths]


@contextmanager
def flock(lock_path, timeout=300):
    """
    Context manager that acquires and releases a file-based lock.
    """
    start_time = time.time()
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL)
            break
        except OSError:
            if (time.time() - start_time) >= timeout:
                raise Exception("Timeout occurred.")

    try:
        yield fd
    finally:
        if fd:
            os.close(fd)
            os.remove(lock_path)

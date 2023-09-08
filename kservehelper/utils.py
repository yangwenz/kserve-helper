import json
import aiohttp
import asyncio
import aiofiles
import requests
import concurrent.futures
from typing import List
from kservehelper.types import Path


def upload_files(webhook_url: str, paths: List[Path], timeout=60):
    # outputs = asyncio.run(_upload_v2(webhook_url, paths, timeout))
    # return outputs
    outputs = _upload_multithread(webhook_url, paths, timeout)
    return {"urls": outputs}


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


def _upload_multithread(webhook_url: str, paths: List[Path], timeout):
    def _make_request(filepath):
        with open(filepath, "rb") as f:
            files = {"file": (filepath, f)}
            try:
                response = requests.post(webhook_url, files=files, timeout=timeout)
            except Exception as e:
                print(f"ERROR: {e}")
                return None
        if response.status_code != 200:
            return None
        r = json.loads(response.text)
        return filepath, r["url"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        jobs = [executor.submit(_make_request, str(path)) for path in paths]
        outputs = [job.result() for job in concurrent.futures.as_completed(jobs)]
        if None in outputs:
            raise RuntimeError("failed to call upload webhook")

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

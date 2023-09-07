import json
import requests
from typing import List
from kservehelper.types import Path


def upload_files(webhook_url: str, paths: List[Path], timeout=60):
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

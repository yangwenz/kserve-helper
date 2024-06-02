import json
import base64
import io
import requests
from PIL import Image


def test():
    url = "http://localhost:8080/v1/models/stable-diffusion:predict"
    with open("input.json", "r") as f:
        payload = json.load(f)
    response = requests.post(url, json=payload)
    r = response.json()

    im_binary = base64.b64decode(r["image"])
    image = Image.open(io.BytesIO(im_binary))
    image.save("test.png", format="PNG")


if __name__ == "__main__":
    test()

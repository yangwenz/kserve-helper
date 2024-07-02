import requests

url = "http://localhost:8080/v1/models/streaming:generate"
with requests.post(url, stream=True, json={"repeat": 5}) as r:
    for chunk in r.iter_content(16):
        print(chunk)

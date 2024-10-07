import requests

url = "http://localhost:8080/v1/models/streaming:predict"
with requests.post(url, stream=True, json={"repeat": 5}) as r:
    for chunk in r.iter_lines():
        print(chunk)

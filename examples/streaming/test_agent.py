import requests

url = "http://localhost:8000/v1/generate"
data = {
    "model_name": "streaming",
    "inputs": {"repeat": 5}
}
with requests.post(url, stream=True, json=data) as r:
    for chunk in r.iter_lines():
        print(chunk)

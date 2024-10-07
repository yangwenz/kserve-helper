import requests

url = "http://localhost:8001/v1/generate"
# url = "http://34.87.171.145:8001/v1/generate"
data = {
    "model_name": "streaming",
    "inputs": {"repeat": 5}
}
with requests.post(url, stream=True, json=data, headers={"UID": "12345"}) as r:
    print(r.status_code)
    for chunk in r.iter_lines():
        print(chunk)

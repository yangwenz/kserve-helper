import unittest
from typing import Dict
from kservehelper.model import KServeModel


class CustomModel:

    def __init__(self):
        self.device = None
        self.text2img_pipe = None

    def load(self):
        pass

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        prompt = payload["prompt"]
        return {"outputs": prompt}


class TestKServeModel(unittest.TestCase):

    def test(self):
        payload = {"prompt": "test test"}
        model = KServeModel("test", CustomModel)
        print(model.predict(payload))


if __name__ == "__main__":
    unittest.main()

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


class CustomTransform:

    def preprocess(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        prompt = payload["prompt"]
        return {"outputs": prompt.upper()}

    def postprocess(self, infer_response: Dict, headers: Dict[str, str] = None) -> Dict:
        outputs = infer_response["outputs"] + " ABC"
        return {"outputs": outputs}


class TestKServeModel(unittest.TestCase):

    def test_model(self):
        payload = {"prompt": "test test"}
        model = KServeModel("test", CustomModel)
        outputs = model.predict(payload)
        self.assertDictEqual(outputs, {"outputs": "test test"})

    def test_tansform(self):
        payload = {"prompt": "test test"}
        model = KServeModel("test", CustomTransform)
        outputs = model.postprocess(model.preprocess(payload))
        self.assertDictEqual(outputs, {"outputs": "TEST TEST ABC"})


if __name__ == "__main__":
    unittest.main()

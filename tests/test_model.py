import unittest
from typing import Dict, List
from kservehelper.model import KServeModel
from kservehelper.types import Input, Path


class CustomModel:

    def __init__(self):
        self.device = None
        self.text2img_pipe = None

    def load(self):
        pass

    def predict(
            self,
            prompt: str = Input(
                description="Input prompt",
                default="standing, (full body)++"
            )
    ) -> Dict:
        return {"outputs": prompt}


class CustomTransform:

    def preprocess(
            self,
            prompt: str = Input(
                description="Input prompt",
                default="standing, (full body)++"
            )
    ) -> Dict:
        return {"outputs": prompt.upper()}

    def postprocess(self, infer_response: Dict) -> Dict:
        outputs = infer_response["outputs"] + " ABC"
        return {"outputs": outputs}


class TestKServeModel(unittest.TestCase):

    def test_model(self):
        payload = {"prompt": "test test", "upload_webhook": "http://localhost"}
        model = KServeModel("test", CustomModel)
        outputs = model.predict(payload)
        self.assertDictEqual(outputs, {"outputs": "test test"})

    def xxx_test_tansform(self):
        payload = {"prompt": "test test", "upload_webhook": "http://localhost"}
        model = KServeModel("test", CustomTransform)
        outputs = model.postprocess(model.preprocess(payload))
        self.assertDictEqual(outputs, {"outputs": "TEST TEST ABC"})


if __name__ == "__main__":
    unittest.main()

import unittest
from typing import Dict
from kservehelper.model import KServeModel
from kservehelper.types import Input


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

    def test_tansform(self):
        payload = {"prompt": "test test", "upload_webhook": "http://localhost"}
        model = KServeModel("test", CustomTransform)
        outputs = model.postprocess(model.preprocess(payload))
        self.assertEqual(outputs["outputs"], "TEST TEST ABC")


if __name__ == "__main__":
    unittest.main()

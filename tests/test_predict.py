import unittest
from typing import Dict
from kservehelper.model import KServeModel
from kservehelper.types import Input


class CustomModel:

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


class TestKServeModel(unittest.TestCase):

    def test_model(self):
        payload = {"prompt": "test test", "upload_webhook": "http://localhost"}
        model = KServeModel("test", CustomModel)
        outputs = model.predict(payload)
        self.assertDictEqual(outputs, {"outputs": "test test"})


if __name__ == "__main__":
    unittest.main()
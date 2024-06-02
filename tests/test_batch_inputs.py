import unittest
from typing import Dict
from kservehelper.model import KServeModel
from kservehelper.types import Input


class CustomModel:

    def load(self):
        pass

    def predict(
            self,
            param: str = Input(
                description="global param",
                default="test_param"
            ),
            batch: list = [{
                "prompt": Input(
                    description="Input prompt",
                    default="standing, (full body)++"
                ),
                "param_a": Input(
                    description="param",
                    default=1
                ),
                "param_b": Input(
                    description="param",
                    default=1,
                    ge=0,
                    le=2
                ),
            }]
    ) -> Dict:
        return {"outputs": batch, "extra": param}

    def after_predict(self, outputs):
        outputs["after_predict"] = True
        return outputs


class TestKServeModel(unittest.TestCase):

    def test_model(self):
        batch = [
            {"prompt": "test a", "param_a": 3},
            {"prompt": "test b", "param_b": 2}
        ]
        payload = {"batch": batch, "param": "test", "upload_webhook": "http://localhost"}
        model = KServeModel("test", CustomModel)
        outputs = model.predict(payload)
        print(outputs)
        self.assertEqual(outputs["extra"], "test")

        values = outputs["outputs"]
        self.assertEqual(outputs["after_predict"], True)
        self.assertEqual(len(values), 2)
        self.assertDictEqual(values[0], {"prompt": "test a", "param_a": 3, "param_b": 1})
        self.assertDictEqual(values[1], {"prompt": "test b", "param_a": 1, "param_b": 2})


if __name__ == "__main__":
    unittest.main()

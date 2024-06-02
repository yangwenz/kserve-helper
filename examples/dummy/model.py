from typing import Dict
from kservehelper.model import KServeModel
from kservehelper.types import Input


class Model:

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


if __name__ == "__main__":
    KServeModel.serve("test-dummy", Model)

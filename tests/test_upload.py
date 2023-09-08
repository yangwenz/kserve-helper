import os
import pytest
import unittest
from typing import List
from kservehelper.model import KServeModel
from kservehelper.types import Input, Path


class CustomModel:

    def load(self):
        pass

    def predict(
            self,
            angle: int = Input(
                description="Rotation angle",
                default=90
            )
    ) -> List[Path]:
        from PIL import Image

        paths = []
        folder = os.path.dirname(os.path.abspath(__file__))
        input_image = Image.open(os.path.join(folder, "dog.jpg"))
        for i in range(4):
            image = input_image.rotate(angle + i * 10)
            output_path = f"/tmp/dog_{i}.jpg"
            image.save(output_path)
            paths.append(Path(output_path))
        return paths


class TestKServeModel(unittest.TestCase):

    @pytest.mark.skip
    def test_model(self):
        payload = {"angle": 45, "upload_webhook": "http://localhost:12000/upload"}
        model = KServeModel("test", CustomModel)
        outputs = model.predict(payload)
        print(outputs)


if __name__ == "__main__":
    unittest.main()

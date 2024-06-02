import os
from PIL import Image
from kservehelper import KServeModel
from kservehelper.types import Input, Path


class Model:

    def load(self):
        pass

    def predict(
            self,
            angle: int = Input(
                description="Rotation angle",
                default=90
            )
    ) -> Path:
        folder = os.path.dirname(os.path.abspath(__file__))
        input_image = Image.open(os.path.join(folder, "dog.jpg"))
        image = input_image.rotate(angle)
        output_path = KServeModel.generate_filepath("dog.jpg")
        image.save(output_path)
        return Path(output_path)


if __name__ == "__main__":
    KServeModel.serve("test-rotate-image", Model)

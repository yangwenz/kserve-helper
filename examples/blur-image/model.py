import base64
import io
from PIL import Image, ImageFilter
from kservehelper import KServeModel
from kservehelper.types import Input, Path


class Model:

    def load(self):
        pass

    def predict(
            self,
            image: str = Input(
                description="Base64 encoded image",
                default=""
            ),
            radius: float = Input(
                description="Standard deviation of the Gaussian kernel",
                default=2
            )
    ) -> Path:
        if image == "":
            raise ValueError("The input image is not set")
        im_binary = base64.b64decode(image)
        input_image = Image.open(io.BytesIO(im_binary))
        output_image = input_image.filter(ImageFilter.GaussianBlur(radius))
        output_path = KServeModel.generate_filepath("image.jpg")
        output_image.save(output_path)
        return Path(output_path)


if __name__ == "__main__":
    KServeModel.serve("test-blur-image", Model)

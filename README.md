# kserve-helper

*kserve-helper* is a toolkit for building docker images for ML models built on KServe. It supports
model input validation, uploading generated files to S3 or GCS, building model images, etc.
[Here](https://github.com/HyperGAI/kserve-helper/tree/main/examples) are some basic examples.

## Implement a Model Class for Serving
To build a docker image for serving, we only need to implement a model class with `load` and `predict`
methods:
```python
class Model:

    def load(self):
        # Load the model
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
```
The `load` function will be called during the model initialization step, which will be only called once.
The `predict` function will be called for each request. The input parameter information is specified by
the `Input` class. This `Input` class allows us to set parameter descriptions, default values and
constraints (e.g., 0 <= input value <= 1). 

The output typing of the `predict` function is important. If the output type is `Path` or 
`List[Path]`, the webhook for uploading files will be called after `predict` is finished. In this case,
the input request should also contain an additional key "upload_webhook" to specify the webhook server
address (an [example](https://github.com/HyperGAI/kserve-helper/tree/main/examples/rotate-image)).
If the output type is not `Path`, the results will be returned directly without calling the webhook.

If streaming outputs are required, the output of `predict` should be an iterator:
```python
class Model:

    def load(self):
        pass

    def predict(
            self,
            repeat: int = Input(
                description="The number of repeats",
                default=5
            )
    ):
        def _generator():
            for i in range(repeat):
                yield "Hello World!"
                time.sleep(1)

        return KServeModel.wrap_generator(_generator)
```
Note that we combine streaming and non-streaming APIs together as `predict` when using KServe >= 0.13.1. 
For KServe <= 0.10.2, we seperate streaming and non-streaming APIs, i.e.,
```python
class Model:

    def load(self):
        pass

    def predict(
            self,
            repeat: int = Input(
                description="The number of repeats",
                default=5
            )
    ):
        time.sleep(repeat)
        return {"output": " ".join(["Hello World!"] * repeat)}

    def generate(
            self,
            repeat: int = Input(
                description="The number of repeats",
                default=5
            )
    ):
        def _generator():
            for i in range(repeat):
                yield "Hello World!"
                time.sleep(1)

        return KServeModel.wrap_generator(_generator)
```

## Write a Config for Building Docker Image

To build the corresponding docker image for serving, we only need to write a config file:
```yaml
build:
  python_version: "3.10"
  cuda: "11.7"

  # a list of commands (optional)
  commands:
    - "apt install -y software-properties-common"

  # a list of ubuntu apt packages to install (optional)
  system_packages:
    - "git"
    - "python3-opencv"

  # choose requirements.txt (optional)
  python_requirements:
    - "requirements.txt"

  # a list of packages in the format <package-name>==<version>
  python_packages:
    - "kservehelper>=1.1.0"
    - "salesforce_lavis-1.1.0-py3-none-any.whl"
    - "git+https://github.com/huggingface/diffusers.git"
    - "controlnet_aux==0.0.7"
    - "opencv-python==4.8.0.74"
    - "Pillow"
    - "tensorboard"
    - "mediapipe"
    - "accelerate"
    - "bitsandbytes"

# The name given to built Docker images
image: "<DOCKER-IMAGE-NAME:TAG>"

# model.py defines the entrypoint
entrypoint: "model.py"
```
In the config file, we can choose python version, cuda version (and whether to use NGC images), 
system packages and python packages. We need to set the docker image name and the entrypoint. 
The entrypoint is just the file that defines the model class above.

To build the docker image, we can simply run in the folder containing the config file:
```shell
kservehelper build .
```
To push the docker image, run this command:
```shell
kservehelper push .
```

For more details, please check the implementations in the [repo](https://github.com/yangwenz/Hyperbooth).

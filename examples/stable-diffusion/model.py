import torch
from typing import List
from torch import autocast
from diffusers import StableDiffusionPipeline
from kservehelper import KServeModel
from kservehelper.types import Input, Path


class StableDiffusion:

    def __init__(self):
        self.device = None
        self.text2img_pipe = None

    def load(self):
        model_path = "/mnt/models/stable-diffusion-v1-4"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text2img_pipe = StableDiffusionPipeline.from_pretrained(
            model_path).to(self.device)

    def predict(
            self,
            prompt: str = Input(
                description="Input prompt",
                default="a dog playing on the street"
            ),
            guidance_scale: float = Input(
                description="Guidance scale",
                default=7.5
            ),
            height: int = Input(
                description="Image height",
                default=512
            ),
            width: int = Input(
                description="Image width",
                default=512
            ),
            num_inference_steps: int = Input(
                description="The number of inference steps",
                default=50
            ),
            num_images_per_prompt: int = Input(
                description="The number of images to generate per prompt",
                default=1
            ),
            seed: int = Input(
                description="Random seed",
                default=12345
            ),
            safety_check: bool = Input(
                description="Do safety check",
                default=False
            )
    ) -> List[Path]:
        if not safety_check:
            self.text2img_pipe.safety_checker = lambda images, **kwargs: (images, False)
        generator = torch.Generator(self.device)
        generator.manual_seed(seed)

        with autocast(self.device):
            images = self.text2img_pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images

        output_paths = []
        for i, image in enumerate(images):
            output_path = KServeModel.generate_filepath(f"image_{i}.jpg")
            image.save(output_path)
            output_paths.append(Path(output_path))
        return output_paths


if __name__ == "__main__":
    KServeModel.serve("stable-diffusion", StableDiffusion)

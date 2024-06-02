import os
import torch
from typing import List
from diffusers import DiffusionPipeline
from kservehelper import KServeModel
from kservehelper.types import Input, Path


# https://huggingface.co/docs/diffusers/v0.21.0/using-diffusers/sdxl


class StableDiffusionXL:

    def __init__(self):
        self.device = None
        self.base = None
        self.refiner = None
        self.use_refiner = os.getenv("USE_REFINER")
        self.use_torch_compile = os.getenv("USE_TORCH_COMPILE")

    def load(self):
        model_path = "/mnt/models/stable-diffusion-xl-base-1.0"
        refiner_path = "/mnt/models/stable-diffusion-xl-refiner-1.0"

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base = DiffusionPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        ).to(self.device)

        if self.use_torch_compile:
            self.base.unet = torch.compile(
                self.base.unet, mode="reduce-overhead", fullgraph=True
            )

        if self.use_refiner:
            self.refiner = DiffusionPipeline.from_pretrained(
                refiner_path,
                text_encoder_2=self.base.text_encoder_2,
                vae=self.base.vae,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            ).to(self.device)

    def predict(
            self,
            prompt: str = Input(
                description="Input prompt",
                default="a dog playing on the street"
            ),
            guidance_scale: float = Input(
                description="Guidance scale",
                default=5.0
            ),
            height: int = Input(
                description="Image height",
                default=1024
            ),
            width: int = Input(
                description="Image width",
                default=1024
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
            )
    ) -> List[Path]:
        generator = torch.Generator(self.device)
        generator.manual_seed(seed)

        if self.use_refiner:
            image = self.base(
                prompt=prompt,
                denoising_end=0.8,
                output_type="latent",
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images
            images = self.refiner(
                prompt=prompt,
                denoising_start=0.8,
                image=image,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator
            ).images
        else:
            images = self.base(
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
    KServeModel.serve("stable-diffusion-xl", StableDiffusionXL)

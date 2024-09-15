# This script generates a synthetic image dataset using ComfyUI.
# --------------------------------------------------------------------------------


import json
import random
import time
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel

import const
from comfyui_adapter import ComfyUIAdapter
import argparse
from datetime import datetime

c_timestamp_format = "%Y%m%d_%H%M%S"

class PromptPool(BaseModel):
	prefix: str
	suffix: str
	prompts: List[str]

class ImageData(BaseModel):
	prompt: str
	prompt_stem: Optional[str] = ""
	image_path: str
	seed: int

class ImageDataset(BaseModel):
	images: List[ImageData] = []
	metadata: dict = {}

def load_prompt_pool(file_path: str) -> PromptPool:
	with open(file_path, 'r') as f:
		data = json.load(f)
	return PromptPool(**data)

def get_dataset_outdir(base_dir: Path) -> Path:
	timestamp = datetime.now().strftime(c_timestamp_format)
	out_dataset = base_dir / f"dataset_{timestamp}"
	out_dataset.mkdir(parents=True, exist_ok=True)
	return out_dataset

def save_dataset(dataset: ImageDataset, file_path: Path):
	with open(file_path, 'w') as f:
		json.dump(dataset.dict(), f, indent=2)

def generate_images(comfy: ComfyUIAdapter,
                    prompt_pool: PromptPool,
                    num_images: int,
                    output_dir: Path) -> ImageDataset:
	dataset = ImageDataset()
	images_dir = output_dir / "images"
	images_dir.mkdir(exist_ok=True)

	for i in range(num_images):
		# Randomly select a prompt
		prompt = random.choice(prompt_pool.prompts)
		full_prompt = f"{prompt_pool.prefix}{prompt}{prompt_pool.suffix}".strip()
		print(f"Generating image {i + 1}/{num_images}")
		print(f"Full prompt: {full_prompt}")

		# Set a random seed
		random_seed = random.randint(1, 999999)  # Full range of 32-bit integer
		print(f"Using seed: {random_seed}")

		# Generate the image using txt2img method
		img = comfy.txt2img(prompt=full_prompt, seed=random_seed)

		# Save the image
		image_path = images_dir / f"image_{i:04d}.png"
		img.save(image_path)

		# Add to the dataset
		dataset.images.append(ImageData(
			prompt=full_prompt,
			prompt_stem=prompt,
			image_path=str(image_path.relative_to(output_dir)),
			seed=random_seed
		))

		print(f"Image saved: {image_path}")

		time.sleep(0.1)

	return dataset

def main(args):
	dir_generated_datasets = Path(args.output_dir)
	prompt_pool = load_prompt_pool(args.prompt_pool)

	# Initialize the ComfyUIAdapter
	comfy = ComfyUIAdapter()

	out_dataset = get_dataset_outdir(dir_generated_datasets)
	print(f"Creating new dataset in: {out_dataset}")

	dataset = generate_images(comfy, prompt_pool, args.num_images, out_dataset)

	# Add metadata
	dataset.metadata = {
		"created_at" : datetime.now().isoformat(),
		"num_images" : len(dataset.images),
		"prompt_pool": args.prompt_pool,
	}

	# Save the dataset
	out_dataset_json = out_dataset / "dataset.json"
	save_dataset(dataset, out_dataset_json)
	print(f"Dataset saved to: {out_dataset_json}")

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate a structured sparse image collection for LLM text-to-image tasks")
	parser.add_argument("--prompts", type=str, default="prompts.json", help="Path to the prompt pool JSON file")
	parser.add_argument("--out", type=str, default=const.dir_image_datasets, help="Base directory for generated output datasets")
	parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate")

	args = parser.parse_args()
	main(args)

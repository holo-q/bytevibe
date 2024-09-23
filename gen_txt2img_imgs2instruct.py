# This script takes an existing image dataset generated using gen_txt2img_imgs.py
# and generates an instruct dataset.
# It can optionally compress the images using the compression estimation models
# which we have trained here in this repo.
# --------------------------------------------------------------------------------

import argparse
import base64
import json
import math
import random
import shutil
from pathlib import Path
from typing import List, Optional

import cv2
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

import const
from gen_txt2img_imgs import ImageDataset
from src_models.img_compression_estimation_models import NeuralNetworkCompressor
from utils import try_compression

# Magic byte sequence to announce change in modality
MAGIC_BYTES = b'\xF0\x9F\x8C\x88'  # Unicode rainbow emoji as magic bytes

_console = Console()

DATASET_PREFIX = "txt2img"

USER_PROMPTS = [
	"Can you create an image of {prompt}?",
	"I'd like to see a picture of {prompt}. Can you make that?",
	"Generate an image showing {prompt}.",
	"Could you produce an image depicting {prompt}?",
	"I'm curious about {prompt}. Can you create an image of it?",
	"txt2img({prompt})",
	"txt2{format}({prompt})",
]

ASSISTANT_PROMPTS = [
	"Certainly! I'll create an image based on your request.",
	"Of course! I'll generate the image for you.",
	"Sure thing! I'll create an image based on this request.",
	"Absolutely! I'll generate an image for you.",
	"I'm ready to create an image based on your request.",
]

out_workdir = const.dir_instruct_datasets / ".llm_txt2img_dataset"

class CompressedImageData(BaseModel):
	file_path: str
	compression_format: str
	compression_params: dict

class InstructSample(BaseModel):
	id: int
	user_prompt: str
	image_prompt: str
	original_image_path: str
	compressed_image: Optional[CompressedImageData] = None
	seed: Optional[int] = None
	instruct_sample_path: Optional[str] = None

class InstructDataset(BaseModel):
	samples: List[InstructSample] = Field(default_factory=list)
	metadata: dict = Field(default_factory=dict)

def load_json(in_file_path: str) -> dict:
	try:
		with open(in_file_path, 'r') as f:
			return json.load(f)
	except FileNotFoundError:
		_console.print(f"[bold red]Error: File {in_file_path} not found.")
		raise
	except json.JSONDecodeError:
		_console.print(f"[bold red]Error: File {in_file_path} is not valid JSON.")
		raise

def save_json(data: dict, out_file_path: str):
	try:
		with open(out_file_path, 'w') as f:
			json.dump(data, f, indent=2)
	except IOError:
		_console.print(f"[bold red]Error: Unable to write to file {out_file_path}.")
		raise

def parse_range(value: str) -> List[int]:
	list_result = []
	for part in value.split(','):
		if '-' in part:
			start, end = map(int, part.split('-'))
			list_result.extend(range(start, end + 1))
		else:
			list_result.append(int(part))
	return list_result

def overfitting_guided_compression(in_image_path,
                                   out_compressed_images,
                                   cfg_compression_formats,
                                   cfg_compression_params,
                                   cfg_resize,
                                   byte_range,
                                   console,
                                   compressor):
	in_img = cv2.imread(str(in_image_path))
	quality_range = parse_range(cfg_compression_params.get("quality", "2-100"))
	width_range = parse_range(cfg_resize) if cfg_resize else [in_img.shape[1]]
	min_bytes, max_bytes = byte_range

	max_iterations = 50
	samples_per_iteration = 5

	for iteration in range(max_iterations):
		console.print(f"[cyan]Iteration {iteration + 1}[/cyan]")

		new_data = []
		for _ in range(samples_per_iteration):
			fmt = random.choice(cfg_compression_formats)
			quality = random.randint(min(quality_range), max(quality_range))
			width = random.randint(min(width_range), max(width_range))

			try:
				compressed, used_quality, used_width = try_compression(in_img, fmt, quality, width)
				size = len(compressed)
				new_data.append((used_quality, used_width, size))
				console.print(f"[dim]Sample: {fmt} q={used_quality} w={used_width}\t{size}b[/dim]")

				if min_bytes <= size <= max_bytes:
					compressor.update_model(new_data)
					return CompressedImageData(
						file_path=save_compressed_image(in_image_path, out_compressed_images, compressed, fmt, used_quality, used_width),
						compression_format=fmt,
						compression_params={"quality": used_quality,
						                    "resize" : used_width}
					)

			except Exception as e:
				console.print(f"[red]Compression failed: {fmt} q={quality} w={width}\t{str(e)}[/red]")

		compressor.update_model(new_data)

		# Model-guided optimization
		optimized_params = compressor.get_optimized_params(quality_range, width_range, (
			                                                                               min_bytes + max_bytes) / 2)

		for quality, width in optimized_params:
			try:
				compressed, used_quality, used_width = try_compression(in_img, fmt, quality, width)
				size = len(compressed)
				if min_bytes <= size <= max_bytes:
					return CompressedImageData(
						file_path=save_compressed_image(in_image_path, out_compressed_images, compressed, fmt, used_quality, used_width),
						compression_format=fmt,
						compression_params={"quality": used_quality,
						                    "resize" : used_width}
					)
			except Exception as e:
				console.print(f"[red]Optimized compression failed: q={quality} w={width}\t{str(e)}[/red]")

	raise ValueError(f"Unable to compress image within {min_bytes}-{max_bytes} bytes after {max_iterations} iterations")

def save_compressed_image(in_image_path: str,
                          out_compressed_images: Path,
                          compressed: bytes,
                          fmt: str,
                          quality: int,
                          width: int) -> str:
	stem = Path(in_image_path).stem
	out_filename = f"{stem}_{fmt}_q{quality}_w{width}"
	out_path = out_compressed_images / f"{out_filename}.{fmt}"
	with open(out_path, "wb") as f:
		f.write(compressed)
	return str(out_path)

def create_instruct_sample(cfg_user_prompt: str,
                           cfg_assistant_acceptance: str,
                           in_image_path: str) -> bytes:
	sample_start = (
		f"User: {cfg_user_prompt}\n\n"
		f"Assistant: {cfg_assistant_acceptance}\n\n"
	).encode('utf-8')

	with open(in_image_path, 'rb') as f:
		image_bytes = f.read()

	sample_end = (
		f"\n\nHere's the image I've generated based on your request."
	).encode('utf-8')

	return sample_start + MAGIC_BYTES + image_bytes + MAGIC_BYTES + sample_end

def write_instruct_sample(sample: bytes, out_path: Path) -> None:
	with open(out_path, 'wb') as f:
		f.write(sample)

def generate_dataset(src_dataset: ImageDataset,
                     in_dataset_stem: str,
                     cfg_compression_formats: List[str],
                     cfg_compression_params: dict,
                     cfg_resize: Optional[str],
                     byte_range: Optional[int] = None) -> InstructDataset:
	compressor = NeuralNetworkCompressor(checkpoint_dir=const.dir_checkpoints)
	compressor.load_checkpoint()

	out_dataset = InstructDataset()
	shutil.rmtree(out_workdir, ignore_errors=True)
	out_workdir.mkdir(exist_ok=True, parents=True)
	b_use_compression = len(cfg_compression_formats) > 0 and cfg_compression_formats != [
		"none"]

	out_compressed_images = out_workdir / "compressed_images" if b_use_compression else None
	if b_use_compression:
		out_compressed_images.mkdir(exist_ok=True)

	out_instruct_samples = const.dir_instruct_datasets / f"{DATASET_PREFIX}_{in_dataset_stem}"
	out_instruct_samples.mkdir(exist_ok=True)

	with Progress() as progress:
		task = progress.add_task("[cyan]Generating instruct samples...", total=len(src_dataset.images))
		for i, entry in enumerate(src_dataset.images):
			img_prompt = entry.prompt
			user_prompt = random.choice(USER_PROMPTS)
			assistant_prompt = random.choice(ASSISTANT_PROMPTS)

			in_sample_image = const.dir_image_datasets / Path(entry.image_path)
			if not in_sample_image.exists():
				progress.console.print(f"[yellow]Warning: Skipping item {i} due to missing image file: {entry.image_path}")
				continue

			compressed_image = None
			if b_use_compression:
				compressed_image = overfitting_guided_compression(
					in_image_path=in_sample_image,
					out_compressed_images=out_compressed_images,
					cfg_compression_formats=cfg_compression_formats,
					cfg_compression_params=cfg_compression_params,
					cfg_resize=cfg_resize,
					byte_range=byte_range,
					console=progress.console,
					compressor=compressor
				)

			in_image_path = compressed_image.file_path if compressed_image else const.dir_image_datasets / entry.image_path

			user_prompt = user_prompt.replace("{prompt}", img_prompt.lower())
			user_prompt = user_prompt.replace("{compression}", compressed_image.compression_format if b_use_compression else "img")
			instruct_sample = create_instruct_sample(user_prompt, assistant_prompt, in_image_path)

			out_instruct_sample_path = out_instruct_samples / f"sample_{i:04d}.bin"
			write_instruct_sample(instruct_sample, out_instruct_sample_path)

			instruct_sample = InstructSample(
				id=i,
				user_prompt=user_prompt,
				image_prompt=img_prompt,
				original_image_path=entry.image_path,
				compressed_image=compressed_image,
				seed=entry.seed,
				instruct_sample_path=str(out_instruct_sample_path.relative_to(const.dir_instruct_datasets))
			)

			out_dataset.samples.append(instruct_sample)

			progress.update(task, advance=1, description=f"[cyan]Processing: {out_instruct_sample_path.name}")
			progress.console.print(f"[green]{out_instruct_sample_path.name}[/green] -> [yellow]{entry.prompt}[/yellow]")

	compressor.save_checkpoint()
	return out_dataset

def main(args):
	_console.print(Panel.fit("Instruct Image Dataset Generator", style="bold magenta"))
	b_use_compression = args.compression != "none"

	try:
		in_dataset_stem = Path(args.input_dataset).stem
		in_dataset_json = const.dir_image_datasets / f"{in_dataset_stem}.json"
		out_json = const.dir_instruct_datasets / f"txt2img_{in_dataset_stem}.json"

		with _console.status("[bold green]Loading existing dataset...") as status:
			src_dataset = ImageDataset(**load_json(in_dataset_json.as_posix()))
			_console.log(f"[green]Loaded existing dataset with {len(src_dataset.images)} images")

		_console.print("[bold green]Generating instruct samples...")
		cfg_compression_params = {
			"quality"          : args.quality,
			"compression_level": args.compression_level
		}
		out_dataset = generate_dataset(
			src_dataset,
			in_dataset_stem,
			args.compression,
			cfg_compression_params,
			args.resize,
			(args.min_bytes, args.max_bytes)
		)

		with _console.status("[bold green]Saving dataset...") as status:
			out_dataset.metadata = {
				'num_samples'       : len(out_dataset.samples),
				'magic_bytes'       : base64.b64encode(MAGIC_BYTES).decode('ascii'),
				'format_version'    : '1.0',
				'description'       : 'Instruct dataset with text prompts and compressed images',
				'compression'       : args.compression if b_use_compression else 'none',
				'compression_params': cfg_compression_params if b_use_compression else {},
				'use_compression'   : b_use_compression,
				'resize'            : args.resize
			}

			save_json(out_dataset.dict(), out_json)
			_console.log(f"[green]Saved dataset to {out_json}")

		_console.print(Panel.fit("Dataset generation complete!", style="bold green"))
		_console.print(f"[yellow]Total samples: {len(out_dataset.samples)}")
		_console.print(f"[yellow]Output file: {out_json}")
		_console.print(f"[yellow]Work directory: {out_workdir}")

	except Exception as e:
		_console.print(f"[bold red]An error occurred: {str(e)}")
		import traceback
		_console.print(Panel(traceback.format_exc(), title="Error Details", expand=False, border_style="red"))

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Generate an instruct dataset with optional compressed images")
	parser.add_argument("--input-dataset", type=str, required=True, help="Name of the input dataset file (without .json extension)")
	parser.add_argument("--output-dir", type=str, default=const.dir_instruct_datasets.relative_to(const.project_root), help="Directory to save the output files")
	parser.add_argument("--compression", type=str, nargs='+', choices=["jpeg",
	                                                                   "png",
	                                                                   "webp"], default=[
		"none"], help="Image compression format(s) (if compression is enabled)")
	parser.add_argument("--quality", type=str, default="90", help="Quality for JPEG and WebP compression (0-100, can use ranges and commas)")
	parser.add_argument("--compression-level", type=str, default="3", help="Compression level for PNG (0-9, can use ranges and commas)")
	parser.add_argument("--resize", type=str, help="Resize images by width (can use ranges and commas)")
	parser.add_argument("--min-bytes", type=int, default=0, help="Minimum required size for compressed images in bytes")
	parser.add_argument("--max-bytes", type=int, default=math.inf, help="Maximum allowed size for compressed images in bytes")

	args = parser.parse_args()

	if args.output_dir:
		const.dir_instruct_datasets = Path(args.output_dir)
		const.dir_instruct_datasets.mkdir(exist_ok=True, parents=True)

	main(args)

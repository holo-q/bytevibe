"""
# RetNPhi: Byte-Level Hybrid of Phi-3.5 and RetNet

RetNPhi is an experimental architecture that transforms Phi-3.5 into a byte-level language model, incorporating RetNet-inspired mechanisms. This innovative approach enables the model to process raw byte sequences, allowing for universal file type handling.

## Key Features:

1. **Byte-Level Processing**: Operates directly on raw byte sequences, enabling universal application to any file type.
2. **RetNet Integration**: Incorporates RetNet's multi-scale exponential decay and group normalization for efficient long-range dependency modeling.
3. **Dual-mode Processing**: Supports parallel mode for efficient training and recurrent mode for inference.
4. **Selective Fine-tuning**: Trains only specific layers (e.g., token embedding, post-attention layer normalizations) while keeping most of the original Phi-3.5 weights frozen.
5. **Weight-Decomposed Low-Rank Adaptation (DoRA)**: Applies DoRA to self-attention output projections for efficient adaptation while preserving pretrained knowledge.

## Implementation Strategy:

- **Weight Reuse**: Utilizes frozen weights from the original Phi-3.5 model for most layers.
- **Flexible DoRA Application**: Allows configuration of which layers and targets to apply DoRA.
- **Configurable Architecture**: Supports both retention-based and original attention mechanisms.
- **Untied Embeddings Option**: Provides the ability to use separate input and output embeddings.

## Training and Inference:

- Implements efficient training loops with customizable learning rate schedules.
- Supports both training from scratch and fine-tuning from a checkpoint.
- Provides a generation function for text completion tasks.

## Goals:

- Explore the potential of retention-like mechanisms in a byte-level Phi architecture.
- Leverage dual-mode processing for efficient training and inference.
- Develop a universal model capable of processing any file type.

Note: This is a highly experimental implementation, designed for research and exploration rather than production use. It demonstrates the potential of combining pretrained models with novel architectures and efficient fine-tuning techniques.

Author: Josef Albers
Date: Aug 28, 2024
"""

import argparse
import gc
import json
import math
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download
from pyarrow import Table
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskID, TextColumn
from rich.prompt import Prompt
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from lion_pytorch import Lion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_console = Console()

# autograd.set_detect_anomaly(True)

class Tokenizer:
	def __init__(self, file_path=None):
		if file_path is None:
			self.vocab = list(range(256))
		else:
			with open(file_path, 'r') as f:
				content = f.read().lower().encode('utf-8')
			self.vocab = sorted(set(content))
		self.vocab_size = len(self.vocab)
		self.byte_to_index = {byte: index for index, byte in enumerate(self.vocab)}
		self.index_to_byte = {index: byte for index, byte in enumerate(self.vocab)}

	def encode(self, text):
		byte_seq = text.encode('utf-8')
		return [self.byte_to_index.get(byte, 0) for byte in byte_seq]

	def decode(self, indices):
		byte_seq = bytes([self.index_to_byte.get(index, 0) for index in
		                  indices])
		return byte_seq.decode('utf-8', errors='ignore')

class SuRoPE(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dim = config.hidden_size // config.num_attention_heads
		self.original_max_position_embeddings = config.original_max_position_embeddings
		self.rope_theta = config.rope_theta  # Ensure this is a float, not a tensor
		self.scaling_factor = math.sqrt(
			1 + math.log(config.max_position_embeddings / config.original_max_position_embeddings) / math.log(config.original_max_position_embeddings)
		)
		self.register_buffer('long_factor', torch.tensor(config.rope_scaling["long_factor"], dtype=torch.float32))
		self.register_buffer('short_factor', torch.tensor(config.rope_scaling["short_factor"], dtype=torch.float32))

		# Precompute inv_freq if possible
		inv_freq = 1.0 / (self.short_factor * torch.pow(self.rope_theta, (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)))
		self.register_buffer('inv_freq', inv_freq)

	def forward(self, q, k, position_ids):
		cos, sin = self._get_cos_sin(position_ids)
		q = (q * cos) + (self._rotate_half(q) * sin)
		k = (k * cos) + (self._rotate_half(k) * sin)
		return q, k

	def _get_cos_sin(self, position_ids):
		su_factor = self.short_factor  # Tensor scalar

		# Compute inv_freq for the given position_ids
		# Ensure device and dtype consistency
		device = position_ids.device
		dtype = position_ids.dtype

		# Create a range tensor based on position_ids
		# Assuming position_ids shape: (batch_size, seq_len)
		batch_size, seq_len = position_ids.shape

		# Create a tensor of exponents: (dim/2,)
		exponents = torch.pow(self.rope_theta, (torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
		inv_freq = 1.0 / (su_factor * exponents)  # (dim/2,)

		# Expand inv_freq to match batch and seq_len
		inv_freq = inv_freq.unsqueeze(0).unsqueeze(-1)  # (1, dim/2, 1)
		freqs = torch.matmul(inv_freq, position_ids.unsqueeze(1))  # (batch_size, dim/2, seq_len)
		freqs = freqs.transpose(1, 2)  # (batch_size, seq_len, dim/2)

		emb = torch.cat([freqs, freqs], dim=-1)  # (batch_size, seq_len, dim)
		cos = torch.cos(emb) * self.scaling_factor  # (batch_size, seq_len, dim)
		sin = torch.sin(emb) * self.scaling_factor  # (batch_size, seq_len, dim)

		cos = cos.unsqueeze(1)  # (batch_size, 1, seq_len, dim)
		sin = sin.unsqueeze(1)  # (batch_size, 1, seq_len, dim)

		return cos, sin

	def _rotate_half(self, x):
		midpoint = x.size(-1) // 2
		x1 = x[..., :midpoint]
		x2 = x[..., midpoint:]
		return torch.cat([-x2, x1], dim=-1)

class Phi3Attention(nn.Module):
	def __init__(self, config):
		super().__init__()
		dim = config.hidden_size
		self.n_heads = n_heads = config.num_attention_heads
		self.n_kv_heads = n_kv_heads = config.num_key_value_heads
		self.num_hidden_layers = config.num_hidden_layers
		self.head_dim = head_dim = config.hidden_size // n_heads
		self.scale = head_dim ** -0.5
		chop_1 = self.n_heads * self.head_dim
		chop_2 = chop_1 + self.n_kv_heads * self.head_dim
		self.chop = (chop_1, chop_2)
		op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
		self.qkv_proj = nn.Linear(dim, op_size, bias=False)
		self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
		self.rope = SuRoPE(config)

	def forward(self,
	            x,
	            position_ids=None,
	            attention_mask=None,
	            cache=None,
	            use_recurrent_mode=False):
		B, L, _ = x.shape
		qkv = self.qkv_proj(x)
		q, k, v = torch.split(qkv, [self.chop[0], self.chop[1] - self.chop[0],
		                            qkv.size(-1) - self.chop[1]], dim=-1)
		q = q.view(B, L, self.n_heads, -1).transpose(1, 2)  # (B, n_heads, L, head_dim)
		k = k.view(B, L, self.n_kv_heads, -1).transpose(1, 2)  # (B, n_kv_heads, L, head_dim)
		v = v.view(B, L, self.n_kv_heads, -1).transpose(1, 2)  # (B, n_kv_heads, L, head_dim)
		if cache is None:
			if position_ids is None:
				position_ids = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0)
			q, k = self.rope(q, k, position_ids)
			mask = torch.full((
				L, L), float('-inf'), device=x.device).triu(diagonal=1)
			if attention_mask is not None:
				mask = mask + (1.0 - attention_mask[:, None, :]) * float('-inf')
				mask = mask.unsqueeze(1)  # (B, 1, L, L)
			else:
				mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)
		else:
			past_k, past_v, past_p, past_m = cache
			position_ids = past_p[:, -1:] + 1
			q, k = self.rope(q, k, position_ids)
			k = torch.cat([past_k, k], dim=2)
			v = torch.cat([past_v, v], dim=2)
			mask = F.pad(past_m[:, :, -1:, :], (0, 1), value=float('-inf'))
		cache = (k, v, position_ids, mask)
		attn_weights = torch.matmul(q * self.scale, k.transpose(-2, -1))
		attn_weights += mask
		attn_weights = F.softmax(attn_weights, dim=-1)
		o = torch.matmul(attn_weights, v)
		o = o.transpose(1, 2).contiguous().view(B, L, -1)
		return self.o_proj(o).type_as(x), cache

class Phi3Retention(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.dim = dim = config.hidden_size
		self.n_heads = n_heads = config.num_attention_heads
		self.n_kv_heads = n_kv_heads = config.num_key_value_heads
		self.num_hidden_layers = config.num_hidden_layers
		self.head_dim = head_dim = config.hidden_size // n_heads
		self.scale = head_dim ** -0.5
		chop_1 = self.n_heads * self.head_dim
		chop_2 = chop_1 + self.n_kv_heads * self.head_dim
		self.chop = (chop_1, chop_2)
		op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
		self.qkv_proj = nn.Linear(dim, op_size, bias=False)
		self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
		self.rope = SuRoPE(config)
		xmin, xmax = math.log(1 / 32), math.log(1 / 512)
		x = torch.linspace(xmin, xmax, steps=n_heads)

		# self.gamma = 1 - torch.exp(x).to(dtype=torch.float32)

		gamma = 1 - torch.exp(x)
		self.register_buffer('gamma', gamma, persistent=False)
		self.gn = nn.GroupNorm(num_groups=self.head_dim, num_channels=self.head_dim * self.n_heads, affine=False)

	def forward(self,
	            x,
	            position_ids=None,
	            attention_mask=None,
	            cache=None,
	            use_recurrent_mode=False):
		if use_recurrent_mode:
			return self.recurrent_mode(x, cache)
		B, L, _ = x.shape
		qkv = self.qkv_proj(x)
		q, k, v = torch.split(qkv, [self.chop[0], self.chop[1] - self.chop[0],
		                            qkv.size(-1) - self.chop[1]], dim=-1)
		q = q.view(B, L, self.n_heads, -1).transpose(1, 2)  # (B, n_heads, L, head_dim)
		k = k.view(B, L, self.n_kv_heads, -1).transpose(1, 2)  # (B, n_kv_heads, L, head_dim)
		v = v.view(B, L, self.n_kv_heads, -1).transpose(1, 2)  # (B, n_kv_heads, L, head_dim)
		if position_ids is None:
			position_ids = torch.arange(L, device=x.device, dtype=torch.float32).unsqueeze(0)
		q, k = self.rope(q, k, position_ids)
		cache = None
		w = torch.matmul(q * self.scale, k.transpose(-2, -1))
		decay = self._decay(L).to(x.device)
		w = w * decay
		o = torch.matmul(w, v)
		o = o.transpose(1, 2).contiguous().view(B * L, -1)
		o = self.gn(o).view(B, L, -1)
		return self.o_proj(o).type_as(x), cache

	def recurrent_mode(self, x, cache):
		if cache is None:
			s = torch.zeros(1, self.n_heads, self.head_dim, self.head_dim, device=x.device, dtype=x.dtype)
			n = 0
		else:
			s, n = cache
		qkv = self.qkv_proj(x)
		q, k, v = torch.split(qkv, [
			self.chop[0],
			self.chop[1] - self.chop[0],
			qkv.size(-1) - self.chop[1]
		], dim=-1)
		q = q.view(1, 1, self.n_heads, -1).transpose(1, 2)  # (1, n_heads, 1, head_dim)
		k = k.view(1, 1, self.n_kv_heads, -1).transpose(1, 2)
		v = v.view(1, 1, self.n_kv_heads, -1).transpose(1, 2)
		position_ids = torch.tensor([[n]], device=x.device, dtype=torch.float32)
		q, k = self.rope(q, k, position_ids)
		k = k * self.scale
		s = self.gamma.view(1, -1, 1, 1) * s + torch.matmul(k.transpose(-2, -1), v)
		o = torch.matmul(q, s)
		o = o.transpose(1, 2).contiguous().view(1, -1)
		o = self.gn(o).view(1, 1, -1)
		o = self.o_proj(o).type_as(x)
		return o, (s, n + 1)

	def _decay(self, sequence_length):
		n = torch.arange(sequence_length).unsqueeze(1)
		m = torch.arange(sequence_length).unsqueeze(0)
		D = (self.gamma.view(-1, 1, 1) ** (n - m).clamp(min=0)).unsqueeze(0)
		return D

class Phi3MLP(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.gate_up_proj = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
		self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

	def forward(self, x):
		x = self.gate_up_proj(x)
		gate, x = torch.chunk(x, 2, dim=-1)
		return self.down_proj(F.silu(gate) * x)

class Phi3DecoderLayer(nn.Module):
	def __init__(self, config):
		super().__init__()
		if config.use_retention:
			self.self_attn = Phi3Retention(config)
		else:
			self.self_attn = Phi3Attention(config)
		self.mlp = Phi3MLP(config)
		self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
		self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self,
	            x,
	            position_ids=None,
	            attention_mask=None,
	            cache=None,
	            use_recurrent_mode=False):
		r, cache = self.self_attn(self.input_layernorm(x), position_ids, attention_mask, cache, use_recurrent_mode)
		h = x + r
		r = self.mlp(self.post_attention_layernorm(h))
		return h + r, cache

class Phi3Model(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.embed_new = nn.Embedding(config.vocab_size, config.hidden_size)
		self.layers = nn.ModuleList([Phi3DecoderLayer(config) for _ in range(config.num_hidden_layers)])
		self.norm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

	def forward(self,
	            input_ids,
	            pixel_values=None,
	            image_sizes=None,
	            position_ids=None,
	            attention_mask=None,
	            cache=None,
	            use_recurrent_mode=False):
		x = self.embed_new(input_ids).to(input_ids.device)
		if cache is None:
			cache = [None] * len(self.layers)
		outputs = []
		for i, layer in enumerate(self.layers):
			x, layer_cache = layer(x, position_ids, attention_mask, cache[i], use_recurrent_mode)
			cache[i] = layer_cache
		x = self.norm(x)
		return x, cache

class Phi3ForCausalLM(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.model = Phi3Model(config)
		if config.untie_embedding:
			self.lm_new = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
			self.untie = True
		else:
			self.untie = False

	def forward(self,
	            input_ids,
	            pixel_values=None,
	            image_sizes=None,
	            position_ids=None,
	            attention_mask=None,
	            cache=None,
	            use_recurrent_mode=False):
		x, cache = self.model(input_ids, pixel_values, image_sizes, position_ids, attention_mask, cache, use_recurrent_mode)
		if self.untie:
			logits = self.lm_new(x)
		else:
			logits = F.linear(x, self.model.embed_new.weight)
		return logits, cache

	@property
	def layers(self):
		return self.model.layers

class DoRALinear(nn.Module):
	def __init__(self,
	             input_dims,
	             output_dims,
	             r,
	             alpha,
	             scale,
	             dropout,
	             bias=False):
		super().__init__()
		self.linear = nn.Linear(input_dims, output_dims, bias=bias)
		self.dropout = nn.Dropout(p=dropout)
		self.scale = scale * (alpha / r)
		init_scale = 1 / math.sqrt(input_dims)
		self.lora_a = nn.Parameter(torch.empty(input_dims, r).uniform_(-init_scale, init_scale))
		self.lora_b = nn.Parameter(torch.zeros(r, output_dims))
		self.register_buffer('m', torch.linalg.norm(self.linear.weight.data, dim=1).to(dtype=torch.float32))

	@staticmethod
	def from_linear(linear, r, alpha, scale, dropout):
		output_dims, input_dims = linear.weight.shape
		lora_lin = DoRALinear(input_dims=input_dims, output_dims=output_dims, r=r, alpha=alpha, scale=scale, dropout=dropout, bias=linear.bias is not None)
		lora_lin.linear = linear
		return lora_lin

	def forward(self, x):
		y = self.linear(x)
		z = (self.dropout(x) @ self.lora_a) @ self.lora_b
		z = y + (self.scale * z)
		adapted = self.linear.weight + (self.scale * self.lora_b.t()) @ self.lora_a.t()
		denom = (self.m / torch.linalg.norm(adapted, dim=1)).detach()
		z = (self.m / denom) * z
		return z.type_as(x)

def linear_to_lora_layers(model,
                          lora_layers,
                          lora_targets,
                          lora_rank,
                          lora_scale,
                          lora_dropout):
	_console.print(f"\n[bold green]Applying LoRA...[/bold green]")

	model_layers = model.layers
	if lora_layers == 'all':
		lora_layers = model_layers
	elif isinstance(lora_layers, int):
		lora_layers = model_layers[-lora_layers:]
	elif isinstance(lora_layers, list):
		lora_layers = [model_layers[i] for i in lora_layers]
	else:
		raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")

	def to_lora(layer):
		return DoRALinear.from_linear(layer, r=lora_rank, alpha=lora_rank, scale=lora_scale, dropout=lora_dropout)

	for l in lora_layers:
		for name in lora_targets:
			parts = name.split('.')
			parent = l
			for part in parts[:-1]:
				parent = getattr(parent, part)
			_console.print(f"  Applying LoRA to {name}")
			original_module = getattr(parent, parts[-1])
			lora = to_lora(original_module)
			setattr(parent, parts[-1], lora)

def load_model_for_nnviz():
	model_cfg = {
		'vocab_size'     : 256,
		'use_retention'  : True,
		'untie_embedding': True,
	}
	model = load_base_model(model_cfg, init=False)
	return model

def print_weights_status(model, stage):
	from rich.table import Table

	table = Table(title=f"Weights ({stage}):", show_header=True, header_style="bold magenta")
	table.add_column("Name", justify="left", style="cyan", no_wrap=True)
	table.add_column("Mean", justify="right", style="green")
	table.add_column("Std", justify="right", style="blue")
	table.add_column("Trainable", justify="center", style="yellow")

	for name, param in list(model.named_parameters()):
		is_trainable = "Yes" if param.requires_grad else "No"
		table.add_row(name, f"{param.mean().item():.4f}", f"{param.std().item():.4f}", is_trainable)

	_console.print(table)
	_console.print("\n")


def load_base_model(model_cfg, init=False):
	model_id = 'microsoft/Phi-3.5-mini-instruct'
	model_path = snapshot_download(model_id, allow_patterns=["*.safetensors",
	                                                         "config.json"])

	with open(f"{model_path}/config.json", "r") as f:
		config = json.load(f)
	config = config | model_cfg
	model_config = SimpleNamespace(**config)
	model = Phi3ForCausalLM(model_config)

	from safetensors.torch import load_file

	# Load split safetensors files
	state_dict = {}
	for i in range(1, 3):  # Assuming there are 2 parts, adjust if needed
		file_path = f"{model_path}/model-0000{i}-of-00002.safetensors"
		state_dict.update(load_file(file_path))

	# print_weights_status(model, "zero")
	model.load_state_dict(state_dict, strict=False)
	# print_weights_status(model, "base")

	if init:
		for name, module in model.named_modules():
			if name.endswith('embed_new'):
				nn.init.normal_(module.weight, mean=-0.000453949, std=0.0344238)

		if model_config.untie_embedding:
			for name, module in model.named_modules():
				if name.endswith('lm_new'):
					nn.init.normal_(module.weight, mean=-0.000231743, std=0.043457)

	# Apply quantization
	# model = quantize_model(model, bits=4, group_size=64)

	return model

def load_model_for_training(lora_cfg, model_cfg, thaws, from_path=None):
	model = load_base_model(model_cfg, init=False)
	if from_path:
		state_dict = torch.load(from_path, map_location='cpu')
		model.load_state_dict(state_dict, strict=False)

	# Freeze all parameters
	for param in model.parameters():
		param.requires_grad = False

	# Unfreeze specified modules
	def unfreeze_matching_modules(module, thaws):
		for name, child in module.named_children():
			if any(name.endswith(t) for t in thaws):
				for parameter in child.parameters():
					parameter.requires_grad = True
				print(f"Unfrozen: {name}")
			else:
				unfreeze_matching_modules(child, thaws)

	_console.print(f"\n")

	unfreeze_matching_modules(model, thaws)

	# Apply LoRA after freezing/unfreezing
	_console.print(f"\n[bold green]LoRA Config:[/bold green]")
	_console.print(f"  Layers: {lora_cfg['layers']}")
	_console.print(f"  Targets: {lora_cfg['targets']}")
	_console.print(f"  Rank: {lora_cfg['rank']}")
	_console.print(f"  Scale: {lora_cfg['scale']}")
	_console.print(f"  Dropout: {lora_cfg['dropout']}")
	if len(lora_cfg['targets']) >= 1:
		linear_to_lora_layers(model,
			lora_layers=lora_cfg['layers'],
			lora_targets=lora_cfg['targets'],
			lora_rank=lora_cfg['rank'],
			lora_scale=lora_cfg['scale'],
			lora_dropout=lora_cfg['dropout'])

	# Print trainable parameters
	model.train()
	return model

def load_model_for_inference(model_path, model_cfg, lora_cfg):
	model = load_base_model(model_cfg, init=False)
	if len(lora_cfg['targets']) >= 1:
		linear_to_lora_layers(model,
			lora_layers=lora_cfg['layers'],
			lora_targets=lora_cfg['targets'],
			lora_rank=lora_cfg['rank'],
			lora_scale=lora_cfg['scale'],
			lora_dropout=lora_cfg['dropout'])
	state_dict = torch.load(model_path, map_location='cpu')
	model.load_state_dict(state_dict, strict=False)
	model.eval()

	# Print weights to verify they're loaded correctly
	print_weights_status(model, "byte-lora")

	# _console.print("Model Configuration:")
	# _console.print(f"Vocab Size: {model.config.vocab_size}")
	# _console.print(f"Hidden Size: {model.config.hidden_size}")
	# _console.print(f"Num Layers: {model.config.num_hidden_layers}")
	# _console.print(f"Num Attention Heads: {model.config.num_attention_heads}")
	# _console.print(f"Use Retention: {model.config.use_retention}")
	# _console.print(f"Untie Embedding: {model.config.untie_embedding}")

	return model

def save_delta_changes(model, unfrozen_params, path):
	delta_dict = {}
	for name, param in model.named_parameters():
		if name in unfrozen_params:
			delta_dict[name] = param.data

	torch.save(delta_dict, path)
	_console.print(f"Delta changes saved to {path}")
	_console.print(f"Number of saved parameter tensors: {len(delta_dict)}")


def generate(model,
             tokenizer,
             prompt,
             max_tokens=50,
             model_cfg=None,
             verbose=True):
	input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

	if model_cfg['use_retention']:
		cache = None
		for i in input_ids[0]:
			logits, cache = model(i.unsqueeze(0).unsqueeze(0), cache=cache, use_recurrent_mode=True)
	else:
		logits, cache = model(input_ids)

	token = torch.argmax(logits[:, -1, :], dim=-1)
	list_tokens = token.tolist()

	for _ in range(max_tokens):
		logits, cache = model(token.unsqueeze(0), cache=cache, use_recurrent_mode=True)
		token = torch.argmax(logits[:, -1, :], dim=-1)
		list_tokens += token.tolist()
		if tokenizer.decode(list_tokens[-2:]) == '\n\n':
			break

	output = tokenizer.decode(list_tokens)
	if verbose:
		_console.print(f'{prompt=} + {output=}\n-> {prompt + output}')

	return output

# region Training


def create_batches(data, tokenizer, batch_size, seq_length):
	def _encode(x):
		return tokenizer.encode(x['input']), tokenizer.encode(x['output'])

	encoded_data = [_encode(x) for x in data]
	encoded_data = [x for x in encoded_data if len(x[0] + x[1]) <= seq_length + 1]

	if batch_size is None:
		batch_size = min(len(encoded_data), 64)
	else:
		encoded_data = encoded_data[:(len(encoded_data) // batch_size) * batch_size]
		np.random.shuffle(encoded_data)

	for i in range(0, len(encoded_data), batch_size):
		batch = encoded_data[i:i + batch_size]
		max_len = min(max(len(q + a) - 1 for q, a in batch), seq_length)
		x_batch = []
		y_batch = []
		mask_batch = []
		for q, a in batch:
			combined = (q + a)[:max_len + 1]
			x = combined[:-1]
			y = combined[1:]
			pad_length = max_len - len(x)
			x = x + [0] * pad_length
			y = y + [0] * pad_length
			mask = [False] * (len(q) - 1) + [True] * (len(a)) + [
				False] * pad_length
			x_batch.append(x)
			y_batch.append(y)
			mask_batch.append(mask)
		yield torch.tensor(x_batch), torch.tensor(y_batch), torch.tensor(mask_batch)

def get_optimizer(model: nn.Module,
                  tokenizer: Tokenizer,
                  learning_rates: Tuple[float, float],
                  num_epochs: int,
                  train_data: Dataset,
                  batch_size: int,
                  seq_length: int):
	"""
	Initializes the Lion optimizer with a learning rate scheduler that includes
	linear warmup followed by cosine decay.

	Args:
		model (nn.Module): The model to optimize.
		tokenizer (Tokenizer): The tokenizer for the model.
		learning_rates (tuple): A tuple containing (max_lr, min_lr).
		num_epochs (int): Total number of training epochs.
		train_data (Dataset): The training dataset.
		batch_size (int): Size of each training batch.
		seq_length (int): Sequence length for training.

	Returns:
		optimizer (Lion): The initialized Lion optimizer.
		scheduler (LambdaLR): The learning rate scheduler.
		num_steps (int): Total number of training steps.
	"""
	num_batches_per_epoch = len(list(create_batches(train_data, tokenizer, batch_size, seq_length)))
	num_steps = num_epochs * num_batches_per_epoch
	num_warmup = num_steps // 10  # 10% of total steps for warmup

	max_lr, min_lr = learning_rates

	# Initialize the Lion optimizer
	optimizer = Lion(model.parameters(), lr=max_lr, weight_decay=0.1)

	def lr_lambda(current_step):
		"""Learning rate scheduler with linear warmup and cosine decay."""
		if current_step < num_warmup:
			return float(current_step) / float(max(1, num_warmup))
		return 0.5 * (1 + math.cos(math.pi * (
			current_step - num_warmup) / max(1, (
			num_steps - num_warmup))))

	# Initialize the LambdaLR scheduler
	scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

	return optimizer, scheduler, num_steps

def loss_fn(model, X, y, mask):
	logits, _ = model(X)
	logits = logits.float()
	ce = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
	masked_loss = ce * mask.view(-1)
	return masked_loss.sum(), mask.sum()

def evaluate(model, data, tokenizer, seq_length):
	model.eval()
	total_loss = 0
	total_samples = 0
	with torch.no_grad():
		for X, y, mask in create_batches(data, tokenizer, None, seq_length):
			X, y, mask = X.to(device), y.to(device), mask.to(device)
			loss, ntoks = loss_fn(model, X, y, mask)
			total_loss += loss.item()
			total_samples += ntoks.item()
	return total_loss / total_samples if total_samples > 0 else -1

def load_gsm_data(take: int,
                  is_tiny: bool = True,
                  progress: Optional[Progress] = None,
                  task: Optional[TaskID] = None) -> Tuple[Dataset, Dataset]:
	"""
	Load and preprocess the GSM (Grade School Math) dataset.

	This function loads either the TinyGSM dataset or the full GSM8K dataset,
	applies filtering and preprocessing, and splits the data into training and evaluation sets.

	Args:
		take (int): Number of examples to take from the dataset.
		is_tiny (bool): If True, use TinyGSM dataset; if False, use full GSM8K dataset.
		progress (Optional[Progress]): Rich progress bar object for visual feedback.
		task (Optional[TaskID]): Task ID for the progress bar.

	Returns:
		Tuple[Dataset, Dataset]: A tuple containing:
			- train_dataset: Preprocessed training dataset
			- eval_dataset: Preprocessed evaluation dataset

	The function performs the following steps:
	1. Loads the appropriate dataset (TinyGSM or GSM8K)
	2. Applies filtering (for TinyGSM only)
	3. Splits the data into training and evaluation sets
	4. Applies preprocessing to format the examples
	5. Returns the processed datasets
	"""

	def format_example(example: dict) -> dict:
		if is_tiny:
			code_raw = example['code']
			start = code_raw.rfind('\n    """')
			if start == -1:
				_console.print('Wrong format to start')
				return {'input' : example['question'].strip(),
				        'output': code_raw.strip()}
			start += 8
			end = code_raw.rfind('\n    result =')
			if end == -1:
				_console.print('Wrong format to end')
				end = len(code_raw)
			code_block = code_raw[start:end]
			code_lines = code_block.split('\n    ')
			formatted_code = '\n'.join(
				line for line in code_lines if line.strip())
			formatted_code = '\n' + formatted_code.strip() + '\n\n'
			return {'input' : example['question'].strip(),
			        'output': formatted_code}
		else:
			return {
				'input' : example['question'].strip(),
				'output': '\n' + example['answer'].strip() + '\n\n'
			}

	# Load and preprocess the dataset
	if is_tiny:
		data = load_dataset("TinyGSM/TinyGSM")["train"]
		data = data.select(range(take))
		data = data.filter(lambda x: len(x['question']) < 100 and
		                             ':' not in x['question'] and
		                             '-' not in x['question'] and
		                             "'" not in x['code'] and
		                             '\n    result =' in x['code'])
		split_point = int(len(data) * 0.8)
		train_data = data.select(range(split_point))
		eval_data = data.select(range(split_point, len(data)))
	else:
		dataset = load_dataset("openai/gsm8k", "main")
		train_data = dataset["train"].select(range(min(take, len(
			dataset["train"]))))
		eval_data = dataset["test"].select(range(min(take // 5, len(
			dataset["test"]))))

	if progress:
		progress.update(task, advance=25)

	# Apply formatting to the datasets
	train_dataset = train_data.map(format_example)
	eval_dataset = eval_data.map(format_example)

	if progress:
		progress.update(task, advance=25)

	return train_dataset, eval_dataset

class CustomDataset(Dataset):
	def __init__(self, data, arrow_table: Table):
		super().__init__(arrow_table)
		self._data = data

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

def train_gsm(learning_rates,
              num_epochs,
              batch_size,
              seq_length,
              lora_cfg,
              model_cfg,
              thaws,
              take,
              from_path=None,
              save_every_n_epoch=1,
              infer_every_n_epoch=1,
              accumulation_steps=1):
	tokenizer = Tokenizer()
	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	_console.print(f"[bold cyan]Training started at {timestamp}[/bold cyan]")

	with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), BarColumn(), TextColumn("[progress.percentage]{task.percentage:>3.0f}%")) as progress:
		data_task = progress.add_task("[green]Loading data...", total=100)
		train_dataset, eval_dataset = load_gsm_data(take=take, progress=progress, task=data_task)
		train_loader = DataLoader(
			train_dataset,
			batch_size=batch_size,
			shuffle=True,
			num_workers=4,
			pin_memory=True
		)
		progress.update(data_task, completed=100)
		_console.print(f"Train data: {len(train_dataset)} samples, Eval data: {len(eval_dataset)} samples")

		model_task = progress.add_task("[blue]Loading model...", total=100)
		model = load_model_for_training(lora_cfg=lora_cfg, model_cfg=model_cfg, thaws=thaws, from_path=from_path).to(device)
		progress.update(model_task, completed=100)

		optimizer_task = progress.add_task("[yellow]Preparing optimizer...", total=100)
		optimizer, scheduler, num_steps = get_optimizer(model, tokenizer, learning_rates, num_epochs, train_dataset, batch_size, seq_length)
		progress.update(optimizer_task, completed=100)

	warm_params = {name for name, param in model.named_parameters() if
	               param.requires_grad}
	metrics = {'steps'       : [],
	           'lr'          : [],
	           'loss'        : [],
	           'trained_toks': [],
	           'eval_loss'   : []}
	scaler = GradScaler()

	_console.print("[bold]Warm Parameters:[/bold]")
	for name, param in model.named_parameters():
		if param.requires_grad:
			_console.print(f"{name}: {param.data.shape}")

	_console.print("\n[bold]Training Progress:[/bold]")
	for i_epoch in range(num_epochs):
		_console.print(f"\n[bold cyan]Epoch {i_epoch + 1}/{num_epochs}[/bold cyan]")
		epoch_start_time = time.time()
		epoch_loss = 0
		epoch_tokens = 0
		step_losses = []
		clear_memory(prints=True)

		for step, (X, y,
		           mask) in enumerate(create_batches(train_dataset, tokenizer, batch_size, seq_length), 1):
			X, y, mask = (X.to(device, non_blocking=True),
			              y.to(device, non_blocking=True),
			              mask.to(device, non_blocking=True))

			clear_memory(prints=False)

			with autocast():
				loss, ntoks = loss_fn(model, X, y, mask)
				loss = loss / accumulation_steps

			scaler.scale(loss).backward()

			epoch_loss += loss.item() * accumulation_steps
			epoch_tokens += ntoks.item()

			if step % accumulation_steps == 0:
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
				scaler.step(optimizer)
				scaler.update()
				optimizer.zero_grad(set_to_none=True)
				scheduler.step()

			step_losses.append(epoch_loss / epoch_tokens)
			avg_loss = sum(step_losses) / len(step_losses)
			lr = scheduler.get_last_lr()[0]
			_console.print(f"Step: {step}/{num_steps} | Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | LR: {lr:.7f} | Tokens: {ntoks.item()}")

			metrics['steps'].append(step + i_epoch * num_steps)
			# metrics['lr'].append(lr)
			metrics['loss'].append(loss.item())
			metrics['trained_toks'].append(ntoks.item())

			del X, y, mask, loss, ntoks

		# Evaluate at the end of each epoch
		eval_loss = evaluate(model, eval_dataset, tokenizer, seq_length)
		metrics['eval_loss'].append(eval_loss)

		epoch_time = time.time() - epoch_start_time
		avg_epoch_loss = epoch_loss / epoch_tokens
		_console.print(f"[bold green]Epoch {i_epoch + 1} Summary:[/bold green]")
		_console.print(f"  Time: {epoch_time:.2f} seconds")
		_console.print(f"  Training Loss: {avg_epoch_loss:.4f}")
		_console.print(f"  Evaluation Loss: {eval_loss:.4f}")
		_console.print(f"  Tokens Processed: {epoch_tokens}")

		if (i_epoch + 1) % save_every_n_epoch == 0:
			_path = Path(f'trained_retnphi.safetensors' if model_cfg[
				'use_retention'] else f'trained_orgnphi.safetensors')
			_path = _path.with_stem(f'{_path.stem}_epoch-{i_epoch + 1}')
			save_delta_changes(model, warm_params, _path)
			_console.print(f"[bold blue]Checkpoint saved to {_path}[/bold blue]")

		if (i_epoch + 1) % infer_every_n_epoch == 0:
			inference_prompt = "Solve this math problem: 7 * 6 = ?"
			model.eval()
			output_text = generate(model, tokenizer, inference_prompt, model_cfg=model_cfg, verbose=False)
			model.train()
			inference_output = output_text
			_console.print(f"\n[magenta]Inference:[/magenta]")
			_console.print(f"  Input: {inference_prompt}")
			_console.print(f"  Output: {inference_output}\n")

		if (i_epoch + 1) % 5 == 0:
			print_weights_status(model, "current")

		torch.cuda.reset_peak_memory_stats()  # Reset peak stats for the next epoch

	# Final model save
	_path = f'trained_retnphi.safetensors' if model_cfg[
		'use_retention'] else f'trained_orgnphi.safetensors'
	save_delta_changes(model, warm_params, _path)
	_console.print(f"\n[bold green]Final model saved to {_path}[/bold green]")

	# Save training log
	log = {
		'args'   : {
			'learning_rates': learning_rates,
			'num_epochs'    : num_epochs,
			'batch_size'    : batch_size,
			'seq_length'    : seq_length,
			'lora_cfg'      : lora_cfg,
			'model_cfg'     : model_cfg,
			'thaws'         : thaws,
			'from_path'     : from_path
		},
		'metrics': metrics
	}
	log_path = f'train_log_{timestamp}.json'
	with open(log_path, 'w') as f:
		json.dump(log, f, indent=2)
	_console.print(f"[bold blue]Training log saved to {log_path}[/bold blue]")

	# Done!
	_console.print("\n[bold green]Training completed successfully![/bold green]")

# Assuming these

# endregion

# region Main
def select_next_token(logits, temperature=0.7):
	logits = logits / temperature
	probs = logits.softmax(dim=-1)
	return torch.multinomial(probs, num_samples=1).item()


def interact_with_model(model, model_cfg, max_tokens=50):
	tokenizer = Tokenizer()

	console = Console()
	console.print("[bold green]Model loaded. You can now interact with it. Type 'exit' to quit.[/bold green]")

	while True:
		user_input = Prompt.ask("[bold cyan]You")
		if user_input.lower() == 'exit':
			break

		input_ids = torch.tensor(tokenizer.encode(user_input)).to(device)
		console.print(f"Input shape: {input_ids.shape}")
		console.print(f"Input tokens: {input_ids.tolist()}")

		if model_cfg['use_retention']:
			cache = None
			for i in input_ids:
				logits, cache = model(i.unsqueeze(0).unsqueeze(0), cache=cache, use_recurrent_mode=True)
		else:
			logits, cache = model(input_ids.unsqueeze(0))

		console.print(f"Initial logits shape: {logits.shape}")

		output_tokens = []
		for _ in range(max_tokens):
			last_logits = logits[0, -1, :]  # Ensure we're working with a 1D tensor
			console.print(f"Last logits shape: {last_logits.shape}")
			console.print(f"Last logits min/max/mean: {last_logits.min().item():.2f}/{last_logits.max().item():.2f}/{last_logits.mean().item():.2f}")

			# Use topk on the 1D tensor
			top_values, top_indices = torch.topk(last_logits, 5)
			console.print(f"Top 5 predicted tokens: {top_indices.tolist()}")
			console.print(f"Top 5 token probabilities: {top_values.softmax(dim=-1).tolist()}")

			token = top_indices[0]  # Select the top token
			output_tokens.append(token.item())

			if len(output_tokens) >= 2 and tokenizer.decode(output_tokens[-2:]) == '\n\n':
				break

			if model_cfg['use_retention']:
				logits, cache = model(token.unsqueeze(0).unsqueeze(0), cache=cache, use_recurrent_mode=True)
			else:
				logits, cache = model(token.unsqueeze(0).unsqueeze(0), cache=cache)

		output_text = tokenizer.decode(output_tokens)
		console.print(f"[bold yellow]Model: {output_text}[/bold yellow]")
		console.print(f"Output tokens: {output_tokens}")

def clear_memory(prints=True):
	if prints:
		print_gpu_memory("before")

	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	if prints:
		print_gpu_memory("after")

def print_gpu_memory(label=''):
	if not torch.cuda.is_available(): return
	header = f"[bold green]{label}[/bold green] " if label else ""
	_console.print(f"{header}"
	               f"GPU memory: allocated {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
	               f"cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB, "
	               f"max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
def main():
	parser = argparse.ArgumentParser(description="Train or test a language model.")
	parser.add_argument("--mode", choices=["train", "chat",
	                                       "prompt"], default="train", help="Mode of operation: train or test")
	parser.add_argument("--prompt", type=str, help="Prompt for prompt mode", default="Solve this math problem: 7 * 6 = ?")
	parser.add_argument("--ckpt", type=str, help="Path to the model for testing or continuing training")
	parser.add_argument("--learning_rate_max", type=float, default=1e-4, help="Maximum learning rate")
	parser.add_argument("--learning_rate_min", type=float, default=1e-5, help="Minimum learning rate")
	parser.add_argument("--num_epochs", type=int, default=90, help="Number of training epochs")
	parser.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
	parser.add_argument("--seq_length", type=int, default=256, help="Sequence length for training")
	parser.add_argument("--lora_rank", type=int, default=32, help="Rank for LoRA")
	parser.add_argument("--lora_scale", type=float, default=0.1, help="Scale for LoRA")
	parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout for LoRA")
	parser.add_argument("--use_retention", type=bool, default=True, help="Use retention mechanism")
	parser.add_argument("--untie_embedding", type=bool, default=True, help="Untie embedding")
	parser.add_argument("--take", type=int, default=1000, help="Number of examples to take from the dataset")
	parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
	parser.add_argument("--infer_every", type=int, default=1, help="Run inference every N epoch")
	parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate")
	args = parser.parse_args()
	lora_cfg = {
		'layers' : 'all',
		'targets': ["self_attn.o_proj"],
		'rank'   : args.lora_rank,
		'scale'  : args.lora_scale,
		'dropout': args.lora_dropout
	}
	model_cfg = {
		'vocab_size'     : 256,
		'use_retention'  : args.use_retention,
		'untie_embedding': args.untie_embedding
	}
	if args.mode == "train":
		# torch.backends.cuda.matmul.allow_tf32 = True
		# torch.backends.cudnn.allow_tf32 = True
		train_gsm(
			learning_rates=(args.learning_rate_max, args.learning_rate_min),
			num_epochs=args.num_epochs,
			batch_size=args.batch_size,
			seq_length=args.seq_length,
			lora_cfg=lora_cfg,
			model_cfg=model_cfg,
			thaws=['new', 'post_attention_layernorm'],
			take=args.take,
			from_path=args.ckpt,
			save_every_n_epoch=args.save_every,
			infer_every_n_epoch=args.infer_every
		)
	elif args.mode == "chat":
		if not args.ckpt:
			raise ValueError("Model path must be provided for testing mode.")

		model = load_model_for_inference(args.ckpt, model_cfg, lora_cfg).to('cuda')
		interact_with_model(model, model_cfg, max_tokens=args.max_tokens)
	elif args.mode == "prompt":
		if not args.ckpt:
			raise ValueError("Model path must be provided for testing mode.")

		model = load_model_for_inference(args.ckpt, model_cfg, lora_cfg).to('cuda')
		tokenizer = Tokenizer()
		output = generate(model, tokenizer, args.prompt, max_tokens=args.max_tokens, model_cfg=model_cfg, verbose=False)
		_console.print(f"[bold yellow]Model: {output}[/bold yellow]")

if __name__ == "__main__":
	main()
# endregion

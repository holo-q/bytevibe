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

import glob
import json
import math
import time
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from datasets import load_dataset
from lion_pytorch import Lion  # Ensure you have installed lion-pytorch package
from torch import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Tokenizer:
	def __init__(self, file_path=None):
		if file_path is None:
			self.vocab = list(range(256))
		else:
			with open(file_path, 'r') as f:
				content = f.read().lower().encode('utf-8')
			self.vocab = sorted(set(content))
		self.vocab_size = len(self.vocab)
		self.byte_to_index = {byte: index for index, byte in
		                      enumerate(self.vocab)}
		self.index_to_byte = {index: byte for index, byte in
		                      enumerate(self.vocab)}

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
		self.register_buffer('long_factor', torch.tensor(
			config.rope_scaling["long_factor"], dtype=torch.float32))
		self.register_buffer('short_factor', torch.tensor(
			config.rope_scaling["short_factor"], dtype=torch.float32))

		# Precompute inv_freq if possible
		inv_freq = 1.0 / (self.short_factor * torch.pow(self.rope_theta, (
			torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim)))
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
		exponents = torch.pow(self.rope_theta, (
			torch.arange(0, self.dim, 2, device=device, dtype=torch.float32) / self.dim))
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
		q, k, v = torch.split(qkv, [self.chop[0], self.chop[1] - self.chop[0],
		                            qkv.size(-1) - self.chop[1]], dim=-1)
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
		self.layers = nn.ModuleList([Phi3DecoderLayer(config) for _ in
		                             range(config.num_hidden_layers)])
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
			x, layer_cache = layer(x, position_ids, attention_mask,
				cache[i], use_recurrent_mode)
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
		adapted = self.linear.weight + (
			self.scale * self.lora_b.t()) @ self.lora_a.t()
		denom = (self.m / torch.linalg.norm(adapted, dim=1)).detach()
		z = (self.m / denom) * z
		return z.type_as(x)

def linear_to_lora_layers(model,
                          lora_layers,
                          lora_targets,
                          lora_rank,
                          lora_scale,
                          lora_dropout):
	if lora_layers == 'all':
		lora_layers = model.layers
	elif isinstance(lora_layers, int):
		lora_layers = model.layers[-lora_layers:]
	elif isinstance(lora_layers, list):
		lora_layers = [model.layers[i] for i in lora_layers]
	else:
		raise ValueError("Invalid type for lora_layers. Expected int (number of layers) or list (layer indices or names).")

	def to_lora(module):
		return DoRALinear.from_linear(module, r=lora_rank, alpha=lora_rank, scale=lora_scale, dropout=lora_dropout)

	for layer in lora_layers:
		for name, module in layer.named_modules():
			if name in lora_targets:
				parent_module = layer
				name_parts = name.split('.')
				for part in name_parts[:-1]:
					parent_module = getattr(parent_module, part)
				setattr(parent_module, name_parts[-1], to_lora(module))

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

	model.float()

	if init:
		nn.init.normal_(model.model.embed_new.weight, mean=-0.000453949, std=0.0344238)
		if model_config.untie_embedding:
			nn.init.normal_(model.lm_new.weight, mean=-0.000231743, std=0.043457)

	for name, module in model.named_modules():
		if isinstance(module, nn.Linear) and not name.endswith('new'):
			module = quantize_linear_module(module)

	return model

def quantize_linear_module(module, bits=4):
	# Placeholder function: Implement your quantization here if needed
	return module  # Returning the module as-is for this placeholder

def load_model_for_training(lora_cfg, model_cfg, thaws, from_path=None):
	model = load_base_model(model_cfg, init=False)
	if from_path:
		state_dict = torch.load(from_path, map_location='cuda')
		model.load_state_dict(state_dict, strict=False)
	for param in model.parameters():
		param.requires_grad = False

	if len(lora_cfg['targets']) > 0:
		linear_to_lora_layers(
			model,
			lora_layers=lora_cfg['layers'],
			lora_targets=lora_cfg['targets'],
			lora_rank=lora_cfg['rank'],
			lora_scale=lora_cfg['scale'],
			lora_dropout=lora_cfg['dropout']
		)
	for name, param in model.named_parameters():
		if any(t in name for t in thaws):
			param.requires_grad = True

	model.train()
	return model


def load_model_for_inference(lora_cfg, model_cfg):
	# Load the model in evaluation mode and with reduced precision
	model = load_base_model(model_cfg, init=False)
	model.half()
	model.eval()

	if len(lora_cfg['targets']) > 0:
		linear_to_lora_layers(
			model,
			lora_layers=lora_cfg['layers'],
			lora_targets=lora_cfg['targets'],
			lora_rank=lora_cfg['rank'],
			lora_scale=lora_cfg['scale'],
			lora_dropout=lora_cfg['dropout']
		)

	_path = 'trained_retnphi.pt' if model_cfg[
		'use_retention'] else 'trained_orgnphi.pt'

	# Load state dict with reduced memory usage
	with torch.no_grad():
		state_dict = torch.load(_path, map_location='cpu')
		model.load_state_dict(state_dict, strict=False)
		del state_dict  # Free up memory

	# Move model to GPU if available, otherwise keep on CPU
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model.to(device)

	return model

def generate(prompt, lora_cfg, model_cfg, max_tokens=50, verbose=True):
	tokenizer = Tokenizer()
	model = load_model_for_inference(lora_cfg=lora_cfg, model_cfg=model_cfg)
	model.eval()
	input_ids = torch.tensor([
		tokenizer.encode(prompt)], dtype=torch.long).to(device)
	cache = None

	with autocast(enabled=torch.cuda.is_available(), dtype=torch.float16, device_type="cuda"):
		if model_cfg.get('use_retention', False):
			for i in input_ids[0]:
				logits, cache = model(i.unsqueeze(0).unsqueeze(0), cache=cache, use_recurrent_mode=True)
		else:
			logits, cache = model(input_ids)
		token = torch.argmax(logits[:, -1, :], dim=-1)
		generated_tokens = token.tolist()
		for _ in range(max_tokens):
			logits, cache = model(token.unsqueeze(0), cache=cache, use_recurrent_mode=True)
			token = torch.argmax(logits[:, -1, :], dim=-1)
			generated_tokens.extend(token.tolist())
			if tokenizer.decode(generated_tokens[-2:]) == '\n\n':
				break

	output = tokenizer.decode(generated_tokens)
	if verbose:
		print(f'prompt={repr(prompt)} + output={repr(output)}\n-> {prompt + output}')
	del model
	return output

def train_gsm(learning_rates,
              num_epochs,
              batch_size,
              seq_length,
              lora_cfg,
              model_cfg,
              thaws,
              take,
              from_path=None):
	tokenizer = Tokenizer()

	def load_gsm_data(tokenizer, is_tiny=True):
		if is_tiny:
			data = load_dataset("TinyGSM/TinyGSM")["train"]
			data = data.select(range(take))
			data = data.filter(lambda x: len(x['question']) < 100 and ':' not in
			                             x['question'] and '-' not in x[
				                             'question'] and "'" not in x[
				                             'code'] and '\n    result =' in x[
				                             'code'])
			split_point = int(len(data) * 0.8)
			train_data = data.select(range(split_point))
			eval_data = data.select(range(split_point, len(data)))

			def format_example(example):
				code_raw = example['code']
				start = code_raw.rfind('\n    """')
				if start == -1:
					return code_raw.strip()
				start = start + 8
				end = code_raw.rfind('\n    result =')
				if end == -1:
					end = len(code_raw)
				code_block = code_raw[start:end]
				code_lines = code_block.split('\n    ')
				formatted_code = '\n'.join(
					line for line in code_lines if line.strip())
				formatted_code = '\n' + formatted_code.strip() + '\n\n'
				result = (example['question'].strip(), formatted_code)
				return result
		else:
			dataset = load_dataset("openai/gsm8k", "main")
			train_data = dataset["train"]
			eval_data = dataset["test"]

			def format_example(example):
				return (example['question'].strip(),
				        '\n' + example['answer'].strip() + '\n\n')
		train_formatted = [format_example(ex) for ex in train_data]
		eval_formatted = [format_example(ex) for ex in eval_data]
		return train_formatted, eval_formatted

	def create_batches(data, tokenizer, batch_size, seq_length):
		encoded_data = [[tokenizer.encode(q), tokenizer.encode(a)] for q, a in
		                data]
		encoded_data = [x for x in encoded_data if
		                len(x[0] + x[1]) <= seq_length + 1]
		if batch_size is None:
			batch_size = min(len(encoded_data), 64)
		else:
			total_batches = len(encoded_data) // batch_size
			encoded_data = encoded_data[:total_batches * batch_size]
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
				mask = [0] * (len(q) - 1) + [1] * len(a) + [0] * pad_length
				x_batch.append(x)
				y_batch.append(y)
				mask_batch.append(mask)
			yield (torch.tensor(x_batch, dtype=torch.long),
			       torch.tensor(y_batch, dtype=torch.long),
			       torch.tensor(mask_batch, dtype=torch.float32))

	def loss_fn(model, X, y, mask):
		logits, _ = model(X)
		logits = logits.float()
		ce = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), reduction='none')
		masked_loss = ce * mask.view(-1)
		return masked_loss.sum(), mask.sum()

	def evaluate(model, data, tokenizer, seq_length):
		model.eval()
		total_loss = 0
		total_samples = 0
		with torch.no_grad():
			for X, y, mask in create_batches(data, tokenizer, None, seq_length):
				loss, ntoks = loss_fn(model, X, y, mask)
				total_loss += loss.item()
				total_samples += ntoks.item()
		return total_loss / total_samples if total_samples > 0 else -1

	def get_optimizer(train_data):
		num_batches_per_epoch = len(list(create_batches(train_data, tokenizer, batch_size, seq_length)))
		num_steps = num_epochs * num_batches_per_epoch
		num_warmup = num_steps // 10
		max_lr, min_lr = learning_rates

		def lr_lambda(step):
			if step < num_warmup:
				return step / max(1, num_warmup)
			else:
				progress = float(step - num_warmup) / max(1, num_steps - num_warmup)
				return 0.5 * (1.0 + math.cos(math.pi * progress))

		optimizer = Lion([p for p in model.parameters() if p.requires_grad], lr=max_lr)
		scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
		return optimizer, scheduler, num_steps

	timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
	print(f'--- {timestamp} ---')

	print("Loading dat...")
	train_data, eval_data = load_gsm_data(tokenizer=tokenizer)

	print("Loading model...")
	model = load_model_for_training(lora_cfg=lora_cfg, model_cfg=model_cfg, thaws=thaws, from_path=from_path).to(device)

	print("Loading optimizer...")
	optimizer, scheduler, num_steps = get_optimizer(train_data)

	metrics = {
		'steps'           : [],
		'learning_rates'  : [],
		'all_train_losses': [],
		'avg_train_losses': [],
		'val_losses'      : [],
		'trained_toks'    : [],
	}
	step = 0
	trained_toks = 0
	losses = []
	tic = time.perf_counter()
	for epoch in range(num_epochs):
		for X, y, loss_mask in create_batches(train_data, tokenizer, batch_size, seq_length):
			X = X.to(device)
			y = y.to(device)
			loss_mask = loss_mask.to(device)

			model.train()
			optimizer.zero_grad()
			loss, ntoks = loss_fn(model, X, y, loss_mask)
			loss.backward()
			optimizer.step()
			scheduler.step()
			losses.append(loss.item())
			trained_toks += ntoks.item()
			if step % (num_steps // 30) == 0:
				avg_train_loss = np.mean(losses)
				lr = scheduler.get_last_lr()[0]
				tic = time.perf_counter()
				metrics['steps'].append(step)
				metrics['learning_rates'].append(lr)
				metrics['all_train_losses'].extend(losses)
				metrics['avg_train_losses'].append(avg_train_loss)
				metrics['trained_toks'].append(trained_toks)
				losses = []
				trained_toks = 0
				print(f"{avg_train_loss:8.4f} @ {step // (num_steps // 30):2}/30 w/ {lr:.2e} ({time.perf_counter() - tic:.2f} sec)")
			step += 1

	_path = f'trained_retnphi.pt' if model_cfg[
		'use_retention'] else f'trained_orgnphi.pt'
	torch.save(model.state_dict(), _path)
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
	with open(f'train_log_{timestamp}.json', 'w') as f:
		json.dump(log, f, indent=2)
	del model

def main(
	take=1000,
	layers='all',
	targets=["self_attn.o_proj"],
	thaws=['new', 'post_attention_layernorm'],
	rank=32,
	scale=0.1,
	dropout=0.0,
	lr_max=1e-4,
	lr_min=1e-5,
	num_epochs=90,
	batch_size=1,
	seq_length=256,
	vocab_size=256,
	use_retention=True,
	untie_embedding=True,
	prompt='There are 8 candies in a carton. How many candies will be in 5 cartons?'
):
	lora_cfg = dict(layers=layers, targets=targets, rank=rank, scale=scale, dropout=dropout)
	model_cfg = dict(
		vocab_size=vocab_size,
		use_retention=use_retention,
		untie_embedding=untie_embedding,
		# hidden_size=768,  # Example size, adjust according to actual config
		# num_attention_heads=12,
		# num_key_value_heads=12,
		# num_hidden_layers=12,
		# intermediate_size=3072,
		# rms_norm_eps=1e-6,
		# original_max_position_embeddings=2048,
		# max_position_embeddings=2048,
		# rope_theta=10000.0,
		# rope_scaling={'long_factor':1.0, 'short_factor':1.0},
	)
	# train_gsm(
	# 	learning_rates=(lr_max, lr_min),
	# 	num_epochs=num_epochs,
	# 	batch_size=batch_size,
	# 	seq_length=seq_length,
	# 	lora_cfg=lora_cfg,
	# 	model_cfg=model_cfg,
	# 	thaws=thaws,
	# 	take=take
	# )
	generate(
		prompt=prompt,
		lora_cfg=lora_cfg,
		model_cfg=model_cfg,
		max_tokens=(seq_length - len(prompt))
	)

if __name__ == "__main__":
	main()

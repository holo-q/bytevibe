# Neural network compression estimation models
# This model learns the optimal compression parameters for
# each image to land an accurate target size, discovering the underlying
# mechanics of compression processes.
#
# It is a great testbed to explore new divergent training methods, of which we have
# implemented two:
#
# 1. Rhythmic annealing: a rhythmic fractal function is used to renoise the weights. (model search)
# 2. Growth-based annealing: the model is trained to grow in size and then crinkled with a falloff the further away from growth centers. (model scaling)
# --------------------------------------------------------------------------------

import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.optimize import minimize
from torch.utils.data import DataLoader, IterableDataset

import const
import utils
from const import _console

def generate_rhythm_pattern(length, fractal_dimension=1.5):
	t = np.linspace(0, 1, length)
	pattern = np.zeros_like(t)
	for i in range(1, int(length / 2)):
		pattern += np.sin(2 * np.pi * (2 ** i) * t) / (i ** fractal_dimension)
	return (pattern - pattern.min()) / (pattern.max() - pattern.min())

class FractalCompressorNetwork(nn.Module):
	def __init__(self,
	             input_size=2,
	             initial_hidden_size=64,
	             num_layers=3,
	             output_size=1):
		super(FractalCompressorNetwork, self).__init__()
		self.input_size = input_size
		self.hidden_size = initial_hidden_size
		self.num_layers = num_layers
		self.output_size = output_size

		self.layers = nn.ModuleList([nn.Linear(input_size, initial_hidden_size)])
		self.layers.extend([nn.Linear(initial_hidden_size, initial_hidden_size) for _ in range(num_layers - 2)])
		self.layers.append(nn.Linear(initial_hidden_size, output_size))
		self.activation = nn.ReLU()

	def get_config(self):
		return {
			'model_type'         : 'FractalRhythmicCompressor',
			'initial_hidden_size': self.model.hidden_size,
			'learning_rate'      : self.learning_rate,
			'optimizer'          : self.optimizer_name,
			'checkpoint_dir'     : str(self.checkpoint_dir),
			'num_layers'         : self.model.num_layers,
			'input_size'         : self.model.input_size,
			'output_size'        : self.model.output_size
		}


	def forward(self, x):
		for layer in self.layers[:-1]:
			x = self.activation(layer(x))
		return self.layers[-1](x)

	def grow_network(self, new_hidden_size=None, new_layer=False):
		if new_hidden_size is None:
			new_hidden_size = int(self.hidden_size * 1.5)

		if new_layer:
			insert_index = len(self.layers) // 2
			new_layer = nn.Linear(self.hidden_size, self.hidden_size)

			# Initialize new layer as interpolation of neighbors
			with torch.no_grad():
				prev_layer = self.layers[insert_index - 1]
				next_layer = self.layers[insert_index]
				if prev_layer.weight.size(1) != next_layer.weight.size(0):
					# If sizes don't match, we can't interpolate. Initialize randomly instead.
					nn.init.xavier_uniform_(new_layer.weight)
					nn.init.zeros_(new_layer.bias)
				else:
					new_layer.weight.data = (
						                        prev_layer.weight.data + next_layer.weight.data.t()) / 2
					new_layer.bias.data = (
						                      prev_layer.bias.data + next_layer.bias.data) / 2

			self.layers.insert(insert_index, new_layer)
			self.num_layers += 1
			return insert_index, insert_index

		# Grow width of all hidden layers
		new_layers = nn.ModuleList()
		for i, layer in enumerate(self.layers):
			if i == 0:  # Input layer
				new_layer = nn.Linear(self.input_size, new_hidden_size)
				new_layer.weight.data[:, :self.input_size] = layer.weight.data
				new_layer.bias.data = layer.bias.data
			elif i == len(self.layers) - 1:  # Output layer
				new_layer = nn.Linear(new_hidden_size, self.output_size)
				new_layer.weight.data[:, :self.hidden_size] = layer.weight.data
				new_layer.bias.data = layer.bias.data
			else:  # Hidden layers
				new_layer = nn.Linear(new_hidden_size, new_hidden_size)
				new_layer.weight.data[:self.hidden_size,
				:self.hidden_size] = layer.weight.data
				new_layer.bias.data[:self.hidden_size] = layer.bias.data

			# Initialize new weights and biases
			if new_hidden_size > self.hidden_size:
				nn.init.xavier_uniform_(new_layer.weight[self.hidden_size:, :])
				nn.init.xavier_uniform_(new_layer.weight[:, self.hidden_size:])
				nn.init.zeros_(new_layer.bias[self.hidden_size:])

			new_layers.append(new_layer)

		self.layers = new_layers
		self.hidden_size = new_hidden_size
		return None, None  # No specific layers were inserted

	def crinkle_parameters(self, amplitude, center_layers=None):
		num_layers = len(self.layers)
		with torch.no_grad():
			for i, layer in enumerate(self.layers):
				if center_layers:
					# Calculate distance from the center layers
					dist = min(abs(i - center_layers[0]), abs(i - center_layers[
						1]))
					# Apply exponential falloff
					falloff = np.exp(-dist / (
						num_layers / 4))  # Adjust the denominator to control falloff rate
					layer_amplitude = amplitude * falloff
				else:
					layer_amplitude = amplitude

				layer.weight.data += torch.randn_like(layer.weight.data) * layer_amplitude
				layer.bias.data += torch.randn_like(layer.bias.data) * layer_amplitude


class FractalRhythmicCompressor:
	def __init__(self,
	             initial_hidden_size=64,
	             learning_rate=0.001,
	             optimizer='adam',
	             checkpoint_dir='checkpoints'):
		self.model = FractalCompressorNetwork(initial_hidden_size=initial_hidden_size)
		self.learning_rate = learning_rate
		self.optimizer_name = optimizer
		self.set_optimizer(optimizer)
		self.criterion = nn.MSELoss()
		self.checkpoint_dir = Path(checkpoint_dir)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.loss_history = []
		self.complexity_history = []

	def set_optimizer(self, optimizer_name):
		if optimizer_name.lower() == 'adam':
			self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
		elif optimizer_name.lower() == 'sgd':
			self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9)
		elif optimizer_name.lower() == 'rmsprop':
			self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
		else:
			raise ValueError(f"Unsupported optimizer: {optimizer_name}")

	def train_model(self, dataset, epochs, batch_size, samples_per_epoch):
		dataloader = DataLoader(dataset, batch_size=batch_size)

		rhythm_pattern = generate_rhythm_pattern(epochs)
		epochs_without_improvement = 0
		best_loss = float('inf')

		for epoch in range(epochs):
			self.model.train()
			epoch_loss = 0
			samples_processed = 0

			rhythm_factor = rhythm_pattern[epoch]

			for batch_X, batch_y in dataloader:
				self.optimizer.zero_grad()
				outputs = self.model(batch_X)
				loss = self.criterion(outputs, batch_y)
				loss.backward()
				self.optimizer.step()

				epoch_loss += loss.item()
				samples_processed += batch_X.size(0)

				if samples_processed >= samples_per_epoch:
					break

			avg_loss = epoch_loss * batch_size / samples_per_epoch
			self.loss_history.append(avg_loss)
			self.complexity_history.append(self.model.hidden_size)
			_console.print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.2f}, Rhythm: {rhythm_factor:.4f}")

			if avg_loss < best_loss:
				best_loss = avg_loss
				epochs_without_improvement = 0
			else:
				epochs_without_improvement += 1

			if epochs_without_improvement >= 10:
				self.grow_and_crinkle(rhythm_factor)
				epochs_without_improvement = 0

		# Apply rhythmic annealing
		# crinkle_amplitude = self.calculate_crinkle_amplitude(rhythm_factor, avg_loss)
		# self.model.crinkle_parameters(crinkle_amplitude)

	def grow_and_crinkle(self, rhythm_factor):
		add_layer = self.should_add_layer()
		if add_layer:
			new_hidden_size = self.model.hidden_size
		else:
			new_hidden_size = self.calculate_new_hidden_size()

		center_layers = self.model.grow_network(new_hidden_size=new_hidden_size, new_layer=add_layer)

		crinkle_amplitude = self.calculate_crinkle_amplitude(rhythm_factor,
			self.loss_history[-1])
		self.model.crinkle_parameters(crinkle_amplitude, center_layers)

		self.set_optimizer(self.optimizer_name)  # Reset the optimizer for the new parameters
		_console.print(f"[green]Grew network to hidden size {self.model.hidden_size}, layers: {self.model.num_layers}[/green]")

	def calculate_crinkle_amplitude(self, rhythm_factor, current_loss):
		# This is a placeholder function. You may want to develop a more sophisticated approach.
		base_amplitude = 0.01 * rhythm_factor
		loss_factor = np.log(current_loss + 1)  # Prevent log(0)
		return base_amplitude * loss_factor

	def calculate_new_hidden_size(self):
		current_performance = self.loss_history[-1]
		model_complexity = self.model.hidden_size * self.model.num_layers

		# Base growth factor
		growth_factor = 1.2

		# Adjust growth factor based on recent performance
		if len(self.loss_history) > 5:
			recent_improvement = (self.loss_history[-5] - current_performance) / \
			                     self.loss_history[-5]
			growth_factor += recent_improvement

		# Limit growth based on model complexity
		max_growth = 10000 / model_complexity
		growth_factor = min(growth_factor, max_growth)

		new_hidden_size = int(self.model.hidden_size * growth_factor)
		return max(new_hidden_size, self.model.hidden_size + 1)  # Ensure at least 1 node is added

	def should_add_layer(self):
		if self.model.num_layers >= 10:  # Set a maximum number of layers
			return False

		current_performance = self.loss_history[-1]
		avg_layer_size = self.model.hidden_size

		# Check if adding a layer might be beneficial
		if len(self.loss_history) > 10:
			recent_improvement = (self.loss_history[
				                      -10] - current_performance) / \
			                     self.loss_history[-10]
			if recent_improvement < 0.01:  # If improvement is less than 1%
				return True

		# Add a layer if the model is too wide compared to its depth
		if avg_layer_size > 5 * self.model.num_layers:
			return True

		return False

	def predict_size(self, quality, width):
		self.model.eval()
		with torch.no_grad():
			input_tensor = torch.tensor([[quality, width]], dtype=torch.float32)
			return self.model(input_tensor).item()

	def plot_performance(self, save_path=None):
		plt.figure(figsize=(12, 4))
		plt.subplot(121)
		plt.plot(self.loss_history)
		plt.title('Loss History')
		plt.xlabel('Update')
		plt.ylabel('Mean Squared Error')

		plt.subplot(122)
		plt.plot(self.complexity_history)
		plt.title('Model Complexity')
		plt.xlabel('Update')
		plt.ylabel('Hidden Layer Size')

		plt.tight_layout()
		# plt.show()
		if save_path:
			plt.savefig(save_path)

	def save_checkpoint(self, filename='compressor_model.pth'):
		path = self.checkpoint_dir / filename
		torch.save({
			'model_state_dict'    : self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'loss_history'        : self.loss_history,
			'complexity_history'  : self.complexity_history,
			'data'                : self.data
		}, path)

	def load_checkpoint(self, filename='compressor_model.pth'):
		path = self.checkpoint_dir / filename
		if not path.exists():
			print(f"Checkpoint file {path} not found. Starting with a new model.")
			return

		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.loss_history = checkpoint['loss_history']
		self.complexity_history = checkpoint['complexity_history']
		self.data = checkpoint['data']
		self.model.hidden_size = self.complexity_history[
			-1] if self.complexity_history else 10
		print(f"Loaded checkpoint from {path}")

	def get_optimized_params(self,
	                         quality_range,
	                         width_range,
	                         target_size,
	                         num_attempts=3):
		self.model.eval()

		def objective(params):
			quality, width = params
			with torch.no_grad():
				input_tensor = torch.tensor([
					[quality, width]], dtype=torch.float32)
				predicted_size = self.model(input_tensor).item()
			return abs(predicted_size - target_size)

		optimized_params = []
		for _ in range(num_attempts):
			initial_guess = [
				random.uniform(quality_range[0], quality_range[1]),
				random.uniform(width_range[0], width_range[1])
			]

			result = minimize(objective,
				x0=initial_guess,
				bounds=[quality_range, width_range],
				method='L-BFGS-B')

			if result.success:
				quality, width = map(int, np.round(result.x))
				optimized_params.append((quality, width))
			else:
				print(f"Optimization failed: {result.message}")

		# Sort the optimized parameters by their objective function value
		optimized_params.sort(key=lambda params: objective(params))

		return optimized_params

class PartiallyNormalizedCompressionDataset(IterableDataset):
	def __init__(self, image_dataset, formats, quality_range, width_range):
		self.image_dataset = image_dataset
		self.formats = formats
		self.quality_range = quality_range
		self.width_range = width_range
		self.max_quality = max(quality_range)

	def __iter__(self):
		while True:
			entry = random.choice(self.image_dataset['images'])
			img_path = const.dir_image_datasets / entry['image_path']
			img = cv2.imread(str(img_path))

			if img is None:
				continue

			fmt = random.choice(self.formats)
			quality = random.randint(*self.quality_range)
			width = random.randint(*self.width_range)

			try:
				compressed = utils.try_compression(img, fmt, quality, width)
				size = len(compressed)

				# Normalize only the quality
				norm_quality = quality / self.max_quality

				yield torch.tensor([norm_quality,
				                    width], dtype=torch.float32), torch.tensor([
					size], dtype=torch.float32)
			except Exception:
				continue

import dearpygui.dearpygui as dpg
import torch.optim as optim
from torch import nn

from model_visualization import ModelVisualization
from training_history_graph import TrainingHistoryGraph
from training_loop import TrainingLoop
from data_loader import get_data_loader
from utils import calculate_model_complexity, format_number
from src_dashboard.run_manager import singleton as run_manager
from src_dashboard.model_registry import singleton as model_registry

class TrainingWindow:
	def __init__(self):
		self.run = None

	# noinspection PyArgumentList
	def setup(self):
		with dpg.window(label="Training Window", width=800, height=600):
			with dpg.group():
				# dpg.add_combo(label="Run", items=run_manager.get_active_runs(), callback=self.set_run)
				# dpg.add_combo(label="Model", items=self.model_registry.get_model_names(), default_value="FractalRhythmicCompressor", callback=self.set_model)
				dpg.add_spacer(height=8)

				dpg.add_text("Drugs")
				dpg.add_slider_int(label="horizontal", min_value=1, max_value=10, default_value=1)
				dpg.add_slider_int(label="vertical", min_value=1, max_value=10, default_value=1)
				dpg.add_slider_float(label="Local Annealing", min_value=0, max_value=1, default_value=0.1)
				dpg.add_button(label="Neurogenesis", callback=self.grow_model)
				dpg.add_spacer(height=8)

				dpg.add_slider_float(label="Strength", min_value=0, max_value=1, default_value=0.1)
				dpg.add_button(label="Crinkle", callback=self.crinkle_model)
				dpg.add_spacer(height=8)

				dpg.add_slider_float(label="Rhythmic Global Annealing", min_value=0, max_value=2, default_value=1.0, callback=self.set_rhythmic_annealing_strength)
				dpg.add_spacer(height=8)

			with dpg.group():
				with dpg.collapsing_header(label="Training Progress", default_open=True):
					dpg.add_button(label="Add", callback=self.start_training)
					dpg.add_same_line()
					dpg.add_input_int(label="Epochs to Train", default_value=10, callback=self.set_epochs)

					dpg.add_button(label="Abort", callback=self.abort_training)
					dpg.add_same_line()
					self.progress_bar = dpg.add_progress_bar(label="Progress", overlay="0/0 epochs")

			with dpg.plot(label="Loss History", height=200, width=-1):
				dpg.add_plot_legend()
				dpg.add_plot_axis(dpg.mvXAxis, label="Epoch")
				dpg.add_plot_axis(dpg.mvYAxis, label="Loss")
				self.loss_line_series = dpg.add_line_series([], [], parent=dpg.last_item(), label="Training Loss")

			with dpg.collapsing_header(label="Model Architecture", default_open=False):
				self.model_visualization = ModelVisualization()
				self.model_visualization.setup()

			with dpg.collapsing_header(label="Checkpoint History", default_open=False):
				self.training_history_graph = TrainingHistoryGraph(self.run.checkpoint_manager)
				self.training_history_graph.setup()

		self.refresh_model_viz()

	def set_run(self, run):
		self.training_history_graph.update_checkpoint_manager(self.run.checkpoint_manager)
		if not self.load_latest_checkpoint():
			self.clear_model()
			# self.set_model(model_registry.instantiate(run.model_config[''])))
			self.run.model = model_registry.get_model("FractalRhythmicCompressor")
			self.refresh_model_viz()
			self.run = run

	def clear_model(self):
		self.run.model = None
		self.optimizer = None
		self.loss_fn = None
		self.data_loader = None
		self.training_loop = None
		self.scheduled_epochs = 0
		self.current_epoch = 0
		self.training_active = False
		self.model_visualization = None
		self.training_history_graph = None
		self.rhythmic_annealing_strength = 1.0
		self.loss_history = []
		self.refresh_model_viz()

	def set_epochs(self, sender, app_data):
		self.scheduled_epochs += app_data

	def set_rhythmic_annealing_strength(self, sender, app_data):
		self.rhythmic_annealing_strength = app_data

	def start_training(self):
		if not self.training_active:
			self.training_active = True
			self.current_epoch = 0
			self.loss_history = []

			# Create a new run if one doesn't exist
			if self.run is None:
				run_name = f"run_{len(run_manager.get_active_runs()) + 1}"
				model_config = self.run.model.get_config()
				dataset_config = {
					'dataset_name' : 'your_dataset_name',
					'formats'      : ["webp", "jpeg"],
					'quality_range': (2, 100),
					'width_range'  : (64, 2048)
				}
				run_config = {**model_config, **dataset_config}
				run_manager.create_run(run_name, run_config)

			self.optimizer = optim.Adam(self.run.model.parameters(), lr=self.run.model.learning_rate)
			self.loss_fn = nn.MSELoss()
			dataset = run_manager.get_run_dataset()
			self.data_loader = get_data_loader(dataset, ["webp", "jpeg"], (
				2, 100), (64, 2048))
			self.training_loop = TrainingLoop(self.run.model, self.optimizer, self.loss_fn, self.data_loader)

	def abort_training(self):
		self.training_active = False
		self.scheduled_epochs = 0

	def crinkle_model(self, sender, app_data):
		crinkle_strength = app_data
		if self.run.model:
			self.run.model.crinkle_parameters(crinkle_strength)
			self.refresh_model_viz()

	def grow_model(self):
		if self.run.model:
			self.run.model.grow_network()
			self.refresh_model_viz()

	def save_state(self):
		if self.run.model and self.loss_history:
			self.run.checkpoint_manager.save_checkpoint(self.run.model, self.optimizer, self.current_epoch, self.loss_history)
			self.training_history_graph.update()


	def load_checkpoint(self, checkpoint_id):
		checkpoint, checkpoint_info = self.run.checkpoint_manager.load_checkpoint(checkpoint_id)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		self.current_epoch = checkpoint_info['epoch']
		self.loss_history = checkpoint_info['loss_history']
		self.refresh_loss_plot()
		self.refresh_model_viz()

	def load_latest_checkpoint(self):
		latest_checkpoint = self.run.checkpoint_manager.get_latest_checkpoint()
		if latest_checkpoint:
			checkpoint, _ = self.run.checkpoint_manager.load_checkpoint(
				latest_checkpoint['id'])
			self.model.load_state_dict(checkpoint['model_state_dict'])
			self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			self.current_epoch = checkpoint['epoch']
			self.loss_history = checkpoint['loss_history']
			self.refresh_loss_plot()
			self.refresh_model_viz()
			return True

		return False

	def update(self):
		if self.training_active and self.scheduled_epochs > 0:
			loss = self.training_loop.train_epoch(self.rhythmic_annealing_strength, self.current_epoch)
			self.loss_history.append(loss)
			self.current_epoch += 1
			self.scheduled_epochs -= 1
			self.refresh_progress()
			self.refresh_loss_plot()
			self.refresh_model_viz()

		if self.scheduled_epochs == 0:
			self.training_active = False
			self.save_state()

	def refresh_progress(self):
		progress = self.current_epoch / self.scheduled_epochs
		dpg.set_value(self.progress_bar, progress)

	# dpg.set_overlay(self.progress_bar, f"{self.current_epoch}/{self.scheduled_epochs} epochs")

	def refresh_loss_plot(self):
		dpg.set_value(self.loss_line_series, [
			list(range(len(self.loss_history))), self.loss_history])

	def refresh_model_viz(self):
		if self.model:
			self.model_visualization.update(self.model)
			complexity = calculate_model_complexity(self.model)
			dpg.set_value("model_complexity", f"Model Complexity: {format_number(complexity)} parameters")


	# noinspection PyArgumentList
	def compare_checkpoints(self, checkpoint_id1, checkpoint_id2):
		comparison = self.run.checkpoint_manager.compare_checkpoints(checkpoint_id1, checkpoint_id2)
		# Display comparison results in the UI
		with dpg.window(label="Checkpoint Comparison"):
			dpg.add_text(f"Epoch difference: {comparison['epoch_diff']}")
			dpg.add_text(f"Loss difference: {comparison['loss_diff']:.4f}")
			dpg.add_text(f"Checkpoint 1 path: {comparison['path1']}")
			dpg.add_text(f"Checkpoint 2 path: {comparison['path2']}")

	# noinspection PyArgumentList
	def find_best_checkpoint(self):
		best_checkpoint = self.run.checkpoint_manager.find_best_checkpoint()
		# Display best checkpoint info in the UI
		with dpg.window(label="Best Checkpoint"):
			dpg.add_text(f"Checkpoint ID: {best_checkpoint['id']}")
			dpg.add_text(f"Epoch: {best_checkpoint['epoch']}")
			dpg.add_text(f"Loss: {best_checkpoint['loss_history'][-1]:.4f}")
			dpg.add_text(f"Path: {best_checkpoint['path']}")

	def prune_checkpoints(self):
		keep_best_n = dpg.get_value("keep_best_n")
		self.run.checkpoint_manager.prune_checkpoints(keep_best_n)
		self.training_history_graph.update()

	def export_checkpoint_graph(self):
		filename = dpg.get_value("export_filename")
		self.training_history_graph.export_graph(filename)
		dpg.add_text(f"Checkpoint graph exported to {filename}")

# noinspection PyArgumentList
# def setup_additional_ui(self):
# 	with dpg.collapsing_header(label="Checkpoint Management", default_open=False):
# 		dpg.add_input_int(label="Keep Best N Checkpoints", default_value=5, tag="keep_best_n")
# 		dpg.add_button(label="Prune Checkpoints", callback=self.prune_checkpoints)
# 		dpg.add_input_text(label="Export Filename", default_value="checkpoint_graph.png", tag="export_filename")
# 		dpg.add_button(label="Export Checkpoint Graph", callback=self.export_checkpoint_graph)
#
# 		self.setup_additional_ui()

from pathlib import Path
from typing import List

from src_dashboard.checkpoint_manager import CheckpointManager
from src_dashboard.training_history_graph import TrainingHistoryGraph

class Run:
	def __init__(self, run_name, model_name, path):
		self.run_name = run_name
		self.run_dir = None
		self.model_name = model_name

		self.model = None
		self.optimizer = None
		self.loss_fn = None
		self.data_loader = None
		self.training_loop = None
		self.current_epoch = 0
		self.scheduled_epochs = 0
		self.initial_epoch_at_train_start = 0
		self.training_active = False
		self.model_visualization = None
		self.training_history_graph = None
		self.rhythmic_annealing_strength = 1.0
		self.loss_history = []

		if path:
			self.load()

	def load(self):
		self.run_dir = Path(self.run_name)
		self.checkpoint_manager = CheckpointManager(self.run_dir / "checkpoints")


class RunManager:
	def __init__(self, runs_dir="runs"):
		self.runs_dir = Path(runs_dir)
		self.runs_dir.mkdir(parents=True, exist_ok=True)
		self.runs : List[Run] = self.read_runs()

	def read_runs(self) -> List[Run]:
		runs = []
		for run_dir in self.runs_dir.iterdir():
			if run_dir.is_dir():
				Run(run_dir.name, None, run_dir)

		return runs

	def load_run(self, run_name):
		if run_name not in self.runs:
			raise ValueError(f"Run '{run_name}' does not exist")

	def create_run(self, run_name, model_name):
		if run_name in self.runs:
			raise ValueError(f"Run '{run_name}' already exists")

		run_dir = self.runs_dir / run_name
		run_dir.mkdir(parents=True, exist_ok=True)

		run = Run(run_name, model_name, None)

		self.runs[run_name] = run

	def delete_run(self, run_name):
		if run_name not in self.runs:
			raise ValueError(f"Run '{run_name}' does not exist")

		run_dir = self.runs_dir / run_name
		for item in run_dir.iterdir():
			if item.is_file():
				item.unlink()
			elif item.is_dir():
				for subitem in item.iterdir():
					subitem.unlink()
				item.rmdir()
		run_dir.rmdir()

		del self.runs[run_name]

singleton = RunManager()
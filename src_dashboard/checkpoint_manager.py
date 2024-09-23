import os
import json
import torch
from pathlib import Path
import networkx as nx

class CheckpointManager:
	def __init__(self, checkpoint_dir="checkpoints"):
		self.checkpoint_dir = Path(checkpoint_dir)
		self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
		self.history_file = self.checkpoint_dir / "history.json"
		self.graph = nx.DiGraph()
		self.load_history()

	def load_history(self):
		if self.history_file.exists():
			with open(self.history_file, "r") as f:
				history = json.load(f)
				for checkpoint in history:
					self.graph.add_node(checkpoint['id'], **checkpoint)
					if checkpoint['parent_id'] is not None:
						self.graph.add_edge(
							checkpoint['parent_id'], checkpoint['id'])
		else:
			self.graph.clear()

	def save_history(self):
		history = [self.graph.nodes[node] for node in self.graph.nodes()]
		with open(self.history_file, "w") as f:
			json.dump(history, f)

	def save_checkpoint(self,
	                    model,
	                    optimizer,
	                    epoch,
	                    loss_history,
	                    parent_id=None):
		checkpoint_id = self.graph.number_of_nodes()
		checkpoint_path = self.checkpoint_dir / f"checkpoint_{checkpoint_id}.pth"

		torch.save({
			'model_state_dict'    : model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'epoch'               : epoch,
			'loss_history'        : loss_history
		}, checkpoint_path)

		checkpoint_info = {
			'id'          : checkpoint_id,
			'epoch'       : epoch,
			'loss_history': loss_history,
			'parent_id'   : parent_id,
			'path'        : str(checkpoint_path)
		}

		self.graph.add_node(checkpoint_id, **checkpoint_info)
		if parent_id is not None:
			self.graph.add_edge(parent_id, checkpoint_id)

		self.save_history()

		return checkpoint_id

	def load_checkpoint(self, checkpoint_id):
		if checkpoint_id not in self.graph.nodes:
			raise ValueError(f"Checkpoint with id {checkpoint_id} not found")

		checkpoint_info = self.graph.nodes[checkpoint_id]
		checkpoint = torch.load(checkpoint_info['path'])
		return checkpoint, checkpoint_info

	def get_checkpoints(self):
		return [self.graph.nodes[node] for node in self.graph.nodes()]

	def get_checkpoint_tree(self):
		return self.graph

	# def compare_checkpoints(self, checkpoint_id1, checkpoint_id2):
	#     checkpoint1 = self.graph.nodes[checkpoint_id1]
	#     checkpoint2 = self.graph.nodes[checkpoint_id2]
	#
	#     comparison = {
	#         'epoch_diff': checkpoint2['epoch'] - checkpoint1['epoch'],
	#         'loss_diff': checkpoint2['loss_history'][-1] - checkpoint1['loss_history'][-1],
	#         'path1': checkpoint1['path'],
	#         'path2': checkpoint2['path']
	#     }
	#
	#     return comparison
	#
	# def find_best_checkpoint(self, metric='loss'):
	#     if metric == 'loss':
	#         best_checkpoint = min(self.graph.nodes, key=lambda x: self.graph.nodes[x]['loss_history'][-1])
	#     else:
	#         raise ValueError(f"Unsupported metric: {metric}")
	#
	#     return self.graph.nodes[best_checkpoint]
	#
	# def prune_checkpoints(self, keep_best_n=5, metric='loss'):
	#     sorted_checkpoints = sorted(self.graph.nodes, key=lambda x: self.graph.nodes[x]['loss_history'][-1])
	#     checkpoints_to_remove = sorted_checkpoints[keep_best_n:]
	#
	#     for checkpoint_id in checkpoints_to_remove:
	#         checkpoint_path = Path(self.graph.nodes[checkpoint_id]['path'])
	#         if checkpoint_path.exists():
	#             checkpoint_path.unlink()
	#         self.graph.remove_node(checkpoint_id)
	#
	#     self.save_history()

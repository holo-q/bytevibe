import dearpygui.dearpygui as dpg
import networkx as nx
import matplotlib.pyplot as plt
from io import BytesIO

class TrainingHistoryGraph:
	def __init__(self, checkpoint_manager):
		self.checkpoint_manager = checkpoint_manager
		self.node_editor = None
		self.nodes = {}

	def setup(self):
		with dpg.group():
			with dpg.node_editor(callback=self.link_callback) as self.node_editor:
				pass  # Nodes will be added dynamically

			with dpg.group(horizontal=True):
				dpg.add_button(label="Zoom In", callback=self.zoom_in)
				dpg.add_button(label="Zoom Out", callback=self.zoom_out)
				dpg.add_button(label="Reset View", callback=self.reset_view)

	def update(self):
		dpg.delete_item(self.node_editor, children_only=True)
		self.nodes = {}

		graph = self.checkpoint_manager.get_checkpoint_tree()
		pos = nx.spring_layout(graph)

		for node, data in graph.nodes(data=True):
			x, y = pos[node]
			node_ui = dpg.add_node(label=f"Epoch {data['epoch']}", pos=(
				x * 500 + 250, y * 500 + 250), parent=self.node_editor)
			with dpg.node_attribute(parent=node_ui):
				dpg.add_text(f"Loss: {data['loss_history'][-1]:.4f}")
			self.nodes[node] = node_ui

			with dpg.tooltip(parent=node_ui):
				dpg.add_text(f"Checkpoint ID: {data['id']}")
				dpg.add_text(f"Epoch: {data['epoch']}")
				dpg.add_text(f"Final Loss: {data['loss_history'][-1]:.4f}")
				dpg.add_text(f"Path: {data['path']}")

		for edge in graph.edges():
			dpg.add_node_link(
				self.nodes[edge[0]],
				self.nodes[edge[1]], parent=self.node_editor)

	def link_callback(self, sender, app_data):
		pass  # We don't want users to add links manually

	def zoom_in(self):
		for node, pos in self.nodes.items():
			x, y = dpg.get_item_pos(pos)
			dpg.set_item_pos(pos, (
				(x - 250) * 1.1 + 250, (y - 250) * 1.1 + 250))

	def zoom_out(self):
		for node, pos in self.nodes.items():
			x, y = dpg.get_item_pos(pos)
			dpg.set_item_pos(pos, (
				(x - 250) * 0.9 + 250, (y - 250) * 0.9 + 250))

	def reset_view(self):
		self.update()

	def update_checkpoint_manager(self, checkpoint_manager):
		self.checkpoint_manager = checkpoint_manager
		self.update()

	def export_graph(self, filename):
		graph = self.checkpoint_manager.get_checkpoint_tree()
		pos = nx.spring_layout(graph)

		plt.figure(figsize=(12, 8))
		nx.draw(graph, pos, with_labels=True, node_color='lightblue',
			node_size=1000, font_size=8, arrows=True)

		node_labels = {
			node: f"Epoch: {data['epoch']}\nLoss: {data['loss_history'][-1]:.4f}"
			for node, data in graph.nodes(data=True)}
		nx.draw_networkx_labels(graph, pos, node_labels, font_size=6)

		plt.tight_layout()
		plt.savefig(filename)
		plt.close()

		return filename

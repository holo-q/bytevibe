import dearpygui.dearpygui as dpg
import torch.nn as nn

class ModelVisualization:
	def __init__(self):
		self.node_editor = None
		self.nodes = {}
		self.links = []

	def setup(self):
		with dpg.node_editor(callback=self.link_callback) as self.node_editor:
			pass  # Nodes will be added dynamically

	def update(self, model):
		dpg.delete_item(self.node_editor, children_only=True)
		self.nodes = {}
		self.links = []

		input_node = dpg.add_node(label="Input", parent=self.node_editor)
		dpg.add_node_attribute(label="input", parent=input_node, attribute_type=dpg.mvNode_Attr_Output)
		self.nodes["input"] = input_node

		for name, module in model.named_modules():
			if isinstance(module, (
				nn.Linear, nn.Conv2d, nn.RNN, nn.LSTM, nn.GRU)):
				node = dpg.add_node(label=f"{name}: {module.__class__.__name__}", parent=self.node_editor)
				dpg.add_node_attribute(label="input", parent=node, attribute_type=dpg.mvNode_Attr_Input)
				dpg.add_node_attribute(label="output", parent=node, attribute_type=dpg.mvNode_Attr_Output)

				if isinstance(module, nn.Linear):
					info = f"in: {module.in_features}, out: {module.out_features}"
				elif isinstance(module, nn.Conv2d):
					info = f"in: {module.in_channels}, out: {module.out_channels}, kernel: {module.kernel_size}"
				elif isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
					info = f"input: {module.input_size}, hidden: {module.hidden_size}, layers: {module.num_layers}"
				else:
					info = "Custom layer"

				dpg.add_node_attribute(label=info, parent=node)
				self.nodes[name] = node
			elif isinstance(module, (nn.ReLU, nn.Tanh, nn.Sigmoid)):
				node = dpg.add_node(label=f"{name}: {module.__class__.__name__}", parent=self.node_editor)
				dpg.add_node_attribute(label="input", parent=node, attribute_type=dpg.mvNode_Attr_Input)
				dpg.add_node_attribute(label="output", parent=node, attribute_type=dpg.mvNode_Attr_Output)
				self.nodes[name] = node

		output_node = dpg.add_node(label="Output", parent=self.node_editor)
		dpg.add_node_attribute(label="output", parent=output_node, attribute_type=dpg.mvNode_Attr_Input)
		self.nodes["output"] = output_node

		self.connect_nodes(model)

	def connect_nodes(self, model):
		prev_node = self.nodes["input"]
		for name, module in model.named_modules():
			if name in self.nodes:
				self.links.append(dpg.add_node_link(
					dpg.get_item_children(prev_node, slot=1)[-1],
					dpg.get_item_children(self.nodes[name], slot=1)[0],
					parent=self.node_editor))
				prev_node = self.nodes[name]

		self.links.append(dpg.add_node_link(
			dpg.get_item_children(prev_node, slot=1)[-1],
			dpg.get_item_children(self.nodes["output"], slot=1)[0],
			parent=self.node_editor))

	def link_callback(self, sender, app_data):
		self.links.append(app_data)

from src_models.img_compression_estimation_models import FractalRhythmicCompressor

class ModelRegistry:
	def __init__(self):
		self.models = {
			"FractalRhythmicCompressor": FractalRhythmicCompressor
		}

	def register_model(self, name, model_class):
		self.models[name] = model_class

	def get_model(self, name, **kwargs):
		if name not in self.models:
			raise ValueError(f"Model {name} not found in registry")
		return self.models[name](**kwargs)

	def get_model_names(self):
		return list(self.models.keys())

singleton = ModelRegistry()

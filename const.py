from pathlib import Path

from rich.console import Console

project_root = Path(__file__).parent

dir_instruct_datasets = project_root / Path("instruct_datasets")
dir_instruct_datasets.mkdir(exist_ok=True, parents=True)

dir_image_datasets = project_root / Path("image_datasets")
dir_image_datasets.mkdir(exist_ok=True, parents=True)

dir_audio_datasets = project_root / Path("audio_datasets")
dir_audio_datasets.mkdir(exist_ok=True, parents=True)

dir_checkpoints = project_root / Path("checkpoints")
dir_checkpoints.mkdir(exist_ok=True, parents=True)

dir_plots = project_root / Path(".plots")
dir_plots.mkdir(exist_ok=True, parents=True)

_console = Console()

import matplotlib as mpl
mpl.use('TKAgg')

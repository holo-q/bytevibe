import torch
from torch.utils.data import Dataset, DataLoader
import json
import cv2
import random
from pathlib import Path
import const
import utils

class CompressionDataset(Dataset):
    def __init__(self, dataset_file, formats, quality_range, width_range):
        self.dataset = self.load_json(dataset_file)
        self.formats = formats
        self.quality_range = quality_range
        self.width_range = width_range
        self.max_quality = max(quality_range)

    def load_json(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def __len__(self):
        return len(self.dataset['images'])

    def __getitem__(self, idx):
        entry = self.dataset['images'][idx]
        img_path = const.dir_image_datasets / entry['image_path']
        img = cv2.imread(str(img_path))

        if img is None:
            return self.__getitem__(random.randint(0, len(self) - 1))

        fmt = random.choice(self.formats)
        quality = random.randint(*self.quality_range)
        width = random.randint(*self.width_range)

        try:
            compressed = utils.try_compression(img, fmt, quality, width)
            size = len(compressed)

            norm_quality = quality / self.max_quality

            return torch.tensor([norm_quality, width], dtype=torch.float32), torch.tensor([size], dtype=torch.float32)
        except Exception:
            return self.__getitem__(random.randint(0, len(self) - 1))

def get_data_loader(dataset_file, formats, quality_range, width_range, batch_size=32):
    dataset = CompressionDataset(dataset_file, formats, quality_range, width_range)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
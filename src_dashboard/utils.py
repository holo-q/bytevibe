import torch
import numpy as np

def generate_rhythm_pattern(length, fractal_dimension=1.5):
    t = np.linspace(0, 1, length)
    pattern = np.zeros_like(t)
    for i in range(1, int(length / 2)):
        pattern += np.sin(2 * np.pi * (2 ** i) * t) / (i ** fractal_dimension)
    return (pattern - pattern.min()) / (pattern.max() - pattern.min())

def apply_rhythmic_annealing(model, strength, current_epoch):
    rhythm = generate_rhythm_pattern(1000)[current_epoch % 1000]
    for param in model.parameters():
        param.data += torch.randn_like(param.data) * strength * rhythm

def calculate_model_complexity(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_number(num):
    for unit in ['', 'K', 'M', 'B', 'T']:
        if abs(num) < 1000.0:
            return f"{num:.1f}{unit}"
        num /= 1000.0
    return f"{num:.1f}P"
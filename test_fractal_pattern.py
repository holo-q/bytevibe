import numpy as np
import matplotlib.pyplot as plt

def generate_rhythm_pattern(length, fractal_dimension=1.5):
    t = np.linspace(0, 1, length)
    pattern = np.zeros_like(t)
    for i in range(1, int(length / 2)):
        pattern += np.sin(2 * np.pi * (2 ** i) * t) / (i ** fractal_dimension)
    return (pattern - pattern.min()) / (pattern.max() - pattern.min())

# Set the parameters
length = 1000
fractal_dimension = 1.5

# Generate the rhythm pattern
pattern = generate_rhythm_pattern(50, fractal_dimension)

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(np.linspace(0, 1, length), pattern, color='blue')
plt.title(f'Rhythm Pattern (Fractal Dimension: {fractal_dimension})')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.grid(True, linestyle='--', alpha=0.7)

# Add a horizontal line at y=0.5 for reference
plt.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Show the plot
plt.tight_layout()
plt.show()

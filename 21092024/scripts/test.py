import numpy as np
import matplotlib.pyplot as plt

# Generate 1D data
data = np.random.randn(1000)

# Compute the histogram
counts, bins = np.histogram(data, bins=50)

# Create a 2D array for two depths z0 and z1
z0 = counts  # Depth 1
z1 = counts * 0.8  # Depth 2 (example: scaled differently)

# Stack them into a 2D array (two rows, one for each depth)
counts_2d = np.vstack([z0, z1])

# Choose the specific depth you want to visualize (0 for z0, 1 for z1)
selected_depth = 0  # Set to 1 for z1

# Extract the selected row (depth) and reshape to 2D (row vector)
selected_counts = counts_2d[selected_depth].reshape(1, -1)

# Plot the selected depth using imshow
plt.imshow(selected_counts, cmap='viridis', aspect='auto')
plt.colorbar(label='Frequency')
plt.xlabel('Bins')
plt.title(f'Histogram at Depth z{selected_depth}')
plt.show()
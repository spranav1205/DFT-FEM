import numpy as np
import matplotlib.pyplot as plt

# Load the exported matrix from deal.II
matrix = np.loadtxt(r"./build/stiffness_matrix.txt")
matrix_inv = np.linalg.inv(matrix)

# Set up figure
plt.figure(figsize=(10, 8))

# Display as a heatmap (log scale highlights subtle off-diagonal entries)
# Use abs() to prevent log errors on zero/negative values
im = plt.imshow(np.abs(matrix), cmap="viridis", interpolation="none")

# Add a colorbar and labels
cbar = plt.colorbar(im)
cbar.set_label("Absolute Value Magnitude", rotation=270, labelpad=15)

plt.title(f"Stiffness Matrix Heatmap ({matrix.shape[0]}x{matrix.shape[1]})", fontsize=14)
plt.xlabel("Global DoF Column Index")
plt.ylabel("Global DoF Row Index")
plt.grid(False)

plt.tight_layout()
plt.savefig("matrix_heatmap.png", dpi=300)
plt.show()

# Display the inverse matrix
plt.figure(figsize=(10, 8))
im = plt.imshow(np.abs(matrix_inv), cmap="viridis", interpolation="none")
cbar = plt.colorbar(im)
cbar.set_label("Absolute Value Magnitude", rotation=270, labelpad=15)
plt.title(f"Inverse Stiffness Matrix Heatmap ({matrix_inv.shape[0]}x{matrix_inv.shape[1]})", fontsize=14)
plt.xlabel("Global DoF Column Index")
plt.ylabel("Global DoF Row Index")
plt.grid(False)
plt.tight_layout()
plt.savefig("matrix_inv_heatmap.png", dpi=300)
plt.show()

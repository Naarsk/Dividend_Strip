# Re-import necessary packages due to state reset
import matplotlib.pyplot as plt
import numpy as np

# Cobb-Douglas production function parameters
A = 1
alpha = 0.5  # equal share for housing (H) and labor (L)

# Define a grid of housing (H) and labor (L)
H = np.linspace(0.1, 10, 100)
L = np.linspace(0.1, 10, 100)
H, L = np.meshgrid(H, L)

# Production function: Q = A * H^alpha * L^(1-alpha)
Q = A * (H**alpha) * (L**(1 - alpha))

# Plot isoquants for fixed levels of output
plt.figure(figsize=(8, 6))
contours = plt.contour(H, L, Q, levels=[1, 2, 3, 4, 5, 6, 7], cmap="plasma")
plt.clabel(contours, inline=True, fontsize=10)
plt.title('Cobb–Douglas Production Isoquants (α = 0.5)')
plt.xlabel('Housing (H)')
plt.ylabel('Labor (L)')
plt.grid(True)
plt.tight_layout()

# Save plot
plot_path = "Cobb_Douglas_Isoquants_HL.png"
plt.savefig(plot_path)
plt.close()

plot_path

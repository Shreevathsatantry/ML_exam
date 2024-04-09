import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

# Generate data
n = 100
x = np.linspace(0, 2 * np.pi, n)
y = np.sin(x) + 0.3 * np.random.randn(n)  # Corrected the syntax here

# Perform Lowess smoothing (frac controls the smoothness of the line)
y_est = lowess(y, x, frac=0.2)  # Corrected the function call here

# Plot results
plt.plot(x, y, "r.", label="Original Data")
plt.plot(y_est[:, 0], y_est[:, 1], "b-", label="Lowess Smoothing")  # Changed x and y axes for y_est
plt.legend()
plt.show()

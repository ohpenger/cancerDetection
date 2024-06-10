import numpy as np
import numpy as np

x_values = np.array([0, 1, 2])
y_values = np.array([0, 1, 2])

X, Y = np.meshgrid(x_values, y_values)
coordinates = np.array([X.ravel(), Y.ravel()]).T

print(coordinates)

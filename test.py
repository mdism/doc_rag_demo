import math
import numpy as np

# Point A and Point B in 3D space
point_a = (1, 2)
point_b = (4, 5)

# Method 1: Using math.dist (Built-in for Python 3.8+)
distance_math = math.dist(point_a, point_b)

# Method 2: Using numpy (Best for machine learning arrays)
distance_np = np.linalg.norm(np.array(point_a) - np.array(point_b))

print(f"Euclidean Distance: {distance_math}")


x = (4-1)**2
y = (5-2)**2

print(x+y)
import numpy as np
from scipy.linalg import solve_banded

# Define the size and bandwidth of the banded matrix
n = 5  # Size of the matrix
u = 2  # upper Bandwidth of the matrix
l = 1  # lower Bandwidth of the matrix

# Full matrix with band pattern
matrix = np.array([[4, 2, -1, 0, 0], [-1, 3, 1, 2, 0], [0, -2, 5, 1, 0], [0, 0, 1, 3, 
  1], [0, 0, 0, 2, 1]])

# Create the banded matrix
banded_matrix = np.zeros((u + l + 1, n))
for i in range(n):
    for j in range(n):
        if j<i-l or j>i+u:
            continue
        banded_matrix[u + i - j, j] = matrix[i,j]

print("Banded Matrix:")
print(banded_matrix)

# Create the right-hand side of the linear system
b = np.array([0,1,2,3,4])

# Solve the linear system using the scipy solver
x = solve_banded((l, u), banded_matrix, b)

print("Solution:")
print(x)
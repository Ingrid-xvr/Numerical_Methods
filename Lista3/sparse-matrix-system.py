import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve

# Create a sparse matrix
data = np.array([1, 2, 3, 4, 5])
col = np.array([0, 1, 2, 3, 4])
ptr = np.array([0, 1, 2, 3, 4, 5])

sparse_matrix = csr_matrix((data, col, ptr), shape=(5, 5))

print("Sparse Matrix:")
print(sparse_matrix.toarray())
# Create a right-hand side vector
b = np.array([1, 2, 3, 4, 5])

# Solve the linear system
x = spsolve(sparse_matrix, b)

print("Solution:")
print(x)
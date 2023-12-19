import numpy as np

# Define the matrices
matrix1 = np.array([[1, 2, 3], [6, 5, 4], [7, 8, 9]])
matrix2 = np.array([[5, 6], [7, 8]])

# Calculate the Kronecker product
result = np.kron(matrix1, matrix2)

# Print the result
print(result)

import numpy as np

# Define the matrices
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

# Calculate the Kronecker product
kronecker_product = np.kron(matrix1, matrix2)

# Print the result
print(kronecker_product)

mat = [[1, 2, 3],
       [6, 5, 4],
       [7, 8, 9]]

def matrix_to_string(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    matrix_str = ""

    # Find the maximum length of each element in the matrix
    max_length = max(len(str(matrix[i][j])) for i in range(rows) for j in range(cols))

    for i in range(rows):
        matrix_str += "["
        for j in range(cols):
            # Right-align each element by adding spaces before it
            element_str = str(matrix[i][j]).rjust(max_length)
            matrix_str += element_str + " "
        matrix_str += "]\n"
    return matrix_str

print(matrix_to_string(mat))

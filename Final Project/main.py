# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix as mm


def print_m(data):
    for row in data:
        print(row)
    return


# Test your code here!

# ~~~~~~~~~~~~~~1. Basic Test~~~~~~~~~~~~~~~
# 1. define the matrix
mat = mm.Matrix(data=[[1, 2, 3], [6, 5, 4], [7, 8, 9]])
print(f"mat:\n{mat}\n")

# 2. shape
print(f"矩阵的形状是：{mat.shape()}\n")

# 3. reshape
reshaped_mat1 = mat.reshape((1, 9))
reshaped_mat2 = mat.reshape((9, 1))
print(f"矩阵变形为1*9是:\n{reshaped_mat1}\n")
print(f"矩阵变形为9*1是:\n{reshaped_mat2}\n")


# 4. dot product
mat_tmp = mm.Matrix([[3], [2], [1]])
dot_product = mat.dot(mat_tmp)
print(f"乘积结果是:\n{dot_product}\n")

# 5. transpose
trans_mat = mat.T()
print(f"矩阵转置是:\n{trans_mat}\n")

# 6. self sum
self_sum_all = mat.sum(axis=None)
self_sum_row = mat.sum(axis=1)  # 对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
self_sum_col = mat.sum(axis=0)  # 对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
print(f"矩阵自身全部求和是:\n{self_sum_all}\n")
print(f"矩阵按行求和是:\n{self_sum_row}\n")
print(f"矩阵按列求和是:\n{self_sum_col}\n")

# 7. copy
copy_mat = mat.copy()
print(f"矩阵复制是:\n{copy_mat}\n")

# 8. Kronecker_product
mat_tmp = mm.Matrix([[5, 6], [7, 8]])
kp_mat = mat.Kronecker_product(mat_tmp)
print(f"矩阵的Kronecker积为\n{kp_mat}\n")

# 9. get_item
print(f"mat[1, 2]:\n{mat[1, 2]}\n")

print(f"mat[:2, 1:]:\n{mat[:2, 1:]}\n")

# 10. set_item
copy_mat[1, 2] = 0
print(f"将(2,3)变为0后的矩阵为:\n{copy_mat}\n")

copy_mat = mat.copy()
copy_mat[1:, 2:] = mm.Matrix([[0], [0]])
print(f"另一赋值测试\n{copy_mat}\n")


print(mat)
# # ~~~~~~~~~~~~~~2. arrange() test~~~~~~~~~~~~~~~

# # ~~~~~~~~~~~~~~3. zeros() test~~~~~~~~~~~~~~~

# # ~~~~~~~~~~~~~~4. ones() test~~~~~~~~~~~~~~~

# # ~~~~~~~~~~~~~~5. nrandom() test~~~~~~~~~~~~~~~

# # ~~~~~~~~~~~~~~6. 最小二乘问题~~~~~~~~~~~~~~~


# # The following code is only for your reference
# # Please write the test code yourself


# # a = mm.narray([4, 5])

# # a = mm.I(10)
# # print(a)

# # print(a)
# # print(a.shape())
# # print(a.reshape([2, 6]))

# # ma1 = mm.Matrix(2,3)
# # ma2 = mm.Matrix(3,4)

# # print(ma1 * ma2)

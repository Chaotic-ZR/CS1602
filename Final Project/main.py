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
print(f"将[1:, 2:]变为0矩阵:\n{copy_mat}\n")

# 11. __pow__
print(f"矩阵的3次方是:\n{mat**3}\n")

# 12. __add__
other = mm.Matrix(data=[[2, 3, 4], [5, 8, 9], [7, 2, 5]])
print(f"两矩阵之和为：\n{mat+other}\n")

# 13. __sub__
print(f"两矩阵之差为：\n{mat-other}\n")

# 14. __mul__
print(f"两矩阵的“乘积”为：\n{mat*other}\n")

# 15. __len__
print(f"矩阵元素的数目为:\n{len(mat)}\n")

# 16. __str__
print(f"矩阵为：\n{mat}\n")

# 17. det
print(f"矩阵的行列式为：\n{mat.det()}\n")

# 18. inverse
# print(f"逆矩阵为：\n{mat.inverse()}\n")
new_mat = mm.Matrix(data=[[1, 2, 3], [89, 24, 46], [48, 2, 558]])
print(f"new_mat的逆矩阵为：\n{new_mat.inverse()}\n")

# 19. rank
print(f"矩阵的秩为：\n{mat.rank()}\n")

# 20. I(5)
print(f"5阶单位矩阵为：\n{mm.I(5)}\n")

print(mat)
#21. narray
print(f"三行二列的矩阵为:\n{mm.narray([3,2])}\n")

#22. arange
print(f"range(1,9,2)的矩阵为:\n{mm.arange(1,9,2)}\n")

#23. zeros
print(f"三行二列的零矩阵为:\n{mm.zeros([3,2])}\n")

#24. zeros_like
print(f"与mat形状相同的零矩阵为:\n{mm.zeros_like(mat)}\n")

#25. ones
print(f"三行二列的一矩阵为:\n{mm.ones([3,2])}\n")

#26. ones_like
print(f"与mat形状相同的一矩阵为:\n{mm.ones_like(mat)}\n")

#27. nrandom
print(f"三行二列的随机矩阵为:\n{mm.nrandom([3,2])}\n")

#28. nrandom_like
print(f"与mat形状相同的随机矩阵为:\n{mm.nrandom_like(mat)}\n")

#29. concatenate
A, B = mm.Matrix([[0, 1, 2]]), mm.Matrix([[3, 4, 5]])
print(f"A,B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])纵向拼接的结果为:\n{mm.concatenate((A,B))}\n")
print(f"A,B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])横向拼接A,B,A的结果为:\n{mm.concatenate((A,B,A),1)}\n")

#30. vectorize
def func(x):
    return x**2
print(f"将函数f(x)=x**2作用于mat的结果为:\n{mm.vectorize(func)(mat)}\n")

# ~~~~~~~~~~~~~~2. arrange() test~~~~~~~~~~~~~~~
m24 = mm.arange(0, 24, 1)
print(f"m24为:\n{m24}\n")
print(f"m24变换为(3, 8)是:\n{m24.reshape((3, 8))}\n")
print(f"m24变换为(24, 1)是:\n{m24.reshape((24, 1))}\n")
print(f"m24变换为(4, 6)是:\n{m24.reshape((4, 6))}\n")

# ~~~~~~~~~~~~~~3. zeros() test~~~~~~~~~~~~~~~
print(f"3*3维的0矩阵是:\n{mm.zeros((3, 3))}\n")
print(f"m24的zeros_like为:\n{mm.zeros_like(m24)}\n")


# ~~~~~~~~~~~~~~4. ones() test~~~~~~~~~~~~~~~
print(f"(3, 3)的全1矩阵是:\n{mm.ones((3, 3))}\n")
print(f"ones_like(m24)是:\n{mm.ones_like(m24)}\n")

# ~~~~~~~~~~~~~~5. nrandom() test~~~~~~~~~~~~~~~
print(f"3*3的随机矩阵:\n{mm.nrandom((3, 3))}\n")
print(f"m24的nrandom_like为:\n{mm.nrandom_like(m24)}\n")


# # ~~~~~~~~~~~~~~6. 最小二乘问题~~~~~~~~~~~~~~~
# m = 1000
# n = 100
# X = mm.nrandom((m, n))
# w = mm.nrandom((n, 1))
# e = mm.nrandom((m, 1))

# # 将e变成零均值
# e_value_list = [row[0] for row in e.data]
# sum_e = sum(e_value_list) - e_value_list[m-1]
# e[m-1, 0] = sum_e

# # 计算w_hat
# Y = X.dot(w) + e
# print(w)
# w_hat = (X.T().dot(X)).inverse().dot(X.T()).dot(Y)
# dif_w = w_hat - w
# print(dif_w)


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

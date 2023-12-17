# Test code for IEEE course final project
# Fan Cheng, 2024

import minimatrix as mm

def print_m(data):
    for row in data:
        print(row)
    return


#Test your code here!

# ~~~~~~~~~~~~~~1. Basic Test~~~~~~~~~~~~~~~
# 1. define the matrix
mat = mm.Matrix(data=[[1, 2, 3],
                      [6, 5, 4],
                      [7, 8, 9]])
print(mat)

# # 2. shape
# print(f"矩阵的形状是： {mat.shape}")

# # 3. reshape
# reshaped_mat1 = mat.reshape((1, 9))
# reshaped_mat2 = mat.reshape((9, 1))
# print("矩阵变形为1*9是:")
# print_m(reshaped_mat1.data)
# print("矩阵变形为9*1是:")
# print_m(reshaped_mat2.data)

# # 4. dot product
# mat_tmp = mm.Matrix([[3],
#                      [2],
#                      [1]])
# dot_product = mat.dot(mat_tmp)
# print("乘积结果是")
# print_m(dot_product.data)

# # 5. transpose
# trans_mat = mat.T()
# print("矩阵转置是: ")
# print_m(trans_mat.data)

# # 6. self sum
# self_sum_all = mat.sum(axis=None)
# self_sum_row = mat.sum(axis=1) # 对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
# self_sum_col = mat.sum(axis=0) # 对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
# print("矩阵自身全部求和是: ")
# print_m(self_sum_all.data)

# print("矩阵按行全部求和是: ")
# print_m(self_sum_row.data)

# print("矩阵自身按列求和是: ")
# print_m(self_sum_col.data)


# # 7. copy
# copy_mat = mat.copy()
# print("矩阵复制: ")
# print_m(copy_mat.data)

# # 8. Kronecker_product
# mat_tmp = mm.Matrix([[5, 6], 
#                      [7, 8]])
# kp_mat = mat.Kronecker_product(mat_tmp)
# print("矩阵的Kronecker积为")
# print_m(kp_mat.data)

# # 9. get_item
# print(f"mat[1, 2]: {mat[1, 2]}")

# print("mat[:2, 1:]: ")
# print_m(mat[:2, 1:].data)

# # 10. set_item
# copy_mat[1, 2] = 0
# print_m(copy_mat.data)
# copy_mat[1:, 2:] = mm.Matrix([[0],
#                               [0]])
# print_m(copy_mat.data)






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




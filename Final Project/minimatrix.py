# Framework for IEEE course final project
# Fan Cheng, 2022

import random


class Matrix:
    r"""
    自定义的二维矩阵类

    Args:
        data: 一个二维的嵌套列表，表示矩阵的数据。即 data[i][j] 表示矩阵第 i+1 行第 j+1 列处的元素。
              当参数 data 不为 None 时，应根据参数 data 确定矩阵的形状。默认值: None
        dim: 一个元组 (n, m) 表示矩阵是 n 行 m 列, 当参数 data 为 None 时，根据该参数确定矩阵的形状；
             当参数 data 不为 None 时，忽略该参数。如果 data 和 dim 同时为 None, 应抛出异常。默认值: None
        init_value: 当提供的 data 参数为 None 时，使用该 init_value 初始化一个 n 行 m 列的矩阵，
                    即矩阵各元素均为 init_value. 当参数 data 不为 None 时，忽略该参数。 默认值: 0

    Attributes:
        dim: 一个元组 (n, m) 表示矩阵的形状
        data: 一个二维的嵌套列表，表示矩阵的数据

    Examples:
        >>> mat1 = Matrix(dim=(2, 3), init_value=0)
        >>> print(mat1)
        >>> [[0 0 0]
             [0 0 0]]
        >>> mat2 = Matrix(data=[[0, 1], [1, 2], [2, 3]])
        >>> print(mat2)
        >>> [[0 1]
             [1 2]
             [2 3]]
    """

    def __init__(self, data=None, dim=None, init_value=0):
        if data != None:  # 输入data不为空的情况
            self.data = data
            self.dim = (len(data), len(data[0]))
        elif dim == None:  # data和dim全为空的情况
            raise ValueError("data和dim不能全为None")
        else:  # data为空，dim和init_value决定矩阵的情况
            self.data = [
                [init_value for column_num in range(0, dim[1])]
                for row_num in range(0, dim[0])
            ]
            self.dim = dim

    def shape(self):
        r"""
        返回矩阵的形状 dim
        """
        return self.dim

    def reshape(self, newdim):
        r"""
        将矩阵从(m,n)维拉伸为newdim=(m1,n1)
        该函数不改变 self

        Args:
            newdim: 一个元组 (m1, n1) 表示拉伸后的矩阵形状。如果 m1 * n1 不等于 self.dim[0] * self.dim[1],
                    应抛出异常

        Returns:
            Matrix: 一个 Matrix 类型的返回结果, 表示 reshape 得到的结果
        """
        if newdim[0] * newdim[1] != self.dim[0] * self.dim[1]:
            raise ValueError("拉伸前后元素数量不相等")

        single_row_matrix = [
            self.data[row_num][column_num]
            for row_num in range(0, self.dim[0])
            for column_num in range(0, self.dim[1])
        ]  # 把原矩阵化为单层list
        new_matrix = []
        i = 0
        for row_num in range(0, newdim[0]):
            new_matrix.append([])
            for column_num in range(0, newdim[1]):
                new_matrix[row_num].append(single_row_matrix[i])
                i += 1
        return new_matrix

    def dot(self, other):
        r"""
        矩阵乘法：矩阵乘以矩阵
        按照公式 A[i, j] = \sum_k B[i, k] * C[k, j] 计算 A = B.dot(C)

        Args:
            other: 参与运算的另一个 Matrix 实例

        Returns:
            Matrix: 计算结果

        Examples:
            >>> A = Matrix(data=[[1, 2], [3, 4]])
            >>> A.dot(A)
            >>> [[ 7 10]
                 [15 22]]
        """
        if self.dim[1] != other.dim[0]:  # 排除特殊情况
            raise ValueError("矩阵无法相乘")

        new_matrix = Matrix(None, (self.dim[0], other.dim[1]), 0)
        for n_row1 in range(0, self.dim[0]):
            for n_col2 in range(0, other.dim[1]):
                for joint_i in range(0, self.dim[1]):
                    new_matrix.data[n_row1][n_col2] += (
                        self.data[n_row1][joint_i] * other.data[joint_i][n_col2]
                    )
        return new_matrix.data

    def T(self):
        r"""
        矩阵的转置

        Returns:
            Matrix: 矩阵的转置

        Examples:
            >>> A = Matrix(data=[[1, 2], [3, 4]])
            >>> A.T()
            >>> [[1 3]
                 [2 4]]
            >>> B = Matrix(data=[[1, 2, 3], [4, 5, 6]])
            >>> B.T()
            >>> [[1 4]
                 [2 5]
                 [3 6]]
        """
        new_matrix = [
            [row[num_col] for row in self.data] for num_col in range(self.dim[1])
        ]
        return new_matrix

    def sum(self, axis=None):
        r"""
        根据指定的坐标轴对矩阵元素进行求和

        Args:
            axis: 一个整数，或者 None. 默认值: None
                  axis = 0 表示对矩阵进行按列求和，得到形状为 (1, self.dim[1]) 的矩阵
                  axis = 1 表示对矩阵进行按行求和，得到形状为 (self.dim[0], 1) 的矩阵
                  axis = None 表示对矩阵全部元素进行求和，得到形状为 (1, 1) 的矩阵

        Returns:
            Matrix: 一个 Matrix 类的实例，表示求和结果

        Examples:
            >>> A = Matrix(data=[[1, 2, 3], [4, 5, 6]])
            >>> A.sum()
            >>> [[21]]
            >>> A.sum(axis=0)
            >>> [[5 7 9]]
            >>> A.sum(axis=1)
            >>> [[6]
                 [15]]
        """
        # sum all
        if axis == None:
            list_of_element = [
                row[num_col] for row in self.data for num_col in range(self.dim[1])
            ]
            sum_matrix = [[sum(list_of_element)]]
        # sum by column(not finished, wait for the transpose)
        elif axis == 0:
            sum_matrix = [[]]

        # sum by row
        elif axis == 1:
            sum_matrix = [
                [sum(row) for row in self.data] for num_col in range(self.dim[1])
            ]

    def copy(self):
        r"""
        返回matrix的一个备份

        Returns:
            Matrix: 一个self的备份
        """
        new_matrix = Matrix(data=self.data)
        return new_matrix

    def Kronecker_product(self, other):
        r"""
        计算两个矩阵的Kronecker积，具体定义可以搜索，https://baike.baidu.com/item/克罗内克积/6282573

        Args:
            other: 参与运算的另一个 Matrix

        Returns:
            Matrix: Kronecke product 的计算结果
        """
        new_matrix = [
            [x * y for x in row1 for y in row2]
            for row1 in self.data
            for row2 in other.data
        ]
        return new_matrix

    def __getitem__(self, key):
        r"""
        实现 Matrix 的索引功能，即 Matrix 实例可以通过 [] 获取矩阵中的元素（或子矩阵）

        x[key] 具备以下基本特性：
        1. 单值索引
            x[a, b] 返回 Matrix 实例 x 的第 a 行, 第 b 列处的元素 (从 0 开始编号)
        2. 矩阵切片
            x[a:b, c:d] 返回 Matrix 实例 x 的一个由 第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵
            特别地, 需要支持省略切片左(右)端点参数的写法, 如 x 是一个 n 行 m 列矩阵, 那么
            x[:b, c:] 的语义等价于 x[0:b, c:m]
            x[:, :] 的语义等价于 x[0:n, 0:m]

        Args:
            key: 一个元组，表示索引

        Returns:
            索引结果，单个元素或者矩阵切片

        Examples:
            >>> x = Matrix(data=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 0, 1]
                    ])
            >>> x[1, 2]
            >>> 6
            >>> x[0:2, 1:4]
            >>> [[1 2 3]
                 [5 6 7]]
            >>> x[:, :2]
            >>> [[0 1]
                 [4 5]
                 [8 9]]
        """

        row_slice, col_slice = key
        # single element
        if not isinstance(row_slice, slice) and not isinstance(col_slice, slice):
            return self.data[row_slice][col_slice]

        # slice
        # initialize the slice
        r_st, r_end, c_st, c_end = (
            row_slice.start,
            row_slice.stop,
            col_slice.start,
            col_slice.stop,
        )
        if r_st == None:
            r_st = 0
        if r_end == None:
            r_end = self.dim[0]
        if c_st == None:
            c_st = 0
        if c_end == None:
            c_end = self.dim[1]

        # apply the slice
        new_matrix = Matrix(dim=(r_end - r_st, c_end - c_st))
        for row_num in range(r_st, r_end):
            for col_num in range(c_st, c_end):
                new_matrix.data[row_num - r_st][col_num - c_st] = self.data[row_num][
                    col_num
                ]
        return new_matrix

    def __setitem__(self, key, value):
        r"""
        实现 Matrix 的赋值功能, 通过 x[key] = value 进行赋值的功能

        类似于 __getitem__ , 需要具备以下基本特性:
        1. 单元素赋值
            x[a, b] = k 的含义为，将 Matrix 实例 x 的 第 a 行, 第 b 处的元素赋值为 k (从 0 开始编号)
        2. 对矩阵切片赋值
            x[a:b, c:d] = value 其中 value 是一个 (b-a)行(d-c)列的 Matrix 实例
            含义为, 将由 Matrix 实例 x 的第 a, a+1, ..., b-1 行, 第 c, c+1, ..., d-1 列元素构成的子矩阵 赋值为 value 矩阵
            即 子矩阵的 (i, j) 位置赋值为 value[i, j]
            同样地, 这里也需要支持如 x[:b, c:] = value, x[:, :] = value 等省略写法

        Args:
            key: 一个元组，表示索引
            value: 赋值运算的右值，即要赋的值

        Examples:
            >>> x = Matrix(data=[
                        [0, 1, 2, 3],
                        [4, 5, 6, 7],
                        [8, 9, 0, 1]
                    ])
            >>> x[1, 2] = 0
            >>> x
            >>> [[0 1 2 3]
                 [4 5 0 7]
                 [8 9 0 1]]
            >>> x[1:, 2:] = Matrix(data=[[1, 2], [3, 4]])
            >>> x
            >>> [[0 1 2 3]
                 [4 5 1 2]
                 [8 9 3 4]]
        """
        row_slice, col_slice = key
        # single element
        if not isinstance(row_slice, slice) and not isinstance(col_slice, slice):
            self.data[row_slice][col_slice] = value
            return

        # slice
        # initialize the slice
        r_st, r_end, c_st, c_end = (
            row_slice.start,
            row_slice.stop,
            col_slice.start,
            col_slice.stop,
        )
        if r_st == None:
            r_st = 0
        if r_end == None:
            r_end = self.dim[0]
        if c_st == None:
            c_st = 0
        if c_end == None:
            c_end = self.dim[1]

        # apply the slice
        # new_matrix = Matrix(dim=(r_end - r_st, c_end - c_st))
        for row_num in range(r_st, r_end):
            for col_num in range(c_st, c_end):
                self.data[row_num][col_num] = value.data[row_num - r_st][col_num - c_st]
        return

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __pow__(self, n):
        r"""
        矩阵的n次幂，n为自然数
        该函数应当不改变 self 的内容

        Args:
            n: int, 自然数

        Returns:
            Matrix: 运算结果
        """
        # 需调用dot函数
        # B=self
        # for i in range(n-1):
        # 	B=Matirx.dot(self)
        # return B

    def __add__(self, other):
        r"""
        两个矩阵相加
        该函数应当不改变 self 和 other 的内容

        Args:
            other: 一个 Matrix 实例

        Returns:
            Matrix: 运算结果
        """
        # 与__mul__方法相同

    def __sub__(self, other):
        r"""
        两个矩阵相减
        该函数应当不改变 self 和 other 的内容

        Args:
            other: 一个 Matrix 实例

        Returns:
            Matrix: 运算结果
        """
        # 与__mul__方法相同

    def __mul__(self, other):
        r"""
        两个矩阵 对应位置 元素  相乘
        注意 不是矩阵乘法dot
        该函数应当不改变 self 和 other 的内容

        Args:
            other: 一个 Matrix 实例

        Returns:
            Matrix: 运算结果

        Examples:
            >>> Matrix(data=[[1, 2]]) * Matrix(data=[[3, 4]])
            >>> [[3 8]]
        """
        # lst=self.copy()
        # for i in range(行数):
        # 	for j in range(列数):
        # 		lst[i][j]=self[i][j]*other[i][j]
        # return lst

    def __len__(self):
        r"""
        返回矩阵元素的数目

        Returns:
            int: 元素数目，即 行数 * 列数
        """
        # self.dim!=None:return dim[0]*dim[1]
        # self.dim==None:return len(Matrix)*len(Matrix[0])

    def __str__(self):
        r"""
        按照
        [[  0   1   4   9  16  25  36  49]
          [ 64  81 100 121 144 169 196 225]
          [256 289 324 361 400 441 484 529]]
         的格式将矩阵表示为一个 字符串
         ！！！ 注意返回值是字符串
        """
        # 需调用T函数
        # a=[0]*lie
        # for i in range(n):
        # 	a[i]=max(A^T[i])
        # for i in range(hang):
        # 	print("[")
        # 	for j in range(lie):
        # 		print(str(lst[i][j]).rjust(a[j]),end=" ")
        # 	print("]\n")
        # 再补上首尾方括号即可
        pass

    def det(self):
        r"""
        计算方阵的行列式。对于非方阵的情形应抛出异常。
        要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
        提示: Gauss消元

        Returns:
            一个 Python int 或者 float, 表示计算结果
        """
        #
        """def hangcheng(lst,k):
            for i in range(len(lst)):
                lst[i]*=k
            return lst
        def hangjian(lst1,lst2):#lst2-lst1
            for i in range(len(lst1)):
                lst2[i]-=lst1[i]
            return lst2
        def judge(self.data):
            for i in range(hang):
                for j in range(i+1:lie):
                    if self.data[i][j]!=0:
                        return False
            return True    
        num=0
        while not judge(self.data):
            for n in range(len(self.data)):
                if self.data[num][n]==0:
                    t=self.data[num].copy()
                    for i in range(num+1,hang):
                        if self.data[i][n]!=0:
                            self.data[num]=self.data[i]
                            self.data[i]=lt
                        break                               #将首个首项非零的一行置顶


                for i in range(num+1,hang):
                    if self.data[i][n]!=0:
                        k=self.data[i][n]/self.data[num][n]
                        lst=hangcheng(self.data[num],k)
                        self.data[i]=hangjian(lst,self.data[i])
            num+=1
        return self.data"""

    def inverse(self):
        r"""
        计算非奇异方阵的逆矩阵。对于非方阵或奇异阵的情形应抛出异常。
        要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
        提示: Gauss消元

        Returns:
            Matrix: 一个 Matrix 实例，表示逆矩阵
        """
        n = len(self.data)
        lst = self.data.copy()
        lst0 = lst.copy()
        storage = [[0 if i != j else 1 for j in range(n)] for i in range(n)]
        for i in range(n):
            lst[i].extend(storage[i])

        def hangcheng(lst, k):
            for i in range(len(lst)):
                lst[i] *= k
            return lst

        def hangjian(lst1, lst2):  # lst2-lst1
            for i in range(len(lst1)):
                lst2[i] -= lst1[i]
            return lst2

        num = 0
        while num < n + 1:
            for t in range(n):
                if self.data[0][t] == 0:
                    lt = self.data[0].copy()
                    for i in range(1, n):
                        if self.data[i][t] != 0:
                            lst[0] = self.data[i]
                            lst0[0] = self.data[i]
                            lst[i] = lt
                            lst0[i] = lt
                        break
                for i in range(num + 1, n) and range(0, num):
                    if lst[i][t] != 0:
                        k = lst[i][t] / lst[num][t]
                        help_lst = hangcheng(lst[num], k)
                        lst[i] = hangjian(help_lst, lst[i])
            num += 1
        return lst

    def rank(self):
        r"""
        计算矩阵的秩
        要求: 该函数应不改变 self 的内容; 该函数的时间复杂度应该不超过 O(n**3).
        提示: Gauss消元

        Returns:
            一个 Python int 表示计算结果
        """
        # if hang<lie:
        """	
        def hangcheng(lst,k):
            for i in range(len(lst)):
                lst[i]*=k
            return lst
        def hangjian(lst1,lst2):#lst2-lst1
            for i in range(len(lst1)):
                lst2[i]-=lst1[i]
            return lst2
        def judge(self.data):
            for i in range(hang):
                for j in range(i+1:lie):
                    if self.data[i][j]!=0:
                         return False
            return True    
        num=0
        while not judge(self.data):
            for n in range(len(self.data)):
            if self.data[num][n]==0:
                t=self.data[num].copy()
                   for i in range(num+1,hang):
                    if self.data[i][n]!=0:
                        self.data[num]=self.data[i]
                        self.data[i]=lt
                    break                               #将首个首项非零的一行置顶

            for i in range(num+1,hang):
                if self.data[i][n]!=0:
                    k=self.data[i][n]/self.data[num][n]
                    lst=hangcheng(self.data[n],k)
                    self.data[i]=hangjian(lst,self.data[n])
            num+=1
        new_lst=self.data
        r=0
        while new_lst[r].count(0)!=lie:
            r+=1
        return r"""
        # if hang>lie:
        # self取转置重新讨论即可


def I(n):
    """
    return an n*n unit matrix
    """
    return [[0 if i != j else 1 for j in range(n)] for i in range(n)]


def narray(dim, init_value=1):  # dim (,,,,,), init为矩阵元素初始值
    r"""
    返回一个matrix，维数为dim，初始值为init_value

    Args:
        dim: Tuple[int, int] 表示矩阵形状
        init_value: 表示初始值，默认值: 1

    Returns:
        Matrix: 一个 Matrix 类型的实例
    """

    Matrix = []
    if dim[0] == 0 or dim[1] == 0:
        return None
    else:
        for i in range(dim[0]):
            Matrix.append([init_value] * dim[1])
        return Matrix  # 返回一个matrix,维数为dim,初始值为init_value
    # return Matrix(dim, None, init_value)


def arange(start, end, step):
    r"""
    返回一个1*n 的 narray 其中的元素类同 range(start, end, step)

    Args:
        start: 起始点(包含)
        end: 终止点(不包含)
        step: 步长

    Returns:
        Matrix: 一个 Matrix 实例
    """
    Narray = []
    for i in range(start, end, step):
        Narray.append(i)
    return Narray  # 返回一个1*n 的 narray 其中的元素类同 range(start, end, step)


def zeros(dim):
    r"""
    返回一个维数为dim 的全0 narray

    Args:
        dim: Tuple[int, int] 表示矩阵形状

    Returns:
        Matrix: 一个 Matrix 类型的实例
    """
    return narray(dim, 0)  # 返回一个维数为dim 的全0 narray


def zeros_like(matrix):
    r"""
    返回一个形状和matrix一样 的全0 narray

    Args:
        matrix: 一个 Matrix 实例

    Returns:
        Matrix: 一个 Matrix 类型的实例

    Examples:
        >>> A = Matrix(data=[[1, 2, 3], [2, 3, 4]])
        >>> zeros_like(A)
        >>> [[0 0 0]
             [0 0 0]]
    """
    if matrix == None:
        return None
    else:
        m = len(matrix)
        n = len(matrix[0])
        return zeros([m, n])  # 返回一个形状和matrix一样 的全0 narray


def ones(dim):
    r"""
    返回一个维数为dim 的全1 narray
    类同 zeros
    """
    return narray(dim, 1)  # 返回一个维数为dim 的全1 narray


def ones_like(matrix):
    r"""
    返回一个维数和matrix一样 的全1 narray
    类同 zeros_like
    """
    if matrix == None:
        return None
    else:
        m = len(matrix)
        n = len(matrix[0])
        return ones([m, n])  # 返回一个维数和matrix一样 的全1 narray


def nrandom(dim):
    r"""
    返回一个维数为dim 的随机 narray
    参数与返回值类型同 zeros
    """
    a = random.random()
    return narray(dim, a)


def nrandom_like(matrix):
    r"""
    返回一个维数和matrix一样 的随机 narray
    参数与返回值类型同 zeros_like
    """
    if matrix == None:
        return None
    else:
        m = len(matrix)
        n = len(matrix[0])
        return nrandom([m, n])  # 返回一个维数为dim 的随机 narray


def concatenate(items, axis=0):
    r"""
    将若干矩阵按照指定的方向拼接起来
    若给定的输入在形状上不对应，应抛出异常
    该函数应当不改变 items 中的元素

    Args:
        items: 一个可迭代的对象，其中的元素为 Matrix 类型。
        axis: 一个取值为 0 或 1 的整数，表示拼接方向，默认值 0.
              0 表示在第0维即行上进行拼接
              1 表示在第1维即列上进行拼接

    Returns:
        Matrix: 一个 Matrix 类型的拼接结果

    Examples:
        >>> A, B = Matrix([[0, 1, 2]]), Matrix([[3, 4, 5]])
        >>> concatenate((A, B))
        >>> [[0 1 2]
             [3 4 5]]
        >>> concatenate((A, B, A), axis=1)
        >>> [[0 1 2 3 4 5 0 1 2]]
    """

    def concatenate_0(A, B):
        if A == None and B == None:
            return None
        elif not (A == None or B == None):
            if len(A[0]) == len(B[0]):
                return A + B
            else:
                return "error"
        else:
            return "error"  # axis=0时,两矩阵列数不变拼接

    def concatenate_1(A, B):
        if A == None and B == None:
            return None
        elif not (A == None or B == None):
            if len(A) == len(B):
                answer = []
                for i in range(len(A)):
                    answer.append(A[i] + B[i])
                return answer
            else:
                return "error"
        else:
            return "error"  # axis=1时,两矩阵行数不变拼接

    answer = items[0]
    if axis == 0:
        for i in range(1, len(items)):
            if concatenate_0(answer, items[i]) == "error":
                return "error"
            else:
                answer = concatenate_0(answer, items[i])
        return answer  # axis=0时,多矩阵列数不变拼接
    elif axis == 1:
        for i in range(1, len(items)):
            if concatenate_1(answer, items[i]) == "error":
                return "error"
            else:
                answer = concatenate_1(answer, items[i])
        return answer  # axis=1时,多矩阵行数不变拼接
    else:
        return "error"


def vectorize(func):
    r"""
    将给定函数进行向量化

    Args:
        func: 一个Python函数

    Returns:
        一个向量化的函数 F: Matrix -> Matrix, 它的参数是一个 Matrix 实例 x, 返回值也是一个 Matrix 实例；
        它将函数 func 作用在 参数 x 的每一个元素上

    Examples:
        >>> def func(x):
                return x ** 2
        >>> F = vectorize(func)
        >>> x = Matrix([[1, 2, 3],[2, 3, 1]])
        >>> F(x)
        >>> [[1 4 9]
             [4 9 1]]
        >>>
        >>> @vectorize
        >>> def my_abs(x):
                if x < 0:
                    return -x
                else:
                    return x
        >>> y = Matrix([[-1, 1], [2, -2]])
        >>> my_abs(y)
        >>> [[1, 1]
             [2, 2]]
    """

    def F(x):
        if x == None:
            return None
        else:
            answer = []
            m = len(x)
            n = len(x[0])
            for i in range(m):
                list1 = []
                for j in range(n):
                    list1.append(func(x[i][j]))
                answer.append(list1)  # 对列表中每一个元素进行func运算
        return answer

    return F


if __name__ == "__main__":
    print("test here")
    pass


m1 = Matrix([[1, 2, 3], [2, 3, 4]])
m2 = Matrix([[6, 6]])
m1[:1, :2] = m2
print(m1.data)
# m2 = Matrix([[1],
# 			 [1],
# 			 [1]])
# print(m1.dot(m2).data)

# 1. 加减法函数
## 1.1 加减法判断函数
### 加减法判断函数
`add_sub_init(str1, str2, out_flag)`
- 将输入的两个数转换为list
- 通过两数的符号及原先的加减函数, 判断是要进行绝对值加法还是绝对值减法, 并确定答案的正负性
- 最终将数据传递到加减法核心函数(list形式)
## 1.2 加法函数
### 加法核心函数
`add_core(lst1, lst2, flag, l_valid)`
- 对两个非负整数(list形式)进行加法

### 加法过渡函数
`add(str1, str2)`
- 用于接受输入, 并将内容传递到加减法判断函数
- 最后将结果从list形式转换为string形式
## 1.3 减法函数
### 减法核心函数
`sub_core(lst1, lst2, flag, l_valid)`
- 对两个非负整数(list形式)进行减法(lst1大于lst2)
### 减法过渡函数
`sub(str1, str2)`
- 用于接受输入, 并将内容传递到加减法判断函数
- 最后将结果从list形式转换为string形式

# 2. 乘法函数
### 单次乘法函数
`single_mul(lst, a, front_blank, back_blank)`
- 用于将一个list代表的大数与另一大数的单独一位相乘
- 返回乘积的结果

### 乘法核心函数
`mul_core(lst1, lst2)`
- 用于两个大数(list形式)的乘积计算

### 乘法过渡函数
`mul(str1, str2)`
- 用于将大数从string形式转换为list形式
- 判断结果正负性
- 将结果转换为string

# 3. 除法函数
### 除法核心函数
`div_core(lst1, lst2)`
- 利用升位的减法模拟除法

### 除法过渡函数
`div(str1, str2)`
- 将大数从string形式转换为list形式
- 将结果转换为string

# 4. 幂函数
### 幂过渡函数
`pow(str1, n)`
- 将大数从string形式转换为list形式
- 将结果转换成string

### 幂核心函数
`pow_core(lst1, n)`
- 计算幂的递归函数

# 5. 转换及其他功能函数
### 字符串转列表函数
`s_to_l(s, out_flag)`
- 将字符串转换成列表与单独的正负号

### 列表转字符串函数
`l_to_s(lst, flag)`
- 将列表和单独的正负号转换成字符串

### 初始转换函数
`two_init(str1, str2, out_flag)`
- 初始转换函数(string->list)

### 大小比较函数
`cmp(lst1, lst2)`
- 比较两数大小(lst1>=lst2返回True)




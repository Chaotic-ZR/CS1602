# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 加减法函数
def add_sub_init(str1, str2, out_flag): # 加减法判断函数
    lst1, flag1, lst2, flag2 = two_init(str1, str2, out_flag)
    l_valid = max(len(lst1), len(lst2))
    
    # 加法判断及处理
    if flag1 == flag2:
        if len(lst1) > len(lst2):
            lst2 += [0] * (len(lst1) - len(lst2))
            return add_core(lst1, lst2, flag1, l_valid)
        elif len(lst1) == len(lst2):
            return add_core(lst1, lst2, flag1, l_valid)
        else:
            lst1 += [0] * (len(lst2) - len(lst1))
            return add_core(lst2, lst1, flag2, l_valid) 
        
    # 减法判断及处理
    if len(lst1) > len(lst2):
        lst2 += [0] * (len(lst1) - len(lst2))
        return sub_core(lst1, lst2, flag1, l_valid)
    elif len(lst1) < len(lst2):
        lst1 += [0] * (len(lst2) - len(lst1))
        return sub_core(lst2, lst1, flag2, l_valid)
    else:
        if cmp(lst1, lst2):
            return sub_core(lst1, lst2, flag1, l_valid)
        else:
            return sub_core(lst2, lst1, flag2, l_valid)
            
        
def add_core(lst1, lst2, flag, l_valid): # list加法主体
    # 初始化
    lst_sum = []
    sum_more = 0
    
    # 相加
    for i in range(0, l_valid):
        sum_tmp = lst1[i] + lst2[i] + sum_more
        sum_digit = sum_tmp % 10
        sum_more = sum_tmp // 10
        lst_sum.append(sum_digit)
    
    return lst_sum, flag


def add(str1, str2): # 加法过渡函数
    out_flag = False
    lst_ans, flag = add_sub_init(str1, str2, out_flag)
    return l_to_s(lst_ans, flag)


def sub_core(lst1, lst2, flag, l_valid): # list减法主体
    # 初始化 
    lst_sub = []
    sub_more = 0
    
    # 逐位相减
    for i in range(0, l_valid):
        sub_tmp = 10 + lst1[i] - lst2[i] - sub_more
        sub_digit = sub_tmp % 10
        sub_more = 1 - (sub_tmp // 10)
        lst_sub.append(sub_digit)
        
    return lst_sub, flag


def sub(str1, str2): # 减法过渡函数
    out_flag = True
    lst_ans, flag = add_sub_init(str1, str2, out_flag)
    return l_to_s(lst_ans, flag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 乘法函数
def single_mul(lst, a, front_blank, back_blank): # 用于lst1和单个数相乘
    new_lst = [0] * front_blank
    lst_tmp = [x*a for x in lst]
    new_lst += lst_tmp
    new_lst += [0] * back_blank
    return new_lst


def mul_core(lst1, lst2): # list乘积主体
    # 获取每一组(行）数据
    row = []
    l_blank = len(lst2) - 1
    for i in range(0, len(lst2)):
        row.append(single_mul(lst1, lst2[i], i, l_blank-i))
    
    # 将每一组数据的列分别相加
    column_num = len(lst1) + len(lst2) - 1
    column_sum = [0] * column_num 
    for i in range(0, column_num):
        for x in row:
            column_sum[i] += x[i]
    
    # 完成每一列的进位
    lst_mul = []
    sum_more = 0
    for y in column_sum:
        sum_tmp = y + sum_more 
        lst_mul.append(sum_tmp%10)
        sum_more = sum_tmp // 10
    if sum_more != 0:
        lst_mul.append(sum_more)
        
    return lst_mul


def mul(str1, str2): # 乘积过渡函数
    # 初始化
    out_flag = False
    lst1, flag1, lst2, flag2= two_init(str1, str2, out_flag)
    flag = (flag1 != flag2)
    
    # 计算乘积并返回字符串
    return l_to_s(mul_core(lst1, lst2), flag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 除法函数
def div_core(lst1, lst2): # list除法主体
    quotient = [0] * len(lst1)
    
    # 逐级进行减法(lst1的位数高于lst2时)
    while len(lst1) > len(lst2):  
        k = 0 
        up_digit = 0
        divisor_tmp = []
        while len(lst1) > len(divisor_tmp) + len(lst2): # 对减数升位以简化运算
            divisor_tmp.append(0)
            up_digit += 1
        if cmp(lst1, divisor_tmp+lst2) == False:
            del divisor_tmp[-1]
            up_digit -= 1
        divisor_tmp += lst2 
        divisor_tmp += [0] * (len(lst1) - len(divisor_tmp))
        while cmp(lst1, divisor_tmp):
            lst1, flag_tmp = sub_core(lst1, divisor_tmp, False, len(lst1))
            k += 1
        quotient[up_digit] = k
        
        while len(lst1) > len(lst2): # 消去lst1中多余的0
            if lst1[-1] == 0:
                del lst1[-1]
            else:
                break
            
    # 最后进行的同级减法
    k = 0
    while cmp(lst1, lst2):
        lst1, flag_tmp = sub_core(lst1, lst2, False, len(lst1))
        k += 1
    quotient[0] = k
    
    return quotient, lst1


def div(str1, str2): # 除法过渡函数
    if str2 == '0':
        print("除数不为0")
        return
    lst1, flag1, lst2, flag2= two_init(str1, str2, False)
    quotient, remainder= div_core(lst1, lst2)
    return l_to_s(quotient, False), l_to_s(remainder, False)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 幂函数
def pow(str1, n): # 幂过渡函数
    if str1 == '0':
        return '0'
    # 转换string
    lst1, flag_tmp = s_to_l(str1, False)
    
    # 转换计算得到的list为string
    return l_to_s(pow_core(lst1, n), False)
    

def pow_core(lst1, n):
    # 基本情况
    if n == 0:
        return [1]
    
    # 递归主体部分
    tmp = pow_core(lst1, n//2)
    tmp = mul_core(tmp, tmp)
    return tmp if n%2 == 0 else mul_core(tmp, lst1)
    

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 转换及其他功能函数
def s_to_l(s, out_flag): # 字符串转换为列表
    # 判断是否是负数并把数字逆序
    flag = out_flag
    if s[0] == '-':
        flag = not flag
        s_reversed = s[:0:-1]
    elif s[0] == '+':
        s_reversed = s[:0:-1]
    else:
        s_reversed = s[::-1]
        
    # 将字符串转化成list(reversed)
    lst = [int(x) for x in s_reversed]
    return lst, flag


def l_to_s(lst, flag): # 列表转换成字符串
    # 删除多余的0
    l = len(lst)
    for i in range(l-1, -1, -1):
        if lst[i] != 0:
            break
        else:
            del lst[i]
            
    # 将list转换回string
    if flag == True:
        lst.append('-')
    lst_reversed = lst[::-1]
    s = ''.join(str(x) for x in lst_reversed)
    return '0' if s == '' or s == '-' else s


def two_init(str1, str2, out_flag): # 初始化函数
    lst1, flag1= s_to_l(str1, False)
    lst2, flag2= s_to_l(str2, out_flag)
    return lst1, flag1, lst2, flag2


def cmp(lst1, lst2): # 比较大小(lst1>=lst2返回True)
    if len(lst1) > len(lst2):
        return True
    elif len(lst1) == len(lst2):
        for i in range(len(lst1)-1, -1, -1):
            if lst1[i] > lst2[i]:
                return True
            elif lst1[i] < lst2[i]:
                return False
            else:
                continue
        return True    


# 样例测试
print(f"22222222222222+8773849905050505={add("22222222222222", "8773849905050505")}")
print(f"11111111-9877344555={sub("11111111", "9877344555")}")
print(f"345676778778-222222={sub("345676778778", "222222")}")
print(f"123456*789={mul("123456", "789")}")
print(f"8773849905050505//123={div("8773849905050505", "123")}")
print(f"2**66={pow("2", 66)}")
print(f"2**100+3**50={add(pow("2", 100), pow("3", 50))}")

a = add(mul("2", "100"), mul("123456", "789"))
b, c = div("8773849905050505", "123")
print(f"2*100+123456*789-8773849905050505//123={sub(a, b)}")
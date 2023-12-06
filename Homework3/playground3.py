# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 幂函数
def pow(str1, n):
    # 0. 转换str
    lst1, flag_tmp = s_to_l(str1, False)
    # 转换计算得到的list为string
    return l_to_s(pow_core(lst1, n))
    


def pow_core(lst1, n):
    # to-do: 利用二分法求解pow
    
    # 1. 基本情况
    if n == 0:
        return 1
    # 2. 利用递归求出一半核心函数的幂
    # 要求tmp仍然为list
    tmp = pow_core(lst1, n//2)
    
    # 3. 递归主体部分
    tmp = mul(tmp, tmp)
    return tmp if n%2 == 0 else mul(tmp, lst1)
    
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 除法函数
def div_result(quotient, remainder):
    # quo_tmp = ''.join(str(x) for x in quotient)
    quo_tmp = l_to_s(quotient, False)
    rem_tmp = l_to_s(remainder, False)
    if quo_tmp == '' :
        quo_tmp = '0'
    if rem_tmp == '':
        rem_tmp = '0'
    return quo_tmp, rem_tmp


def cmp(lst1, lst2):
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
    
    
def div(str1, str2):
    # 初始化
    lst1, flag1, lst2, flag2= two_init(str1, str2, False)
    quotient = [0] * len(lst1)
    
    # 逐级进行减法(lst1的位数高于lst2时)
    while len(lst1) > len(lst2):  
        k = 0 
        divisor_tmp = []
        while len(lst1) > len(divisor_tmp) + len(lst2): # 对减数升位以简化运算
            divisor_tmp.append(0)
        if cmp(lst1, divisor_tmp+lst2) == False:
            del divisor_tmp[-1]
        divisor_tmp += lst2
        while len(lst1) > len(divisor_tmp): # 两数位数不等时的减法
            lst1 , flag_tmp = sub_core(lst1, divisor_tmp, False, len(divisor_tmp))
            if lst1[-1] == 0: lst1.pop(-1) 
            k += 1
        while True: # 在两数位数相等时的减法
            if cmp(lst1, divisor_tmp):
                lst1 , flag_tmp =  sub_core(lst1, divisor_tmp, False, len(lst1))
                k += 1
                continue
            else:
                break
        quotient[len(divisor_tmp) - len(lst2)] = k 
        while len(lst1) > len(lst2): # 消去lst1中多余的0
            if lst1[-1] == 0:
                del lst1[-1]
            else:
                break

    # 最后进行的同级减法
    k = 0
    while True:
        if cmp(lst1, lst2):
            lst1 , flag_tmp =  sub_core(lst1, lst2, False, len(lst1))
            k += 1
            continue
        else:
            break
    quotient[0] = k
    return div_result(quotient, lst1)    


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 乘法函数
def single_mul(lst, a, front_blank, back_blank): # 用于lst1和单个数相乘
    new_lst = [0] * front_blank
    lst_tmp = [x*a for x in lst]
    new_lst += lst_tmp
    new_lst += [0] * back_blank
    return new_lst


def mul_core(lst1, lst2): # lst1和lst2相乘
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

def mul(str1, str2):
    # 初始化
    out_flag = False
    lst1, flag1, lst2, flag2= two_init(str1, str2, out_flag)
    flag = (flag1 != flag2)
    
    # 计算乘积并返回字符串
    return l_to_s(mul_core(lst1, lst2), flag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 加法函数
def add_sub_init(str1, str2, out_flag):
    lst1, flag1, lst2, flag2= two_init(str1, str2, out_flag)
    l_valid = max(len(lst1), len(lst2))
    # 加法判断
    if flag1 == flag2:
        
        if len(lst1) > len(lst2):
            lst2 += [0] * (len(lst1) - len(lst2))
            return add_core(lst1, lst2, flag1, l_valid)
        elif len(lst1) == len(lst2):
            return add_core(lst1, lst2, flag1, l_valid)
        else:
            lst1 += [0] * (len(lst2) - len(lst1))
            return add_core(lst2, lst1, flag2, l_valid) 
    # 减法判断
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
            
        
def add_core(lst1, lst2, flag, l_valid):
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


def add(str1, str2):
    out_flag = False
    lst_ans, flag = add_sub_init(str1, str2, out_flag)
    return l_to_s(lst_ans, flag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 减法函数
def sub_core(lst1, lst2, flag, l_valid):
    # 初始化
    lst_sub = []
    sub_more = 0
    
    # 逐位相减
    for i in range(0, l_valid):
        sub_tmp = 10 + lst1[i] - lst2[i] - sub_more
        sub_digit = sub_tmp % 10
        sub_more = 1 - (sub_tmp // 10)
        lst_sub.append(sub_digit)
        
    # 多余位处理
    if len(lst1) == len(lst2):
        pass
    else:
        lst_sub.append(lst1[l_valid]-sub_more)
        try:
            lst_sub += lst1[l_valid+1:]
        except IndexError:
            pass
    return lst_sub, flag


def sub(str1, str2):
    out_flag = True
    lst_ans, flag = add_sub_init(str1, str2, out_flag)
    return l_to_s(lst_ans, flag)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 转换及初始化函数
def s_to_l(s, out_flag):
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


def l_to_s(lst, flag):
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
    return s


def two_init(str1, str2, out_flag):
    lst1, flag1= s_to_l(str1, False)
    lst2, flag2= s_to_l(str2, out_flag)
    return lst1, flag1, lst2, flag2


s1 = '123456'
s2 = '789'
print(mul(s1, s2))

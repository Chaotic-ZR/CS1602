import time
# 1. Brute Force
def bf_prime(n):
    if n == 2:
        return True

    for i in range(2, n):
        if n % i == 0:
            return False
    return True


def count1(n):
    ans = 0
    for i in range(2, n+1):
        if bf_prime(i):
            ans += 1
    return ans


# 2. Optimize Brute Force
def obf_prime(n):
    if n == 2:
        return True
    
    k = int(n**(1/2)) + 1
    for i in range(2, k):
        if n % i == 0:
            return False
    return True


def count2(n):
    ans = 0
    for i in range(2, n+1):
        if obf_prime(i):
            ans += 1
    return ans


# 3. Optimize Factor
def of_prime(n):
    if n == 2:
        return True
    
    k = int(n**(1/2)) + 1
    for i in range(2, k):
        if n % i == 0:
            return False
    return True


def op_factor(n): # 先删去部分数字
    b = [2, 3]
    c = [x for x in range(5, n+1) if x % 6 == 1 or x % 6 == 5]
    a = b + c
    return a


def count3(n):
    k = op_factor(n)
    a = [x for x in k if obf_prime(x)]
    return len(a)


# 4. Sieve of Eratosthenes
# 筛子函数
def sieve(n, lst):
    k = lst.index(n**2)
    cut_lst = lst[k:]
    
    a = [x for x in cut_lst if x % n != 0]
    
    c = lst[1:k] + a 
    return c


# 主函数
def soe_prime(n):
    # 初始化
    ans = [2]
    a = [x for x in range(3, n+1) if x % 2 != 0]
    tmp = 0
    
    # 筛选过程
    while True:
        tmp = a[0]
        if tmp * tmp > n:
            break
        ans.append(tmp)
        a = sieve(tmp, a)
        
    # 合并结果
    ans += a
    return ans


def count4(n):
    return len(soe_prime(n))


# 5. Miller-Rabin
def mr_test(a, n):
    # 确认a, n互质
    if n % a == 0:
        return False
    
    # 计算t与u
    tmp = n - 1
    t = 0
    while tmp % 2 == 0:
        tmp //= 2
        t += 1
    u = tmp
    
    # 判断
    y = pow(a, u, n)
    # y = (a**u) % n
    if y == 1 or y == n - 1: # 特殊情况：k=1
        return True
    k = 2
    while k <= t+1: # k>=2时的条件二判断
        y = (y * y) % n
        if y == 1:
            return False
        if y == n-1:
            return True
        k += 1
    return False


def mr_prime(n):
    if n == 2 or n == 3:
        return True
    
    # 选a进行判断
    lst = [2, 3]
    for a in lst:
        if mr_test(a, n) == False:
            return False
    return True


def count5(n):
    ans = 0
    for i in range(2, n+1):
        if mr_prime(i):
            ans += 1
    return ans


# 计时器
def  find_prime(n):
    time_start = time.time()
    number_pn = 0

    
        
    #Write down Your code Here
    number_pn = count1(n)
    # number_pn = count2(n)
    # number_pn = count3(n)
    # number_pn = count4(n)
    # number_pn = count5(n)

    time_end = time.time()    
    return time_end - time_start, number_pn


t, ans = find_prime(1000000)
print(f"time is {t}s, number is {ans}")
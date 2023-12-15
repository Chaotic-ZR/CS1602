# 1
# 递归主体
# hanoi(阶数, 起点, 中间点, 终点)
def hanoi(n, x, y, z):
    if n == 1: 
        print(f"{x}->{y}")
        print(f"{y}->{z}")
        return
        
    # n至n-1阶转化
    hanoi(n-1, x, y, z) # 将n-1阶全部移动至z
    print(f"{x}->{y}") # 将最下面一个移动至y
    
    hanoi(n-1, z, y, x) # 将n-1阶全部移回x
    print(f"{y}->{z}") # 将最下面一个移动至z
    
    # n-1阶hanoi
    hanoi(n-1, x, y, z)


n = 3
x = 'A'
y = 'B'
z = 'C'

print(f"hanoi_plus({n}, {x}, {y}, {z})")
hanoi(n, x, y, z)




# 2 the josephus problem
# 2.1
def select(n):
    a = [i for i in range(1, n+1)]
    flag = 0 # 是否移除的标志
    
    # 反复对列表a进行移除, 直到剩下最后一个
    while len(a) > 1:
        aa = a.copy()
        for x in aa:
            # 间隔一个移除一个
            if flag:
                a.remove(x)
                flag = 0
            else:
                flag = 1
                
    return a[0]


n = 4
print(select(n))


# 2.2
def t(n):
    if n == 1: return 1 # 基本状态
    
    # 降阶
    if n % 2 == 0:
        return 2 * t(n/2) - 1
    else:
        return 2 * t(n//2) + 1
    

print(t(5))


# 2.3
import math
def t(n):
    m = int(math.log(n, 2))
    l = n - 2**m
    ans = 2 * l + 1
    return ans


print(t(16))




# 3
# 填格子
def w1(sr, sc, l): # 填1号格子
    sr = sr + l - 1
    sc = sc + l - 1
    a[sr-1][sc-1] = 1
    a[sr][sc-1] = 1
    a[sr-1][sc] = 1
    
    
def w2(sr, sc, l): # 填2号格子
    sr = sr + l - 1
    sc = sc + l - 1
    a[sr-1][sc-1] = 2
    a[sr-1][sc] = 2
    a[sr][sc] = 2
    
    
def w3(sr, sc, l): # 填3号格子
    sr = sr + l - 1
    sc = sc + l - 1
    a[sr-1][sc-1] = 3
    a[sr][sc-1] = 3
    a[sr][sc] = 3
    
    
def w4(sr, sc, l): # 填4号格子
    sr = sr + l - 1
    sc = sc + l - 1
    a[sr][sc] = 4
    a[sr][sc-1] = 4
    a[sr-1][sc] = 4


# 递归主体
def cover(sr, sc, i, j, l): 
    # 2*2时填格子
    if l == 2:
        # for r in a:
        #     print(r)
        # print()
        l = l // 2
        if a[sr-1][sc-1] != -1:
            w4(sr, sc, l)
        elif a[sr-1][sc] != -1:
            w3(sr, sc, l)
        elif a[sr][sc-1] != -1:
            w2(sr, sc, l)
        else:
            w1(sr, sc, l)
        return
    # 大于2*2时分割格子
    l = l // 2
    # 1.黑色在左上
    if i < sr + l and j < sc + l:
        cover(sr, sc, i, j, l)
        w4(sr, sc, l)
        # 其他三个区域的覆盖
        cover(sr+l, sc, sr+l, sc+l-1, l)
        cover(sr, sc+l, sr+l-1, sc+l, l)
        cover(sr+l, sc+l, sr+l, sc+l, l)
    # 2.黑色在左下
    if i >= sr + l and j < sc + l:
        cover(sr+l, sc, i, j, l)
        w2(sr, sc, l)
        # 其他三个区域的覆盖
        cover(sr, sc, sr+l-1, sc+l-1, l)
        cover(sr+l, sc+l, sr+l, sc+l, l)
        cover(sr, sc+l, sr+l-1, sc+l, l)
    # 3.黑色在右上
    if i < sr + l and j >= sc + l:
        cover(sr, sc+l, i, j, l)
        w3(sr, sc, l)
        # 其他三个区域的覆盖
        cover(sr, sc, sr+l-1, sc+l-1, l)
        cover(sr+l, sc, sr+l, sc+l-1, l)
        cover(sr+l, sc+l, sr+l, sc+l, l)
    # 4.黑色在右下
    if i >= sr + l and j >= sc + l:
        cover(sr+l, sc+l, i, j, l)
        w1(sr, sc, l)
        # 其他三个区域的覆盖
        cover(sr+l, sc, sr+l, sc+l-1, l)
        cover(sr, sc+l, sr+l-1, sc+l, l)
        cover(sr, sc, sr+l-1, sc+l-1, l)
        
    return
        
        
def GridCover(k, i, j):
    # 初始化棋盘
    global a
    l = 2**k
    a = [[-1 for c in range(l)] for r in range(l)]
    a[i-1][j-1] = 0
    
    # 进入递归
    cover(1, 1, i, j, l)
    
    # 打印
    for r in a:
        print(r) 
        
        
k = 2
i = 1
j = 4

GridCover(k, i, j)



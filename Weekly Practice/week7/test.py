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
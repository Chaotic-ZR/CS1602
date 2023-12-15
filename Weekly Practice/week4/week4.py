# 1
def is_triangle(a, b, c):
    if a + b > c and a + c > b and b + c > a:
        return True
    else:
        return False


print(f"{is_triangle(1, 2, 2)}")



# 2
def delta(a, b, c):
    if a == 0: 
        if b == 0 and c != 0:
            return 0, 0, 0
        elif b == 0 and c == 0:
            return -1, 0, 0
        elif b != 0:
            x = c / b
            return 1, x, 0
    else: 
        d = b**2 - 4 * a * c 
        if d < 0:
            return 0, 0, 0
        elif d == 0:
            x = (-b) / (2 * a)
            return 1, x, 0
        else:
            x1 = (-b + d**(1/2)) / (2 * a)
            x2 = (-b - d**(1/2)) / (2 * a)
            return 2, x1, x2


a = int(input("a: "))
b = int(input("b: "))
c = int(input("c: "))

num, x1, x2 = delta(a, b, c)

if num == -1:
    print("有无穷多个解")
elif num == 0:
    print("无解")
elif num == 1:
    print(f"有1个解, x={x1}")
elif num == 2:
    print(f"有2个解, x1={x1}, x2={x2}")


# 3
def is_prime(n):
    if n == 1:
        print(f"1既不是质数也不是合数")
        return
    m = int(n**(1/2)) + 1
    for x in range(2, m):
        if n % x == 0:
            print(f"{n}不是质数")
            break
    else:
        print(f"{n}是质数")
    return
        
    

n = int(input("n: "))
is_prime(n)



# 4
def is_palindrome_year(year):
    if year[0] == year[3] and year[1] == year[2]:
        return True
    else:
        return False

year = input("year: ")

if is_palindrome_year(year) == True:
    print(f"{year}是回文年")
else:
    print(f"{year}不是回文年")
        


# 5
import math
def f(x):
    if x < -2:
        return x**4
    elif x > 2:
        return math.exp(x)
    elif x != 2:
        return math.sin(x)
    
x = int(input("x: "))

print(f"f(x)={f(x):.6f}")

# 6
def rev(a):
    l = len(a)
    b = ""
    for i in range(l):
        b += a[l - i -1]
    return b


def main():
    while True:
        x = input("x: ")
        x1 = int(x)
        if x1 == 0:
            break
        elif x1 == -1:
            continue
        else:
            print(f"{rev(x)}")


main()


# 7
def judge(x):
    l = len(x) - 1
    x = int(x)
    for i in range(l, 0, -1):
        t = 10**i
        if x % t == 0:
            return i
    return 0


def main():
    x = input("x: ")
    print(f"{judge(x)}")


main()



# 8
def check(a, b):
    c = str(bin(a ^ b))
    num = 0
    for letter in c:
        if letter == "1":
            num +=1
    return num


def main():
    # input
    a = int(input("a: "))
    b = int(input("b: "))
    
    print(f"有{check(a, b)}位不同")
        
        
main()





# 9
def check(n):
    while n != 1:
        if n % 2 == 0:
            n /= 2
        else:
            n = 3 * n + 1
    return True


def main():
    for n in range(1, 1000001):
        if check(n) == False:
            print("不成立")
            break
    else:
        print("成立")
    
        
main()




# 10
# x=2+3n=3+5m=2+7k
# x=2+21n=3+5m
# x=23+21n=23+5m
# x=23+105n

x = 23
while x <= 100000:
    print(f"{x}")
    x += 105
    
    


# 11
def fractal(n):
    if n == 1:
        return 1
    else:
        return n * fractal(n - 1)
    
    

def choose(n, k):
    if k == 0 or n == k or n == 0:
        return 1
    else:
        return int(fractal(n) / (fractal(k) * fractal(n - k)))

def out(n):
    for i in range(n + 1):
        for j in range(i+1):
            print(choose(i, j), end=" ")
        print()
    
def main():
    n = int(input("n: "))
    out(n)
    
main()

# # 1
# def list_sum(lst):
#     sum1 = 0
#     multi = 1
#     for i in lst:
#         sum1 += i
#         multi *= i
#     return sum1, multi

# a = [1, 2, 3, 4]
# print(list_sum(a))





# # 2
# def compare_list(lst1, lst2):
#     flag1 = [1 for i in lst1 if i not in lst2]
#     flag2 = [1 for i in lst2 if i not in lst1]
#     if flag1 and flag2:
#         return False
#     elif flag2:
#         return True, "list2"
#     else:
#         return True, "list1"
    

# a = [1, 2, 3]
# b = [1, 2, 3]
# print(compare_list(a, b))




# # 3
# def find():
#     a = []
#     for n in range(1000, 3001):
#         lst = list(str(n))
#         for c in lst:
#             if int(c) % 2 != 0:
#                 break
#         else:
#             a.append(n)
#     return a


# a = find()
# l = len(a)
# for i in range(l - 1):
#     print(a[i], end=",")
# print(a[l-1])




# # 4
# def rm_d(lst):
#     new = []
#     [new.append(x) for x in lst if x not in new]
#     return new    


# a = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
# print(rm_d(a))





# # 5
# def pos_int(n):
#     lst = list(str(n))
#     lst.reverse()
#     return lst


# def neg_int(n):
#     lst = list(str(n))
#     lst = [lst[0]] + lst[:0:-1]
#     return lst

# n = 12345
# n_ = pos_int(n)
# for x in n_:
#     print(x, end="")
# print()

# k = -12345
# k_ = neg_int(k)
# for x in k_:
#     print(x, end="")
# print()





# # 6
# import re
# def inter(a, b):
#     lst1 = re.split(',|\[|\]|\s', a)
#     lst2 = re.split(',|\[|\]|\s', b)
#     sect = [int(x) for x in lst1 if x in lst2 and x != ""]
#     if sect == []:
#         print("no intersection")
#         return False
#     return sect

# a = input("a: ")
# b = input("b: ")


# print(inter(a, b))





# # 7
# def pat(n):
#     for i in range(n):
#         for j in range(i+1):
#             print("*", end=" ")
#         print()
#     for i in range(n-2, -1, -1):
#         for j in range(i+1):
#             print("*", end=" ")
#         print()


# n = int(input("n: "))
# pat(n)





# # 8
# def cnt(n):
#     # 1: count every digit
#     # n = str(n)
#     l = len(n)
#     a = [0 for i in range(10)]
#     for i in range(l):
#         index = int(n[i])
#         a[index] += 1
#     # 2: return a list containing the the number of 0-9 
#     return a


# n = input("n: ")
# print(cnt(n))




# # 9
# def sum_of_two(n):
#     total = 0
#     for i in range(1, n+1):
#         x = n - i
#         total += i * 10**x
#     total *= 2
#     return total

# n = int(input("n: "))
# print(sum_of_two(n))





# # 10
# def create():
#     a = [[x for x in range(1, 11)] for y in range(1, 11)]
#     b = [[y for i in range(1, 11)] for y in range(1, 11)]
#     return a, b


# a, b = create()
# print(a)
# print(b)





# # 11
# def find_min(a):
#     m = max(a)
#     index = a.index(m)
#     return index


# a = [1, 2, 3, 4, 5, 5, 4, 2, 0]
# print(find_min(a))




# # 12
# def is_prime(x):
#     n = int(x**(1/2))
#     for i in range(2, n + 1):
#         if x % i == 0:
#             return False
#     return True


# def add():
#     a = [x for x in range(2, 1000000) if is_prime(x)]
#     return a


# print(add())

# 12
def judge(n):
    a = [1 for i in range(n+1)]
    a[0] = 0
    a[1] = 0
    for i in range(2, n+1):
        if a[i]:
            a[i+i: :i] = [0] * (n // i - 1)
    b = [i for i, value in enumerate(a) if value]
    return b

print(judge(100))
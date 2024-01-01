# def f(a):
#     return lambda x: x + a


# a = 1
# g = f(a)
# h = lambda x: x + a
# a = 2
# print(g(0), h(0))


# def mypow():
#     plist = []
#     for i in range(3):
#         plist.append(lambda x: x**i)
#     return plist

# for i in range(3):
#     print(mypow()[i](2), end="")


import minimatrix as mm

m1 = mm.Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
m2 = m1.T()
m = mm.Matrix([[10]])
m2[0:1, 0:1] = m
print(m1)
print(m2)

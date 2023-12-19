import random
import minimatrix as mm
import time

matrix1 = []
for _ in range(50):
    row = []
    for _ in range(50):
        row.append(random.randint(0, 100))
    matrix1.append(row)

m1 = mm.Matrix(data=matrix1)

matrix2 = []
for _ in range(100):
    row = []
    for _ in range(100):
        row.append(random.randint(0, 100))
    matrix2.append(row)

m2 = mm.Matrix(matrix2)

start1 = time.time()
m11 = m1.inverse()
end1 = time.time()

start2 = time.time()
m22 = m2.inverse()
end2 = time.time()

t1 = end1 - start1
t2 = end2 - start2
print(t2/t1)
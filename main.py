import numpy as np

a = np.arange(10)
b = a[2:7:2]

# these are your vector slicing operations
# access x consecutive elements (start:stop:step)

print(a)
print(b)

c = a**2
print(c)

double = np.array([[1],
                  [2],
                  [3]])

print(double)
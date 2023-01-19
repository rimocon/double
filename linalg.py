import numpy as np

a = np.array([[5, -1],
              [2,  3]])
b = np.array([9, 7])
b = b.reshape(2,1)
print("a",a)
print("a shape",a.shape)
print("b",b)
print("b shape",b.shape)
x = np.linalg.solve(a,b)
print("x",x)
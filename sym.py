import sympy as sp
import numpy as np

# x,y,z = sp.symbols('x, y, z')

# def f(x,y,z):
#     return x**2 + 2*y + z

# print(f(x,y,z))

# a = sp.Symbol('a')
# x = a
# print("a ver = ",f(x,y,z))

sym_p = sp.MatrixSymbol('p',2,1)
sym_z = sp.MatrixSymbol('z',2,1)
def y(z,p):
    ret = sp.Matrix([
        z[0]*z[0] + 2*p[0] + z[1]*z[1],
        2*z[1] + p[1]*p[1]
    ])
    return ret
# print("test",y(z,p))
# # p = z[2:4,0]
# print("pshape",p.shape)
# p[0:1,0] = z[2:3,0]
# print("z_ver",y(z,p))
F = y(sym_z,sym_p)
J = F.jacobian(sym_z)
print(F)
print(J)
z = np.array([1,2,3,4]).reshape(4,1)
print(z)
# s = sp.Matrix([1,2,3,4]).reshape(4,1)
s = sp.Matrix(z)
# J_sub = J.subs([sym_z,z])
J_sub = J.subs(sym_z,s)
print(J_sub)

Jac = np.array([[1,0,0],[0,0,1]])
print("JAC",Jac)
# a = np.array([[1,2,3,4],
#             [5,6,7,8],
#             [9,10,11,12],
#             [13,14,15,16]])
# print("a= ",a)
# print("a 1retume",a[0:4,0])
# print("a 2retume",a[0:4,1])
# sp_a = sp.Matrix([a[0:4,0]])
# print("sp_a",sp_a)
# jac_a = sp_a.jacobian(z)
# print(jac_a)
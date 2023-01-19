from re import A
import numpy as np
import sympy as sp
from sympy import sin, cos
import matplotlib.pyplot as plt

def map(x, p, c):

    # double pendulum  equation
    # x:variable, p:parameters, c:constant

    M1 = c[0]
    M2 = c[1]
    L1 = c[2]
    L2 = c[3]
    G = c[4]

    a11 = M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12 = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21 = a12
    a22 = M2 * L2 * L2
    b1 = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2 = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    delta = a11 * a22 - a12 * a21

    ret = sp.Matrix([x[1],
                    (b1 * a22 - b2 * a12) / delta,
                    x[3],
                    (b2 * a11 - b1 * a21) / delta
                    ])
    return ret


class DynamicalSystem:

    def __init__(self, json):
        self.x0 = sp.Matrix(json['x0'])
        self.params = sp.Matrix(json['params'])
        self.const = sp.Matrix([json['const']])
        self.xdim = json['xdim']
        self.delta = json['delta']

        self.x_alpha = np.array([0,0,0,0])
        self.x_omega = np.array([0,0,0,0])
        self.mu_alpha = 0
        self.mu_omega = 0
        self.sym_x = sp.MatrixSymbol('x', self.xdim, 1)
        self.sym_p = sp.MatrixSymbol('p', sp.shape(self.params)[0],1)
        
        self.F = map(self.sym_x, self.sym_p, self.const)
        self.dFdx = self.F.jacobian(self.sym_x)
        self.dFdlambda = self.F.jacobian(self.sym_p)
        
        # for newton condition
        self.sym_z = sp.MatrixSymbol('z', 16, 1)
        self.sym_xx = sp.MatrixSymbol('x', self.xdim, 1)
        self.sym_pp = sp.MatrixSymbol('p', sp.shape(self.params)[0],1)
        self.sym_xx = self.sym_z[0:4,0]
        self.sym_pp = self.sym_z[12:16,0]
        self.F_n = map(self.sym_xx, self.sym_pp, self.const)
        self.dFdx_n = self.F_n.jacobian(self.sym_xx)
        self.var = json['var']

        #self.sym_p = self.sym_z[8:12,0]
        #for solver
        self.duration = [0,json['tau']]
        self.duration_m = [-json['tau'],0]
        self.tick = 0.01
        self.vari_ini = np.eye(4).reshape(16,)
        self.vari_param_ini = np.zeros(4)
        self.vari_ini = np.concatenate([self.vari_ini,self.vari_param_ini])

        # for newton method
        self.ite_max = 100
        self.eps = 1e-8
        self.explode = 100


        
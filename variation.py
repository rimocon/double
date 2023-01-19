import sys
import json
import dynamical_system
import ds_func
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
# from sympy import sin,cos
from numpy import sin,cos
from scipy import linalg


if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)

fd = open(sys.argv[1], 'r')
json_data = json.load(fd)
fd.close()

# input data to constructor
ds = dynamical_system.DynamicalSystem(json_data)

def main():
    solve()

def solve():
    ####solve####
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    # calculate equilibrium points
    eq = ds_func.equilibrium(ds)
    # convert to numpy
    ds.eq = ds_func.sp2np(eq)

    ds.state0 = ds.eq[0,:].flatten()
    vari_ini = np.eye(4).reshape(16,)
    ds.state0 = np.concatenate([ds.state0,vari_ini])
    print("initial value",ds.state0)
    state = solve_ivp(func, ds.duration, ds.state0,
        method='RK45', args = (p, c), t_eval = ds.t_eval,
        rtol=1e-12)
    print(state.t)
    print(state.y)




def func(t, x, p, c):
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

    ret = np.array([
        x[1],
        (b1 * a22 - b2 * a12) / delta,
        x[3],
        (b2 * a11 - b1 * a21) / delta
    ])
    # phi = sp.Matrix([[ret[0]],
    #                 [ret[1]],
    #                 [ret[2]],
    #                 [ret[3]]])
    # sp_p = sp.Matrix([[p[0]],
    #                   [p[1]],
    #                   [p[2]],
    #                   [p[3]]])
    ## 初期値に関する変分方程式((phi_0 ~ phi_4) * (x00~x04)で16個)
    #ここもパラメタ代入できるように
    # print(ds.dFdx)

    # dFdx = ds_func.dFdx(phi,sp_p,ds)
    dFdx = np.array([[0, 1, 0, 0],
    # 2行目
    [(9.80665*(-1.0*cos(x[2]) - 1.0)*sin(x[0] + x[2]) + 9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0) ,
    (-2.0*(-1.0*cos(x[2]) - 1.0)*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3] - 1.0*p[0])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    0.111111111111111*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))*(-(1.0*cos(x[2]) + 1.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]) + 2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - 1.0*p[0]*x[1] + 1.0*p[2])/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(-1.0*cos(x[2]) - 1.0) + 1.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (-(-1.0*cos(x[2]) - 1.0)*p[1] + 2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
    [0, 0, 0, 1],
    # ４行目
    [((9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))*(-1.0*cos(x[2]) - 1.0) + 9.80665*(2.0*cos(x[2]) + 3.0)*sin(x[0] + x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    ((2.0*sin(x[2])*x[3] - p[0])*(-1.0*cos(x[2]) - 1.0) - 2.0*(2.0*cos(x[2]) + 3.0)*sin(x[2])*x[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    0.111111111111111*(-(1.0*cos(x[2]) + 1.0)*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2]) + (2.0*cos(x[2]) + 3.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]))*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(2.0*cos(x[2]) + 3.0) + (-1.0*cos(x[2]) - 1.0)*(9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2) - 2.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 1.0*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2])*sin(x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    ((2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])*(-1.0*cos(x[2]) - 1.0) - (2.0*cos(x[2]) + 3.0)*p[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
    ])

    print("dFdx",dFdx)
    dphidx = np.array(x[4:24])
    ## dFdx @ dphidx0 (要はtheta_10に関するやつ)
    i_0 = (dFdx @ dphidx.reshape(20,1)[0:4]).reshape(4,)
    ## dFdx @ dphidx1 (omega_10に関するやつ)
    i_1 = (dFdx @ dphidx.reshape(20,1)[4:8]).reshape(4,)
    ## dFdx @ dphidx2
    i_2 = (dFdx @ dphidx.reshape(20,1)[8:12]).reshape(4,)
    ## dFdx @ dphidx3
    i_3 = (dFdx @ dphidx.reshape(20,1)[12:16]).reshape(4,)

    # ## パラメタに関する変分方程式
    # dFdlambda = ds.dFdlambda.subs([(ds.sym_x, phi),(ds.sym_p, sp_p)])
    # dFdlambda = ds_func.sp2np(dFdlambda)
    # print("dfdlabda",dFdlambda)

    dFdl = np.array([[0, 0, 0, 0],
    # 2行目
    [-1.0*x[1]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    -(-1.0*cos(x[2]) - 1.0)*x[3]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    1.0/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (-1.0*cos(x[2]) - 1.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
    [0, 0 ,0, 0],
    # 4行目
    [-(-1.0*cos(x[2]) - 1.0)*x[1]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    -(2.0*cos(x[2]) + 3.0)*x[3]/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
    (-1.0*cos(x[2]) - 1.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
    (2.0*cos(x[2]) + 3.0)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
    ])
    print("dfdl",dFdl)

    ##この4本から3本だけでいい
    p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdl[0:4,ds.var].reshape(4,1)).reshape(4,)
    # lambda0に関する変分方程式(要はdphidk1 k1に関するやつ)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,0].reshape(4,1)).reshape(4,)
    # # lambda1に関する変分方程式(dphidk2,k2に関するやつ)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,1].reshape(4,1)).reshape(4,)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,2].reshape(4,1)).reshape(4,)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,3].reshape(4,1)).reshape(4,)
    ## 元の微分方程式+変分方程式に結合
    ret = np.concatenate([ret,i_0,i_1,i_2,i_3,p])
    # print("ret",ret)
    return ret 


if __name__ == '__main__':
    main()
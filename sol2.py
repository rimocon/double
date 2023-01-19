import sys
import json
import dynamical_system
import dp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from sympy import sin,cos
import ds_func


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



def func(t, x, p, c, ds):
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
    phi = sp.Matrix([[ret[0]],
                    [ret[1]],
                    [ret[2]],
                    [ret[3]]])
    sp_p = sp.Matrix([[p[0]],
                      [p[1]],
                      [p[2]],
                      [p[3]]])
    ## 初期値に関する変分方程式((phi_0 ~ phi_4) * (x00~x04)で16個)
    #ここもパラメタ代入できるように
    print(ds_func.dFdx)
    dFdx = ds_func.dFdx(phi,sp_p,ds)
    # print("dFdx",dFdx)
    dphidx = np.array(x[4:24])
    ## dFdx @ dphidx0 (要はtheta_10に関するやつ)
    i_0 = (dFdx @ dphidx.reshape(20,1)[0:4]).reshape(4,)
    ## dFdx @ dphidx1 (omega_10に関するやつ)
    i_1 = (dFdx @ dphidx.reshape(20,1)[4:8]).reshape(4,)
    ## dFdx @ dphidx2
    i_2 = (dFdx @ dphidx.reshape(20,1)[8:12]).reshape(4,)
    ## dFdx @ dphidx3
    i_3 = (dFdx @ dphidx.reshape(20,1)[12:16]).reshape(4,)

    ############initialだけ確認したい時用#####
    # dphidx = np.array(x[4:20])
    # ## dFdx @ dphidx0 (要はtheta_10に関するやつ)
    # i_0 = (dFdx @ dphidx.reshape(16,1)[0:4]).reshape(4,)
    # ## dFdx @ dphidx1 (omega_10に関するやつ)
    # i_1 = (dFdx @ dphidx.reshape(16,1)[4:8]).reshape(4,)
    # ## dFdx @ dphidx2
    # i_2 = (dFdx @ dphidx.reshape(16,1)[8:12]).reshape(4,)
    # ## dFdx @ dphidx3
    # i_3 = (dFdx @ dphidx.reshape(16,1)[12:16]).reshape(4,)


    # ## パラメタに関する変分方程式

    dFdlambda = ds.dFdlambda.subs([(ds.sym_x, phi),(ds.sym_p, sp_p)])
    dFdlambda = ds_func.sp2np(dFdlambda)
    # print("dfdlabda",dFdlambda)

    ##この4本から3本だけでいい
    p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,ds.var].reshape(4,1)).reshape(4,)
    # lambda0に関する変分方程式(要はdphidk1 k1に関するやつ)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,0].reshape(4,1)).reshape(4,)
    # # lambda1に関する変分方程式(dphidk2,k2に関するやつ)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,1].reshape(4,1)).reshape(4,)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,2].reshape(4,1)).reshape(4,)
    # p = (dFdx @ dphidx.reshape(20,1)[16:20] + dFdlambda[0:4,3].reshape(4,1)).reshape(4,)
    ## 元の微分方程式+変分方程式に結合
    ret = np.block([ret,i_0,i_1,i_2,i_3,p])
    # print("ret",ret)
    return ret 


if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)

fd = open(sys.argv[1], 'r')
json_data = json.load(fd)
fd.close()

# input data to constructor
ds = dynamical_system.DynamicalSystem(json_data)

sym_x = sp.MatrixSymbol('x', 4, 1)
sym_p = sp.MatrixSymbol('p', 4,1)
sym_z = sp.MatrixSymbol('z', 12, 1)
sym_pp = sp.MatrixSymbol('p', 4,1)
sym_pp = sym_z[8:12,0]
const = sp.Matrix([json_data['const']])


F = map(sym_x, sym_p, const)
dFdx = F.jacobian(sym_x)
dFdlambda = F.jacobian(sym_p)

# print("dFdx",dFdx)
print("dFdlambda",dFdlambda)

ds.p = ds_func.sp2np(ds.params).flatten()
# import numpy constant
ds.c = ds_func.sp2np(ds.const).flatten()
# convert initial value to numpy
# ds.state0 = ds_func.sp2np(ds.x0).flatten()
ds.x0 = dp.Eq_check(ds.params,ds)
dp.Eigen(ds.x0,ds.params,ds)
vari_ini = np.eye(4).reshape(16,)
vari_param_ini = np.zeros(4)
print("initial param",vari_param_ini)
vari_ini = np.concatenate([vari_ini,vari_param_ini])
print("initial vari",vari_ini)
ds.x0_p = np.concatenate([ds.x_alpha,vari_ini])
ds.x0_m = np.concatenate([ds.x_omega,vari_ini])
print("initial value",ds.x0_p)

ds.state_p = solve_ivp(func, ds.duration, ds.x0_p,
        method='RK45', args = (ds.p, ds.c,ds),
        rtol=1e-12)
ds.state_m = solve_ivp(func, ds.duration_m, ds.x0_m,
        method='RK45', args = (ds.p, ds.c,ds), 
        rtol=1e-12)
print("y_p=",ds.state_p.y)
print("y_m=",ds.state_m.y)
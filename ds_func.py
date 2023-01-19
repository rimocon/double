from re import A
import numpy as np
import sympy as sp
from numpy import sin,cos
from scipy import linalg


def set_x0(p,c):
    t1 = sp.acos((p[2] - p[3]) / ((c[0] + c[1]) * c[2] * c[4]))
    t2 = sp.acos(p[3] / (c[1] * c[3] * c[4])) 

    eqpoints = sp.Matrix([
        [t1, 0, t2 - t1, 0],
        [-t1, 0, t2 + t1, 0],
        [t1, 0, -t2 - t1, 0],
        [-t1, 0, -t2 + t1, 0]
    ])
    return eqpoints

def eigen(x0, p, ds):

    # subs eqpoints for jacobian(df/dx(x0)) 
    # jac = ds.dFdx.subs([(ds.sym_x, x0), (ds.sym_p, ds.params)])
    # for all eqpoints
    p = sp.Matrix(p)
    # print(x0)
    jac = ds.dFdx.subs([(ds.sym_x, x0), (ds.sym_p, p)])
    # convert to numpy
    np_jac = sp2np(jac)
    # print("#############jac",np_jac)
    # calculate eigen values,eigen vectors
    eig_vals,eig_vl, eig_vr = linalg.eig(np_jac, left=True,right=True)
    
    return eig_vals,eig_vl,eig_vr
    
def sp2np(x):
    return sp.matrix2numpy(x, dtype = np.float64)


def dFdx(x, p, ds):
    dfdx = ds.dFdx.subs([(ds.sym_x, x), (ds.sym_p, p)])
    return sp2np(dfdx)


def newton_F(z, ds):

    ##ここちょっと分かってない
    # print("##########dFdx_n",ds.dFdx_n)
    x0 = sp.Matrix([z[0],z[1],z[2],z[3]])
    dfdx = ds.dFdx_n.subs([(ds.sym_x, ds.sp_x0)])
    # dfdx = ds.dFdx_n
    # print("dfdx",dfdx)
    I = sp.eye(4)
    x_alpha = sp.Matrix([z[4],z[5],z[6],z[7]])
    x_omega = sp.Matrix([z[8],z[9],z[10],z[11]])
    homo = ds.state_p.y[0:4,-1] - ds.state_m.y[0:4,-1]
    ret = sp.Matrix([
        # 固有ベクトル上にx_alphaが存在する条件(dfdx(x0,lambda) - mu_alpha I)@(x_alpha - x0)
        ((dfdx - ds.mu_alpha * I) @ (x_alpha - x0))[0:4,0],
        # deltaだけ離れた点である条件
        ((x_alpha - x0).T @ (x_alpha - x0))[0,0] - ds.delta * ds.delta,
        # x_omegaも同様
        ((dfdx - ds.mu_omega * I) @ (x_omega- x0))[0:4,0],
        ((x_omega- x0).T @ (x_omega- x0))[0,0] - ds.delta * ds.delta,
        # 解が一致する条件
        [homo[0] - 2*np.pi],
        [homo[1]],
        [homo[2]],
        [homo[3]]
    ])
    # print("F",ret)
    print("F shape ",ret.shape)

    return ret
# def F(vp,x0,p,ds):

#     x = x0
#     dFdx = np.array([[0, 1, 0, 0],
#     # 2行目
#     [(9.80665*(-1.0*cos(x[2]) - 1.0)*sin(x[0] + x[2]) + 9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0) ,
#     (-2.0*(-1.0*cos(x[2]) - 1.0)*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3] - 1.0*p[0])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
#     0.111111111111111*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))*(-(1.0*cos(x[2]) + 1.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]) + 2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - 1.0*p[0]*x[1] + 1.0*p[2])/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(-1.0*cos(x[2]) - 1.0) + 1.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2)/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0),
#     (-(-1.0*cos(x[2]) - 1.0)*p[1] + 2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)],
#     [0, 0, 0, 1],
#     # ４行目
#     [((9.80665*sin(x[0] + x[2]) + 19.6133*sin(x[0]))*(-1.0*cos(x[2]) - 1.0) + 9.80665*(2.0*cos(x[2]) + 3.0)*sin(x[0] + x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
#     ((2.0*sin(x[2])*x[3] - p[0])*(-1.0*cos(x[2]) - 1.0) - 2.0*(2.0*cos(x[2]) + 3.0)*sin(x[2])*x[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
#     0.111111111111111*(-(1.0*cos(x[2]) + 1.0)*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2]) + (2.0*cos(x[2]) + 3.0)*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3]))*(-2.0*(1.0*cos(x[2]) + 1.0)*sin(x[2]) + 2.0*sin(x[2]))/(-0.333333333333333*(1.0*cos(x[2]) + 1.0)**2 + 0.666666666666667*cos(x[2]) + 1)**2 + ((9.80665*sin(x[0] + x[2]) - 1.0*cos(x[2])*x[1]**2)*(2.0*cos(x[2]) + 3.0) + (-1.0*cos(x[2]) - 1.0)*(9.80665*sin(x[0] + x[2]) + 2.0*cos(x[2])*x[1]*x[3] + 1.0*cos(x[2])*x[3]**2) - 2.0*(-1.0*sin(x[2])*x[1]**2 - 9.80665*cos(x[0] + x[2]) - p[1]*x[3] + p[3])*sin(x[2]) + 1.0*(2.0*sin(x[2])*x[1]*x[3] + 1.0*sin(x[2])*x[3]**2 - 9.80665*cos(x[0] + x[2]) - 19.6133*cos(x[0]) - p[0]*x[1] + p[2])*sin(x[2]))/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0), 
#     ((2.0*sin(x[2])*x[1] + 2.0*sin(x[2])*x[3])*(-1.0*cos(x[2]) - 1.0) - (2.0*cos(x[2]) + 3.0)*p[1])/(-(1.0*cos(x[2]) + 1.0)**2 + 2.0*cos(x[2]) + 3.0)]
#     ])
#     print("dFdx",dFdx)
#     x0 = x0.reshape(4,1)
#     I = np.eye(4)
#     x_alpha = np.array(vp[0:4]).reshape(4,1)
#     x_omega = np.array(vp[4:8]).reshape(4,1)
#     # 固有ベクトル条件
#     f1 = ((dFdx - ds.mu_alpha * I) @ (x_alpha - x0)).flatten()
#     # デルタだけ離れた条件
#     f2 = ((x_alpha - x0).T @ (x_alpha - x0))[0,0] - ds.delta * ds.delta
#     f3 = ((dFdx - ds.mu_omega * I) @ (x_omega- x0)).flatten()
#     f4 = ((x_omega- x0).T @ (x_omega- x0))[0,0] - ds.delta * ds.delta
#     # 解が一致する条件
#     phi = ds.state_p.y[0:4,-1] - ds.state_m.y[0:4,-1]
#     ret = np.array([f1[0],
#                     f1[1],
#                     f1[2],
#                     f2,
#                     f3[0],
#                     f3[1],
#                     f3[2],
#                     f4,
#                     phi[0] - 2*np.pi,
#                     phi[1],
#                     # phi[2],
#                     # phi[3]
#                     ]).reshape(10,1)
#     # ret_2 = np.array([
#     # 2.42505416465942*x0[0, 0] + x0[1, 0] - 1.21252708232971*np.pi,
#     # (2.42505416465942 - 1.0*x0[8, 0])*x0[1, 0] + 9.80665*x0[0, 0] - 9.80665*x0[2, 0] + 2.0*x0[3, 0]*x0[9, 0] - 4.903325*np.pi,
#     # (x0[0, 0] - np.pi/2)**2 + x0[1, 0]**2 + x0[2, 0]**2 + x0[3, 0]**2 - 0.0001, 
#     # -2.42636235352655*x0[4, 0] + x0[5, 0] + 1.21318117676327*np.pi,
#     # (-1.0*x0[8, 0] - 2.42636235352655)*x0[5, 0] + 9.80665*x0[4, 0] - 9.80665*x0[6, 0] + 2.0*x0[7, 0]*x0[9, 0] - 4.903325*np.pi,
#     # (x0[4, 0] - np.pi/2)**2 + x0[5, 0]**2 + x0[6, 0]**2 + x0[7, 0]**2 - 0.0001, 
#     # -0.0554300059461896,
#     # -0.0224364443835218,
#     # 0.00252148503919230, 
#     # 0.00123372766257275])
#     return ret


# def J(vp,p,ds):
#     J = np.array([[-ds.mu_alpha, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#            [ds.c[4], -ds.mu_alpha - 1.0*p[0], -ds.c[4], 2.0*p[1], 0, 0, 0, 0, 0, 0],
#            [0, 0, -ds.mu_alpha, 1, 0, 0, 0, 0, 0, 0],
#            [2*vp[0] - np.pi, 2*vp[1], 2*vp[2], 2*vp[3], 0, 0, 0, 0, 0, 0],
#            [0, 0, 0, 0, -ds.mu_omega, 1, 0, 0, 0, 0],
#            [0, 0, 0, 0, ds.c[4], -1.0*p[0] - ds.mu_omega, -ds.c[4], 2.0*p[1], 0, 0],
#            [0, 0, 0, 0, 0, 0, -ds.mu_omega, 1, 0, 0],
#            [0, 0, 0, 0, 2*vp[4] - np.pi, 2*vp[5], 2*vp[6], 2*vp[7], 0, 0],
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
#            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
#     # print("J=",J)
#     # print("Jshape",J.shape)
#     for i in range(0,2):
#         # dphidlambda(plus) - dphidlambda(minus)
#         J[i+8][8] = ds.state_p.y[20+i,-1] - ds.state_m.y[20+i,-1]
#         for j in range(0,4):
#             # dphidxalpha ~ dphidxomega
#             J[i+8][j] = ds.state_p.y[4+i+4*j,-1]
#             J[i+8][j+4] = ds.state_m.y[4+i+4*j,-1]
    
#     M1 = ds.c[0]
#     M2 = ds.c[1]
#     L1 = ds.c[2]
#     L2 = ds.c[3]
#     G = ds.c[4]
#     x = []
#     x[0:4] = ds.state_p.y[0:4,-1]
#     a11p= M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
#     a12p = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
#     a21p = a12p
#     a22p = M2 * L2 * L2
#     b1p = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
#         + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
#         - M2 * L2 * G * cos(x[0] + x[2])
#         - (M1 + M2) * L1 * G * cos(x[0])
#         - p[0] * x[1])
#     b2p = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
#         - M2 * L2 * G * cos(x[0] + x[2]) 
#         - p[1] * x[3])
#     deltap = a11p * a22p - a12p * a21p

#     x[0:4] = ds.state_m.y[0:4,-1]
#     a11m= M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
#     a12m = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
#     a21m = a12m
#     a22m = M2 * L2 * L2
#     b1m = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
#         + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
#         - M2 * L2 * G * cos(x[0] + x[2])
#         - (M1 + M2) * L1 * G * cos(x[0])
#         - p[0] * x[1])
#     b2m = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
#         - M2 * L2 * G * cos(x[0] + x[2]) 
#         - p[1] * x[3])
#     deltam = a11m * a22m - a12m * a21m
#     ### ここプラスかマイナスどっち？
#     # J[6][9] = ds.state_p.y[1,-1] + ds.state_m.y[1,-1]
#     # J[7][9] = (b1p * a22p - b2p * a12p) / deltap + (b1m * a22m - b2m * a12m) / deltam
#     # J[8][9] = ds.state_p.y[3,-1] + ds.state_m.y[3,-1]
#     # J[9][9] = (b2p * a11p - b1p * a21p) / deltap + (b2m * a11m - b1m * a21m) / deltam
#     J[8][9] = ds.state_p.y[1,-1] + ds.state_m.y[1,-1]
#     J[9][9] = ((b1p * a22p - b2p * a12p) / deltap) + ((b1m * a22m - b2m * a12m) / deltam)
#     # print("J",J)
#     return J
from re import A
import sys
import json
import dynamical_system
import ds_func
# import variation
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import sin,cos
from scipy import linalg


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
    # 初期値に関する変分方程式
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

    dphidx = np.array(x[4:20])
    ## dFdx @ dphidx0 (要はtheta_10に関するやつ)
    i_0 = (dFdx @ dphidx.reshape(16,1)[0:4]).reshape(4,)
    ## dFdx @ dphidx1 (omega_10に関するやつ)
    i_1 = (dFdx @ dphidx.reshape(16,1)[4:8]).reshape(4,)
    ## dFdx @ dphidx2
    i_2 = (dFdx @ dphidx.reshape(16,1)[8:12]).reshape(4,)
    ## dFdx @ dphidx3
    i_3 = (dFdx @ dphidx.reshape(16,1)[12:16]).reshape(4,)

    # ## パラメタに関する変分方程式
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

    ##この4本から1本だけでいい
    dphidl = np.array(x[20:24])
    # ここの0:4,3←ここは選択するパラメタによって変える
    p = (dFdx @ dphidl.reshape(4,1) + dFdl[0:4,2].reshape(4,1)).reshape(4,)
    ## 元の微分方程式+変分方程式に結合
    ret = np.concatenate([ret,i_0,i_1,i_2,i_3,p])
    # print("ret",ret)
    return ret 

def main():
    # load data from json file
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} filename")
        sys.exit(0)

    fd = open(sys.argv[1], 'r')
    json_data = json.load(fd)
    fd.close()

    # input data to constructor
    ds = dynamical_system.DynamicalSystem(json_data)
    # import numpy parameter
    ds.p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    ds.c = ds_func.sp2np(ds.const).flatten()

    # ここで平衡点+固有値などセット
    spp = sp.Matrix(ds.p)
    ds.sp_x0 = Eq_check(spp,ds)
    ds.x0 = ds_func.sp2np(ds.sp_x0)
    x0 = ds.x0.flatten()
    Eigen(ds.sp_x0,spp,ds)


    vp = np.block([x0, ds.x_alpha, ds.x_omega, ds.p[ds.var], ds.duration[1]])
    solve(vp,ds)
    # F = ds_func.F(vp, x0, ds.p, ds)
    # J = ds_func.J(vp, ds.p, ds)
    # print("F=",F)
    # print("J=",J)
    ds.sym_F = ds_func.newton_F(ds.sym_z,ds)
    # print("sym_F= ",ds.sym_F)
    ds.sym_J = ds.sym_F.jacobian(ds.sym_z)
    # print("sym_J= ",ds.sym_J)
    print("sym_J= ",ds.sym_J.shape)
    newton_method(vp,ds)
   

def Condition(z,ds):
    F = ds.sym_F.subs(ds.sym_z,z)
    J = ds.sym_J.subs(ds.sym_z,z)
    F = ds_func.sp2np(F)
    J = ds_func.sp2np(J)
    # print("F subs",F)
    # print("J subs",J)
    dFdlambda = J[:,12+ds.var].reshape(14,1)
    dFdtau = np.zeros((14,1))
    J = np.block([[J[:,0:12],dFdlambda,dFdtau]])
    for i in range(0,4):
        # dphidlambda(plus) - dphidlambda(minus)
        J[i+10][12] = ds.state_p.y[20+i,-1] - ds.state_m.y[20+i,-1]
        for j in range(0,4):
            # dphidxalpha ~ dphidxomega
            J[i+10][j+4] = ds.state_p.y[4+i+4*j,-1]
            J[i+10][j+8] = ds.state_m.y[4+i+4*j,-1]
    p = ds.p
    p[ds.var] = z[12+ds.var]


    M1 = ds.c[0]
    M2 = ds.c[1]
    L1 = ds.c[2]
    L2 = ds.c[3]
    G = ds.c[4]
    x = []
    x[0:4] = ds.state_p.y[0:4,-1]
    a11p= M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12p = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21p = a12p
    a22p = M2 * L2 * L2
    b1p = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2p = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    deltap = a11p * a22p - a12p * a21p

    x[0:4] = ds.state_m.y[0:4,-1]
    a11m= M2 * L2 * L2 + (M1 + M2) * L1 * L1 + 2 * M2 * L1 * L2 * cos(x[2])
    a12m = M2 * L2 * L2 + M2 * L1 * L2 * cos(x[2])
    a21m = a12m
    a22m = M2 * L2 * L2
    b1m = (p[2] + M2 * L1 * L2 * sin(x[2]) * x[3] * x[3]
        + 2 * M2 * L1 * L2 * sin(x[2]) * x[1] * x[3]
        - M2 * L2 * G * cos(x[0] + x[2])
        - (M1 + M2) * L1 * G * cos(x[0])
        - p[0] * x[1])
    b2m = (p[3] - M2 * L1 * L2 * sin(x[2]) * x[1] * x[1]
        - M2 * L2 * G * cos(x[0] + x[2]) 
        - p[1] * x[3])
    deltam = a11m * a22m - a12m * a21m
    ### ここプラスかマイナスどっち？
    J[10][13] = ds.state_p.y[1,-1] + ds.state_m.y[1,-1]
    J[11][13] = (b1p * a22p - b2p * a12p) / deltap + (b1m * a22m - b2m * a12m) / deltam
    J[12][13] = ds.state_p.y[3,-1] + ds.state_m.y[3,-1]
    J[13][13] = (b2p * a11p - b1p * a21p) / deltap + (b2m * a11m - b1m * a21m) / deltam
    # J[8][9] = ds.state_p.y[1,-1] + ds.state_m.y[1,-1]
    # J[9][9] = ((b1p * a22p - b2p * a12p) / deltap) + ((b1m * a22m - b2m * a12m) / deltam)
    # print("J[6][9]",J[6][9])
    # print("J[7][9]",J[7][9])
    # print("J[8][9]",J[8][9])
    # print("J[9][9]",J[9][9])
    return F,J


def Eigen(x0, p, ds):
    #パラメータに依存するように
    eig,eig_vl,eig_vr = ds_func.eigen(x0, p, ds)
    print("eigenvalue\n", eig)
    ds.mu_alpha = eig[1].real
    ds.mu_omega = eig[2].real
    eig_vr = eig_vr * (-1)
    print("eigen_vector",*eig_vr[:,].T,sep='\n')
    delta = eig_vr[:,].T * ds.delta
    # print("delta",delta)
    np_x0 = ds_func.sp2np(x0)
    ####ここは８本から２本選択
    ds.x_alpha = (np_x0 + delta[1,:].reshape(4,1)).flatten()
    ds.x_omega = (np_x0 + delta[2,:].reshape(4,1)).flatten()
    print("x_alpha", ds.x_alpha)
    print("x_omega", ds.x_omega)

def Eq_check(p,ds):
    eq = ds_func.set_x0(p,ds.c)
    vp = eq[0,:].T
    for i in range(ds.ite_max):
        F = ds.F.subs([(ds.sym_x, vp), (ds.sym_p, p)])
        J = ds.dFdx.subs([(ds.sym_x, vp), (ds.sym_p, p)])
        F = ds_func.sp2np(F)
        J = ds_func.sp2np(J)
        dif = abs(np.linalg.norm(F))
        # print("dif=",dif)
        if dif < ds.eps:
            # print("success!!!")
            print("solve vp = ",vp)
            return vp
        
        if dif > ds.explode:
            print("Exploded")
            exit()
            # vn = xk+1
            # print("vp=",vp)
        vn = np.linalg.solve(J,-F) + vp
        print("vn=",vn)
        vp = vn
        # if vn[5] > 1.0:
        #   print("B0 is too high")




def solve(vp,ds):
    ####solve####
    # import numpy parameter
    # パラメータ
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    p[ds.var] = vp[12]
    # print("ppppppp",p)
    # print("vp",vp)
    # ds.x00 = np.concatenate([vp[0:4],ds.vari_ini])
    ds.x0_p = np.concatenate([vp[4:8],ds.vari_ini])
    ds.x0_m = np.concatenate([vp[8:12],ds.vari_ini])
    # print("initial value",ds.x0_p)
    # ds.state_x0 = solve_ivp(func, [0,vp[13]], ds.x00,
    #     method='RK45', args = (p, c),
    #     rtol=1e-12,atol=1e-5)
    # for plus
    ds.state_p = solve_ivp(func, [0,vp[13]], ds.x0_p,
        method='RK45', args = (p, c),
        rtol=1e-12,atol=1e-5)
    # print("y_p",ds.state_p.y)
    ## for minus
    ds.state_m = solve_ivp(func, [-vp[13],0], ds.x0_m,
        method='RK45', args = (p, c), 
        rtol=1e-12,atol=1e-5)
    # print("t_m",ds.state_m.t)
    # print("y_m",ds.state_m.y)

# ニュートン法
def newton_method(vp,ds):
    p = ds.p
    for i in range(ds.ite_max):
        print(f"###################iteration:{i}#######################")
        # パラメタだけセット
        p[ds.var] = vp[12]
        # # パラメタによって平衡点は変化するのでセットしなおし
        # eq = ds_func.set_x0(p,ds.c)
        # # ds.x0 = (ds_func.set_x0(p,ds.c)[0,:]).reshape(4,1)
        spp = sp.Matrix(p)
        # ds.sp_x0 = Eq_check(spp,ds)
        sp_x0 = sp.Matrix(vp[0:4])
        Eigen(sp_x0,spp,ds)
        # x0 = ds_func.sp2np(ds.sp_x0).flatten()
        
        # # print("x0",ds.x0)
        # Eigen(eq,p,ds)

        # 微分方程式+変分方程式を初期値vpで解く
        solve(vp,ds)
        z = np.block([vp[0:4],vp[4:8],vp[8:12],p])
        # # print("z=",z)
        z = sp.Matrix(z.reshape(16,1))
        ################################3
        # F = ds_func.F(vp, x0, p, ds)
        # J = ds_func.J(vp, p, ds)
        F,J = Condition(z,ds)
        print("F",F)
        print("J",J)
        dif = abs(np.linalg.norm(F))
        print("diff=",dif)
        if dif < ds.eps:
            print("success!!!")
            print("solve vp = ",vp)
            return vp
        if dif > ds.explode:
            print("Exploded")
            exit()
        test = np.linalg.solve(J,-F)
        print("solve(J,-F) = ",test)
        vn = np.linalg.solve(J,-F).flatten() + vp
        print("vn=",vn)
        vp = vn
        

if __name__ == '__main__':
    main()
import sys
import json
import dynamical_system
import ds_func
import dp
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from sympy import sin,cos
from numpy import sin,cos
# # プロット用設定
# plt.rcParams["font.family"] = "Nimbus Roman"    #全体のフォントを設定
# plt.rcParams['text.usetex'] = True              #描画にTeXを利用
# plt.rcParams['text.latex.preamble'] = r'''\usepackage{amsmath}
#                                           \usepackage{amssymb}
#                                           \usepackage[T1]{fontenc}
#                                           \usepackage{bm}
#                                           '''
# plt.rcParams["figure.autolayout"] = False       #レイアウト自動調整をするかどうか
# plt.rcParams["font.size"] = 22                  #フォントの大きさ
# plt.rcParams["xtick.direction"] = "in"          #x軸の目盛線を内向きへ
# plt.rcParams["ytick.direction"] = "in"          #y軸の目盛線を内向きへ
# # plt.rcParams["xtick.minor.visible"] = True      #x軸補助目盛りの追加
# # plt.rcParams["ytick.minor.visible"] = True      #y軸補助目盛りの追加
# plt.rcParams["xtick.major.width"] = 1.0         #x軸主目盛り線の線幅
# plt.rcParams["ytick.major.width"] = 1.0         #y軸主目盛り線の線幅
# plt.rcParams["xtick.minor.width"] = 0.5         #x軸補助目盛り線の線幅
# plt.rcParams["ytick.minor.width"] = 0.5         #y軸補助目盛り線の線幅
# plt.rcParams["xtick.major.size"] = 20           #x軸主目盛り線の長さ
# plt.rcParams["ytick.major.size"] = 20          #y軸主目盛り線の長さ
# plt.rcParams["xtick.minor.size"] = 10            #x軸補助目盛り線の長さ
# plt.rcParams["ytick.minor.size"] = 10            #y軸補助目盛り線の長さ
# plt.rcParams["xtick.major.pad"] = 14             #x軸と目盛数値のマージン
# plt.rcParams["ytick.major.pad"] = 14             #y軸と目盛数値のマージン
# plt.rcParams["axes.linewidth"] = 2            #囲みの太さ



'''
# if use sympy matrix
def func(t, x, ds):
    # sympy F
    sp_f = dynamical_system.map(sp.Matrix(x), ds.params, ds.const)
    # convert to numpy
    ret = ds_func.sp2np(sp_f).flatten()
    return ret
'''
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
    # calculate equilibrium points
    eq = ds_func.equilibrium(ds)
    # convert to numpy
    ds.eq = ds_func.sp2np(eq)
    ds.state0 = ds.eq[0,:].flatten()
    dp.Eigen(ds)
    temp = np.array([-2*np.pi,0,0,0])
    ds.state0_a = ds.x_a
    ds.state0_b = ds.x_b
    ds.state0_c = ds.x_c
    ds.state0_d = ds.x_d

    ds.state0_e = ds.x_e
    ds.state0_f = ds.x_f
    ds.state0_g = ds.x_g
    ds.state0_h = ds.x_h

    print(ds.x_a)
    print("kore")
    #print("e",ds.state0_e)
    print("a",ds.state0_a)


    # calculate orbit
    duration = 15
    tick = 0.01
    matplotinit(ds)
    # import numpy parameter
    p = ds_func.sp2np(ds.params).flatten()
    # import numpy constant
    c = ds_func.sp2np(ds.const).flatten()
    show_params(ds)

    while ds.running == True:
        state = solve_ivp(func, (0, duration), ds.state0,
             method='RK45', args = (p, c), max_step=tick,
             rtol=1e-12, vectorized = True) 
        state_a = solve_ivp(func, (0, duration), ds.state0_a,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)  
        state_b = solve_ivp(func, (0, duration), ds.state0_b,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)
        state_c = solve_ivp(func, (0, duration), ds.state0_c,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)
        state_d = solve_ivp(func, (0, duration), ds.state0_d,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)
        # state_x = solve_ivp(func, (0, duration), ds.state0_x,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)
        state_a_m = solve_ivp(func, (-duration,0), ds.x_a - temp,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)  
        state_b_m = solve_ivp(func, (-duration,0), ds.x_b - temp,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)
        state_c_m = solve_ivp(func, (-duration,0), ds.x_c - temp,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)
        state_d_m = solve_ivp(func, (-duration,0), ds.x_d - temp,
            method='RK45', args = (p, c), max_step=tick,
            rtol=1e-12, vectorized = True)
        # state_e = solve_ivp(func, (0,duration), ds.state0_e,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)  
        # state_f = solve_ivp(func, (0,duration), ds.state0_f,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)
        # state_g = solve_ivp(func, (0,duration), ds.state0_g,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)
        # state_h = solve_ivp(func, (0,duration), ds.state0_h,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)

        # state_minus = solve_ivp(func, (-duration,0), ds.state0_minus,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True) 
        # state_alpha_minus = solve_ivp(func, (-duration,0), ds.alpha0_minus,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)  
        # state_alpha_plus = solve_ivp(func, (0, duration), ds.alpha0,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True)  
        # state_omega_plus = solve_ivp(func, (0, duration), ds.omega0,
        #     method='RK45', args = (p, c), max_step=tick,
        #     rtol=1e-12, vectorized = True) 

        ds.ax2.plot(state.y[0,:], state.y[1,:],
                linewidth=1,color = (0.0, 0.0, 0.0),
               ls="-")
        # ds.ax2.plot(state_a.y[0,:], state_a.y[1,:],
        #         linewidth=1,color = (1.0, 0.0, 0.0),
        #        ls="-")
        # ds.ax2.plot(state_b.y[0,:], state_b.y[1,:],
        #         linewidth=1,color = (1.0, 0.0, 0.0),
        #         ls="-")
        ds.ax2.plot(state_c.y[0,:], state_c.y[1,:],
                linewidth=1,color = (0.0, 0.0, 1.0),
               ls="-")
        ds.ax2.plot(state_d.y[0,:], state_d.y[1,:],
                linewidth=1,color = (0.0, 0.0, 1.0),
               ls="-")
        # ds.ax2.plot(state_a_m.y[0,:], state_a_m.y[1,:],
        #         linewidth=1,color = (1.0, 0.0, 0.0),
        #        ls="-")
        # ds.ax2.plot(state_b_m.y[0,:], state_b_m.y[1,:],
        #         linewidth=1,color = (1.0, 0.0, 0.0),
        #         ls="-")
        ds.ax2.plot(state_c_m.y[0,:], state_c_m.y[1,:],
                linewidth=1,color = (0.0, 0.0, 1.0),
               ls="-")
        ds.ax2.plot(state_d_m.y[0,:], state_d_m.y[1,:],
                linewidth=1,color = (0.0, 0.0, 1.0),
               ls="-")


        # ds.ax2.plot(state_e.y[0,:], state_e.y[1,:],
        #         linewidth=1,
        #        ls="-")
        # ds.ax2.plot(state_f.y[0,:], state_f.y[1,:],
        #         linewidth=1,
        #         ls="-")
        # ds.ax2.plot(state_g.y[0,:], state_g.y[1,:],
        #         linewidth=1,
        #        ls="-")
        # ds.ax2.plot(state_h.y[0,:], state_h.y[1,:],
        #         linewidth=1,
        #        ls="-")
        # ds.ax2.plot(state.y[2,:], state.y[3,:],
        #         linewidth=1, color=(0.3, 0.6, 0.6),
        # #         ls="-")
        plt.pause(0.001)  # REQIRED

def matplotinit(ds):
    redraw(ds)
    plt.connect('button_press_event',
                lambda event: on_click(event, ds))
    plt.connect('key_press_event',
                lambda event: keyin(event, ds))

def redraw(ds):
    show_params(ds)
    ds.ax2.set_xlim(ds.xrange)
    ds.ax2.set_ylim(ds.yrange)
    ds.ax2.grid(c='gainsboro', ls='--', zorder=9)
    ds.ax2.set_xlabel(r"$\theta_{1}$")
    # ds.ax.set_xlabel(r"$\theta_{2}$")
    ds.ax2.set_ylabel(r"$\dot{\theta_{1}}$")
    # ds.ax.set_ylabel(r"$\dot{\theta_{2}}$")

def eq_change(ds):
    ds.x_ptr += 1
    if(ds.x_ptr >= ds.xdim):
        ds.x_ptr = 0
    ds.state0 = ds.eq[ds.x_ptr, :].flatten()
    print(ds.x_ptr)

def on_click(event, ds):
    #left click
    if event.xdata == None or event.ydata == None:
        return
    plt.cla()
    redraw(ds)
    if ds.dim_ptr <= 0:
        ds.state0[0] = event.xdata
        ds.state0[1] = event.ydata
    if ds.dim_ptr >= 1:
        ds.state0[2] = event.xdata
        ds.state0[3] = event.ydata

def state_reset(ds):
    ds.state0 = ds.eq[ds.x_ptr,:].flatten()

def keyin(event, ds):
    if event.key == 'q':
        ds.running = False
        plt.clf()
        plt.close('all')
        print("quit")
        sys.exit()
    elif event.key == ' ':
        ds.ax2.cla()
        redraw(ds)
        state_reset(ds)
    elif event.key == 'x':
        ds.ax2.cla()
        eq_change(ds)
        redraw(ds)
    elif event.key == 'c':
        ds.dim_ptr += 1
        if(ds.dim_ptr >= 2):
            ds.dim_ptr = 0
        if(ds.dim_ptr <= 0):
            print("change on_click dimension to theta_1, theta_1 dot")
        elif(ds.dim_ptr >= 1):
            print("change on_click dimension to theta_2, theta_2 dot")
    elif event.key == 'p':
        # change parameter
        print("change parameter")
    elif event.key == 's':
        print("now writing...")
        pdf = PdfPages('snapshot.pdf')
        pdf.savefig()
        pdf.close()
        print("done.")
    

def show_params(ds):
    s = ""
    p = ""
    params = ds_func.sp2np(ds.params).flatten().tolist()
    x0 = ds.state0.flatten().tolist()
    for i in range(len(params)):
        s += f"x{i}:{x0[i]:.5f},"
        p += f"p{i}:{params[i]:.4f},"
    plt.title(s+"\n"+p, color = 'blue')

if __name__ == '__main__':
    main()
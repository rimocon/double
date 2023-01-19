import sys
import json
import dynamical_system
import ds_func
import numpy as np
from numpy import sin, cos
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# プロット用設定
plt.rcParams["font.family"] = "Nimbus Roman"    #全体のフォントを設定
plt.rcParams['text.usetex'] = True              #描画にTeXを利用
plt.rcParams['text.latex.preamble'] = r'''\usepackage{amsmath}
                                          \usepackage{amssymb}
                                          \usepackage[T1]{fontenc}
                                          \usepackage{bm}
                                          '''
plt.rcParams["figure.autolayout"] = False       #レイアウト自動調整をするかどうか
plt.rcParams["font.size"] = 24                  #フォントの大きさ
plt.rcParams["xtick.direction"] = "in"          #x軸の目盛線を内向きへ
plt.rcParams["ytick.direction"] = "in"          #y軸の目盛線を内向きへ
# plt.rcParams["xtick.minor.visible"] = True      #x軸補助目盛りの追加
# plt.rcParams["ytick.minor.visible"] = True      #y軸補助目盛りの追加
plt.rcParams["xtick.major.width"] = 1.0         #x軸主目盛り線の線幅
plt.rcParams["ytick.major.width"] = 1.0         #y軸主目盛り線の線幅
plt.rcParams["xtick.minor.width"] = 0.5         #x軸補助目盛り線の線幅
plt.rcParams["ytick.minor.width"] = 0.5         #y軸補助目盛り線の線幅
plt.rcParams["xtick.major.size"] = 20           #x軸主目盛り線の長さ
plt.rcParams["ytick.major.size"] = 20          #y軸主目盛り線の長さ
plt.rcParams["xtick.minor.size"] = 10            #x軸補助目盛り線の長さ
plt.rcParams["ytick.minor.size"] = 10            #y軸補助目盛り線の長さ
plt.rcParams["xtick.major.pad"] = 16             #x軸と目盛数値のマージン
plt.rcParams["ytick.major.pad"] = 16             #y軸と目盛数値のマージン
plt.rcParams["axes.linewidth"] = 2            #囲みの太さ

# 運動方程式
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

def solver(ds):
    # solver
    ds.state = solve_ivp(func, (0, ds.duration), ds.state0,
        method='RK45', args = (ds.p, ds.c), max_step=ds.tick,
        rtol=1e-12, vectorized = True)

def animate(data):
    t, x1, y1, x2, y2 = data
    ds.xlocus.append(x2)
    ds.ylocus.append(y2)

    ds.locus.set_data(ds.xlocus, ds.ylocus)
    ds.line.set_data([0, x1, x2], [0, y1, y2])
    ds.time_text.set_text(ds.time_template % (t))

# generator
def gen(ds):
    for tt, th1, th2 in zip(ds.state.t, ds.state.y[0,:], ds.state.y[2,:]):
        x1 = ds.c[0] * cos(th1)
        y1 = ds.c[0] * sin(th1)
        x2 = x1 + ds.c[1] * cos(th1 + th2)
        y2 = y1 + ds.c[1] * sin(th1 + th2)
        yield tt, x1, y1, x2, y2

def set(ds):
    ds.ax.set_xlim(-(ds.c[0]+ds.c[1]),ds.c[0]+ds.c[1])
    ds.ax.set_ylim(-(ds.c[0]+ds.c[1]),ds.c[0]+ds.c[1])
    ds.ax.set_xticks([-(ds.c[0]+ds.c[1]),-(ds.c[0]+ds.c[1])/2, 0, (ds.c[0]+ds.c[1])/2, ds.c[0]+ds.c[1]])
    ds.ax.set_yticks([-(ds.c[0]+ds.c[1]),-(ds.c[0]+ds.c[1])/2, 0, (ds.c[0]+ds.c[1])/2, ds.c[0]+ds.c[1]])
    ds.ax.set_aspect('equal')
    ds.ax.grid()

def keyin(event, ds):
    if event.key == 'q':
        plt.cla()
        plt.close('all')
        print("quit")
        sys.exit()
    elif event.key == 'x':
        print("xxxxxxxxxxxx")
        plt.cla()
        set(ds)
        ds.x_ptr += 1
        if(ds.x_ptr >= ds.xdim):
            ds.x_ptr = 0
        ds.state0 = ds.eq[ds.x_ptr,:].flatten()
        print("change eq point")       
        print(ds.state0)

        solver(ds)
        locus(ds)
        ds.ani.new_frame_seq()
        # plt.show()
        
        
    #     ds.ax.cla()
    #     eq_change(ds)
    #     redraw(ds)
    # #  elif event.key == 'p':
    #      # change parameter
    #      print("change parameter")
    #  elif event.key == 's':
    #     ds.ani.save('double_pendulum.gif', writer='pillow', fps=15)

def locus(ds):

    ds.locus, = ds.ax.plot([], [], 'r-', linewidth=2)
    ds.line, = ds.ax.plot([], [], 'o-', linewidth=2)
    ds.time_template = 'time = %.1fs'
    ds.time_text = ds.ax.text(0.05, 0.9, '', transform = ds.ax.transAxes)

    ds.xlocus, ds.ylocus = [], []

# load data from json file
if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} filename")
fd = open(sys.argv[1], 'r')
json_data = json.load(fd)
fd.close()

ds = dynamical_system.DynamicalSystem(json_data)
ds.x_ptr = 3
# input data to constructor
eq = ds_func.equilibrium(ds)

# convert to numpy
ds.eq = ds_func.sp2np(eq)
ds.state0 = ds.eq[ds.x_ptr,:].flatten()

# calculate orbit
ds.duration = 10
ds.tick = 0.05

# import numpy parameter
ds.p = ds_func.sp2np(ds.params).flatten()

# import numpy constant
ds.c = ds_func.sp2np(ds.const).flatten()

solver(ds)
# graph
plt.connect('key_press_event',
    lambda event: keyin(event, ds))
set(ds)
locus(ds)
ds.ani = FuncAnimation(ds.fig, animate, gen(ds),interval=50)
    # ani.save('double_pendulum.gif', writer='pillow', fps=15)
plt.show()
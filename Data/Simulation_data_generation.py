import os
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import matplotlib
import matplotlib.pyplot as plt

def ODE_CLorenz(t, x, sigma, b, r, C, N, t_step, noise):
    t_idx = int(t / t_step)
    rho = r[t_idx] if isinstance(r, np.ndarray) else r
    nt = noise[:, t_idx] if isinstance(noise, np.ndarray) else noise
    dxdt = np.zeros(3*N)
    dxdt[0] = sigma * (x[1] - x[0]) + C * x[(N - 1) * 3]**2+ nt[0]
    dxdt[1] = rho * x[0] - x[1] - x[0] * x[2] + nt[1]
    dxdt[2] = -b * x[2] + x[0] * x[1] + nt[2]
    for j in range(1, N):
        dxdt[3 * j] = sigma * (x[1 + 3 * j] - x[3 * j]) + C * x[3 * (j - 1)]**2 + nt[3 * j]
        dxdt[1 + 3 * j] = rho * x[3 * j] - x[1 + 3 * j] - x[3 * j] * x[2 + 3 * j] + nt[1+3 * j]
        dxdt[2 + 3 * j] = -b * x[2 + 3 * j] + x[3 * j] * x[1 + 3 * j] + nt[2+3 * j]
    return dxdt
def Coupled_Lorenz(r=3.4, t_step=0.1, t_end=12000,
                   x0=[2, 2, -2],ns=float(0), N: int=3):
    if type(r) == float:
        r = np.repeat(r, t_end)
    x0 = np.repeat(x0, N)
    sigma=4; b=8/3; C=0.1
    t_span = (0, t_step*(t_end-1))  # 时间范围
    t_eval = np.linspace(*t_span, t_end)
    noise = ns * np.random.randn(3*N, t_end)
    sol = solve_ivp(ODE_CLorenz, t_span, x0, args=(sigma, b, r, C, N, t_step, noise), t_eval=t_eval)
    return sol

def M_Henon(r, b=0.3, N=3, C=0.02, x0=np.array([0.8, 1.21]), ns=float(0)):
    data = np.zeros((len(r) + 1, 2*N))
    for j in range(N):
        data[0, 2*j:2*(j+1)] = x0 + np.random.rand(2) # 均匀分布，在一定范围内随机生成初始状态
    M_r = 0.1 + 0.2 * np.random.rand(N-1) # 在0.1-0.3中随机取值，对应若干维不发生分岔
    for i in range(len(r)):
        data[i + 1, 0] = 1 - r[i] * data[i, 0] ** 2 + data[i, 1] + ns * np.random.randn()
        data[i + 1, 1] = b * data[i, 0] + ns * np.random.randn()
        for j in range(1, N, 1):
            data[i + 1, 2*j+0] = 1 - M_r[j-1] * data[i, 2*j+0] ** 2 + data[i, 2*j+1] + C * data[i, 0] + ns * np.random.randn()
            data[i + 1, 2*j+1] = b * data[i, 2*j+0] + ns * np.random.randn()
    return data[1000:, :]


def ODE_cadvp(t, y, v, mu, alpha, beta, C, N, t_step, noise):
    t_idx = int(t / t_step)
    dxdt = np.zeros(3 * N)
    vi = v[t_idx] if isinstance(v, np.ndarray) else v
    mui = mu[t_idx] if isinstance(mu, np.ndarray) else mu
    betai = beta[t_idx] if isinstance(beta, np.ndarray) else beta
    alphai = alpha[t_idx] if isinstance(alpha, np.ndarray) else alpha
    nt = noise[:, t_idx] if isinstance(noise, np.ndarray) else noise
    dxdt[0] = -vi*(y[0]**3-mui*y[0]-y[1]) + nt[0]
    dxdt[1] = y[0]-alphai*y[1]-y[2] + nt[1]
    dxdt[2] = betai*y[1] + nt[2]
    for j in range(1, N):
        dxdt[3 * j] = -vi * (y[3 * j] ** 3 - mui * y[3 * j] - y[1+3 * j]) + C * y[3 * (j - 1)] + nt[3 * j]
        dxdt[1 + 3 * j] = y[3 * j] - alphai * y[1+3 * j] - y[2 + 3 * j] + nt[1 + 3 * j]
        dxdt[2 + 3 * j] = betai * y[1+3 * j] + nt[2+3 * j]
    return dxdt
def Coupled_ADVP(alpha=3.4, t_step=0.05, t_end=16000,
                 x0=[0.2,0.5,-0.1], ns=float(0), N: int=3):
    if type(alpha) == float:
        alpha = np.repeat(alpha, t_end+1)
    x0 = np.repeat(x0, N)
    v=100; mu=-0.1; beta=200; C=0.1 #mu<0
    t_span = (0, t_step * (t_end))  # 时间范围
    t_eval = np.linspace(*t_span, t_end)
    noise = ns * np.random.randn(3*N, t_end + 1)
    sol = solve_ivp(ODE_cadvp, t_span, x0,
                    args=(v, mu, alpha, beta, C, N, t_step, noise), t_eval=t_eval)
    return sol


def data_generation(model, **arg):
    if model == 'M_Henon':
        bif = 0 + 0.368 / 10000 * np.arange(0, 16000, 1)  # r=0.3675处倍周期分岔
        ns = 0.02
        ts_data = M_Henon(r=bif, ns=ns)
        comments = """# bif = 0 + 0.368 / 10000 * np.arange(0, 16000, 1)  # r=0.3675处倍周期分岔
# coupling strength C = 0.02
# ns = 0.2 
# t_step = 1"""
    elif model == 'Coupled_Lorenz':
        bif = -3 + 4 / 10000 * np.arange(0, 16001, 1) # r=1, 在10000处transcritical
        ns = 0.5 # ns:0.001
        ts = Coupled_Lorenz(r=bif, t_step=0.5, t_end=len(bif), ns=ns)  # t_step:0.1, 0.5
        ts_data = ts.y[:, 1000:].T  #r=1, 在9000处transcritical
        comments = """# bif = -3 + 4 / 10000 * np.arange(0, 16001, 1) # r = 1处transcritical
# ns = 0.5
# coupling strength C = 0.1
# t_step=0.5"""
    elif model == 'Coupled_ADVP':
        bif = 4.65 - 0.508 / 10000 * np.arange(0, 16001, 1)  #4.1421, NS bifurcation
        ns = 0.001
        ts = Coupled_ADVP(alpha=bif, ns=ns, t_step=0.1)
        ts_data = ts.y[:, 1000:].T
        comments = """# bif = 4.65 - 0.5 / 10000 * np.arange(0, 16001, 1)  #4.1421, NS bifurcation
# ns = 0.001
# coupling strength C = 0.1
# t_step=0.1"""
    else:
        exit()
    return ts_data, ns, comments


if __name__ == '__main__':
    model='Coupled_Lorenz'
    data, ns, comments = data_generation(model=model)
    path = "/home/yangna/JetBrains"
    if not os.path.exists(path):
        os.makedirs(path)
    matplotlib.use('Agg')  # 设置后端为 Agg
    # 单维变化图
    plt.figure(figsize=(16, 8))
    plt.plot(data[:, :], linewidth=0.5, marker=".", markersize=3)
    plt.axvline(x=9000, color='red', linestyle='--', alpha=0.6)
    # plt.plot(data[:, 2], linewidth=0.5, c='red', marker=".", markersize=3)
    plt.title(model + ' Data', fontsize=20)
    plt.savefig(path + '/Data/%s_ns=%f_1d.png' % (model, ns))
    plt.close()
    np.savetxt("../Data/%s_ns=%f.txt" % (model, ns), data,
               header=comments, comments='',
               fmt='%f', delimiter=',')

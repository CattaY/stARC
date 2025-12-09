import numpy as np
import pandas as pd
import torch
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.colors import LogNorm
from scipy.stats import norm
from scipy.special import logsumexp
import matplotlib
matplotlib.use('Agg')  # 设置后端为 Agg

# -----------------------------------------------------------------------------
def bocd(data, model, hazard):
    """Return run length posterior using Algorithm 1 in Adams & MacKay 2007.
    """
    # 1. Initialize lower triangular matrix representing the posterior as
    #    function of time. Model parameters are initialized in the model class.
    #
    #    When we exponentiate R at the end, exp(-inf) --> 0, which is nice for
    #    visualization.
    #
    T = len(data)
    log_R = -np.inf * np.ones((T + 1, T + 1))
    log_R[0, 0] = 0  # log 0 == 1
    pmean = np.empty(T)  # Model's predictive mean.
    pvar = np.empty(T)  # Model's predictive variance.
    log_message = np.array([0])  # log 0 == 1
    log_H = np.log(hazard)
    log_1mH = np.log(1 - hazard)

    for t in range(1, T + 1):
        # 2. Observe new datum.
        x = data[t - 1]

        # Make model predictions.
        pmean[t - 1] = np.sum(np.exp(log_R[t - 1, :t]) * model.mean_params[:t])
        pvar[t - 1] = np.sum(np.exp(log_R[t - 1, :t]) * model.var_params[:t])

        # 3. Evaluate predictive probabilities.
        log_pis = model.log_pred_prob(t, x)

        # 4. Calculate growth probabilities.
        log_growth_probs = log_pis + log_message + log_1mH

        # 5. Calculate changepoint probabilities.
        log_cp_prob = logsumexp(log_pis + log_message + log_H)

        # 6. Calculate evidence
        new_log_joint = np.append(log_cp_prob, log_growth_probs)

        # 7. Determine run length distribution.
        log_R[t, :t + 1] = new_log_joint
        log_R[t, :t + 1] -= logsumexp(new_log_joint)

        # 8. Update sufficient statistics.
        model.update_params(t, x)

        # Pass message.
        log_message = new_log_joint

    R = np.exp(log_R)
    return R, pmean, pvar

# -----------------------------------------------------------------------------
class GaussianUnknownMean:

    def __init__(self, mean0, var0, varx):
        """Initialize model.

        meanx is unknown; varx is known
        p(meanx) = N(mean0, var0)
        p(x) = N(meanx, varx)
        """
        self.mean0 = mean0
        self.var0 = var0
        self.varx = varx
        self.mean_params = np.array([mean0])
        self.prec_params = np.array([1 / var0])

    def log_pred_prob(self, t, x):
        """Compute predictive probabilities \pi, i.e. the posterior predictive
        for each run length hypothesis.
        """
        # Posterior predictive: see eq. 40 in (Murphy 2007).
        post_means = self.mean_params[:t]
        post_stds = np.sqrt(self.var_params[:t])
        return norm(post_means, post_stds).logpdf(x)

    def update_params(self, t, x):
        """Upon observing a new datum x at time t, update all run length
        hypotheses.
        """
        # See eq. 19 in (Murphy 2007).
        new_prec_params = self.prec_params + (1 / self.varx)
        self.prec_params = np.append([1 / self.var0], new_prec_params)
        # See eq. 24 in (Murphy 2007).
        new_mean_params = (self.mean_params * self.prec_params[:-1] +
                           (x / self.varx)) / new_prec_params
        self.mean_params = np.append([self.mean0], new_mean_params)

    @property
    def var_params(self):
        """Helper function for computing the posterior variance.
        """
        return 1. / self.prec_params + self.varx

# -----------------------------------------------------------------------------
def plot_posterior(T, data, cps, R, pmean, pvar, path, label='N'):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    ax1, ax2 = axes
    ax1.scatter(range(0, T), data)
    ax1.plot(range(0, T), data)
    ax1.set_xlim([0, T])
    ax1.margins(0)
    # Plot predictions.
    ax1.plot(range(0, T), pmean, c='k')
    _2std = 2 * np.sqrt(pvar)
    ax1.plot(range(0, T), pmean - _2std, c='k', ls='--')
    ax1.plot(range(0, T), pmean + _2std, c='k', ls='--')

    ax2.imshow(np.rot90(R), aspect='auto', cmap='gray_r',
               norm=LogNorm(vmin=0.0001, vmax=1))
    ax2.set_xlim([0, T])
    ax2.margins(0)
    if type(cps) == int or np.int64 or np.int16 or np.int32:
        ax1.axvline(cps, c='red', ls='dotted')
        ax2.axvline(cps, c='red', ls='dotted')
    else:
        for cp in cps:
            ax1.axvline(cp, c='red', ls='dotted')
            ax2.axvline(cp, c='red', ls='dotted')
    plt.savefig(path + '/bocd_%s.svg' % label)
    plt.close()

# -----------------------------------------------------------------------------
def EWS_bocd(sdZ, path='', bif_window=-1, p_bocd = 1, step=10, if_plot=True, label='N'):
    hazard = 1 / 1000  # Constant prior on changepoint probability.
    T = len(sdZ) # Number of observations.
    mean0 = np.mean(sdZ[:T])  # The prior mean on the mean parameter.
    var0 = np.std(sdZ[:T])  # The prior variance for mean parameter.
    varx = np.std(sdZ[:T])  # The known variance of the data.
    model = GaussianUnknownMean(mean0, var0, varx)
    R, pmean, pvar = bocd(sdZ[:T], model, hazard)
    maxp_idx = np.array([np.argmax(R[i, :]) for i in range(T)])
    for i in range(5, T, 1):
        if maxp_idx[i] != i and i - maxp_idx[i]+1 > 5:
            bif_window = i - maxp_idx[i]+1
            p_bocd = R[i, maxp_idx[i]]
            break
    if if_plot == True:
        plot_posterior(T, sdZ[:T], bif_window, R, pmean, pvar, path, label)
    return bif_window, p_bocd


def type_identification(devs1, devs2, EWS_idx, sigma=10):
    if EWS_idx < 0:
        return 'Null'
    if isinstance(devs1, np.ndarray):
        devs1 = torch.from_numpy(devs1)
        devs2 = torch.from_numpy(devs2)
    # 计算每个元素与1的绝对距离
    dis_devs1 = torch.abs(torch.abs(devs1[EWS_idx:]) - 1)
    dis_devs1 = dis_devs1.view(-1)
    _, indices1 = torch.topk(-dis_devs1, min(len(dis_devs1), sigma))  # 使用topk寻找最大值
    test1 = devs1[EWS_idx:][indices1]
    # test1 = test1_tmp[torch.abs(test1_tmp) > 0.9]
    test2 = devs2[EWS_idx:][indices1]
    # test2 = test2_tmp[torch.abs(test1_tmp) > 0.9]
    if abs(torch.median(torch.real(test1)-torch.real(test2))) <= 0.1 \
            and abs(torch.median(torch.imag(test1)+torch.imag(test2))) <= 0.1\
            and abs(torch.median(torch.imag(test1))) > 0.05:
        return "NS"
    elif torch.median(torch.real(test1)) >= 0.9:
        return "Trans"
    elif torch.median(torch.real(test1)) <= -0.9:
        return "PD"
    else:
        return "Null"


def plot_perf_stARC(stARC_results, data_show, ews_point, ews_window,
                    data_xlims, win_xlims, path, color_map=[],
                    num_dev=1, subplot=None, if_axinsert=True,
                    if_invert=[False, False, False, False], legend=True, **arg):
    if subplot is None:
        subplot = [True, True, True, True]
    if len(color_map) < 5:
        color_map = ['#DEF056', '#56A2F0', '#F06156', 'red', 0.2]
    if 'devs1_str' in stARC_results.columns:
        devs1 = np.array([complex(item) for item in stARC_results['devs1_str']])
        devs2 = np.array([complex(item) for item in stARC_results['devs2_str']])
    else:
        print('No dev column in stARC_results.')
        exit()
    devs1_real, devs1_imag = np.real(devs1), np.imag(devs1)
    devs2_real, devs2_imag = np.real(devs2), np.imag(devs2)
    sdZ = stARC_results["sd(Z)"]
    bif_point_para = arg['bif_point_para']
    if bif_point_para > 0 and bif_point_para <= arg['win_m']+arg['win_step']*(arg['num_win']-1):
        bif_window_para = int(np.ceil((bif_point_para-arg['win_m'])/arg['win_step']+1))
    else:
        bif_window_para = -1
    bif_point_obsvt = arg['bif_point_obsvt']

    # #绘制数据变化
    if subplot[0]:
        # 绘制 data, zorder控制元素位置，值越大元素越靠前
        fig, ax = plt.subplots()
        if type(data_show) == pd.DataFrame:
            colors=['#2372a9', '#2ca02c', '#9467bd', '#e377c2', '#17becf'] #blue, green, purple, pink, cran
            for i, col in enumerate(data_show.columns):
                color = colors[(i % len(colors))]  # 防止超出 colormap 范围自动循环
                ax.plot(data_show.index, data_show[col], label=col, color=color)
            if arg['dataset'] == "Dante_cave":
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.1f}"))
        else:
            ax.plot(data_xlims, data_show, linewidth=2)

        if bif_point_obsvt > 0:
            ax.axvspan(bif_point_obsvt, data_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(x=bif_point_obsvt, color=color_map[2], linestyle='--', linewidth=1.5, label="observed bifurcation")
        if bif_point_para > 0:
            ax.axvline(x=bif_point_para, color=color_map[1], linestyle='--', linewidth=0.75, label="parameter bifurcation")
            if bif_point_obsvt < 0:
                ax.axvspan(bif_point_para, data_xlims[-1], color=color_map[3], alpha=color_map[4])
        if ews_window > 0:
            if bif_point_para < 0 and bif_point_obsvt < 0:
                ax.axvspan(ews_point, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(x=ews_point, color=color_map[0], linestyle='--', linewidth=1.5, label="early warning signal")
            if arg['dataset'] == "Dante_cave":
                ymin, ymax = ax.get_ylim()
                ax.text(ews_point, ymin - 0.02 * (ymax - ymin),
                        f"{ews_point / 1000:.1f}",
                        ha='center', va='top', fontsize=10, fontweight='bold', color=color_map[0])

        ax.set_title(arg['dataset'])
        if if_invert[0]:
            ax.invert_xaxis()
        if legend:
            ax.legend()
        plt.savefig(path + '/%s data.svg' % arg['dataset'])
        plt.close()

    # 绘制 sd(Z)
    if subplot[1]:
        fig, ax = plt.subplots()
        # plt.xlim(0, arg["num_win"] + 1)  # 设置x轴范围
        ax.plot(win_xlims, sdZ, linewidth=2, c="#2372a9")
        if arg['dataset'] == "Dante_cave":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.1f}"))
        if bif_point_obsvt > 0:
            ax.axvspan(bif_point_obsvt, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(x=bif_point_obsvt, color=color_map[2], linestyle='--', linewidth=1.5, label="observed bifurcation")
        if bif_point_para > 0:
            ax.axvline(x=bif_point_para, color=color_map[1], linestyle='--', linewidth=0.75, label="parameter bifurcation")
            if bif_point_obsvt < 0:
                ax.axvspan(bif_point_para, win_xlims[-1], color=color_map[3], alpha=color_map[4])
        if ews_window > 0:
            if bif_point_para < 0 and bif_point_obsvt < 0:
                ax.axvspan(ews_point, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            # 获取当前 y 轴范围
            y_min, y_max = ax.get_ylim()  # 可能是 [-0.1, 2.1] 等自动扩展范围
            # 计算归一化的 ymax
            ymax_norm = float((sdZ[ews_window] - y_min) / (y_max - y_min))
            # 绘制垂直线
            ax.axvline(x=ews_point, color=color_map[0], linestyle='--', linewidth=1.5,
                       ymin=0,  # ymin=0 表示从 x 轴开始
                       ymax=ymax_norm,  # 归一化到当前 y 轴范围
                       label="early warning signal")
            ax.scatter(ews_point, sdZ[ews_window], color=color_map[0], s=150, zorder=5, marker="*")  # s 控制点大小
        # ax.set_xlim(left=0)
        if legend:
            ax.legend()
        ax.set_title('sd(Z) with %i neighbors' % (arg['nn']))
        if if_invert[1]:
            ax.invert_xaxis()
        if ews_window > 0 and if_axinsert==True:
            # # 定义要放大的区域（x范围）
            if bif_window_para>0:
                zoom_win1, zoom_win2 = ews_window - 20, bif_window_para + 5
            else:
                zoom_win1, zoom_win2 = ews_window - 20, ews_window + 5
            # # 创建插入的子图（放大图）
            ax_inset = inset_axes(ax, width="40%", height="30%", loc='center left')
            ax_inset.plot(win_xlims[zoom_win1:zoom_win2 + 1], sdZ[zoom_win1:zoom_win2 + 1], c="#2372a9")
            if bif_point_para>0:
                ax_inset.axvline(x=bif_point_para, color=color_map[1], linestyle='--', linewidth=1.5, label="parameter bifurcation")
            y_min, y_max = ax_inset.get_ylim()
            ymax_norm = float((sdZ[ews_window] - y_min) / (y_max - y_min))
            ax_inset.axvline(x=ews_point, color=color_map[0], linestyle='--', linewidth=1.5,
                       ymin=0,  # ymin=0 表示从 x 轴开始
                       ymax=ymax_norm)  # 归一化到当前 y 轴范围
            ax_inset.scatter(ews_point, sdZ[ews_window], color=color_map[0], s=80, zorder=5, marker="*")  # s 控制点大小
            # ax_inset.set_yticks([0, np.ceil(max(sdZ[zoom_win1:zoom_win2 + 1]))])
            ax_inset.set_yticks([])
        plt.savefig(path + '/sd(Z).svg')
        plt.close()
      
    cmin = 0; cmax = 1             
    # 绘制DEV的图
    if subplot[2]:
        plt.figure()
        if ews_window < 0:
            show_dev = len(devs1_real)
            Re, Im = devs1_real[:show_dev], devs1_imag[:show_dev]
        else:
            show_dev = ews_window
            Re, Im = devs1_real[:show_dev], devs1_imag[:show_dev]
        theta = np.linspace(0, 2 * np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), linewidth=1, color="black")
        plt.plot([0, 1], [0, 0], linewidth=0.5, color='black')
        plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
        plt.scatter(Re[::-1], Im[::-1],
                    c=np.linspace(cmin, cmax, show_dev), cmap='Blues',
                    vmin=0, vmax=1,
                    s=10)
        if num_dev==2:
            plt.scatter(devs2_real[:show_dev], devs2_imag[:show_dev],
                        c=np.linspace(cmin, cmax, show_dev), cmap='Blues',
                        vmin=0, vmax=1,
                        s=10)
        plt.axis('equal')
        plt.xlabel("Re(DEV)")
        plt.ylabel("Im(DEV)")
        plt.title('DEV in complex panel', fontsize=20)
        plt.savefig(path + '/DEV.svg')
        plt.close()


    # 绘制DEV模的图
    if subplot[3]:
        fig, ax = plt.subplots()
        if arg['dataset'] == "Dante_cave":
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x / 1000:.1f}"))
        if bif_point_obsvt > 0:
            ax.axvspan(bif_point_obsvt, win_xlims[-1], color=color_map[3], alpha=color_map[4])
            ax.axvline(bif_point_obsvt, color=color_map[2], linestyle='--', linewidth=1.5, label="observed bifurcation")
        if bif_point_para > 0:
            ax.axvline(bif_point_para, color=color_map[1], linestyle='--', linewidth=0.75, label="parameter bifurcation")
            if bif_point_obsvt < 0:
                ax.axvspan(bif_point_para, win_xlims[-1], color=color_map[3], alpha=color_map[4])
        if ews_point > 0:
            ax.axvline(ews_point, color=color_map[0], linestyle='--', linewidth=1.5, label="early warning signal")
            if arg['dataset'] == "Dante_cave":
                ymin, ymax = ax.get_ylim()
                ax.text(ews_point, ymin - 0.02 * (ymax - ymin),
                        f"{ews_point / 1000:.1f}",
                        ha='center', va='top', fontsize=10, fontweight='bold', color=color_map[0])
            if bif_point_para < 0 and bif_point_obsvt < 0:
                ax.axvspan(ews_point, win_xlims[-1], color=color_map[3], alpha=color_map[4])
        plt.axhline(y=1, color='black', linestyle='--', linewidth=1)
        sc=plt.scatter(win_xlims, abs(devs1), linewidth=0,  # c='blue',
                    c=np.linspace(cmin, cmax, len(devs1)), cmap='Blues',
                    vmin=0, vmax=1,
                    s=20, zorder=5)
        plt.title('mod(DEV)', fontsize=20)
        plt.ylim([0, 1.2])
        # plt.xlim(left=0)
        if if_invert[3]:
            ax.invert_xaxis()
        if legend:
            plt.legend()
        cbar=fig.colorbar(sc, ax=ax)
        cbar.ax.set_ylim(cmin, cmax)
        cbar.set_ticks([cmin, cmax])
        cbar.set_ticklabels([])  # 标签表示时间早晚
        plt.savefig(path + '/mod(DEV).svg')
        plt.close()
    print("✅ 可视化完成！")
    return


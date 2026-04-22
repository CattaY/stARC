# %% #import part
import importlib
import pandas as pd
importlib.invalidate_caches()
import os
import sys
import func_stARC_gpu as fst
import func_perf_assess as fpa
import datetime
import numpy as np
from scipy.stats import zscore
import torch
#%% Settings
arg, groups = fst.args()
timestamp = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
print('\nStart time: ', timestamp)

# ##################### Data Mark ######################
arg['dataset'] = 'Peter_lake'
num_dev = 2 # num of dominant eigenvalues
num_dev = 2
all_data = pd.read_csv('../Data/%s.csv' % arg['dataset'],
                        delimiter=',', header=0, comment='#')
tsid = 2009;
# #########建立存储结果文件夹########
path = "../Figures/RealDataset/[%s] %s_%i" % (timestamp, arg['dataset'], tsid)
if not os.path.exists(path):
    os.makedirs(path)
arg['label'] = str(tsid)
arg['bif_point_obsvt'] = 185 # 185 for 2009
data_selected = all_data[(all_data['lake'] == 'Peter') & (all_data['year'] == tsid)]
data_raw = data_selected[['DOY', 'Chla', 'ZBgrav', 'LMinnCPE']]#.iloc[::-1]
comments = """鱼类群落结构数据 - 数组\n# 行向量为data的多维变量\n# 数据开始：\n"""
with open(path + "/data_noise_%s.txt" % arg['label'], "w") as f:
    f.write(f"# {comments}")  # 写注释
    f.write(",".join(data_raw.columns) + "\n")  # 写列名
    np.savetxt(f, data_raw.values, delimiter=",", fmt="%.6f")  # 写数值
data = (data_raw[['Chla', 'ZBgrav', 'LMinnCPE']].values.copy()).T
data = zscore(data, axis=1)
arg['win_step'] = 1;
step = arg['win_step']
arg['num_win'] = -1
# ##################### Algorithm Setting ######################
arg['GPUid'] = '0'
arg['num_iter'] = 1
arg['if_z'] = True
arg['neigh'] = True
arg['NNkey'] = 'RC'  
device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
# ####################### Reservoir ########################
arg['fun_act'] = 'tanh'
arg['warmup_steps'] = 20
arg['aa'] = 1
arg['alpha'] = 0.5  # (1-alpha)r^t+alpha*f()
arg['nodes'] = 100
arg['deg'] = 0.2
arg['rho'] = 0.8
# ######################### stPCA ##########################
arg['win_m'] = 30
arg['L'] = 10
# ####################### neighbor #########################
arg['nn'] = 10
arg['theta'] = 0.1  # [0, 1], 值越小受到邻居的影响越小
arg['nn'] = 0 if not arg['neigh'] else arg['nn']
print("✅ 参数设置成功！")
# %% 主要部分

[n, m] = np.shape(data)  # input_dimensions, total_length
length_used = arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1)
data_noise = np.array(
    [data[i, :] + arg["ns"] * np.std(data[i, :length_used])*np.random.randn(m) for i in range(n)])
comments = """# data with noise - 数组\n# 行向量为data的多维变量\n# 数据开始："""
np.savetxt(path + '/data_noise_%s.txt' % arg["label"], data_noise.T,
            header=comments, comments='',  # 禁用默认的注释符
            delimiter=',',
            fmt='%.6f')
main_results, arg = fst.stARC(data_noise, path, groups, **arg)
devs1 = main_results["devs1"]; devs2 = main_results["devs2"]
sdZ = main_results["sd(Z)"]
print("✅ stARC 计算完成！")
#%% ################################      保存结果        ####################################
fst.Save_mainresluts(main_results, path, **arg)
summary = pd.read_csv(path + '/summary_%s.txt' % arg['label'], sep=',',
                            header=0,  # 使用第一行作为列名: devs1_real,devs1_imag,devs1_str,
                            # devs2_real,devs2_imag,devs2_str,sd(Z)
                            dtype=None)  # 自动推断类型
print("✅ stARC 结果已保存！")
# %% EWS
bif_point_obsvt = arg['bif_point_obsvt']
bif_point_para = arg['bif_point_para']
if bif_point_obsvt > 0 and bif_point_obsvt <= arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1):
    bif_window_obsvt = int(np.ceil((bif_point_obsvt - arg['win_m']) / arg['win_step'] + 1))
else:
    bif_window_obsvt = -1
if bif_point_para > 0 and bif_point_para <= arg['win_m'] + arg['win_step'] * (arg['num_win'] - 1):
    bif_window_para = int(np.ceil((bif_point_para - arg['win_m']) / arg['win_step'] + 1))
else:
    bif_window_para = -1
sdZ = sdZ.cpu().numpy()
sdZ_ews = sdZ[:bif_window_obsvt] if bif_window_obsvt>0 else sdZ
EWS_idx, pvalue = fpa.EWS_bocd(sdZ_ews, path=path)
type_bocd = fpa.type_identification(devs1, devs2, EWS_idx, sigma=100)
print("Early warning window: %i,\nBifurcation window: %i\n" % (EWS_idx, bif_window_para),
        "Bifurcation type is %s and the confidence level is %f"%(type_bocd, pvalue))
ews_point = EWS_idx * arg["win_step"] + arg["win_m"] ##
print("Early warning signal: %i,\nParameter bifurcation point: %i,\nObserved bifurcation point: %i\n"
        %(ews_point, arg['bif_point_para'], arg['bif_point_obsvt']))
# %% 可视化
# 图像元素
show_length = arg['win_m']+arg['win_step']*(arg['num_win']-1)
data_xlims = np.arange(0, show_length, 1) #(arg['num_win']-1)* arg["win_step"] + arg["win_m"]+1
win_xlims = np.arange(arg['win_m'], show_length+1, arg['win_step'])
data_show = data_noise[0, :show_length].T
step = arg['win_step']
ews_window = EWS_idx
ews_point = EWS_idx * arg["win_step"] + arg["win_m"]
#       绘制图表        #
color_map = ['#f4a116', '#2c92d1', '#e60012', 'red', 0.2]
fpa.plot_perf_stARC(summary, data_show, ews_point, ews_window,
                    data_xlims, win_xlims, path, color_map, num_dev=num_dev, **arg)
print("✅ 单次实验操作全部完成！")


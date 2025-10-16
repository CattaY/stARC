import os
import json
import argparse
import time
import numpy as np
import random
import pandas as pd
import pyEDM
import torch
import matplotlib.pyplot as plt
from numpy import isfinite, corrcoef, sqrt, mean
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import pickle

def args():
    parser = argparse.ArgumentParser(description="arguments")
    parser.add_argument('-GPUid', default='0', type=str, help="GPU used")
    parser.add_argument('-num_iter', default=1, type=int, help="Number of iterations")
    parser.add_argument('-win_step', default=1, type=int, help="Window step")
    parser.add_argument('-num_win', default=-1, type=int, help="Number of windows")
    parser.add_argument('-if_z', default=True, type=bool, help="Dimension reduction or not")
    parser.add_argument('-neigh', default=False, type=bool, help="Number of neighbor")
    parser.add_argument('-NNkey', default='N', type=str,
                        choices=['N', 'kernel', 'RC0'],
                        help="Type of neural network")
    grouped_args = {}
    #### data mark
    parser.add_argument('-dataset', default='UU_16nodes', type=str, help="Dataset")
    parser.add_argument('-label', default='N', type=str, help="Sample label")
    parser.add_argument('-datashape', default=(0, 0), type=tuple, help="Shape of data, dim * timelength")
    parser.add_argument('-ns', default=0, type=float, help="Noise strength")
    parser.add_argument('-bif_point_para', default=-1, type=int, help="Bifurcation point based on parameter")
    parser.add_argument('-bif_point_obsvt', default=-1, type=int, help="Bifurcation point based on observation")
    grouped_args['data_group'] = ['dataset', 'label', 'datashape', 'ns', 'bif_point_para', 'bif_point_obsvt']
    #### z
    z_group = parser.add_argument_group('parameters related to stPCA')
    z_group.add_argument('-ran_idx', default=0, type=int, help="Select some dimension for DEV analysis")
    z_group.add_argument('-win_m', default=10, type=int, help="length of each window")
    z_group.add_argument('-L', default=3, type=int, help="Embedding dimension")
    z_group.add_argument('-lam', default=0.2, type=float, help="1-lambda in Manuscript")
    grouped_args['z_group'] = ['win_m', 'L', 'lam', 'ran_idx']
    #### neighbor
    neighbor_group = parser.add_argument_group('parameters related to CCM neighbor')
    neighbor_group.add_argument('-nn', default=3, type=int, help="Number of neighbor")
    neighbor_group.add_argument('-dm', default='euclidean', type=str, help="Distance measurement")
    neighbor_group.add_argument('-theta', default=0.6, type=float, help="Neighboring regularization parameter")
    grouped_args['neighbor_group'] = ['nn', 'dm', 'theta']
    #### Smap coefficient
    Smap_group = parser.add_argument_group('parameters related to Smap')
    Smap_group.add_argument('-dreg', default=3, type=float,
                            help="Degree of regression localized to the region of state space")
    Smap_group.add_argument('-dlag', default=10, type=int, help="Dimension of lagged coordinate embedding")
    grouped_args['Smap_group'] = ['dreg', 'dlag']
    #### Reservoir Computing Hyperparameters
    RC_group = parser.add_argument_group('parameters related to RC, when NNkey==\'RC0\'')
    RC_group.add_argument('-fun_act', default='tanh', type=str, help="Activation function")
    RC_group.add_argument('-nodes', default=800, type=int, help="Number of reservoir nodes")
    RC_group.add_argument('-deg', default=0.1, type=float, help="Average degree of reservoir")
    RC_group.add_argument('-aa', default=5, type=float, help="Scaler of reservoir")
    RC_group.add_argument('-alpha', default=1, type=float, help="Leakage factor of reservoir")
    RC_group.add_argument('-rho', default=1, type=float, help="Spectral radius of reservoir")
    RC_group.add_argument('-warmup_steps', default=0, type=int, help="Warmup steps of reservoir")
    grouped_args['RC_group'] = ['fun_act', 'nodes', 'deg', 'aa', 'alpha', 'rho', 'warmup_steps']

    args, unknown = parser.parse_known_args()
    return vars(args), grouped_args

"""根据分组信息组织参数"""
def group_args(args, groups):
    grouped_dict = {}
    # 未分组参数
    grouped_args_set = {arg for arg_names in groups.values() for arg in arg_names}
    ungrouped_args = {arg: value for arg, value in args.items() if arg not in grouped_args_set}
    grouped_dict.update(ungrouped_args) # 将未分组参数添加到根级别
    # 分组参数
    for group_name, arg_names in groups.items():
        grouped_dict[group_name] = {arg: args[arg] for arg in arg_names}
    return grouped_dict

##########defining defining W_in, W_r, W_b of RC##########
def RC_ge(**arg):
    flag = [arg['nodes'], arg['aa'], arg['datashape'][0]]
    if arg['NNkey'] == 'RC0':
        [n, a, dim] = flag
        n = int(n)
        dim = int(dim)
        #######defining W_in and W_b
        W_in = np.zeros((n, dim))
        n_win = n - n % dim
        index = np.random.permutation(range(n))
        index = index[:n_win]
        index = np.reshape(index, [int(n_win / dim), dim])
        for d in range(dim):
            W_in[index[:, d], d] = a * (2 * np.random.rand(int(n_win / dim)) - 1)
        W_b = a * (2 * np.random.rand(n) - 1)
        sample = random.randint(1, 10000)
        W_r = np.loadtxt("/home/yangna/JetBrains/Data/Wr_groups/k=%.2f/%in%i/Wr_%in%i_a5_%i.txt"
                         % (arg['deg'], int(arg['deg'] * arg['nodes']), arg['nodes'],
                            int(arg['deg'] * arg['nodes']), arg['nodes'], sample),
                         delimiter=",")
        W_r = arg['rho'] * W_r
    else:
        n = arg['nodes']
        dim = arg['datashape'][0]
        a = arg['aa']
        #######defining W_in and W_b
        W_in = np.zeros((n, dim))
        W_r = np.zeros((n, n))
        n_win = n - n % dim
        n_gr = int(n_win / dim)  ## 每组神经元的个数
        for d in range(dim):
            W_in[d * n_gr:(d + 1) * n_gr, d] = a * (2 * np.random.rand(n_gr) - 1)
        W_b = a * (2 * np.random.rand(n) - 1)
        for k in range(dim):
            sample = random.randint(1, 10000)
            tmp = np.loadtxt("/home/yangna/JetBrains/Data/Wr_groups/k=%.2f/%in%i/Wr_%in%i_a5_%i.txt" % (
            arg['deg'], int(arg['deg'] * n_gr), n_gr, int(arg['deg'] * n_gr), n_gr, sample), delimiter=",")
            W_r[k * n_gr:(k + 1) * n_gr, k * n_gr:(k + 1) * n_gr] = tmp
    return {'W_in': W_in, 'W_b': W_b, 'W_r': W_r}

def afunc(x, **kwargs):
    if kwargs.get("fun_act") == 'ReLu':
        return torch.where(x < 0, 0, x)/10e4
    elif kwargs.get("fun_act") == 'softplus':
        return torch.log(1 + torch.exp(x))/10e4
    elif kwargs.get("fun_act") == 'ELU':
        return torch.where(x > 0, x/10e4, 10e4 * (torch.exp(x) - 1)/10e4)
    elif kwargs.get("fun_act") == 'tanh':
        # return torch.tanh(x)
        return torch.tanh(x/10e20)
    else:
        return x

# 计算储备池状态（输入含 warm up data，输出时舍去）, 储备池输入：udata
def R_fun_yes(udata: torch.Tensor, W_in, W_r,  r0=False, *W_b, **kwargs):
    device = torch.device("cuda:" + kwargs['GPUid'] if torch.cuda.is_available() else "cpu")
    alpha = kwargs.get("alpha")
    n = kwargs.get("nodes")
    len_washout = kwargs.get("warmup_steps")
    r_tmp = torch.zeros((n, udata.shape[1]+1)).to(device)
    if type(r0) == torch.Tensor:
        r_tmp[:, 0] = r0
    for ti in range(udata.shape[1]):
        x1 = torch.matmul(W_r, r_tmp[:, ti])
        x2 = torch.matmul(W_in, udata[:, ti])
        if type(W_b) == torch.Tensor:
            r_tmp[:, ti+1] = (1 - alpha) * r_tmp[:, ti] + alpha * afunc(x1+x2+W_b)
        else:
            r_tmp[:, ti + 1] = (1 - alpha) * r_tmp[:, ti] + alpha * afunc(x1+x2)
    r_all = r_tmp[:, 1+len_washout:]
    return r_all

def Mapping(data: torch.Tensor, **arg):
    device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
    if arg['NNkey'] == 'N':
        input_data = data
    elif arg['NNkey'] == 'Kernel':
        input_data = Kfun(data, **arg)
    else:
        Ws = RC_ge(**arg)
        W_in = torch.from_numpy(Ws["W_in"]).float().to(device)
        W_r = torch.from_numpy(Ws["W_r"]).float().to(device)
        W_b = torch.from_numpy(Ws["W_b"]).float().to(device)
        idx = np.arange(data.shape[1])
        np.random.shuffle(idx)
        warmup_data = data[:, idx[0:arg['warmup_steps']]]
        input_data = R_fun_yes(torch.cat((warmup_data, data), dim=1), W_in=W_in, W_b=W_b, W_r=W_r, **arg)
    return input_data

def best_smap_para(batch_data, batch_ori, **arg):
    if not arg['if_stPCA']:
        flat_z = batch_data[arg['ran_idx'], :]
    else:
        input_data = batch_data
        traindata = input_data - torch.mean(input_data, dim=1, keepdim=True)
        stPCA_results = stPCA(traindata, ori_data=batch_ori,
                              **arg)  # Z.T in Manuscript ## X: n*m, W: n*L, Z: m*L
        flat_z = stPCA_results.get('flat_z')
    #### DEV
    norm_z = (flat_z - torch.mean(flat_z)) # flat_z # (flat_z - torch.mean(flat_z)) #/ torch.std(flat_z)
    norm_z = norm_z.cpu().numpy()
    data_dev = pd.DataFrame({"X": norm_z})
    # train_start, train_end = 1, int(0.5 * len(norm_z))
    # pred_start, pred_end = int(0.75 * len(norm_z)+1), len(norm_z)
    train_start, train_end = 1, len(norm_z)-1
    pred_start, pred_end = 2, len(norm_z)
    # if arg['if_stPCA'] == True:
    #     train_start, train_end = 1, arg['win_m']
    #     pred_start, pred_end = arg['win_m'], len(norm_z)
    # else:
    #     train_start, train_end = 1, len(norm_z) - 1
    #     pred_start, pred_end = 2, len(norm_z)
    Eresults = pyEDM.EmbedDimension(
        dataFrame=data_dev, columns='X', target='X',
        maxE=min(arg['L'], 10),
        lib='%i %i' % (train_start, train_end),
        pred='%i %i' % (pred_start, pred_end),
        Tp=1, tau=-1, noTime=True, ignoreNan=True,
        showPlot=False
    )
    Eresults = Eresults.loc[2:]
    best_dlag, best_dPCC = Eresults.loc[Eresults['rho'].idxmax()]
    # dlag, best_dPCC = Eresults.loc[2] if int(best_dlag) < 3 else int(best_dlag), best_dPCC
    dlag = int(best_dlag)
    Tresults = pyEDM.PredictNonlinear(
        dataFrame=data_dev, columns='X', target='X',
        theta='0.5 1 1.5 2 4 6 10',
        lib='%i %i' % (train_start, train_end),
        pred='%i %i' % (pred_start, pred_end),
        E=dlag, Tp=1, knn=0, tau=-1,
        noTime=True, ignoreNan=True,
        showPlot=False
    )
    best_dreg, best_rPCC = Tresults.loc[Tresults['rho'].idxmax()]
    return {'best_dlag': dlag, 'best_dreg': best_dreg,
            'best_lPCC': best_dPCC, 'best_rPCC': best_rPCC}

def ComputeError(obs, pred, digits = 6 ):
    '''Pearson rho, RMSE, MAE
       Remove nan from obs, pred for corrcoeff.
    '''
    notNan = isfinite(pred)
    if any(~notNan):
        pred = pred[notNan]
        obs = obs[notNan]

    notNan = isfinite(obs)
    if any(~notNan):
        pred = pred[notNan]
        obs = obs[notNan]

    if len(pred) < 5 :
        msg = f'ComputeError(): Not enough data ({len(pred)}) to ' +\
               ' compute error statistics.'
        print(msg)
        return None

    rho = round(corrcoef(obs, pred)[0,1], digits)
    err = obs - pred
    MAE = round(max(err), digits)
    RMSE = round(sqrt(mean(err**2)), digits)
    return np.array([rho, MAE, RMSE])

# ########## 展示Smap效果 ############
def Save_SmapPlot(df, zonei, stats, arg, savepath):
    plt.figure()
    title = arg['dataset'] + "\nE=" + str(arg['dlag']) + " Theta=" + str(arg['dreg']) + \
            "  ρ=" + str(round(stats[0], 3)) + \
            "  RMSE=" + str(round(stats[2], 3))

    plt.plot(df['Time'], df['Observations'],
             linewidth=3, c='blue', label='Obe')
    plt.plot(df['Time'], df['Predictions'],
             linewidth=3, c='orange', label='Pre')
    plt.title(title)
    plt.legend()
    plt.savefig(savepath + '/Predictions of Smap [zone %i].png' % zonei)
    plt.close()
    return

def arg_save(path, arg, groups):
    grouped_arg = group_args(arg, groups)
    with open(os.path.join(path, 'arguments.json'), 'w', encoding='utf-8') as f:
        json.dump(grouped_arg, f,
                  indent=4,
                  ensure_ascii=False,  # 保证非ASCII字符正常保存
                  sort_keys=False,
                  separators=(',', ': '))
    return

class Timer:
    def __init__(self, print_tmpl="Time elapsed: {:.2f} seconds"):
        self.print_tmpl = print_tmpl

    def __enter__(self):
        self.start_time = time.time()  # 记录开始时间
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        elapsed_time = time.time() - self.start_time  # 计算耗时
        print(self.print_tmpl.format(elapsed_time))  # 打印耗时信息

def complex_median(x):
    real_median = torch.nanmedian(x.real)       # 实部中位数
    imag_median = torch.nanmedian(x.imag)       # 虚部中位数
    return torch.complex(real_median, imag_median)
def magnitude_median(x):
    magnitudes = torch.abs(x)                # 计算幅度
    median_idx = torch.median(torch.arange(len(magnitudes)))  # 中位数索引
    return x[median_idx]
def stARC(xx_noise, path, groups, if_plot=True, **arg):
    step = arg['win_step']
    device = torch.device("cuda:" + arg['GPUid'] if torch.cuda.is_available() else "cpu")
    # ##################### data batching ########################

    if type(xx_noise) == np.ndarray:
        xx_noise = torch.from_numpy(xx_noise).float().to(device)
    win_m = arg['win_m']
    [n, m] = xx_noise.shape # input_dimensions, total_length
    arg['datashape'] = (n, m)
    if arg["num_win"] <= 0:
        num_zones = int((m - win_m) / step)  # 180
        arg["num_win"] = num_zones
    else:
        num_zones = arg["num_win"]
    myzones = []
    for i in range(num_zones):
        myzones.append(range(0 + i * step, 0 + i * step + win_m, 1))
    batches_ori = [xx_noise[:, 0 + i * step: 0 + i * step + win_m] for i in range(num_zones)]
    batches_index = torch.arange(1, num_zones + 1).tolist()
    # ########################## algorithm start ###############################
    # Perform_Smap = np.zeros((num_zones, 3))
    Evalues_Smap = {key: [] for key in ['zone', 'Local_J', 'Evalues']}
    xx_input = Mapping(xx_noise, **arg)
    batches = [xx_input[:, 0 + i * step: 0 + i * step + win_m] for i in range(num_zones)]
    smap_para = best_smap_para(batches[0], batches_ori[0], **arg)
    print("Mapping done.\nSmap setting: " + str(smap_para))
    if arg['NNkey'] != 'N' and (smap_para['best_lPCC'] < 0.5 and smap_para['best_rPCC'] < 0.5):
        print('PCC is too low and Restart stARC.')
        return stARC(xx_noise, path, groups, if_plot, **arg)
    else:
        arg['dreg'] = smap_para['best_dreg']; arg['dlag'] = smap_para['best_dlag']
        dreg = arg['dreg']; dlag = arg['dlag']
        arg_save(path, arg, groups)
        print("Arguments: " + str(arg))
        print("-" * 80)

        def parallel(index, batch, batch_ori):
            if not arg['if_stPCA']:
                flat_z = batch[arg['ran_idx'], :]
                temp_var_y = torch.std(flat_z)
            else:
                input_data = batch
                traindata = input_data - torch.mean(input_data, dim=1, keepdim=True)
                stPCA_results = latent(traindata, ori_data=batch_ori,
                                      **arg)  # Z.T in Manuscript ## X: n*m, W: n*L, Z: m*L
                temp_var_y = stPCA_results.get('var_y')
                flat_z = stPCA_results.get('flat_z')
            #### DEV
            norm_z = (flat_z - torch.mean(flat_z))  # flat_z # (flat_z - torch.mean(flat_z)) #/ torch.std(flat_z)
            norm_z = norm_z.cpu().numpy()
            data_dev = pd.DataFrame({"X": norm_z})
            train_start, train_end = 1, len(norm_z) - 1
            pred_start, pred_end = 2, len(norm_z)
            smap_results = pyEDM.SMap(
                dataFrame=data_dev,
                lib="%i %i" % (train_start, train_end),  # 训练数据范围
                pred="%i %i" % (pred_start, pred_end),  # 预测数据范围
                columns="X", target="X",  # 使用的变量, 预测的目标变量
                theta=dreg, E=dlag,  # 控制局部化程度和嵌入维度
                embedded=False, tau=-1, noTime=True,
            )
            Predictions_Smap = smap_results['predictions']
            stats = ComputeError(Predictions_Smap['Observations'],
                                 Predictions_Smap['Predictions'])  # np.array([rho, MAE, RMSE])
            if if_plot == True and ((index >= 0.8 * num_zones and index % 5 == 0) or index <= 3):
                Save_SmapPlot(df=Predictions_Smap, zonei=index, stats=stats, arg=arg, savepath=path)
            local_J = torch.zeros((dlag, dlag)).to(device)
            cos = torch.tensor(smap_results['coefficients'].values).to(device)
            dev_tmp = torch.zeros(len(cos) - 1, dtype=torch.complex64).to(device)
            dev_tmp_2 = torch.zeros(len(cos) - 1, dtype=torch.complex64).to(device)
            devs_tmp = torch.zeros((len(cos) - 1, dlag), dtype=torch.complex64).to(device)
            for k in range(1, len(cos)):
                local_J[0, :] = cos[k, 2:]
                for j in range(1, dlag, 1):
                    local_J[j, j - 1] = 1
                Evalue, _ = torch.linalg.eig(local_J)
                sorted_val, temInd = torch.sort(torch.abs(Evalue), descending=True)  # torch.abs(Evalue)
                devs_tmp[k - 1, :] = sorted_val
                dev_tmp[k - 1] = (Evalue[temInd[0]])
                dev_tmp_2[k - 1] = (Evalue[temInd[1]])
            # return [magnitude_median(dev_tmp), magnitude_median(dev_tmp_2), [index, devs_tmp],
            #             flat_z, temp_var_y, stats]
            # min_abs_value, min_index = torch.abs(dev_tmp).min()
            # min_abs_value_2, min_index_2 = torch.abs(dev_tmp_2).min()
            min_index = torch.argmin(abs(dev_tmp))
            min_index_2 = torch.argmin(abs(dev_tmp_2))
            return [dev_tmp[min_index], dev_tmp_2[min_index_2], [index, devs_tmp],
                    flat_z, temp_var_y, stats]

        with Timer(print_tmpl='Pool() takes {:.1f} seconds'):
            with ThreadPoolExecutor(6) as executor:
                results_pool = list(
                    tqdm(executor.map(
                        parallel, batches_index, batches, batches_ori),
                        total=num_zones, desc="Processing tasks"
                    )
                )
                dev1_pool, dev2_pool, devs_pool, flatz_pool, var_y_pool, stats_pool = zip(*results_pool)
                dev1_pool = torch.tensor(dev1_pool, dtype=torch.complex64)
                dev2_pool = torch.tensor(dev2_pool, dtype=torch.complex64)
                var_y_pool = torch.tensor(var_y_pool)
                flatz_pool = torch.stack(flatz_pool)
                Perform_Smap= np.stack(stats_pool)
                Evalues_Smap['zone'].append(devs_pool[0])
                Evalues_Smap['Evalues'].append(devs_pool[1])
        # 保存字典 Evalues_Smap 到文件
        with open(path + '/Evalues_Smap.pkl', 'wb') as f:
            pickle.dump(Evalues_Smap, f)
        return {"devs1": dev1_pool, "devs2": dev2_pool, "sd(Z)": var_y_pool, "flat_z": flatz_pool}, arg



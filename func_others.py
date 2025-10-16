import numpy as np
import pandas as pd
import pyEDM
import scipy
from sklearn.decomposition import KernelPCA
from sklearn.svm import OneClassSVM
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans

def multiDEV_best_theta(batch_data):
    dim, m = np.shape(batch_data)
    # norm_batch = zscore(batch_data, axis=0)
    norm_batch = np.array([(batch_data[i, :]-np.mean(batch_data[i,:]))/np.std(batch_data[i,:])
                           for i in range(dim)])
    data_dev = pd.DataFrame(norm_batch.T)
    train_start, train_end = 1, m - 1
    pred_start, pred_end = 2, m
    columns = list(range(0, dim))
    # dlag = int(dim)
    Eresults = pyEDM.EmbedDimension(
        dataFrame=data_dev, columns=columns, target=[0],
        maxE=min(m-1, 10),
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
        dataFrame=data_dev, columns=columns, target=[0],
        theta='0.1 0.2 0.5 1 1.5 2 4 6 8 9 10 15 20',
        lib='%i %i' % (train_start, train_end),
        pred='%i %i' % (pred_start, pred_end),
        E=dlag, Tp=1, tau=-1,
        noTime=True, ignoreNan=True,
        showPlot=False, embedded=True,
        # knn=int(train_end-train_start)
    )
    best_dreg, best_rPCC = Tresults.loc[Tresults['rho'].idxmax()]

    return {'best_dlag': dlag, 'best_dreg': best_dreg,
            'best_lPCC': best_dPCC, 'best_rPCC': best_rPCC}

def multi_DEV(xx_input, calls = 1, **arg):
    step = arg['win_step']
    win_m = arg['win_m']
    [n, m] = xx_input.shape  # input_dimensions, total_length
    arg['datashape'] = (n, m)
    if arg["num_win"] <= 0:
        num_zones = int((m - win_m) / step)  # 180
        arg["num_win"] = num_zones
    else:
        num_zones = arg["num_win"]
    myzones = []
    for i in range(num_zones):
        myzones.append(range(0 + i * step, 0 + i * step + win_m, 1))
    # ########################## algorithm start ###############################
    batches = [xx_input[:, 0 + i * step: 0 + i * step + win_m] for i in range(num_zones)]

    # dlag, best_dPCC = Eresults.loc[2] if int(best_dlag) < 3 else int(best_dlag), best_dPCC

    smap_theta = multiDEV_best_theta(batches[0])
    print(smap_theta)
    # if (smap_theta['best_rPCC'] < 0.5) and calls <= 20:
    #     print('PCC is too low and Restart stARC.')
    #     return multi_DEV(xx_input, calls=calls+1, **arg)
    arg['dreg'] = smap_theta['best_dreg']
    arg['dlag'] = smap_theta['best_dlag']
    dev1 = np.zeros(num_zones, dtype=complex)
    dev2 = np.zeros(num_zones, dtype=complex)
    for zone in range(num_zones):
        dreg = arg['dreg'];
        dlag = arg['dlag']
        #### multivariance DEV
        norm_batch = np.array([(batches[zone][i, :] - np.mean(batches[zone][i, :])) / np.std(batches[zone][i, :])
                               for i in range(n)])
        data_dev = pd.DataFrame(norm_batch.T)
        train_start, train_end = 1, win_m - 1
        pred_start, pred_end = 2, win_m
        columns = list(range(0, n))
        cos=[]
        for target in range(n):
            smap_results = pyEDM.SMap(
                dataFrame=data_dev,
                lib="%i %i" % (train_start, train_end),  # 训练数据范围
                pred="%i %i" % (pred_start, pred_end),  # 预测数据范围
                columns=columns, target=[target],  # 使用的变量, 预测的目标变量
                theta=dreg, E=dlag,  # 控制局部化程度和嵌入维度
                embedded=True, tau=-1, noTime=True,
                # knn=int(train_end-train_start-1)
            )
            smap_coe = np.array(smap_results['coefficients'].values)
            cos.append(smap_coe[~np.isnan(smap_coe).any(axis=1), 2:])
        dev1_tmp = []
        dev2_tmp = []
        for k in range(win_m-2):
            local_J = np.zeros((n, n))
            for i in range(len(cos)):
                local_J[i, :dlag] = cos[i][k, :dlag]
             # = np.array([cos[i][k, :dlag] for i in range(len(cos))])
            # Evalue, _ = np.linalg.eig(local_J)

            Evalue, _ = scipy.linalg.eig(local_J)
            temInd = np.argsort(abs(Evalue))
            dev1_tmp.append(Evalue[temInd[-1]])
            dev2_tmp.append(Evalue[temInd[-2]])
        dev1_tmp = np.array(dev1_tmp)
        dev2_tmp = np.array(dev2_tmp)
        dev1[zone] = np.nanmean(dev1_tmp)
        dev2[zone] = np.nanmean(dev2_tmp)
        if (zone+1)%20==0:
            print("Zone " + str(zone+1)+": dev1 = "+ str(dev1[zone]) + ".")

    return {"devs1": dev1, "devs2": dev2}, arg

def find_best_ocsvm(X, n_components, nus, gammas):
    best_model = None
    best_score = -np.inf
    best_params = None
    for nu in nus:
        for gamma in gammas:
            kpca = KernelPCA(n_components=n_components, kernel='rbf', gamma=gamma)  # gamma需调参
            X_kpca = kpca.fit_transform(X)  # 输出形状 [N, 10]
            model = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
            model.fit(X_kpca)
            score = model.decision_function(X_kpca).mean()
            if score > best_score:
                best_score = score
                best_model = model
                best_params = (nu, gamma)

    return best_model, best_params, best_score

class CNNEncoder(nn.Module):
    def __init__(self, latent_dim=64, input_dim=9):
        super(CNNEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # output shape: [batch, 64, 1]
        )
        self.fc = nn.Linear(64, latent_dim)

    def forward(self, x):  # x: [B, 100, 9]
        x = x.permute(0, 2, 1)  # [B, 9, 100]
        x = self.encoder(x)     # [B, 64, 1]
        x = x.squeeze(-1)       # [B, 64]
        x = self.fc(x)          # [B, latent_dim]
        return x

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim=9, latent_dim=64, hidden_dim=128, num_layers=1, bidirectional=False):
        super(LSTMEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.bidirectional = bidirectional
        lstm_out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_out_dim, latent_dim)

    def forward(self, x):  # x: [B, 100, 9]
        output, (h_n, c_n) = self.lstm(x)  # h_n: [num_layers * num_directions, B, H]
        if self.bidirectional:
            h_final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [B, 2*H]
        else:
            h_final = h_n[-1]  # [B, H]
        return self.fc(h_final)  # [B, latent_dim]


def deepcluster_train(X_seq, n_clusters=2, latent_dim=64, n_cycles=10,
                      epochs_per_cycle=3, lr=1e-3, device='cpu', verbose=True):
    """
    DeepCluster training loop for time series [N, 100, 9].
    Args:
        X_seq: np.ndarray, shape [N, 100, 9]
        n_clusters: int, number of clusters
        latent_dim: int, size of latent vector
        n_cycles: int, DeepCluster iterations
        epochs_per_cycle: int, epochs per iteration
        lr: float, learning rate
        device: str, 'cpu' or 'cuda'
        verbose: bool, whether to print logs
    Returns:
        model: trained encoder model
        cluster_labels: final cluster assignments
        latents: final latent vectors
    """
    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(device)
    # model = CNNEncoder(latent_dim=latent_dim, input_dim=np.shape(X_seq)[-1]).to(device)
    model = CNNEncoder(latent_dim=latent_dim, input_dim=np.shape(X_seq)[-1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for cycle in range(n_cycles):
        if verbose:
            print(f"\n🔁 DeepCluster Iteration {cycle+1}/{n_cycles}")

        # Step 1: Extract latent features
        model.eval()
        with torch.no_grad():
            latents = model(X_tensor).cpu().numpy()

        # Step 2: Clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(latents)
        if verbose:
            print(f"  📌 Cluster counts: {np.bincount(cluster_labels)}")

        # Step 3: Train model using pseudo-labels
        model.train()
        targets = torch.tensor(cluster_labels, dtype=torch.long).to(device)
        for epoch in range(epochs_per_cycle):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        if verbose:
            print(f"  ✅ Training loss: {loss.item():.4f}")

    return model, cluster_labels, latents

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=64, num_layers=1, output_dim=9):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: [batch, seq_len, input_dim]
        out, _ = self.lstm(x)       # [batch, seq_len, hidden_dim]
        out = self.fc(out)          # [batch, seq_len, output_dim]
        return out

def train_LSTM(model, train_loader, silence=True, n_epochs=20, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(n_epochs):
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)          # x: [batch, window, 9]
            y = x.clone()                    # 使用自身作为目标

            output = model(x)                # 输出形状 [batch, window, 9]
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if silence==False:
            print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

def detect_anomaly(model, test_tensor, threshold=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    model.to(device)
    model.eval()
    with torch.no_grad():
        x = test_tensor.to(device)  # [batch, window, 9]
        y = x.clone()
        pred = model(x)
        mse = ((pred - y) ** 2).mean(dim=(1, 2))  # 每个样本的平均 MSE
        if threshold is not None:
            anomalies = mse > threshold
            return anomalies.cpu(), mse.cpu()
        else:
            return mse.cpu()


class TemporalCNNAutoencoder(nn.Module):
    def __init__(self, input_channels=9, latent_dim=64):
        super(TemporalCNNAutoencoder, self).__init__()

        # Encoder: Conv1D expects input shape [batch, channels, seq_len]
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # [batch, 32, 50]

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),  # [batch, 64, 25]
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),  # [batch, 64, 50]
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2),  # [batch, 32, 100]
            nn.Conv1d(32, input_channels, kernel_size=5, padding=2),
            nn.Tanh()  # 或者 ReLU, 取决于数据分布
        )

    def forward(self, x):
        # 输入 x: [batch, seq_len, channels]
        x = x.permute(0, 2, 1)  # -> [batch, channels, seq_len]
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat = x_hat.permute(0, 2, 1)  # -> [batch, seq_len, channels]
        return x_hat


def train_cnn_ae(model, train_loader, silence=True, epochs=20, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in train_loader:
            x = batch[0].to(device)  # [batch, seq_len, features]
            x_hat = model(x)
            loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if silence == False:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")


def detect_with_reconstruction_error(model, x, threshold=None, device="cuda"):
    model.eval()
    x = x.to(device)
    with torch.no_grad():
        x_hat = model(x)
        loss = ((x_hat - x)**2).mean(dim=(1, 2))  # 每个样本的重构误差

        if threshold is not None:
            anomaly_flags = loss > threshold
            return anomaly_flags.cpu(), loss.cpu()
        else:
            return loss.cpu()
# import torch
# import torch
# import torch.nn as nn
# from xlstm import (
#     xLSTMBlockStack,
#     xLSTMBlockStackConfig,
#     mLSTMBlockConfig,
#     mLSTMLayerConfig,
#     sLSTMBlockConfig,
#     sLSTMLayerConfig,
#     FeedForwardConfig,
# )

# cfg = xLSTMBlockStackConfig(
#     mlstm_block=mLSTMBlockConfig(
#         mlstm=mLSTMLayerConfig(
#             conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
#         )
#     ),
#     slstm_block=sLSTMBlockConfig(
#         slstm=sLSTMLayerConfig(
#             backend="cuda",
#             num_heads=4,
#             conv1d_kernel_size=4,
#             bias_init="powerlaw_blockdependent",
#         ),
#         feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
#     ),
#     context_length=128,
#     num_blocks=7,
#     embedding_dim=4096,
#     slstm_at=[1],
#     return_last_state=True

# )


# x = torch.randn(64, 12, 4096).to("cuda")
# xlstm_stack=xLSTMBlockStack(cfg)
# # xlstm_stack = xLSTMLMModel(cfg)

# xlstm_stack = xlstm_stack.to("cuda")
# y = xlstm_stack(x)
# print(f'y {y.shape}')
# y.shape == x.shape



import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore') 
 
class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation, **kwargs)
 
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]
 
 
class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
 
        assert out_features % num_blocks == 0
 
        block_out_features = out_features // num_blocks
 
        self.blocks = nn.ModuleList([
            nn.Linear(in_features, block_out_features)
            for _ in range(num_blocks)
        ])
 
    def forward(self, x):
        x = [block(x) for block in self.blocks]
        x = torch.cat(x, dim=-1)
        return x
 
class mLSTMBlock(nn.Module):
    def __init__(self, input_size, head_size, num_heads, proj_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.proj_factor = proj_factor
 
        assert proj_factor > 0
 
        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, self.hidden_size)
        self.down_proj = nn.Linear(self.hidden_size, input_size)
 
        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), self.hidden_size)
 
        self.Wq = BlockDiagonal(int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), self.hidden_size, num_heads)
        self.Wi = nn.Linear(int(input_size * proj_factor), self.hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), self.hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), self.hidden_size)
 
        self.group_norm = nn.GroupNorm(num_heads, self.hidden_size)
 
    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
 
        h_prev = h_prev.to(x.device)
        c_prev = c_prev.to(x.device)
        n_prev = n_prev.to(x.device)
        m_prev = m_prev.to(x.device)
 
        assert x.size(-1) == self.input_size
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)
 
        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)
 
        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)
 
        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))
 
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
 
        c_t = f * c_prev + i * (v * k)  # v @ k.T
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0]  # o * (c @ q) / max{|n.T @ q|, 1}
 
        output = h_t
        h_t1=self.group_norm(h_t)
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x
 
        return final_output, (h_t, c_t, n_t, m_t)
 
class sLSTMBlock(nn.Module):
    def __init__(self, input_size, head_size, num_heads, proj_factor=4/5): #将3/4改为4/5
        super(sLSTMBlock, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.proj_factor = proj_factor
 
        assert proj_factor > 0
 
        self.layer_norm = nn.LayerNorm(input_size)
        self.causal_conv = CausalConv1D(1, 1, 4)
 
        self.Wz = BlockDiagonal(input_size, self.hidden_size, num_heads)
        self.Wi = BlockDiagonal(input_size, self.hidden_size, num_heads)
        self.Wf = BlockDiagonal(input_size, self.hidden_size, num_heads)
        self.Wo = BlockDiagonal(input_size, self.hidden_size, num_heads)
 
        self.Rz = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
        self.Ri = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
        self.Rf = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
        self.Ro = BlockDiagonal(self.hidden_size, self.hidden_size, num_heads)
 
        self.group_norm = nn.GroupNorm(num_heads, self.hidden_size)
 
        self.up_proj_left = nn.Linear(self.hidden_size, int(self.hidden_size * proj_factor))
        self.up_proj_right = nn.Linear(self.hidden_size, int(self.hidden_size * proj_factor))
        self.down_proj = nn.Linear(int(self.hidden_size * proj_factor), input_size)
 
    def forward(self, x, prev_state):
        assert x.size(-1) == self.input_size
        h_prev, c_prev, n_prev, m_prev = prev_state
 
        h_prev = h_prev.to(x.device)
        c_prev = c_prev.to(x.device)
        n_prev = n_prev.to(x.device)
        m_prev = m_prev.to(x.device)
 
        x_norm = self.layer_norm(x)
        x_conv = F.silu(self.causal_conv(x_norm.unsqueeze(1)).squeeze(1))
 
        z = torch.tanh(self.Wz(x_norm) + self.Rz(h_prev))
        o = torch.sigmoid(self.Wo(x_norm) + self.Ro(h_prev))
        i_tilde = self.Wi(x_conv) + self.Ri(h_prev)
        f_tilde = self.Wf(x_conv) + self.Rf(h_prev)
 
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
 
        c_t = f * c_prev + i * z
        n_t = f * n_prev + i
        h_t = o * c_t / n_t
 
        output = h_t
        # print(f'h_t {h_t.shape}')
        h_t1=self.group_norm(h_t)
        # print(f'h_t1 {h_t1.shape}')

        output_norm = self.group_norm(output)
        output_left = self.up_proj_left(output_norm)
        output_right = self.up_proj_right(output_norm)
        output_gated = F.gelu(output_right)
        output = output_left * output_gated
        output = self.down_proj(output)
        final_output = output + x
 
        return final_output, (h_t, c_t, n_t, m_t)
 
class xLSTM(nn.Module):
    def __init__(self, input_size, head_size, num_heads, layers, batch_first=True, proj_factor_slstm=4 / 3,
                 proj_factor_mlstm=2):
        super(xLSTM, self).__init__()
        self.input_size = input_size
        self.head_size = head_size
        self.hidden_size = head_size * num_heads
        self.num_heads = num_heads
        self.layers = layers
        self.num_layers = len(layers)
        self.batch_first = batch_first
        self.proj_factor_slstm = proj_factor_slstm
        self.proj_factor_mlstm = proj_factor_mlstm
 
        self.layers = nn.ModuleList()
        for layer_type in layers:
            if layer_type == 's':
                layer = sLSTMBlock(input_size, head_size, num_heads, proj_factor_slstm)
            elif layer_type == 'm':
                layer = mLSTMBlock(input_size, head_size, num_heads, proj_factor_mlstm)
            else:
                raise ValueError(f"Invalid layer type: {layer_type}. Choose 's' for sLSTM or 'm' for mLSTM.")
            self.layers.append(layer)
 
    def forward(self, x, state=None):
        assert x.ndim == 3
        if self.batch_first: x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
 
        if state is not None:
            state = torch.stack(list(state)).to(x.device)
            assert state.ndim == 4
            num_hidden, state_num_layers, state_batch_size, state_input_size = state.size()
            assert num_hidden == 4
            assert state_num_layers == self.num_layers
            assert state_batch_size == batch_size
            assert state_input_size == self.input_size
            state = state.transpose(0, 1)
        else:
            state = torch.zeros(self.num_layers, 4, batch_size, self.hidden_size, device=x.device)
 
        output = []
        h_t_all=[]
        for t in range(seq_len):
            x_t = x[t]
            for layer in range(self.num_layers):
                x_t, state_tuple = self.layers[layer](x_t, tuple(state[layer].clone()))
                h_t=state_tuple[0]


                # print(f'x_t {x_t.shape}')
                state[layer] = torch.stack(list(state_tuple))
                # h_t_all=torch.stack(list(h_t))
            # print(f'h_t_all {h_t_all.shape}')
            h_t_all.append(h_t)
            # print(f'h_t {h_t.shape}')
            output.append(x_t)
            
            # h_t_all.append(h_t)
        h_t_all = torch.stack(h_t_all)
            
        # print(type(h_t_all)) 
        # h_t_all=torch.tensor(h_t_all)
        # print(f'h_t_all {h_t_all.shape}')
            
 
        output = torch.stack(output)
        # print(f'output1 {output.shape}')
        if self.batch_first:
            output = output.transpose(0, 1)
            h_t_all=h_t_all.transpose(0, 1)
        state = tuple(state.transpose(0, 1))
        return output, h_t_all,state

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# x = torch.randn(32,12,4096).to(device)  # [4, 4096, 12]
# model = xLSTM(input_size=4096, head_size=512, num_heads=2, batch_first=True, layers='ms').to(device)
 


# # 查看输出形状
# output, h_t_all,state = model(x)
# print("Output size:", output.size())
# print("h_t_all,:", h_t_all.shape)
# print("State size:", [s.size() for s in state])
# ###########################################单变量预测模型##############################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = pd.read_excel('单变量.xlsx', engine='python').drop('date',axis=1)
# dataset = data.values.astype('float32')
# batch_size = 32
# seq_len = 8
 
# scaler = MinMaxScaler(feature_range=(0, 1))
# dataset = scaler.fit_transform(dataset)
 
# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
 
# def create_dataset(dataset, seq_len=8):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - seq_len):
#         a = dataset[i:(i + seq_len - 1)]
#         dataX.append(a)
#         dataY.append(dataset[i + seq_len - 1])
#     return torch.Tensor(dataX), torch.Tensor(dataY)
 
# trainX, trainY = create_dataset(train, seq_len)
# testX, testY = create_dataset(test, seq_len)
 
# train_dataset = TensorDataset(trainX, trainY)
# test_dataset = TensorDataset(testX, testY)
 
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
# input_size = 1
# head_size = 32
# num_heads = 2


# model = xLSTM(input_size=4096, head_size=1024, num_heads=2, batch_first=True, layers='ms').to(device)
 
# def train_model(model, epochs=20, learning_rate=0.01):
#     criterion = nn.MSELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
 
#     train_losses = []
#     for epoch in tqdm(range(epochs), desc='Training xLSTM'):
#         model.train()
#         epoch_loss = 0
#         for i, (inputs, targets) in enumerate(train_loader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             optimizer.zero_grad()
#             outputs, _ = model(inputs)
#             outputs = outputs[:, -1, :]
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
 
#             epoch_loss += loss.item()
 
#         train_losses.append(epoch_loss / len(train_loader))
 
#     return model, train_losses
 
# trained_model, train_losses = train_model(model)
# plt.figure()
# plt.plot(train_losses, label='xLSTM')
# plt.title('Training Losses for xLSTM')
# plt.xlabel('Epochs')
# plt.ylabel('MSE Loss')
# plt.legend()
# plt.show()
 
# def evaluate_model(model, data_loader):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for inputs, _ in data_loader:
#             inputs = inputs.to(device)  
#             outputs, _ = model(inputs)
#             predictions.extend(outputs[:, -1, :].cpu().numpy())  
#     return predictions
 
# test_predictions = evaluate_model(trained_model, test_loader)
# preds = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1))
# actual = scaler.inverse_transform(testY.numpy().reshape(-1, 1))
# rmse = mean_squared_error(actual, preds, squared=False)
# mae = mean_absolute_error(actual, preds)
# mape = mean_absolute_percentage_error(actual, preds)
# r2 = r2_score(actual, preds)
 
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"MAPE: {mape:.4f}")
# print(f"R^2: {r2:.4f}")
 
# plt.figure()
# plt.plot(actual, label='Actual')
# plt.plot(preds, label='xLSTM Predictions')
# plt.title('xLSTM Predictions vs Actual')
# plt.legend()
# plt.show()
 
# ###########################################多变量预测模型##############################################
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# data = pd.read_excel('多变量.xlsx').drop('date',axis=1)  
# target_col_name = 'canopy' 
 
# seq_len = 8
# feature_cols = data.columns.drop(target_col_name)
# target_col = data[target_col_name].values.astype('float32')
# features = data[feature_cols].values.astype('float32')
 
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_features = scaler.fit_transform(features)
# scaled_target = scaler.fit_transform(target_col.reshape(-1, 1))
# dataset = np.hstack((scaled_features, scaled_target))
 
# def create_dataset(dataset, seq_len=8, target_col_idx=-1):
#     dataX, dataY = [], []
#     for i in range(len(dataset) - seq_len):
#         a = dataset[i:(i + seq_len - 1), :-1]  
#         dataX.append(a)
#         dataY.append(dataset[i + seq_len - 1, target_col_idx])  
#     return torch.Tensor(dataX), torch.Tensor(dataY)
 
# train_size = int(len(dataset) * 0.8)
# test_size = len(dataset) - train_size
# train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
 
# trainX, trainY = create_dataset(train, seq_len, target_col_idx=-1)
# testX, testY = create_dataset(test, seq_len, target_col_idx=-1)
 
# train_dataset = TensorDataset(trainX, trainY)
# test_dataset = TensorDataset(testX, testY)
 
# batch_size = 32
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
 
# input_size = len(feature_cols) 
# head_size = 32
# num_heads = 2
# model = xLSTM(input_size, head_size, num_heads, batch_first=True, layers='msm').to(device)
 
# def train_model(model, epochs, learning_rate):
#     optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#     criterion = nn.MSELoss()
 
#     train_losses = []
#     for epoch in tqdm(range(epochs), desc='Training xLSTM'):
#         for inputs, targets in train_loader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             model.train()
#             optimizer.zero_grad()
#             outputs, _ = model(inputs)
#             outputs = outputs[:, -1, :]
#             if len(targets.shape) == 2:
#                 targets = targets 
#             elif len(targets.shape) == 1:
#                 targets = targets.unsqueeze(1) 
#             loss = criterion(outputs, targets)
#             loss.backward()
#             optimizer.step()
 
#         train_losses.append(loss.item())
#         if (epoch+1) % 5 == 0:
#             print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
 
#     return model, train_losses
 
# def evaluate_model(model, data_loader):
#     model.eval()
#     predictions = []
#     with torch.no_grad():
#         for inputs, _ in data_loader:
#             inputs = inputs.to(device)
#             outputs, _ = model(inputs)
#             predictions.extend(outputs[:, -1, :].cpu().numpy())
#     return predictions
 
# trained_model, train_losses = train_model(model, epochs=5, learning_rate=0.01)
# test_predictions = evaluate_model(trained_model, test_loader)
# preds = scaler.inverse_transform(np.array(test_predictions).reshape(-1, input_size))
# actual = scaler.inverse_transform(testY.numpy().reshape(-1, 1))
# rmse = mean_squared_error(actual, preds[:, 0], squared=False)
# mae = mean_absolute_error(actual, preds[:, 0])
# mape = mean_absolute_percentage_error(actual, preds[:, 0])
# r2 = r2_score(actual, preds[:, 0])
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE: {mae:.4f}")
# print(f"MAPE: {mape:.4f}")
# print(f"R^2: {r2:.4f}")
 
# plt.figure()
# plt.plot(actual[:, 0], label='Actual')
# plt.plot(preds[:, 0], label='xLSTM Predictions')  
# plt.title('xLSTM Predictions vs Actual')
# plt.legend()
# plt.show()
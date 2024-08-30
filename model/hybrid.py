import torch
import torch.nn as nn
import torchdiffeq
import torch.nn.functional as F
from model_classes import SEIRM

class HybridODE(nn.Module):
    def __init__(self, zx_dim, zy_dim, ze_dim, action_dim, params, learnable_params, device=None):
        super(HybridODE, self).__init__()
        self.device = device or torch.device('cpu')
        
        # 初始化线性层
        self.linear_zx = nn.Linear(zx_dim + zy_dim + action_dim, zx_dim).to(self.device)
        self.linear_zy = nn.Linear(zy_dim + zx_dim + ze_dim + action_dim, zy_dim).to(self.device)
        self.seirm = SEIRM(params, learnable_params, device=self.device)

        # 初始化参数
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.ze_dim = ze_dim
        self.action_dim = action_dim
        
    def forward(self, t, state):
        # 从 state 中提取 zx, zy, ze, a
        zx = state[:, :self.zx_dim]  # (batch_size, zx_dim)
        zy = state[:, self.zx_dim:self.zx_dim + self.zy_dim]  # (batch_size, zy_dim)
        ze = state[:, self.zx_dim + self.zy_dim:self.zx_dim + self.zy_dim + self.ze_dim]  # (batch_size, ze_dim)
        a = state[:, -self.action_dim:]  # (batch_size, action_dim)
        ze_input = ze.squeeze()

        # # 打印当前时间步的 Zx, Zy, Ze
        # print(f"Time: {t}")
        # print(f"Zx: {zx}")
        # print(f"Zy: {zy}")
        # print(f"Ze: {ze}")

        # 计算 Ze(t) 的导数
        dze_dt = self.seirm(t, ze_input).unsqueeze(0)
        
        # 计算 Zx(t) 的导数并应用 tanh 激活函数
        dzx_dt = torch.tanh(self.linear_zx(torch.cat([zx, zy, a], dim=-1)))
        dzx_dt = torch.clamp(dzx_dt, -1e6, 1e6)  # 防止数值过大或过小

        # 使用更新后的 Zx 和 Ze 计算 Zy(t) 的导数并应用 tanh 激活函数
        dzy_dt = torch.tanh(self.linear_zy(torch.cat([zy, zx + dzx_dt, ze + dze_dt, a], dim=-1)))
        dzy_dt = torch.clamp(dzy_dt, -1e6, 1e6)

        # 返回合并后的导数
        dstate_dt = torch.cat([dzx_dt, dzy_dt, dze_dt, a], dim=-1)
        return dstate_dt



class EncoderLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_zx, output_dim_zy, output_dim_ze, device=None):
        super(EncoderLSTM, self).__init__()
        self.device = device or torch.device('cpu')
        self.hidden_dim = hidden_dim

        # LSTM层
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(self.device)

        # 线性层
        self.g_eta = nn.Linear(hidden_dim, output_dim_ze).to(self.device)  # gη
        self.g_xi = nn.Linear(hidden_dim, output_dim_zx).to(self.device)   # gξ
        self.g_zeta = nn.Linear(hidden_dim + output_dim_zx, output_dim_zy).to(self.device) # gζ
        
        # self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True).to(self.device)

        # # 线性层
        # self.g_eta = nn.Linear(input_dim, output_dim_ze).to(self.device)  # gη
        # self.g_xi = nn.Linear(input_dim, output_dim_zx).to(self.device)   # gξ
        # self.g_zeta = nn.Linear(input_dim + output_dim_zx, output_dim_zy).to(self.device) # gζ

    def forward(self, x, a, y):
        # 合并输入
        # x, a, y: torch.Size([sequence_length, 1])
        # 将时间维度进行平均池化，使得输出为 [1, 1] 大小
        x = x.mean(dim=0, keepdim=True)  # torch.Size([1, 1])
        a = a.mean(dim=0, keepdim=True)  # torch.Size([1, 1])
        y = y.mean(dim=0, keepdim=True)  # torch.Size([1, 1])

        # 合并池化后的输入
        input_concat = torch.cat([x, a, y], dim=-1).unsqueeze(0)  # 添加 batch 维度，变为 torch.Size([1, 3])

        # 通过LSTM层
        lstm_out, _ = self.lstm(input_concat)  # (batch_size=1, sequence_length=1, hidden_dim)
        hidden_state = lstm_out.squeeze(1)  # 移除 sequence_length 维度, 变为 torch.Size([1, hidden_dim])

        # Ze(0) = softmax(gη(hidden_state))
        ze_init = F.softmax(self.g_eta(hidden_state), dim=-1)

        # Zx(0) = gξ(hidden_state)
        zx_init = self.g_xi(hidden_state)

        # Zy(0) = gζ([hidden_state, Zx(0)])
        zy_input = torch.cat([hidden_state, zx_init], dim=-1)
        zy_init = self.g_zeta(zy_input)

        return zx_init, zy_init, ze_init


class HybridDecoder(nn.Module):
    def __init__(self, zx_dim, zy_dim, ze_dim, action_dim, y_dim, x_dim, params, learnable_params, batch_size, step_size, device=None):
        super(HybridDecoder, self).__init__()
        self.device = device or torch.device('cpu')
        self.zx_dim = zx_dim
        self.zy_dim = zy_dim
        self.ze_dim = ze_dim

        self.step_size = step_size

        self.hybrid_ode = HybridODE(zx_dim, zy_dim, ze_dim, action_dim, params, learnable_params, device=self.device)

        # 生成 x(t) 和 y(t) 的神经网络
        self.gy = nn.Linear(ze_dim + zy_dim + zx_dim + action_dim, y_dim).to(self.device)  # g_y 网络
        self.gx = nn.Linear(zx_dim + action_dim, x_dim).to(self.device)

    def forward(self, zx_init, zy_init, ze_init, a, input_length):
        # a: torch.Size([49])
        # zx_init: torch.Size([1, 5])
        remaining_steps = 50 - input_length
        num_steps = remaining_steps // self.step_size
        
        # 创建时间序列
        t = torch.arange(0, self.step_size * num_steps, self.step_size, device=self.device, dtype=torch.float32)

        # 合并 zx_init, zy_init, ze_init, 和 a 作为初始状态
        a_init = a[0].unsqueeze(0)
        if a_init.dim() == 1:
            a_init = a_init.unsqueeze(-1)

        initial_state = torch.cat([zx_init, zy_init, ze_init, a_init], dim=-1)

        # 使用 torchdiffeq.odeint 求解 ODE
        state_trajectory = torchdiffeq.odeint(self.hybrid_ode, initial_state, t, method='dopri5')

        # 从状态中提取 zx, zy 和 ze
        zx_output = state_trajectory[:, 0, :self.zx_dim]
        zy_output = state_trajectory[:, 0, self.zx_dim:self.zx_dim + self.zy_dim]
        ze_output = state_trajectory[:, 0, self.zx_dim + self.zy_dim:self.zx_dim + self.zy_dim + self.ze_dim]

        # 生成 x(t) 和 y(t)
        # zx_output: torch.Size([49, 5])
        # a[input_length:].unsqueeze(-1)] torch.Size([49, 1])
        y_t = self.gy(torch.cat([ze_output, zy_output, zx_output, a.unsqueeze(-1)], dim=-1))
        x_t = self.gx(torch.cat([zx_output, a.unsqueeze(-1)], dim=-1))

        return x_t, y_t



class HybridModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(HybridModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # print("Initial model parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"Layer: {name} | Initial Value: {param.data}")

    def forward(self, x, a, y, input_length):
        # 根据 input_length 选择输入的时间点
        x_input = x[:input_length]
        y_input = y[:input_length]
        a_input = a[:input_length]

        # 编码器步骤，根据 input_length 使用前几个时间点生成初始值
        zx_init, zy_init, ze_init = self.encoder(x_input, a_input, y_input)
        
        # 解码器步骤，预测剩余时间点的值
        x_pred, y_pred = self.decoder(zx_init, zy_init, ze_init, a[input_length:], input_length)
        
        return x_pred, y_pred

    def loss(self, x_data, y_data, a_data, input_length):
        # 根据 input_length 计算 loss
        # x_data: torch.Size([50])
        # a: torch.Size([50])
        x_pred, y_pred = self.forward(x_data, a_data, y_data, input_length)
        
        x_pred = x_pred.squeeze()
        y_pred = y_pred.squeeze()
        loss_x = nn.MSELoss()(x_pred, x_data[input_length:])
        loss_y = nn.MSELoss()(y_pred, y_data[input_length:])
        rmse_loss_x = torch.sqrt(loss_x)
        rmse_loss_y = torch.sqrt(loss_y)

        return rmse_loss_x, rmse_loss_y


class ExpertEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim_ze, device=None):
        super(ExpertEncoder, self).__init__()
        self.device = device or torch.device('cpu')
        self.hidden_dim = hidden_dim

        # LSTM layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True).to(self.device)

        # Linear layer for ze_init
        self.g_eta = nn.Linear(hidden_dim, output_dim_ze).to(self.device)

    def forward(self, x, a, y):
        # Combine inputs and mean pooling over time
        x = x.mean(dim=0, keepdim=True)
        a = a.mean(dim=0, keepdim=True)
        y = y.mean(dim=0, keepdim=True)

        # Concatenate the pooled inputs
        input_concat = torch.cat([x, a, y], dim=-1).unsqueeze(0)

        # Pass through LSTM layer
        lstm_out, _ = self.lstm(input_concat)
        hidden_state = lstm_out.squeeze(1)

        # Ze(0) = softmax(gη(hidden_state))
        ze_init = F.softmax(self.g_eta(hidden_state), dim=-1)

        return ze_init

class ExpertDecoder(nn.Module):
    def __init__(self, ze_dim, y_dim, params, learnable_params, device=None):
        super(ExpertDecoder, self).__init__()
        self.device = device or torch.device('cpu')
        self.ze_dim = ze_dim

        self.seirm = SEIRM(params, learnable_params, device=self.device)

        # Neural network to generate y(t)
        self.gy = nn.Linear(ze_dim, y_dim).to(self.device)

    def forward(self, ze_init, t):
        # Solve ODE using SEIRM
        ze_input = ze_init.squeeze(0)
        ze_output = torchdiffeq.odeint(self.seirm, ze_input, t, method='dopri5')

        # Generate y(t)
        y_t = self.gy(ze_output)

        return y_t


class ExpertModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(ExpertModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, a, y, input_length):
        # Select inputs based on input_length
        x_input = x[:input_length]
        y_input = y[:input_length]
        a_input = a[:input_length]

        # Encoder step
        ze_init = self.encoder(x_input, a_input, y_input)

        # Time series
        remaining_steps = 50 - input_length
        t = torch.arange(0, remaining_steps, 1, device=self.encoder.device, dtype=torch.float32)

        # Decoder step
        y_pred = self.decoder(ze_init, t)

        return y_pred

    def loss(self, y_data, a_data, input_length):
        # Compute loss
        y_pred = self.forward(y_data, a_data, y_data, input_length)
        y_pred = y_pred.squeeze()

        loss_y = nn.MSELoss()(y_pred, y_data[input_length:])
        rmse_loss_y = torch.sqrt(loss_y)

        return rmse_loss_y

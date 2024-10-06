import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from model_classes import SEIRM

# 加载配置
from sim_config import OptimConfig

# 加载数据
def load_data(csv_path):
    data = pd.read_csv(csv_path)
    # 提取SEIRM相关数据
    susceptible = torch.tensor(data['Susceptible'].values, dtype=torch.float32)
    exposed = torch.tensor(data['Exposed'].values, dtype=torch.float32)
    infectious_a = torch.tensor(data['Infectious_asymptomatic'].values, dtype=torch.float32)
    recovered = torch.tensor(data['Recovered'].values, dtype=torch.float32)  # 新增的Recovered
    deceased = torch.tensor(data['Deceased'].values, dtype=torch.float32)
    
    # stack成一个tensor: (S, E, I, R, M)
    y_true = torch.stack([susceptible, exposed, infectious_a, recovered, deceased], dim=-1)
    
    # 根据batch size进行reshape, 这里的假设是每个周期为72个时间步长
    num_batches = y_true.shape[0] // 72
    y_true = y_true[:num_batches * 72].reshape(num_batches, 72, 5)
    
    return y_true

# 定义损失函数（MSE）
def compute_loss(y_pred, y_true):
    # 我们只计算M的误差，也就是Deceased
    loss_fn = nn.MSELoss()
    return loss_fn(y_pred[..., 4], y_true[..., 4])

# 训练函数
def train_ode_model(data_path, save_path, plot_save_path=None):
    # 加载配置
    config = OptimConfig()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    y_true = load_data(data_path).to(device)

    # 初始化ODE模型
    learnable_params = {'beta': 0.5, 'alpha': 0.1, 'gamma': 0.05, 'mu': 0.01, 'initial_infections_percentage': 0.001}
    model = SEIRM({}, learnable_params, device).to(device)

    # 初始化优化器
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    # 训练循环
    for epoch in range(config.niters):
        model.train()
        epoch_loss = 0.0
        all_y_pred = []
        all_y_true = []
        
        for batch_idx in range(y_true.size(0)):
            # 取出当前batch的真实数据
            y_batch_true = y_true[batch_idx]
            
            # 第一个 timestep 作为初始状态 (t=0)
            state = y_batch_true[0]  # 形状: torch.Size([1, 5])

            # 定义时间范围 (72个 timestep)
            t = torch.linspace(0, 71, 72, device=device)
            
            # 用 odeint 计算从 t=0 开始，生成接下来的 71 个 timestep 的预测值
            y_pred = odeint(model, state, t, method=config.ode_method).squeeze()
            
            # 计算第1个时间步后的预测值 (即 1~71的值，与ground truth对比)
            loss = compute_loss(y_pred[1:], y_batch_true[1:])
            
            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # 打印每个batch的损失
            print(f"Batch [{batch_idx+1}/{y_true.size(0)}], Batch Loss: {loss.item():.6f}")
            
            # 累加epoch的总损失
            epoch_loss += loss.item()
            
            # 在最后一个epoch保存预测和真实值用于后续可视化
            if epoch == config.niters - 1:
                all_y_pred.append(y_pred[1:, 4])  # 仅保存 M 的预测值
                all_y_true.append(y_batch_true[1:, 4])  # 仅保存 M 的真实值
        
        # 计算并打印平均损失
        average_epoch_loss = epoch_loss / y_true.size(0)
        print(f"Epoch [{epoch+1}/{config.niters}], Average Loss: {average_epoch_loss:.6f}")
        
        # 保存模型
        if epoch == config.niters - 1:
            torch.save(model.state_dict(), save_path)
    
    print(f"模型训练完成，模型已保存到: {save_path}")
    
    # 在最后一个epoch打印预测和真实值并绘制图形
    if all_y_pred and all_y_true:
        all_y_pred = torch.cat(all_y_pred, dim=0)
        all_y_true = torch.cat(all_y_true, dim=0)
        
        # 打印预测值和真实值
        print("Predicted values:", all_y_pred)
        print("Ground Truth values:", all_y_true)

# 主函数，进行训练
if __name__ == "__main__":
    data_path = "/home/zhicao/ODE/data/weekly_data_with_treatment.csv"
    save_path = "/home/zhicao/ODE/model/expert_checkpoint.pth"
    
    train_ode_model(data_path, save_path, plot_save_path)

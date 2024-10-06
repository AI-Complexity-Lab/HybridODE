import torch
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from torchdiffeq import odeint
from model_classes import SEIRM
from sim_config import DataConfig, SEIRMConfig, ModelConfig, OptimConfig
from hybrid import HybridDecoder, EncoderLSTM, HybridModel, ExpertEncoder, ExpertModel, ExpertDecoder

def load_test_data(result_csv, noisy_deceased_csv, device):
    result_df = pd.read_csv(result_csv)
    noisy_deceased_df = pd.read_csv(noisy_deceased_csv)
    
    start_idx = 0
    end_idx = start_idx + 71
    
    y_data = torch.tensor(result_df['Deceased'].values[start_idx:end_idx], dtype=torch.float32).to(device)
    x_data = torch.tensor(noisy_deceased_df['x'].values[start_idx:end_idx], dtype=torch.float32).to(device)
    a_data = torch.tensor(result_df['a'].values[start_idx:end_idx], dtype=torch.float32).to(device)
    
    return x_data, y_data, a_data

def initialize_model(data_config, seirm_config, model_config, batch_size, device, expert=False):
    if expert:
        encoder = ExpertEncoder(
            input_dim=3, 
            hidden_dim=int(data_config.obs_dim * model_config.encoder_latent_ratio),
            output_dim_ze=data_config.latent_dim,
            device=device
        )

        decoder = ExpertDecoder(
            ze_dim=data_config.latent_dim,
            y_dim=data_config.obs_dim,
            params=seirm_config._asdict(),
            learnable_params={
                "beta": torch.tensor(seirm_config.beta).to(device), 
                "alpha": torch.tensor(seirm_config.alpha).to(device), 
                "gamma": torch.tensor(seirm_config.gamma).to(device), 
                "mu": torch.tensor(seirm_config.mu).to(device)
            }, 
            device=device
        )

        model = ExpertModel(encoder, decoder).to(device)
    else:
        encoder = EncoderLSTM(
            input_dim=3, 
            hidden_dim=int(data_config.obs_dim * model_config.encoder_latent_ratio),
            output_dim_zx=data_config.latent_dim,
            output_dim_zy=data_config.latent_dim,
            output_dim_ze=data_config.latent_dim,
            device=device
        )

        decoder = HybridDecoder(
            zx_dim=data_config.latent_dim,
            zy_dim=data_config.latent_dim,
            ze_dim=data_config.latent_dim,
            action_dim=data_config.action_dim,
            y_dim=data_config.obs_dim,
            x_dim=data_config.obs_dim,
            step_size=data_config.step_size,
            params=seirm_config._asdict(),
            batch_size=batch_size,
            learnable_params={
                "beta": torch.tensor(seirm_config.beta).to(device), 
                "alpha": torch.tensor(seirm_config.alpha).to(device), 
                "gamma": torch.tensor(seirm_config.gamma).to(device), 
                "mu": torch.tensor(seirm_config.mu).to(device)
            }, 
            device=device
        )

        model = HybridModel(encoder, decoder).to(device)

    return model


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
    
    return y_true

# 计算MSE损失
def compute_loss(y_pred, y_true):
    """
    计算MSE损失，使用Deceased列 (M)。
    """
    loss_fn = torch.nn.MSELoss()
    return loss_fn(y_pred, y_true)

# 绘制两个模型结果图
def plot_combined_results(y_true, y_pred_expert, y_pred_hybrid, total_population, save_path=None):
    """
    将ExpertODE和HybridODE的预测结果与真实结果结合绘制在同一个图中。
    
    参数：
    - y_true (Tensor): 真实的死亡人数
    - y_pred_expert (Tensor): ExpertODE的预测结果
    - y_pred_hybrid (Tensor): HybridODE的预测结果
    - total_population (int): 总人口数
    - save_path (str, optional): 保存图像的路径。如果为 None，则显示图像。
    """
    # 将Tensor转换为CPU上的NumPy数组
    y_true_np = y_true.cpu().numpy()
    y_pred_expert_np = y_pred_expert.detach().cpu().numpy()
    y_pred_hybrid_np = y_pred_hybrid.detach().cpu().numpy()
    
    # 转换为实际死亡人数
    y_true_counts = y_true_np * total_population
    y_pred_expert_counts = y_pred_expert_np * total_population
    y_pred_hybrid_counts = y_pred_hybrid_np * total_population
    
    # 定义时间步
    time_steps = range(len(y_true_counts))
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_true_counts, label='Ground Truth', marker='o', color='blue')
    plt.plot(time_steps, y_pred_expert_counts, label='ExpertODE', marker='x', color='orange')
    plt.plot(time_steps, y_pred_hybrid_counts, label='HybridODE', marker='x', color='red')
    
    plt.xlabel('Time Step')
    plt.ylabel('Number of Deceased')
    plt.title('ExpertODE vs HybridODE vs Ground Truth')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

def run_expertODE(result_csv, model_path, device):
    # 加载数据
    y_true = load_data(result_csv).to(device)
    y_true_first_71 = y_true[:71]

    # 加载并推理ExpertODE
    learnable_params = {'beta': 0.5, 'alpha': 0.1, 'gamma': 0.05, 'mu': 0.01, 'initial_infections_percentage': 0.001}
    model = SEIRM({}, learnable_params, device).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 使用第一个时间步作为初始状态
    state = y_true_first_71[0]
    t = torch.linspace(0, 70, 71, device=device)
    y_pred_expert = odeint(model, state, t).squeeze()[:, 4]  # 仅取死亡人数
    
    return y_pred_expert, y_true_first_71[:, 4]

def run_hybridODE(result_csv, noisy_deceased_csv, model_path, data_config, seirm_config, model_config, device):
    # 加载测试数据
    x_data, y_data, a_data = load_test_data(result_csv, noisy_deceased_csv, device)
    
    # 初始化HybridODE模型
    model = initialize_model(data_config, seirm_config, model_config, batch_size=50, device=device, expert=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        y_pred, x_pred = model(x_data, y_data, a_data, input_length=10)
    
    return y_pred, y_data

if __name__ == "__main__":
    # 配置路径
    result_csv = "/home/zhicao/ODE/data/weekly_data_with_treatment.csv"
    noisy_deceased_csv = "/home/zhicao/ODE/data/weekly_noisy_deceased.csv"
    expert_model_path = "/home/zhicao/ODE/model/expert_checkpoint.pth"
    hybrid_model_path = "/home/zhicao/ODE/model/trained_model.pth"
    save_plot_path = "/home/zhicao/ODE/model/combined_visualization.png"

    # 配置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # SEIRM配置
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    model_config = ModelConfig()

    # 运行ExpertODE推理
    y_pred_expert, y_true = run_expertODE(result_csv, expert_model_path, device)
    
    # 运行HybridODE推理
    y_pred_hybrid, _ = run_hybridODE(result_csv, noisy_deceased_csv, hybrid_model_path, data_config, seirm_config, model_config, device)

    # 绘制结果
    total_population = 1938000
    plot_combined_results(y_true, y_pred_expert, y_pred_hybrid, total_population, save_plot_path)

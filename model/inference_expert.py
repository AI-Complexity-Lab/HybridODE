import torch
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
    
    return y_true

# 绘制结果图
def plot_results(y_pred, y_true, total_population, save_path=None):
    """
    绘制预测结果与真实结果的对比图。
    
    参数：
    - y_pred (Tensor): 模型预测的死亡比例
    - y_true (Tensor): 真实的死亡比例
    - total_population (int): 总人口数
    - save_path (str, optional): 保存图像的路径。如果为 None，则显示图像。
    """
    # 将Tensor转换为CPU上的NumPy数组
    y_pred_np = y_pred.detach().cpu().numpy()
    y_true_np = y_true.detach().cpu().numpy()
    
    # 转换为实际死亡人数
    y_pred_counts = y_pred_np * total_population
    y_true_counts = y_true_np * total_population
    
    # 定义时间步
    time_steps = range(len(y_true_counts))
    
    # 绘制图形
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, y_true_counts, label='Ground Truth', marker='o')
    plt.plot(time_steps, y_pred_counts, label='Prediction', marker='x')
    plt.xlabel('Time Step')
    plt.ylabel('Number of Deceased')
    plt.title('Prediction vs Ground Truth (First 71 Time Steps)')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

# 计算MSE损失
def compute_loss(y_pred, y_true):
    """
    计算MSE损失，使用Deceased列 (M)。
    """
    loss_fn = torch.nn.MSELoss()
    return loss_fn(y_pred, y_true)

# 加载模型并绘制预测结果
def test_and_plot(model_path, data_path, plot_save_path=None):
    # 加载配置
    config = OptimConfig()

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载数据
    y_true = load_data(data_path).to(device)

    # 取整个数据表格的前71个时间步作为ground truth
    y_true_first_71 = y_true[:71]  # 形状: torch.Size([71, 5])

    # 初始化ODE模型
    learnable_params = {'beta': 0.5, 'alpha': 0.1, 'gamma': 0.05, 'mu': 0.01, 'initial_infections_percentage': 0.001}
    model = SEIRM({}, learnable_params, device).to(device)

    # 加载训练好的模型
    model.load_state_dict(torch.load(model_path))
    model.eval()

    total_population = 1938000  # 使用的总人口数

    # 使用前 71 个时间步中的第一个 timestep 作为初始状态 (t=0)
    state = y_true_first_71[0]  # 形状: torch.Size([5])

    # 定义时间范围 (71个 timestep)
    t = torch.linspace(0, 70, 71, device=device)
    
    # 用 odeint 计算从 t=0 开始，生成接下来的 70 个 timestep 的预测值
    y_pred = odeint(model, state, t, method=config.ode_method).squeeze()
    
    # 仅保存 M (死亡人数) 的预测值和真实值
    y_pred_first_71 = y_pred[:, 4]  # 模型预测的所有71个时间步的死亡人数
    y_true_first_71 = y_true_first_71[:, 4]  # 真实的前71个时间步的死亡人数
    
    # 打印预测值和真实值
    print("Predicted values (first 71):", y_pred_first_71)
    print("Ground Truth values (first 71):", y_true_first_71)

    # 计算测试损失
    test_loss = compute_loss(y_pred_first_71, y_true_first_71)
    print(f"Test Loss: {test_loss.item():.6f}")

    # 绘制并保存图形
    plot_results(y_pred_first_71, y_true_first_71, total_population, plot_save_path)

# 主函数，进行模型加载和绘图
if __name__ == "__main__":
    model_path = "/home/zhicao/ODE/model/expert_checkpoint.pth"
    data_path = "/home/zhicao/ODE/data/weekly_data_with_treatment.csv"
    plot_save_path = "/home/zhicao/ODE/model/visualization_expert.png"
    
    test_and_plot(model_path, data_path, plot_save_path)

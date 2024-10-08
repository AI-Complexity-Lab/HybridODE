import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sim_config import DataConfig, SEIRMConfig, ModelConfig, OptimConfig, EvalConfig
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

def run_inference(result_csv, noisy_deceased_csv, model_path, data_config, seirm_config, model_config, device, expert=False, input_length = 10):
    """
    运行模型推理并返回预测结果和真实数据。

    参数：
    - result_csv (str): 结果CSV文件路径。
    - noisy_deceased_csv (str): 含噪声死亡数据的CSV文件路径。
    - model_path (str): 训练好的模型参数文件路径。
    - data_config (DataConfig): 数据配置。
    - seirm_config (SEIRMConfig): SEIRM模型配置。
    - model_config (ModelConfig): 模型配置。
    - device (torch.device): 运行设备。
    - expert (bool): 是否使用专家模型。

    返回：
    - expert=True: y_pred, y_true
    - expert=False: y_pred, y_true, x_pred, x_true
    """
    # 加载测试数据
    x_data, y_data, a_data = load_test_data(result_csv, noisy_deceased_csv, device)
    
    # 初始化模型
    model = initialize_model(data_config, seirm_config, model_config, batch_size=50, device=device, expert=expert)
    
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    with torch.no_grad():
        if expert:
            y_pred = model(y_data, a_data, y_data, input_length)
            
            # 计算损失
            input_length = 10
            # test_loss_y = model.loss(y_data, a_data, input_length)
            # print(f'Test Loss Y: {test_loss_y.item()}')
            print(y_data)
            return y_pred, y_data
        else:
            y_pred, x_pred = model(x_data, y_data, a_data, input_length)
            input_length = 10
            test_loss_x, test_loss_y = model.loss(x_data, y_data, a_data, input_length)
            print(f'Test Loss X: {test_loss_x.item()}, Test Loss Y: {test_loss_y.item()}')
            return y_pred, y_data, x_pred, x_data

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
    y_pred_np = y_pred.cpu().numpy()
    y_true_np = y_true.cpu().numpy()
    
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
    plt.title('Prediction vs Ground Truth')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--result_csv", default="/home/zhicao/ODE/data/weekly_10_data.csv", type=str, help="Path to result CSV file")
    parser.add_argument("--noisy_deceased_csv", default="/home/zhicao/ODE/data/weekly_10_covariate.csv", type=str, help="Path to noisy deceased CSV file")
    parser.add_argument("--model_path", default="/home/zhicao/ODE/model/trained_model.pth", type=str, help="Path to trained model parameters")
    parser.add_argument("--device", choices=["0", "1", "c"], default="1", type=str, help="Device to use: '0', '1', or 'c' for CPU")
    parser.add_argument("--expert", default=False)
    parser.add_argument("--save_plot", default="/home/zhicao/ODE/model/visualization", type=str, help="Path to save the plot image (e.g., 'plot.png')")
    
    args = parser.parse_args()
    

    if args.device.lower() == "c":
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    model_config = ModelConfig()
    
    
    if args.expert:
        y_pred, y_true = run_inference(
            result_csv=args.result_csv,
            noisy_deceased_csv=args.noisy_deceased_csv,
            model_path=args.model_path,
            data_config=data_config,
            seirm_config=seirm_config,
            model_config=model_config,
            device=device,
            expert=True
        )
        
        total_population = 1938000
        plot_results(y_pred, y_true, total_population, save_path=args.save_plot)
    else:
        y_pred, y_true, x_pred, x_true = run_inference(
            result_csv=args.result_csv,
            noisy_deceased_csv=args.noisy_deceased_csv,
            model_path=args.model_path,
            data_config=data_config,
            seirm_config=seirm_config,
            model_config=model_config,
            device=device,
            expert=False
        )
        
        total_population = 1938000
        plot_results(y_pred, y_true, total_population, save_path=args.save_plot)
        
        # if args.save_plot:
        #     save_path_x = args.save_plot.replace('.png', '_x.png')
        # else:
        #     save_path_x = None
        # plot_results(x_pred, x_true, total_population, save_path=save_path_x)

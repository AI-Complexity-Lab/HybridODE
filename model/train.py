import argparse
import torch
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sim_config import DataConfig, SEIRMConfig, ModelConfig, OptimConfig, EvalConfig
from hybrid import HybridDecoder, EncoderLSTM, HybridModel, ExpertEncoder, ExpertModel, ExpertDecoder

def load_data(result_csv, noisy_deceased_csv, batch_size, device):
    result_df = pd.read_csv(result_csv)
    noisy_deceased_df = pd.read_csv(noisy_deceased_csv)
    
    total_points = 500
    num_batches = total_points // batch_size
    
    x_batches = []
    y_batches = []
    a_batches = []

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        
        y_data = torch.tensor(result_df['Deceased'].values[start_idx:end_idx], dtype=torch.float32).to(device)
        x_data = torch.tensor(noisy_deceased_df['x'].values[start_idx:end_idx], dtype=torch.float32).to(device)
        
        a_data = torch.zeros(batch_size, dtype=torch.float32).to(device)
        if start_idx >= 200:
            a_data[:] = 1.0
        elif end_idx > 200:
            a_data[200 - start_idx:] = 1.0
        
        x_batches.append(x_data)
        y_batches.append(y_data)
        a_batches.append(a_data)
    return x_batches, y_batches, a_batches

def initialize_model(data_config, seirm_config, model_config, batch_size, device, expert):
    if expert:
        print(111111111)
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

def train_and_evaluate(model, x_batches, y_batches, a_batches, batch_size, optim_config, device, expert):
    optimizer = optim.Adam(model.parameters(), lr=optim_config.lr)
    model.train()

    num_batches = len(x_batches)
    input_lengths = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45]

    if expert:
        epoch_losses_y = []

        for epoch in range(optim_config.niters):
            total_loss_y_epoch = 0

            for batch_idx in range(num_batches):
                optimizer.zero_grad()

                y_data = y_batches[batch_idx]
                a_data = a_batches[batch_idx]

                total_loss_y = 0
                
                for input_length in input_lengths:
                    loss_y = model.loss(y_data, a_data, input_length)
                    total_loss_y += loss_y
                
                total_loss_y.backward()
                optimizer.step()

                total_loss_y_epoch += total_loss_y.item()

                print(f'Epoch {epoch}, Batch {batch_idx}, Loss Y: {loss_y.item()}')

            avg_loss_y = total_loss_y_epoch / (num_batches * len(input_lengths))
            epoch_losses_y.append(avg_loss_y)

            print(f'Epoch {epoch} Average Loss Y: {avg_loss_y}')

        # 绘制并保存 Y 损失图
        plt.figure()
        plt.plot(range(optim_config.niters), epoch_losses_y, label="Average Y Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (Y)")
        plt.title("Average Y Loss per Epoch")
        plt.legend()
        plt.savefig('/home/zhicao/ODE/model/expert_average_loss_y_per_epoch.png')
        plt.close()

        # 保存模型参数
        model_save_path = '/home/zhicao/ODE/model/trained_expert_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model parameters saved to {model_save_path}")
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_loss_y = model.loss(y_batches[-1], a_batches[-1], 1)
            print(f'Test Loss Y: {test_loss_y.item()}')
            return epoch_losses_y, test_loss_y

    else:
        epoch_losses_x = []
        epoch_losses_y = []

        for epoch in range(optim_config.niters):
            total_loss_x_epoch = 0
            total_loss_y_epoch = 0

            for batch_idx in range(num_batches):
                optimizer.zero_grad()

                x_data = x_batches[batch_idx]
                y_data = y_batches[batch_idx]
                a_data = a_batches[batch_idx]

                total_loss_x = 0
                total_loss_y = 0
                
                for input_length in input_lengths:
                    loss_x, loss_y = model.loss(x_data, y_data, a_data, input_length)
                    total_loss_x += loss_x
                    total_loss_y += loss_y
                
                total_loss = total_loss_x + total_loss_y
                total_loss.backward()
                optimizer.step()

                total_loss_x_epoch += total_loss_x.item()
                total_loss_y_epoch += total_loss_y.item()

                print(f'Epoch {epoch}, Batch {batch_idx}, Loss X: {loss_x.item()}, Loss Y: {loss_y.item()}')

            avg_loss_x = total_loss_x_epoch / (num_batches * len(input_lengths))
            avg_loss_y = total_loss_y_epoch / (num_batches * len(input_lengths))
            
            epoch_losses_x.append(avg_loss_x)
            epoch_losses_y.append(avg_loss_y)

            print(f'Epoch {epoch} Average Loss X: {avg_loss_x}, Average Loss Y: {avg_loss_y}')

        # 绘制并保存 X 和 Y 损失图
        plt.figure()
        plt.plot(range(optim_config.niters), epoch_losses_x, label="Average X Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (X)")
        plt.title("Average X Loss per Epoch")
        plt.legend()
        plt.savefig('/home/zhicao/ODE/model/average_loss_x_per_epoch.png')
        plt.close()

        plt.figure()
        plt.plot(range(optim_config.niters), epoch_losses_y, label="Average Y Loss per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("RMSE (Y)")
        plt.title("Average Y Loss per Epoch")
        plt.legend()
        plt.savefig('/home/zhicao/ODE/model/average_loss_y_per_epoch.png')
        plt.close()

        # 保存模型参数
        model_save_path = '/home/zhicao/ODE/model/trained_model.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f"Model parameters saved to {model_save_path}")
        
        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_loss_x, test_loss_y = model.loss(x_batches[-1], y_batches[-1], a_batches[-1], 1)
            print(f'Test Loss X: {test_loss_x.item()}, Test Loss Y: {test_loss_y.item()}')
            return epoch_losses_x, epoch_losses_y, test_loss_x, test_loss_y
        

def run(result_csv, noisy_deceased_csv, data_config, seirm_config, model_config, optim_config, eval_config, device, expert):
    batch_size = optim_config.batch_size

    x_batches, y_batches, a_batches = load_data(result_csv, noisy_deceased_csv, batch_size, device)
    
    model = initialize_model(data_config, seirm_config, model_config, batch_size, device=device, expert=expert)
    
    train_and_evaluate(model, x_batches, y_batches, a_batches, batch_size, optim_config=optim_config, device=device, expert=expert)
    
    print("Training completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference and Training")
    parser.add_argument("--result_csv", default="/home/zhicao/ODE/data/result.csv", type=str)
    parser.add_argument("--noisy_deceased_csv", default="/home/zhicao/ODE/data/noisy_deceased.csv", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="1", type=str)
    parser.add_argument("--expert", default=False)
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    model_config = ModelConfig()
    optim_config = OptimConfig()
    eval_config = EvalConfig()
    
    run(
        result_csv=args.result_csv,
        noisy_deceased_csv=args.noisy_deceased_csv,
        data_config=data_config,
        seirm_config=seirm_config,
        model_config=model_config,
        optim_config=optim_config,
        eval_config=eval_config,
        device=device,
        expert=args.expert
    )

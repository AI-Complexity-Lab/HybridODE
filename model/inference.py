import argparse
import torch
import pandas as pd
from sim_config import DataConfig, SEIRMConfig, ModelConfig, OptimConfig, EvalConfig
from hybrid import HybridDecoder, EncoderLSTM, HybridModel, ExpertEncoder, ExpertModel, ExpertDecoder

def load_test_data(result_csv, noisy_deceased_csv, device):
    result_df = pd.read_csv(result_csv)
    noisy_deceased_df = pd.read_csv(noisy_deceased_csv)
    
    start_idx = 500
    end_idx = start_idx + 50
    
    y_data = torch.tensor(result_df['Deceased'].values[start_idx:end_idx], dtype=torch.float32).to(device)
    x_data = torch.tensor(noisy_deceased_df['x'].values[start_idx:end_idx], dtype=torch.float32).to(device)
    
    a_data = torch.zeros(50, dtype=torch.float32).to(device)
    a_data[:] = 1.0
    
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

def run_inference(result_csv, noisy_deceased_csv, model_path, data_config, seirm_config, model_config, device, expert=False):
    # Load test data
    x_data, y_data, a_data = load_test_data(result_csv, noisy_deceased_csv, device)
    
    # Initialize the model
    model = initialize_model(data_config, seirm_config, model_config, batch_size=50, device=device, expert=expert)
    
    # Load the trained model parameters
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    with torch.no_grad():
        input_length = 1  # You can adjust this as needed
        test_loss_x, test_loss_y = model.loss(x_data, y_data, a_data, input_length)
        print(f'Test Loss X: {test_loss_x.item()}, Test Loss Y: {test_loss_y.item()}')

    return test_loss_x, test_loss_y

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Inference")
    parser.add_argument("--result_csv", default="/home/zhicao/ODE/data/result.csv", type=str)
    parser.add_argument("--noisy_deceased_csv", default="/home/zhicao/ODE/data/noisy_deceased.csv", type=str)
    parser.add_argument("--model_path", default="/home/zhicao/ODE/model/trained_model.pth", type=str)
    parser.add_argument("--device", choices=["0", "1", "c"], default="1", type=str)
    parser.add_argument("--expert", action='store_true', help="Use the expert model instead of the hybrid model")
    
    args = parser.parse_args()
    
    device = torch.device("cuda:" + args.device if args.device != "c" and torch.cuda.is_available() else "cpu")
    
    data_config = DataConfig()
    seirm_config = SEIRMConfig()
    model_config = ModelConfig()
    
    run_inference(
        result_csv=args.result_csv,
        noisy_deceased_csv=args.noisy_deceased_csv,
        model_path=args.model_path,
        data_config=data_config,
        seirm_config=seirm_config,
        model_config=model_config,
        device=device,
        expert=args.expert
    )

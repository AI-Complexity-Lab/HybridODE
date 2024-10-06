import torch
from hybrid import ExpertDecoder, ExpertEncoder, ExpertModel
from model_classes import SEIRM

# 加载模型路径
model_path = "/home/zhicao/ODE/model/trained_expert_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化模型
seirm_params = {
    'beta': 0.2,
    'alpha': 0.1,
    'gamma': 0.05,
    'mu': 0.01
}
learnable_params = {
    "beta": torch.tensor(seirm_params['beta'], device=device), 
    "alpha": torch.tensor(seirm_params['alpha'], device=device), 
    "gamma": torch.tensor(seirm_params['gamma'], device=device), 
    "mu": torch.tensor(seirm_params['mu'], device=device)
}

encoder = ExpertEncoder(input_dim=3, hidden_dim=64, output_dim_ze=5, device=device)
decoder = ExpertDecoder(ze_dim=5, y_dim=1, params=seirm_params, learnable_params=learnable_params, device=device)

model = ExpertModel(encoder, decoder).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# 打印更新后的 beta, alpha, gamma, mu 的值
print("Updated beta:", model.decoder.seirm.beta.item())
print("Updated alpha:", model.decoder.seirm.alpha.item())
print("Updated gamma:", model.decoder.seirm.gamma.item())
print("Updated mu:", model.decoder.seirm.mu.item())

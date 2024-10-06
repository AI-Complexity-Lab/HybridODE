import torch
import numpy as np
import pandas as pd
from torchdiffeq import odeint
import torch.nn as nn
import torch.optim as optim
import math

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_initial_conditions():
    populations = [1938000, 1500000, 1000000, 500000, 300000]
    ratios = [pop / 1938000 for pop in populations]
    
    base_conditions = [
        [1937829, 100, 50, 5, 0, 0, 5, 5, 6, 0],
        [1937050, 500, 200, 50, 100, 20, 15, 10, 50, 5],
        [1936085, 1000, 300, 100, 200, 50, 30, 20, 200, 15],
        [1934780, 1500, 500, 200, 300, 100, 50, 40, 500, 30],
        [1933330, 2000, 800, 300, 400, 150, 100, 70, 800, 50]
    ]
    
    initial_conditions = []
    for ratio in ratios:
        for base in base_conditions:
            y0 = [int(value * ratio) if i != 0 else int(populations[ratios.index(ratio)] - sum([int(value * ratio) for i, value in enumerate(base[1:])])) for i, value in enumerate(base)]
            initial_conditions.append(torch.tensor(y0, dtype=torch.float32, device=DEVICE))
    
    while len(initial_conditions) < 25:
        initial_conditions.append(initial_conditions[len(initial_conditions) % len(base_conditions)])
    
    return initial_conditions[:25]

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.N = torch.tensor(1938000.0, device=DEVICE)
        self.Ca = nn.Parameter(torch.tensor(0.425, device=DEVICE))
        self.Cp = torch.tensor(1.0, device=DEVICE)
        self.Cm = torch.tensor(1.0, device=DEVICE)
        self.Cs = torch.tensor(1.0, device=DEVICE)
        self.alpha = torch.tensor(0.4875, device=DEVICE)
        self.delta = nn.Parameter(torch.tensor(0.1375, device=DEVICE))
        self.mu = torch.tensor(0.928125, device=DEVICE)
        self.gamma = torch.tensor(1/3.5, device=DEVICE)
        self.lambdaa = torch.tensor(1/7, device=DEVICE)
        self.lambdap = torch.tensor(1/1.5, device=DEVICE)
        self.lambdam = torch.tensor(1/5.5, device=DEVICE)
        self.lambdas = torch.tensor(1/5.5, device=DEVICE)
        self.rhor = torch.tensor(1/15, device=DEVICE)
        self.rhod = torch.tensor(1/13.3, device=DEVICE)
        self.beta = nn.Parameter(torch.tensor(0.5, device=DEVICE))
        self.t = torch.tensor(0.0, device=DEVICE)

    def set_beta(self, beta):
        self.beta = nn.Parameter(torch.tensor(beta, device=DEVICE))

    def forward(self, t, y):
        S, E, Ia, Ip, Im, Is, Hr, Hd, R, D = y
        dt = t - self.t
        self.t = t
        dSE = S * (1 - torch.exp(-self.beta * (self.Ca * Ia + self.Cp * Ip + self.Cm * Im + self.Cs * Is) * dt / self.N))
        dEIa = E * self.alpha * (1 - torch.exp(-self.gamma * dt))
        dEIp = E * (1 - self.alpha) * (1 - torch.exp(-self.gamma * dt))
        dIaR = Ia * (1 - torch.exp(-self.lambdaa * dt))
        dIpIm = Ip * self.mu * (1 - torch.exp(-self.lambdap * dt))
        dIpIs = Ip * (1 - self.mu) * (1 - torch.exp(-self.lambdap * dt))
        dImR = Im * (1 - torch.exp(-self.lambdam * dt))
        dIsHr = Is * self.delta * (1 - torch.exp(-self.lambdas * dt))
        dIsHd = Is * (1 - self.delta) * (1 - torch.exp(-self.lambdas * dt))
        dHrR = Hr * (1 - torch.exp(-self.rhor * dt))
        dHdD = Hd * (1 - torch.exp(-self.rhod * dt))

        dS = -dSE
        dE = dSE - dEIa - dEIp
        dIa = dEIa - dIaR
        dIp = dEIp - dIpIs - dIpIm
        dIm = dIpIm - dImR
        dIs = dIpIs - dIsHr - dIsHd
        dHr = dIsHr - dHrR
        dHd = dIsHd - dHdD
        dR = dHrR
        dD = dHdD
      
        dy = torch.stack([dS, dE, dIa, dIp, dIm, dIs, dHr, dHd, dR, dD])
        return dy
    
    def reset_t(self):
        self.t = torch.tensor(0.0, device=DEVICE)

def generate_beta_schedule(initial_beta=0.5, lambda_=0.01, total_steps=500, segment_length=50, fixed_steps=20, decay_steps=30):
    """
    生成beta时间表，每50步一个段：
    - 前20步beta保持不变
    - 后30步beta以指数方式衰减
    """
    beta_schedule = []
    current_beta = initial_beta
    num_segments = total_steps // segment_length
    for seg in range(num_segments):
        # 前fixed_steps步beta保持不变
        for _ in range(fixed_steps):
            beta_schedule.append(current_beta)
        # 后decay_steps步beta衰减
        for _ in range(decay_steps):
            current_beta = current_beta * math.exp(-lambda_)  # 使用 math.exp
            beta_schedule.append(current_beta)
    # 处理剩余的步数（如果total_steps不是segment_length的整数倍）
    remaining_steps = total_steps - num_segments * segment_length
    for _ in range(remaining_steps):
        beta_schedule.append(current_beta)
    return torch.tensor(beta_schedule, dtype=torch.float32, device=DEVICE)

# def beta_decay(initial_beta, lambda_, t):
#     beta = torch.ones_like(t) * initial_beta
#     beta[t >= 200] = initial_beta * torch.exp(-lambda_ * (t[t >= 200] - 199))
#     return beta

def generate_inference_csv():
    y0_list = generate_initial_conditions()
    beta_values = [
        generate_beta_schedule(initial_beta=0.5, lambda_=0.01, total_steps=500, segment_length=50, fixed_steps=20, decay_steps=30),
        (0.5, 0.5)
    ]
    Ca_values = [0.425, 0.6]
    delta_value = 0.1375
    time_points = 499
    t = torch.linspace(0, time_points, time_points + 1, device=DEVICE)

    results = []

    for i, initial_y0 in enumerate(y0_list):
        total_population = initial_y0.sum().item()
        
        for beta_value in beta_values:
            for Ca in Ca_values:
                y0 = initial_y0.clone()
                model = ODEFunc().to(DEVICE)
                model.Ca = nn.Parameter(torch.tensor(Ca, device=DEVICE))
                model.delta = nn.Parameter(torch.tensor(delta_value, device=DEVICE))

                if isinstance(beta_value, torch.Tensor):
                    results.append({
                        'Time': t[0].item(),
                        'y0_index': i,
                        'beta': beta_value[0].item(),
                        'Ca': model.Ca.item(),
                        'delta': model.delta.item(),
                        'Susceptible': y0[0].item() / total_population,
                        'Exposed': y0[1].item() / total_population,
                        'Infectious_asymptomatic': y0[2].item() / total_population,
                        'Infectious_pre-symptomatic': y0[3].item() / total_population,
                        'Infectious_mild': y0[4].item() / total_population,
                        'Infectious_severe': y0[5].item() / total_population,
                        'Hospitalized_recovered': y0[6].item() / total_population,
                        'Hospitalized_deceased': y0[7].item() / total_population,
                        'Recovered': y0[8].item() / total_population,
                        'Deceased': y0[9].item() / total_population,
                    })

                    for time_step in range(len(beta_value)-1):
                        t_half = t[time_step:time_step+2].to(DEVICE)
                        with torch.no_grad():
                            Y_pred = odeint(model, y0, t_half, method='rk4')
                        y0 = Y_pred[-1]
                        results.append({
                            'Time': t_half[1].item(),
                            'y0_index': i,
                            'beta': beta_value[time_step].item(),
                            'Ca': model.Ca.item(),
                            'delta': model.delta.item(),
                            'Susceptible': Y_pred[-1, 0].item() / total_population,
                            'Exposed': Y_pred[-1, 1].item() / total_population,
                            'Infectious_asymptomatic': Y_pred[-1, 2].item() / total_population,
                            'Infectious_pre-symptomatic': Y_pred[-1, 3].item() / total_population,
                            'Infectious_mild': Y_pred[-1, 4].item() / total_population,
                            'Infectious_severe': Y_pred[-1, 5].item() / total_population,
                            'Hospitalized_recovered': Y_pred[-1, 6].item() / total_population,
                            'Hospitalized_deceased': Y_pred[-1, 7].item() / total_population,
                            'Recovered': Y_pred[-1, 8].item() / total_population,
                            'Deceased': Y_pred[-1, 9].item() / total_population,
                        })
                else:
                    beta1, beta2 = beta_value
                    model.set_beta(beta1)
                    with torch.no_grad():
                        Y_pred = odeint(model, y0, t, method='rk4')
                    for j, t_val in enumerate(t):
                        results.append({
                            'Time': t_val.item(),
                            'y0_index': i,
                            'beta': beta1,
                            'Ca': model.Ca.item(),
                            'delta': model.delta.item(),
                            'Susceptible': Y_pred[j, 0].item() / total_population,
                            'Exposed': Y_pred[j, 1].item() / total_population,
                            'Infectious_asymptomatic': Y_pred[j, 2].item() / total_population,
                            'Infectious_pre-symptomatic': Y_pred[j, 3].item() / total_population,
                            'Infectious_mild': Y_pred[j, 4].item() / total_population,
                            'Infectious_severe': Y_pred[j, 5].item() / total_population,
                            'Hospitalized_recovered': Y_pred[j, 6].item() / total_population,
                            'Hospitalized_deceased': Y_pred[j, 7].item() / total_population,
                            'Recovered': Y_pred[j, 8].item() / total_population,
                            'Deceased': Y_pred[j, 9].item() / total_population,
                        })

    df = pd.DataFrame(results)
    df.to_csv('/home/zhicao/ODE/data/result1.csv', index=False)
    return df

df = generate_inference_csv()
df.head()

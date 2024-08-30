from typing import NamedTuple

class SEIRMConfig(NamedTuple):
    beta: float = 0.5
    alpha: float = 0.4875
    gamma: float = 1/3.5
    mu: float = 0.928125
    num_agents: int = 1938000  # Total population for SEIRM model

class DataConfig(NamedTuple):
    n_sample: int = 1
    obs_dim: int = 1
    latent_dim: int = 5
    action_dim: int = 1
    t_max: int = 500
    step_size: int = 1
    sparsity: float = 0.5
    output_sigma: float = 0.1

# Example data configurations
dim8_config = DataConfig(obs_dim=40, latent_dim=8, output_sigma=0.2, sparsity=0.625)
dim12_config = DataConfig(obs_dim=80, latent_dim=12, output_sigma=0.2, sparsity=0.75)

class ModelConfig(NamedTuple):
    encoder_latent_ratio: float = 2.0
    expert_only: bool = False
    neural_ode: bool = False
    path: str = "model/"

class OptimConfig(NamedTuple):
    lr: float = 0.01
    ode_method: str = "dopri5"
    niters: int = 10
    batch_size: int = 50
    test_freq: int = 100
    shuffle: bool = True
    n_restart: int = 3
    early_stop: int = 10

class EvalConfig(NamedTuple):
    t0: int = 5

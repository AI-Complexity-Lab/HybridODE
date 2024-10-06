import torch
import torch.nn as nn

cuda = torch.device('cuda')
dtype = torch.float

class ODE(nn.Module):
    def __init__(self, params, device):
        super(ODE, self).__init__()
        self.params = params
        self.device = device
        self.num_agents = 1938000
        self.t = 0

    def reset_t(self):
        self.t = 0


class SEIRM(ODE):
    def __init__(self, params, learnable_params, device):
        super().__init__(params,device)
        self.beta = nn.Parameter(torch.tensor(learnable_params['beta'], device=device))
        self.alpha = nn.Parameter(torch.tensor(learnable_params['alpha'], device=device))
        self.gamma = nn.Parameter(torch.tensor(learnable_params['gamma'], device=device))
        self.mu = nn.Parameter(torch.tensor(learnable_params['mu'], device=device))

        self.new_infections = torch.zeros(100, device=device)
        self.new_deaths = torch.zeros(100, device=device)
    
    def init_compartments(self,learnable_params):
        ''' let's get initial conditions '''
        initial_infections_percentage = learnable_params['initial_infections_percentage']
        initial_conditions = torch.empty((5)).to(self.device)
        no_infected = (initial_infections_percentage / 100) * self.num_agents # 1.0 is ILI
        initial_conditions[2] = no_infected
        initial_conditions[0] = self.num_agents - no_infected
        print('initial infected',no_infected)

        return initial_conditions

    def forward(self, t, state):
        """
        Computes ODE states via equations       
            state is the array of state value (S,E,I,R,M)
        """
        
        state = state * self.num_agents
        # to make the NN predict lower numbers, we can make its prediction to be N-Susceptible
        dSE = self.beta * state[0] * state[2] / self.num_agents 
        dEI = self.alpha * state[1] 
        dIR = self.gamma * state[2] 
        dIM = self.mu * state[2] 

        dS  = -1.0 * dSE
        dE  = dSE - dEI
        dI = dEI - dIR - dIM
        dR  = dIR
        dM  = dIM

        # concat and reshape to make it rows as obs, cols as states
        dstate = torch.stack([dS, dE, dI, dR, dM], 0)

        # update state
        state = state + dstate
        
        state_proportion = state / torch.sum(state)
        
        self.t = t
        return state_proportion

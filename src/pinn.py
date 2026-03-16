#PINN Class, define the network
import torch 
import torch.nn as nn 

class pinn(nn.Module):
    def __init__(self):
        # Needed for tracking weights and bias 
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(), # 2 inputs x,t
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1), # 1 output u, temp
        )
    def forward(self, x, t):
        # stacks x and t column together into a (N,2) tensor, passing through the network to get prediction of (N,1)
        return self.net(torch.cat([x, t], dim=1))

"""Nonlinear inverse corrections that read only the fixed PCA window vector."""

import torch
from torch import nn


class PCALatentResidual(nn.Module):
    def __init__(self,state_dim,output_dim,hidden_dim=128,seed=42):
        super().__init__()
        self.register_buffer('state_scale',torch.ones(state_dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+317)
            self.network=nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,output_dim))
            nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,state):
        return self.network(state/self.state_scale.clamp_min(1e-6))

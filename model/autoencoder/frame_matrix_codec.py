"""A shared per-frame bottleneck; flattening and dimensional reduction are separate."""

import math

import torch
from torch import nn

from model.layers.residual_mlp import ResidualMLPBlock


class FrameMatrixCodec(nn.Module):
    def __init__(self,input_dim,state_dim=1024,kind='linear',hidden_dim=512,seed=42,residual_depth=0):
        super().__init__()
        if kind not in ('linear','nonlinear','residual') or not 0<state_dim<=input_dim:raise ValueError('Invalid frame codec')
        if kind=='residual' and residual_depth<1:raise ValueError('Residual codec requires at least one block')
        self.input_dim,self.state_dim,self.kind=input_dim,state_dim,kind
        self.register_buffer('input_mean',torch.zeros(input_dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1024)
            self.encoder=nn.Linear(input_dim,state_dim,bias=False)
            self.decoder=nn.Linear(state_dim,input_dim,bias=False)
            if kind in ('nonlinear','residual'):
                self.encoder_residual=nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,state_dim))
                self.decoder_residual=nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,input_dim))
                for module in (self.encoder_residual[-1],self.decoder_residual[-1]):
                    nn.init.zeros_(module.weight);nn.init.zeros_(module.bias)
                if kind=='residual':
                    # Build blocks after the old branches, preserving the same
                    # input weights and zero-output warm start at equal width.
                    branches=[]
                    for side,branch in enumerate((self.encoder_residual,self.decoder_residual)):
                        blocks=[]
                        for index in range(residual_depth):
                            with torch.random.fork_rng(devices=[]):
                                torch.manual_seed(seed+2100+side*100+index)
                                blocks.append(ResidualMLPBlock(hidden_dim))
                        branches.append(nn.Sequential(branch[0],branch[1],*blocks,branch[2]))
                    self.encoder_residual,self.decoder_residual=branches

    @torch.no_grad()
    def initialize_pca(self,mean,basis):
        if mean.shape!=(self.input_dim,) or basis.shape!=(self.state_dim,self.input_dim):raise ValueError('PCA shape mismatch')
        self.input_mean.copy_(mean);self.encoder.weight.copy_(basis);self.decoder.weight.copy_(basis.T)

    def encode(self,x):
        x=x-self.input_mean;z=self.encoder(x)
        if self.kind!='linear':z=z+self.encoder_residual(x/math.sqrt(self.input_dim))
        return z

    def decode(self,z):
        x=self.decoder(z)
        if self.kind!='linear':x=x+self.decoder_residual(z/math.sqrt(self.state_dim))
        return x+self.input_mean

    def forward(self,x):
        z=self.encode(x);return dict(matrix=z,reconstruction=self.decode(z))

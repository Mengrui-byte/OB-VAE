"""Two ordered sublevel readouts from the final post-Z CNN feature pairs."""
import torch
from torch import nn
from torch.nn import functional as F


def pair_features(features):
    if features.ndim!=3 or features.shape[1]!=32 or features.shape[-1]%2:raise ValueError('Expected[N,32,even levels]')
    n,c,l=features.shape
    return features.reshape(n,c,l//2,2).permute(0,3,1,2).reshape(n,2*c,l//2)


def log_quantity_residual(quantity,delta,mask):
    if quantity.shape!=delta.shape or quantity.shape!=mask.shape:raise ValueError('Mismatching quantity correction grid')
    bounded=4*torch.tanh(delta.float()/4)
    return (quantity.float().masked_fill(~mask,0)*bounded.exp()).masked_fill(~mask,0)


class PairedQuantityLogitReadout(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight=nn.Parameter(torch.zeros(2,64,1));self.bias=nn.Parameter(torch.zeros(2))

    def forward(self,features,*,phase_specific=False):
        if type(phase_specific) is not bool:raise ValueError('Explicit Boolean phase flag required')
        if features.ndim!=3 or features.shape[1]!=64:raise ValueError('Expected paired[N,64,L/2] features')
        value=F.conv1d(features,self.weight,self.bias)
        if not phase_specific:value=value.mean(1,keepdim=True).expand(-1,2,-1)
        return value.transpose(1,2).reshape(features.shape[0],2*features.shape[-1]).float()

"""Ordered quantity-logit residuals from post-Z local or frame features."""
import torch
from torch import nn
from torch.nn import functional as F


class QuantityLogitReadout(nn.Module):
    def __init__(self,*,feature_source='seed',levels=2048,frame_dim=256,region_width=16):
        super().__init__()
        if feature_source not in ['seed','frame']:raise ValueError('Unknown feature source')
        if levels<=0 or levels%region_width:raise ValueError('Incomplete region grid')
        self.feature_source=feature_source;self.levels=levels;self.regions=levels//region_width
        self.frame_dim=frame_dim;self.region_width=region_width
        dim=16 if feature_source=='seed' else frame_dim
        # One affine per side/region, with a distinct output for every level.
        # Empty zero tensors consume no RNG and do not change source init.
        self.weight=nn.Parameter(torch.zeros(2*self.regions,region_width,dim))
        self.bias=nn.Parameter(torch.zeros(2*self.regions,region_width))

    def forward(self,seed,frame):
        if self.feature_source=='frame':
            if frame.shape[-1]!=self.frame_dim:raise ValueError('Unexpected frame features')
            delta=F.linear(frame,self.weight.reshape(2*self.levels,self.frame_dim),self.bias.flatten())
            return delta.reshape(*frame.shape[:-1],2,self.levels).float()
        if seed.shape[-3:]!=(2,self.regions,16):raise ValueError('Unexpected ordered seed grid')
        batch_shape=seed.shape[:-3]
        x=seed.movedim((-3,-2),(0,1)).reshape(2*self.regions,-1,16)
        delta=torch.baddbmm(self.bias[:,None,:],x,self.weight.transpose(1,2))
        delta=delta.reshape(2,self.regions,*batch_shape,self.region_width)
        return delta.movedim((0,1),(-3,-2)).reshape(*batch_shape,2,self.levels).float()


def add_quantity_logit(base_logit,delta,mask,quantity_scale):
    if base_logit.shape!=delta.shape or mask.shape!=delta.shape:raise ValueError('Logit grid mismatch')
    logit=(base_logit.float()+delta.float()).masked_fill(~mask,0)
    return (quantity_scale*F.softplus(logit)).masked_fill(~mask,0),logit

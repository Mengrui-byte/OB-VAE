"""Latent-only residual corrections for price geometry and quantity amplitude."""

import torch
from torch import nn
from torch.nn import functional as F

from model.dual.book_layers import PositionCode, Residual1d


class LatentSpatialResidual(nn.Module):
    def __init__(self, levels, global_dim, local_dim, extra_channels):
        super().__init__()
        self.position = PositionCode(levels)
        self.global_projection = nn.Linear(global_dim, 4)
        self.projection = nn.Conv1d(local_dim + 4 + 32 + extra_channels, 32, 1)
        self.blocks = nn.ModuleList([Residual1d(32, dilation=d) for d in (1,4,16)])
        self.head = nn.Conv1d(32, 1, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, zg, zl, mask, features, times=None):
        B,T,_,L = mask.shape
        local = zl.permute(0,1,2,4,3).reshape(B*T*2, zl.shape[-1], -1)
        local = F.interpolate(local, size=L, mode="linear", align_corners=False)
        glob = self.global_projection(zg)[:,:,None,:,None].expand(B,T,2,4,L)
        pos = self.position(B,T,L,zg,times)
        inputs = torch.cat([local, glob.reshape(B*T*2,4,L), pos.reshape(B*T*2,32,L),
                            features.reshape(B*T*2,features.shape[-2],L)], dim=1)
        m = mask.reshape(B*T*2,1,L).to(inputs.dtype)
        h = self.projection(inputs*m)*m
        for block in self.blocks:
            h = block(h,m)
        return self.head(h).reshape(B,T,2,L).float()*mask


class PriceGapRefinement(nn.Module):
    def __init__(self, levels, global_dim, local_dim):
        super().__init__()
        self.residual = LatentSpatialResidual(levels,global_dim,local_dim,2)

    def forward(self, zg, zl, original, mask, tick_norm, times=None):
        tick = tick_norm.float()[:,None,None,None]
        padded_gap = F.pad(original["gap_norm"],(1,0))
        features = torch.stack([original["price_norm"],
                                torch.log((padded_gap/tick).clamp_min(1e-6))],-2)
        delta = self.residual(zg,zl,mask,features,times)[...,1:]
        multiplier = torch.exp(2*torch.tanh(delta/2))
        valid = mask[...,1:] & mask[...,:-1]
        gap = (tick + (original["gap_norm"]-tick).clamp_min(0)*multiplier)*valid
        distance = F.pad(torch.cumsum(gap,-1,dtype=torch.float32),(1,0))
        sign = gap.new_tensor([-1,1])[None,None,:,None]
        price = original["mid_norm"][:,:,None,None] + sign*(
            original["spread_norm"][:,:,None,None]/2 + distance)
        return {**original,"gap_norm":gap,"price_norm":price*mask}


class VolumeAmplitudeRefinement(nn.Module):
    def __init__(self, levels, global_dim, local_dim, quantity_scale):
        super().__init__()
        self.register_buffer("q0",torch.tensor(float(quantity_scale)))
        self.residual = LatentSpatialResidual(levels,global_dim,local_dim,3)

    def forward(self, zg, zl, volume, price_output, mask, tick_norm, times=None):
        tick = tick_norm[:,None,None,None]
        log_gap = F.pad(torch.log((price_output["gap_norm"]/tick).clamp_min(1e-6)),(1,0))
        features = torch.stack([torch.log1p(volume.float()/self.q0),
                                price_output["price_norm"],log_gap],-2)
        delta = self.residual(zg,zl,mask,features,times)
        return volume.float()*torch.exp(3*torch.tanh(delta/3))*mask

"""Independent price/quantity decoders from one whole-window latent."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model.dual.book_layers import PositionCode, Residual1d, mask_at, mean_at, sinusoidal


class TemporalExpander(nn.Module):
    def __init__(self, latent_channels=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent_channels,512), nn.GELU(), nn.Linear(512,1024))
        self.refine = Residual1d(256)

    def forward(self, z, times):
        B = z.shape[0]
        h = self.mlp(z).reshape(B,256,256)
        h = h+sinusoidal(times,256).to(h.dtype)
        return self.refine(h.transpose(1,2),h.new_ones(B,1,256)).transpose(1,2)


class SpatialUp(nn.Module):
    def __init__(self, cin, cout, conditional, dilation):
        super().__init__()
        self.conv = nn.Conv1d(cin,cout,3,padding=1)
        self.fuse = nn.Conv1d(cout+32,cout,1) if conditional else None
        self.res = Residual1d(cout,dilation)

    def forward(self, h, mask, condition):
        length = h.shape[-1]*2
        m = mask_at(mask,length)
        h = self.conv(F.interpolate(h,scale_factor=2,mode='nearest')*m)*m
        if self.fuse is not None:
            h = self.fuse(torch.cat((h,mean_at(condition,mask,length)),1))*m
        return self.res(h,m)


class BookSpatialDecoder(nn.Module):
    def __init__(self, levels, conditional, activation_checkpointing):
        super().__init__()
        self.levels, self.conditional = levels, conditional
        self.activation_checkpointing = activation_checkpointing
        self.seed = nn.Linear(256,2*(levels//16)*16)
        self.position = PositionCode(levels)
        self.proj = nn.Conv1d(16+32,256,1)
        self.fuse = nn.Conv1d(288,256,1) if conditional else None
        self.up = nn.ModuleList(SpatialUp(a,b,conditional,d) for a,b,d in zip((256,192,128,64),(192,128,64,32),(2,2,2,1)))

    def _expand(self, h, mask, condition):
        if self.fuse is not None:
            h = self.fuse(torch.cat((h,mean_at(condition,mask,h.shape[-1])),1))*mask_at(mask,h.shape[-1])
        for up in self.up:
            h = up(h,mask,condition)
        return h

    def forward(self, frame, mask, times, condition=None):
        B,T,_ = frame.shape
        K = self.levels//16
        seed = self.seed(frame).reshape(B,T,2,K,16).transpose(-1,-2)
        pos = self.position(B,T,K,frame,times)
        h = self.proj(torch.cat((seed,pos),3).reshape(B*T*2,48,K))
        m = mask.reshape(B*T*2,1,self.levels).float()
        h = h*mask_at(m,K)
        if self.training and self.activation_checkpointing:
            return checkpoint(self._expand,h,m,condition,use_reentrant=False)
        return self._expand(h,m,condition)


class PriceDecoder(nn.Module):
    def __init__(self, levels=2048, activation_checkpointing=True, latent_channels=128):
        super().__init__()
        self.temporal = TemporalExpander(latent_channels)
        self.spatial = BookSpatialDecoder(levels,False,activation_checkpointing)
        self.mid_spread = nn.Sequential(nn.Linear(256,128),nn.GELU(),nn.Linear(128,2))
        self.gap = nn.Conv1d(32,1,1)
        nn.init.normal_(self.gap.weight,std=.001)
        nn.init.normal_(self.mid_spread[-1].weight,std=.001)
        nn.init.zeros_(self.mid_spread[-1].bias)

    def forward(self, z, mask, tick_norm, times):
        B,T,_,L = mask.shape
        frame = self.temporal(z,times)
        feature = self.spatial(frame,mask,times)
        raw_gap = self.gap(feature).reshape(B,T,2,L)[...,1:].float()
        ms = self.mid_spread(frame).float()
        tick = tick_norm.float()[:,None,None,None]
        mid = torch.cat((torch.zeros_like(ms[:,:1,0]),ms[:,1:,0]),1)
        spread = tick_norm.float()[:,None]*(1+F.softplus(ms[...,1]))
        gap = tick*(1+F.softplus(raw_gap))
        distance = F.pad(gap.cumsum(-1,dtype=torch.float32),(1,0))
        sign = z.new_tensor([-1,1]).float()[None,None,:,None]
        price = mid[:,:,None,None]+sign*(spread[:,:,None,None]/2+distance)
        return dict(price_norm=price.masked_fill(~mask,0),mid_norm=mid,spread_norm=spread,gap_norm=gap)


class VolumeDecoder(nn.Module):
    def __init__(self, levels=2048, quantity_scale=1., activation_checkpointing=True, latent_channels=128):
        super().__init__()
        self.register_buffer('q0',torch.tensor(float(quantity_scale)))
        self.temporal = TemporalExpander(latent_channels)
        self.position = PositionCode(levels)
        self.geometry = nn.Sequential(nn.Conv1d(34,32,1),nn.GELU(),nn.Conv1d(32,32,1))
        self.spatial = BookSpatialDecoder(levels,True,activation_checkpointing)
        self.quantity = nn.Conv1d(32,1,1)
        nn.init.normal_(self.quantity.weight,std=.001)

    def forward(self, z, price_output, mask, tick_norm, times):
        B,T,_,L = mask.shape
        p = price_output['price_norm']
        gap = price_output['gap_norm']
        loggap = F.pad(torch.log((gap/tick_norm[:,None,None,None]).clamp_min(1e-6)),(1,0))
        adjacent = F.pad(mask[...,1:] & mask[...,:-1],(1,0))
        loggap = loggap*adjacent
        pos = self.position(B,T,L,z,times)
        geometry = torch.cat((p.unsqueeze(-2),loggap.unsqueeze(-2),pos),3).reshape(B*T*2,34,L)
        m = mask.reshape(B*T*2,1,L).float()
        condition = self.geometry(geometry)*m
        frame = self.temporal(z,times)
        h = self.spatial(frame,mask,times,condition)
        return (self.q0*F.softplus(self.quantity(h).float().reshape(B,T,2,L))).masked_fill(~mask,0)

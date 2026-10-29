"""Regional seeds retain their coordinates through the existing spatial path."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.decoders.unified_book import BookSpatialDecoder,PriceDecoder
from model.decoders.logit_volume import LogitVolumeDecoder
from model.dual.book_layers import mask_at
from model.layers.region_grid import RegionSeed


class RegionSpatialDecoder(BookSpatialDecoder):
    def __init__(self, levels, conditional, activation_checkpointing, seed_hidden=1024):
        super().__init__(levels,conditional,activation_checkpointing)
        self.seed = RegionSeed(levels,seed_hidden)

    def forward(self, frame, mask, times, condition=None, *, z):
        B,T,_ = frame.shape
        K = self.levels//16
        seed = self.seed(z,frame).transpose(-1,-2)
        pos = self.position(B,T,K,frame,times)
        h = self.proj(torch.cat((seed,pos),3).reshape(B*T*2,48,K))
        m = mask.reshape(B*T*2,1,self.levels).float()
        h = h*mask_at(m,K)
        if self.training and self.activation_checkpointing:
            return checkpoint(self._expand,h,m,condition,use_reentrant=False)
        return self._expand(h,m,condition)


class RegionPriceDecoder(PriceDecoder):
    def forward(self, z, mask, tick_norm, times):
        B,T,_,L = mask.shape
        frame = self.temporal(z,times)
        feature = self.spatial(frame,mask,times,z=z)
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


class RegionVolumeDecoder(LogitVolumeDecoder):
    def forward(self, z, price_output, mask, tick_norm, times):
        B,T,_,L = mask.shape
        p = price_output['price_norm']
        gap = price_output['gap_norm']
        loggap = F.pad(torch.log((gap/tick_norm[:,None,None,None]).clamp_min(1e-6)),(1,0))
        loggap = loggap*F.pad(mask[...,1:]&mask[...,:-1],(1,0))
        pos = self.position(B,T,L,z,times)
        geometry = torch.cat((p.unsqueeze(-2),loggap.unsqueeze(-2),pos),3).reshape(B*T*2,34,L)
        m = mask.reshape(B*T*2,1,L).float()
        condition = self.geometry(geometry)*m
        h = self.spatial(self.temporal(z,times),mask,times,condition,z=z)
        r = self.quantity(h).float().reshape(B,T,2,L)
        return dict(volume=(self.q0*F.softplus(r)).masked_fill(~mask,0),quantity_logit=r.masked_fill(~mask,0))

"""Original quantity reconstruction plus an auxiliary per-level peak logit."""
import torch
from torch import nn
from torch.nn import functional as F

from model.decoders.unified_book import VolumeDecoder


class PeakAuxVolumeDecoder(VolumeDecoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.peak_head = nn.Conv1d(32, 1, 1)
        nn.init.zeros_(self.peak_head.weight)
        nn.init.zeros_(self.peak_head.bias)

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
        volume = (self.q0*F.softplus(self.quantity(h).float().reshape(B,T,2,L))).masked_fill(~mask,0)
        logits = self.peak_head(h).float().reshape(B,T,2,L).masked_fill(~mask,0)
        return dict(volume=volume, peak_logits=logits)

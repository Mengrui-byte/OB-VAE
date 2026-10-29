"""Original quantity decoder, exposing its existing pre-Softplus logit."""
import torch
from torch.nn import functional as F
from model.decoders.unified_book import VolumeDecoder


class LogitVolumeDecoder(VolumeDecoder):
    def forward(self,z,price_output,mask,tick_norm,times):
        B,T,_,L=mask.shape;p=price_output['price_norm'];gap=price_output['gap_norm']
        loggap=F.pad(torch.log((gap/tick_norm[:,None,None,None]).clamp_min(1e-6)),(1,0))
        loggap=loggap*F.pad(mask[...,1:]&mask[...,:-1],(1,0))
        pos=self.position(B,T,L,z,times)
        geometry=torch.cat((p.unsqueeze(-2),loggap.unsqueeze(-2),pos),3).reshape(B*T*2,34,L)
        m=mask.reshape(B*T*2,1,L).float();condition=self.geometry(geometry)*m
        h=self.spatial(self.temporal(z,times),mask,times,condition)
        r=self.quantity(h).float().reshape(B,T,2,L)
        return dict(volume=(self.q0*F.softplus(r)).masked_fill(~mask,0),quantity_logit=r.masked_fill(~mask,0))

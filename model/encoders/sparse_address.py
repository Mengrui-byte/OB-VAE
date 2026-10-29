"""One posterior combines neural price coordinates and explicit sparse events."""
import torch
from torch import nn
from model.encoders.frame_aligned import FrameAlignedEncoder
from model.layers.sparse_address import price_dimension


class SparseAddressEncoder(FrameAlignedEncoder):
    def __init__(self,*,peak_slots=2048,frame_ffn=1536,**kwargs):
        super().__init__(frame_ffn=frame_ffn,**kwargs)
        self.peak_slots=peak_slots;self.price_dim=price_dimension(peak_slots)
        self.mu=nn.Linear(256,self.price_dim)

    def forward(self,price,volume,mask,tick_norm,times):
        h=self.encode_context(price,volume,mask,tick_norm,times);B=price.shape[0]
        neural=self.mu(h).float().reshape(B,-1)
        price_lv=self.logvar(h)[...,:self.price_dim].float().clamp(-12,8).reshape(B,-1)
        if self.peak_slots:
            values,indices=volume.float().masked_fill(~mask,-torch.inf).reshape(B,-1).topk(self.peak_slots,dim=-1)
            values=values.masked_fill(~torch.isfinite(values),0)
            mu=torch.cat((neural,indices.float(),values),-1)
            # Physical level-time integer addresses and BTC amplitudes. Small
            # posterior noise is not a learned normal-prior compression claim.
            lv=torch.cat((price_lv,torch.full_like(values,-12.),torch.full_like(values,-12.)),-1)
        else:mu,lv=neural,price_lv
        return mu.reshape(B,64,128),lv.reshape(B,64,128)

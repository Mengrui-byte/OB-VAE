"""Full quantity decoder using nonlinear ordered coordinate queries."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.decoders.unified_book import TemporalExpander
from model.dual.book_layers import PositionCode,mask_at


class QueryVolumeDecoder(nn.Module):
    def __init__(self,levels=2048,quantity_scale=1.,activation_checkpointing=True,
                 latent_channels=128,frame_hidden=1408):
        super().__init__()
        if levels%16:raise ValueError('levels must be a multiple of16')
        self.levels=levels;self.activation_checkpointing=activation_checkpointing
        self.register_buffer('q0',torch.tensor(float(quantity_scale)))
        self.temporal=TemporalExpander(latent_channels)
        self.position=PositionCode(levels)
        self.geometry=nn.Sequential(nn.Conv1d(34,32,1),nn.GELU(),nn.Conv1d(32,32,1))
        self.frame_mlp=nn.Sequential(nn.LayerNorm(256),nn.Linear(256,frame_hidden),nn.GELU(),nn.Linear(frame_hidden,256))
        nn.init.zeros_(self.frame_mlp[-1].weight);nn.init.zeros_(self.frame_mlp[-1].bias)
        self.seed=nn.Linear(256,2*(levels//16)*16)
        self.region=nn.Sequential(nn.Conv1d(48,128,1),nn.GELU())
        self.query=nn.Sequential(nn.Conv1d(128+32+32+16,64,1),nn.GELU(),nn.Conv1d(64,64,1),nn.GELU())
        self.quantity=nn.Conv1d(64,1,1)

    def _query(self,h,condition,position,offset,mask):
        # One ordered feature per16-level region; the query has distinct offset,
        # absolute position and predicted-price geometry at every actual level.
        h=h.repeat_interleave(16,dim=-1)
        x=torch.cat((h,condition,position,offset),1)
        return self.quantity(self.query(x))*mask

    def forward(self,z,price_output,mask,tick_norm,times):
        B,T,_,L=mask.shape
        if L!=self.levels or T!=256:raise ValueError('Complete256-frame book required')
        K=L//16;N=B*T*2;m=mask.reshape(N,1,L).float()
        p,gap=price_output['price_norm'],price_output['gap_norm']
        loggap=F.pad(torch.log((gap/tick_norm[:,None,None,None]).clamp_min(1e-6)),(1,0))
        loggap=loggap*F.pad(mask[...,1:]&mask[...,:-1],(1,0))
        position=self.position(B,T,L,z,times).reshape(N,32,L)
        geometry=torch.cat((p.unsqueeze(-2),loggap.unsqueeze(-2),position.reshape(B,T,2,32,L)),3).reshape(N,34,L)
        condition=self.geometry(geometry)*m
        frame=self.temporal(z,times);frame=frame+self.frame_mlp(frame)
        seed=self.seed(frame).reshape(B,T,2,K,16).transpose(-1,-2)
        regional_position=self.position(B,T,K,frame,times)
        h=self.region(torch.cat((seed,regional_position),3).reshape(N,48,K))*mask_at(m,K)
        offset=F.one_hot(torch.arange(L,device=z.device)%16,16).to(frame.dtype).T[None].expand(N,-1,-1)
        if self.training and self.activation_checkpointing:
            raw=checkpoint(self._query,h,condition,position,offset,m,use_reentrant=False)
        else:raw=self._query(h,condition,position,offset,m)
        return (self.q0*F.softplus(raw.float().reshape(B,T,2,L))).masked_fill(~mask,0)

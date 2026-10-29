"""Write Global into unchanged Local slots and read conditions from Local only."""

import torch
from torch import nn

from model.dual.book_layers import PositionCode,sinusoidal
from model.layers.residual_mlp import ResidualMLPBlock


class GlobalIntoLocal(nn.Module):
    def __init__(self,levels,global_dim=64,local_dim=16,mode='residual',hidden=128):
        super().__init__();self.mode=mode
        if mode not in ('local','residual','attention'):raise ValueError(mode)
        if mode=='local':return
        self.position=PositionCode(levels)
        if mode=='residual':
            self.writer=nn.Sequential(nn.Linear(global_dim+local_dim+32,hidden),nn.GELU(),
                ResidualMLPBlock(hidden),ResidualMLPBlock(hidden),nn.Linear(hidden,local_dim))
            nn.init.zeros_(self.writer[-1].weight);nn.init.zeros_(self.writer[-1].bias)
        else:
            self.local_query=nn.Linear(local_dim,64);self.position_query=nn.Linear(32,64);self.query_norm=nn.LayerNorm(64)
            self.key=nn.Linear(global_dim,64);self.key_norm=nn.LayerNorm(64);self.value=nn.Linear(global_dim,64)
            self.attention=nn.MultiheadAttention(64,4,batch_first=True)
            self.output=nn.Linear(64,local_dim);nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,g,l,latent_mask,times=None):
        if self.mode=='local':return l
        b,t,_,k,d=l.shape
        if times is None:times=torch.arange(t,device=l.device).expand(b,t)
        pos=self.position(b,t,k,l,times).permute(0,1,2,4,3)
        if self.mode=='residual':
            glob=g[:,:,None,None,:].expand(b,t,2,k,-1)
            delta=self.writer(torch.cat([l,glob,pos.to(l.dtype)],-1))
        else:
            query=self.query_norm(self.local_query(l.reshape(b,t*2*k,d))+self.position_query(pos.reshape(b,t*2*k,32)))
            keys=self.key_norm(self.key(g)+sinusoidal(times,64).to(g.dtype))
            update=self.attention(query,keys,self.value(g),need_weights=False)[0]
            delta=self.output(update).reshape_as(l)
        return (l+delta)*latent_mask[...,None]


class GlobalReadoutFromLocal(nn.Module):
    def __init__(self,global_dim=64,local_dim=16,bands=4,hidden=128):
        super().__init__();self.bands=bands
        self.network=nn.Sequential(nn.Linear(2*bands*local_dim+32,hidden),nn.GELU(),nn.Linear(hidden,global_dim))
        nn.init.zeros_(self.network[-1].weight);nn.init.zeros_(self.network[-1].bias)

    def forward(self,l,latent_mask,times=None):
        b,t,_,k,d=l.shape
        if k%self.bands:raise ValueError('Local positions must divide into readout bands')
        valid=latent_mask.reshape(b,t,2,self.bands,k//self.bands,1).to(l.dtype)
        values=l.reshape(b,t,2,self.bands,k//self.bands,d)
        pooled=(values*valid).sum(-2)/valid.sum(-2).clamp_min(1)
        if times is None:times=torch.arange(t,device=l.device).expand(b,t)
        inputs=torch.cat([pooled.flatten(2),sinusoidal(times,32).to(l.dtype)],-1)
        return self.network(inputs)

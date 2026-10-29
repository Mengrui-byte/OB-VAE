"""Decoder-side temporal residuals over the existing compressed latents."""

import torch
from torch import nn

from model.dual.book_layers import Residual1d


class TemporalLatentRefinement(nn.Module):
    def __init__(self, global_dim, local_dim, causal=False):
        super().__init__()
        self.global_blocks = nn.ModuleList([Residual1d(global_dim,dilation=d,causal=causal) for d in (1,2)])
        self.local_blocks = nn.ModuleList([Residual1d(local_dim,dilation=d,causal=causal) for d in (1,2)])
        self.global_head = nn.Conv1d(global_dim,global_dim,1)
        self.local_head = nn.Conv1d(local_dim,local_dim,1)
        for head in (self.global_head,self.local_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self,zg,zl,latent_mask):
        B,T,_,K,D = zl.shape
        local = zl.permute(0,2,3,4,1).reshape(B*2*K,D,T)
        lm = latent_mask.permute(0,2,3,1).reshape(B*2*K,1,T).to(local.dtype)
        h = local
        for block in self.local_blocks:
            h = block(h,lm)
        local = (local+self.local_head(h)*lm).reshape(B,2,K,D,T).permute(0,4,1,2,3)
        gm = latent_mask.any(dim=(2,3))[:,None,:].to(zg.dtype)
        glob = zg.transpose(1,2)
        h = glob
        for block in self.global_blocks:
            h = block(h,gm)
        glob = (glob+self.global_head(h)*gm).transpose(1,2)
        return glob,local*latent_mask[...,None]

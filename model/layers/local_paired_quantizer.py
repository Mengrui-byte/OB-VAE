"""Quantize a pair of matrices at each time/depth position, without scaling."""

import torch
from torch import nn

from model.layers.paired_matrix_quantizer import PairedMatrixQuantizer


class LocalPairedQuantizer(nn.Module):
    def __init__(self,window,positions,global_dim=64,local_dim=16,levels=8,size=16):
        super().__init__()
        self.window,self.positions=window,positions
        self.global_dim,self.local_dim=global_dim,local_dim
        self.register_buffer('global_origin',torch.zeros(window,global_dim))
        self.register_buffer('local_origin',torch.zeros(window,2,positions,local_dim))
        self.book=PairedMatrixQuantizer((1,global_dim),(2,local_dim),levels=levels,size=size,metric='l2',mode='joint')
        # Same state names as the base quantizer, now trainable through decoding.
        for name in ('global_codebook','local_codebook'):
            value=self.book._buffers.pop(name)
            self.book.register_parameter(name,nn.Parameter(value))

    def pairs(self,g,l):
        b,t,k=len(g),g.shape[1],self.positions
        if g.shape[1:]!=self.global_origin.shape or l.shape[1:]!=self.local_origin.shape:
            raise ValueError('Unexpected local quantizer shape')
        rg=(g-self.global_origin)[...,None,:].expand(b,t,k,self.global_dim)
        rl=(l-self.local_origin).permute(0,1,3,2,4)
        return rg.reshape(-1,1,self.global_dim),rl.reshape(-1,2,self.local_dim)

    @torch.no_grad()
    def fit(self,g,l,samples=131072,iterations=12,seed=42,callback=None):
        self.global_origin.copy_(g.mean(0));self.local_origin.copy_(l.mean(0))
        generator=torch.Generator(device=g.device).manual_seed(seed)
        n=min(samples,len(g)*self.window*self.positions)
        index=torch.randint(len(g)*self.window*self.positions,(n,),device=g.device,generator=generator)
        window=index//(self.window*self.positions);time=(index//self.positions)%self.window;position=index%self.positions
        pg=(g[window,time]-self.global_origin[time])[:,None]
        pl=l[window,time,:,position]-self.local_origin[time,:,position]
        return self.book.fit(pg,pl,iterations=iterations,seed=seed,callback=callback)

    def decode_codes(self,codes,present=None):
        if codes.ndim!=4 or codes.shape[1:3]!=(self.window,self.positions):raise ValueError('Expected [B,T,K,R] integer codes')
        b,t,k,depth=codes.shape
        g,l=self.book.decode_codes(codes.reshape(-1,depth))
        g=g.reshape(b,t,k,self.global_dim)
        if present is None:present=torch.ones(b,t,k,device=g.device,dtype=torch.bool)
        # Global appears in each paired patch; reconstruct it by a fixed mean.
        g=(g*present[...,None]).sum(2)/present.sum(2,keepdim=True).clamp_min(1)+self.global_origin
        l=l.reshape(b,t,k,2,self.local_dim).permute(0,1,3,2,4)+self.local_origin
        return g,l

    def forward(self,g,l,present=None,depth=None):
        pg,pl=self.pairs(g,l);out=self.book(pg,pl,depth)
        codes=out['codes'].reshape(len(g),self.window,self.positions,-1)
        gh,lh=self.decode_codes(codes,present)
        latent_mse=((g-gh).square().sum()+(l-lh).square().sum())/(g.numel()+l.numel())
        return dict(global_latent=gh,local_latent=lh,codes=codes,latent_mse=latent_mse)

    def logical_bits(self,depth=None):
        return self.window*self.positions*self.book.logical_bits(depth)

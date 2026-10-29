"""Compress each fused frame to one row, retaining the window time axis."""

import torch
from torch.nn import functional as F

from model.autoencoder.frame_matrix_codec import FrameMatrixCodec
from model.experimental.dual_fusion import FusedOrderBookVAE


class FrameMatrixOrderBookVAE(FusedOrderBookVAE):
    def __init__(self,*,matrix_dim=1024,codec_kind='linear',codec_hidden=512,codec_seed=42,codec_depth=0,**core):
        super().__init__(fusion_variant='cross',fusion_seed=42,**core)
        self.frame_matrix_config=dict(matrix_dim=matrix_dim,codec_kind=codec_kind,codec_hidden=codec_hidden,codec_seed=codec_seed)
        if codec_kind=='residual':self.frame_matrix_config['codec_depth']=codec_depth
        self.global_dim=core.get('global_dim',64);self.local_dim=core.get('local_dim',16)
        self.positions=core.get('levels',2048)//16
        self.frame_codec=FrameMatrixCodec(self.global_dim+2*self.positions*self.local_dim,matrix_dim,codec_kind,codec_hidden,codec_seed,codec_depth)
        self.requires_grad_(False);self.evaluation_mode='compressed';self.train_source=False

    def load_cross_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('frame_codec.') for k in missing):raise ValueError('Frame matrix source differs')

    def configure_training(self,mode):
        if mode not in ('fixed','codec','joint','all'):raise ValueError(mode)
        self.requires_grad_(False)
        if mode!='fixed':self.frame_codec.requires_grad_(True)
        if mode in ('joint','all'):
            for module in (self.price_decoder,self.volume_decoder,self.volume_refinement):module.requires_grad_(True)
        self.train_source=mode=='all'
        if self.train_source:self.encoder.requires_grad_(True);self.fusion.requires_grad_(True)

    @staticmethod
    def pack(g,l):return torch.cat([g,l.flatten(2)],-1)

    def unpack(self,x):
        return x[...,:self.global_dim],x[...,self.global_dim:].reshape(*x.shape[:2],2,self.positions,self.local_dim)

    def decode_latents(self,g,l,mask,tick_norm,times=None):
        b,t,_,levels=mask.shape
        latent_mask=F.adaptive_max_pool1d(mask.reshape(b*t*2,1,levels).float(),self.positions).reshape(b,t,2,self.positions).bool()
        l=l*latent_mask[...,None]
        geometry=self.price_decoder(g,l,mask,tick_norm,times=times)
        quantity=self.volume_decoder(g,l,geometry,mask,tick_norm,times=times)
        quantity=self.volume_refinement(g,l,quantity,geometry,mask,tick_norm,times)
        return dict(geometry,volume=quantity)

    def decode_matrix(self,matrix,mask,tick_norm,times=None):
        if matrix.shape[-1]!=self.frame_codec.state_dim:raise ValueError('Wrong matrix width')
        g,l=self.unpack(self.frame_codec.decode(matrix))
        return self.decode_latents(g,l,mask,tick_norm,times)

    def forward_latents(self,g,l,mask,tick_norm,times=None):
        x=self.pack(g,l)
        if self.evaluation_mode=='flatten':
            gh,lh=self.unpack(x);return dict(self.decode_latents(gh,lh,mask,tick_norm,times),packed_latent=x)
        out=self.frame_codec(x);gh,lh=self.unpack(out['reconstruction'])
        return dict(self.decode_latents(gh,lh,mask,tick_norm,times),matrix_latent=out['matrix'],
                    latent_mse=(out['reconstruction']-x).square().mean())

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        if self.evaluation_mode=='original':return super().forward(price,volume,mask,tick_norm,times=times,sample=False)
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_source):
            p=self.encoder(price,volume,mask,tick_norm,times=times,sample=False)
            f=self.fusion([(p['z_global'],p['z_local'])],p['latent_mask'],times)
        return self.forward_latents(f['global_latent'],f['local_latent'],mask,tick_norm,times)

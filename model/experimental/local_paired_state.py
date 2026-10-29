"""Generate books solely from local paired integer codes and shared metadata."""

import torch
from torch.nn import functional as F

from model.experimental.dual_fusion import FusedOrderBookVAE
from model.layers.local_paired_quantizer import LocalPairedQuantizer


class LocalPairedOrderBookVAE(FusedOrderBookVAE):
    def __init__(self,*,quant_window=16,quant_levels=8,quant_size=16,**core):
        super().__init__(fusion_variant='cross',fusion_seed=42,**core)
        self.local_paired_config=dict(quant_window=quant_window,quant_levels=quant_levels,quant_size=quant_size)
        self.local_quantizer=LocalPairedQuantizer(quant_window,core.get('levels',2048)//16,
            core.get('global_dim',64),core.get('local_dim',16),quant_levels,quant_size)
        self.requires_grad_(False);self.evaluation_mode='quantized';self.quant_depth=quant_levels

    def load_cross_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith('local_quantizer.') for k in missing):raise ValueError('Local paired source mismatch')

    def configure_training(self,mode):
        if mode not in ('frozen','books','joint'):raise ValueError(mode)
        self.requires_grad_(False)
        if mode!='frozen':self.local_quantizer.book.requires_grad_(True)
        if mode=='joint':
            for module in (self.price_decoder,self.volume_decoder,self.volume_refinement):module.requires_grad_(True)

    @staticmethod
    def latent_mask(mask):
        b,t,_,levels=mask.shape
        return F.adaptive_max_pool1d(mask.reshape(b*t*2,1,levels).float(),levels//16).reshape(b,t,2,levels//16).bool()

    def decode_latents(self,g,l,mask,tick_norm,times=None):
        l=l*self.latent_mask(mask)[...,None]
        geometry=self.price_decoder(g,l,mask,tick_norm,times=times)
        q=self.volume_decoder(g,l,geometry,mask,tick_norm,times=times)
        q=self.volume_refinement(g,l,q,geometry,mask,tick_norm,times)
        return dict(geometry,volume=q)

    def decode_codes(self,codes,mask,tick_norm,times=None):
        g,l=self.local_quantizer.decode_codes(codes,self.latent_mask(mask).any(2))
        return self.decode_latents(g,l,mask,tick_norm,times)

    def forward_latents(self,g,l,mask,tick_norm,times=None):
        q=self.local_quantizer(g,l,self.latent_mask(mask).any(2),self.quant_depth)
        return dict(self.decode_latents(q['global_latent'],q['local_latent'],mask,tick_norm,times),
                    local_codes=q['codes'],latent_mse=q['latent_mse'])

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        if self.evaluation_mode=='original':return super().forward(price,volume,mask,tick_norm,times=times,sample=False)
        with torch.no_grad():
            p=self.encoder(price,volume,mask,tick_norm,times=times,sample=False)
            f=self.fusion([(p['z_global'],p['z_local'])],p['latent_mask'],times)
        return self.forward_latents(f['global_latent'],f['local_latent'],mask,tick_norm,times)

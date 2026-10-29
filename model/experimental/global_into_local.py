"""One Local latent for full256minute generation; no independent Global input."""

import torch
from torch.nn import functional as F

from model.experimental.full_window_matrix import FullWindowMatrixOrderBookVAE
from model.layers.global_into_local import GlobalIntoLocal,GlobalReadoutFromLocal


class GlobalIntoLocalOrderBookVAE(FullWindowMatrixOrderBookVAE):
    def __init__(self,*,merge_mode='residual',window_size=256,readout_bands=4,hidden=128,activation_checkpoint=True,**core):
        if merge_mode not in ('reference','local','residual','attention'):raise ValueError(merge_mode)
        super().__init__(window_size=window_size,compress=False,activation_checkpoint=activation_checkpoint,**core)
        self.merge_mode=merge_mode
        bands=min(readout_bands,self.positions)
        self.global_local_merge_config=dict(merge_mode=merge_mode,window_size=window_size,readout_bands=readout_bands,hidden=hidden,activation_checkpoint=activation_checkpoint)
        if merge_mode!='reference':
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42001)
                self.local_readout=GlobalReadoutFromLocal(self.global_dim,self.local_dim,bands,hidden)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42002)
                self.local_merger=GlobalIntoLocal(core.get('levels',2048),self.global_dim,self.local_dim,merge_mode,hidden)

    def load_reference_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not k.startswith(('local_merger.','local_readout.')) for k in missing):raise ValueError('Reference source differs')

    def latent_mask(self,mask):
        b,t,_,levels=mask.shape
        if t!=self.window_size:raise ValueError('Expected full256minute mask')
        return F.adaptive_max_pool1d(mask.reshape(b*t*2,1,levels).float(),self.positions).reshape(b,t,2,self.positions).bool()

    def merge_latents(self,g,l,latent_mask,times=None):
        if self.merge_mode=='reference':raise ValueError('Reference retains both original latents')
        merged=self._run(self.local_merger,g,l,latent_mask,times)
        return merged,self.local_readout(merged,latent_mask,times)

    def decode_local(self,local,mask,tick_norm,times=None):
        if self.merge_mode=='reference':raise ValueError('Reference does not use a Local-only bottleneck')
        expected=(self.window_size,2,self.positions,self.local_dim)
        if local.ndim!=5 or tuple(local.shape[1:])!=expected:raise ValueError('Wrong Local-only latent shape')
        lm=self.latent_mask(mask);local=local*lm[...,None]
        global_readout=self.local_readout(local,lm,times)
        out=self.decode_latents(global_readout,local,mask,tick_norm,times)
        return dict(out,global_readout=global_readout)

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        g,l=self.encode_fused(price,volume,mask,tick_norm,times)
        if self.merge_mode=='reference':return dict(self.decode_latents(g,l,mask,tick_norm,times),global_readout=g,local_latent=l,latent_mse=g.new_zeros(()))
        merged=self._run(self.local_merger,g,l,self.latent_mask(mask),times)
        return dict(self.decode_local(merged,mask,tick_norm,times),local_latent=merged,latent_mse=(merged-l).square().mean())

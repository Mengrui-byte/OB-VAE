"""Full256minute global/local or modality towers through one16x1024 matrix."""

import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.autoencoder.window_matrix_codec import WindowMatrixCodec
from model.experimental.dual_fusion import FusedOrderBookVAE


class FullWindowMatrixOrderBookVAE(FusedOrderBookVAE):
    def __init__(self,*,window_size=256,matrix_rows=16,matrix_dim=1024,frame_dim=256,
                 codec_kind='residual',codec_depth=2,codec_hidden=512,branch_mode='shared',
                 compress=True,activation_checkpoint=True,**core):
        if branch_mode not in ('shared','tower'):raise ValueError(branch_mode)
        super().__init__(fusion_variant='tower_cross' if branch_mode=='tower' else 'cross',fusion_seed=42,**core)
        self.window_size,self.compress,self.activation_checkpoint=window_size,compress,activation_checkpoint
        self.window_matrix_config=dict(window_size=window_size,matrix_rows=matrix_rows,matrix_dim=matrix_dim,
            frame_dim=frame_dim,codec_kind=codec_kind,codec_depth=codec_depth,codec_hidden=codec_hidden,
            branch_mode=branch_mode,compress=compress,activation_checkpoint=activation_checkpoint)
        self.global_dim=core.get('global_dim',64);self.local_dim=core.get('local_dim',16);self.positions=core.get('levels',2048)//16
        self.window_codec=WindowMatrixCodec(self.global_dim+2*self.positions*self.local_dim,window_size,
            matrix_rows,frame_dim,matrix_dim,codec_kind,codec_hidden,codec_depth) if compress else None

    def load_cross_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        allowed=('window_codec.','quantity_encoder.','fusion.')
        if unexpected or any(not key.startswith(allowed) for key in missing):raise ValueError('Source Cross architecture differs')
        if self.two_towers:
            self.quantity_encoder.load_state_dict(self.encoder.state_dict(),strict=True)
            for name in ('global_proj','local_proj','global_norm','local_norm','global_cross','local_cross'):
                modules=getattr(self.fusion,name);modules[1].load_state_dict(modules[0].state_dict(),strict=True)

    def _run(self,module,*args):
        if self.activation_checkpoint and self.training and torch.is_grad_enabled():
            return checkpoint(module,*args,use_reentrant=False)
        return module(*args)

    def encode_fused(self,price,volume,mask,tick_norm,times=None):
        if price.shape[1]!=self.window_size:raise ValueError(f'Expected a complete{self.window_size}minute window')
        posterior=self._run(self.encoder,price,volume,mask,tick_norm,times,False)
        sources=[(posterior['z_global'],posterior['z_local'])]
        if self.two_towers:
            quantity=self._run(self.quantity_encoder,price,volume,mask,tick_norm,times,False)
            sources.append((quantity['z_global'],quantity['z_local']))
        fused=self._run(self.fusion,sources,posterior['latent_mask'],times)
        return fused['global_latent'],fused['local_latent']

    def _unpack(self,x):
        return x[...,:self.global_dim],x[...,self.global_dim:].reshape(x.shape[0],self.window_size,2,self.positions,self.local_dim)

    def decode_latents(self,g,l,mask,tick_norm,times=None):
        b,t,_,levels=mask.shape
        if t!=self.window_size:raise ValueError('Decode mask must cover the complete window')
        lm=F.adaptive_max_pool1d(mask.reshape(b*t*2,1,levels).float(),self.positions).reshape(b,t,2,self.positions).bool()
        l=l*lm[...,None]
        geometry=self._run(self.price_decoder,g,l,mask,tick_norm,times)
        quantity=self._run(self.volume_decoder,g,l,geometry,mask,tick_norm,times)
        quantity=self._run(self.volume_refinement,g,l,quantity,geometry,mask,tick_norm,times)
        return dict(geometry,volume=quantity)

    def decode_matrix(self,matrix,mask,tick_norm,times=None):
        if not self.compress:raise ValueError('Uncompressed reference has no matrix bottleneck')
        g,l=self._unpack(self.window_codec.decode(matrix))
        return self.decode_latents(g,l,mask,tick_norm,times)

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        g,l=self.encode_fused(price,volume,mask,tick_norm,times)
        if not self.compress:return dict(self.decode_latents(g,l,mask,tick_norm,times),latent_mse=g.new_zeros(()))
        x=torch.cat([g,l.flatten(2)],-1);encoded=self.window_codec(x)
        gh,lh=self._unpack(encoded['reconstruction'])
        return dict(self.decode_latents(gh,lh,mask,tick_norm,times),matrix_latent=encoded['matrix'],
                    latent_mse=(encoded['reconstruction']-x).square().mean())

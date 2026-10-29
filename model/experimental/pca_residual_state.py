"""Fixed low-dimensional state, PCA inverse plus a trainable nonlinear inverse."""

import torch
from torch.nn import functional as F

from model.experimental.pca_state import PCAWindowVAE
from model.decoders.pca_residual import PCALatentResidual


class ResidualPCAWindowVAE(PCAWindowVAE):
    def __init__(self,*,hidden_dim=128,residual_seed=42,**config):
        super().__init__(**config)
        self.residual_pca_config=dict(hidden_dim=hidden_dim,residual_seed=residual_seed)
        codec=self.state_codec
        self.latent_residual=PCALatentResidual(codec.state_dim,codec.global_length+codec.local_length,hidden_dim,residual_seed)

    def load_pca_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not key.startswith('latent_residual.') for key in missing):
            raise ValueError(f'PCA warm start mismatch: {missing}, {unexpected}')

    def decode_state(self,state,mask,tick_norm,times=None):
        B,T,_,L=mask.shape;codec=self.state_codec
        latent_mask=F.adaptive_max_pool1d(mask.reshape(B*T*2,1,L).float(),L//16).reshape(B,T,2,L//16).bool()
        zg,zl=codec.decode(state,latent_mask)
        correction=self.latent_residual(state).float()
        dg=correction[:,:codec.global_length]*codec.global_scale
        dl=correction[:,codec.global_length:]*codec.local_scale
        zg=zg+dg.reshape_as(zg);zl=(zl+dl.reshape_as(zl))*latent_mask[...,None]
        geometry=self.price_decoder(zg,zl,mask,tick_norm,times=times)
        quantity=self.volume_decoder(zg,zl,geometry,mask,tick_norm,times=times)
        quantity=self.volume_refinement(zg,zl,quantity,geometry,mask,tick_norm,times)
        return {**geometry,'volume':quantity,'market_state':state}

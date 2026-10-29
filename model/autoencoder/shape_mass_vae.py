"""Fully learned original posterior, with neural shape/mass residual readout."""
import torch
from torch import nn

from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
from model.layers.shape_mass import shape_mass_residual


class ShapeMassOrderBookVAE(LocalHintOrderBookVAE):
    def __init__(self,*,use_shape=False,use_mass=False,detach_price_geometry=False,**kwargs):
        super().__init__(**kwargs);self.use_shape,self.use_mass=use_shape,use_mass
        self.volume_decoder.return_features=True
        self.volume_decoder.detach_price_geometry=detach_price_geometry
        # Appended afterevery originalparameter, includingthehintadapter.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.shape_head=nn.Conv1d(32,1,1,bias=False)
            self.mass_head=nn.Linear(256,2)
        nn.init.zeros_(self.shape_head.weight)
        nn.init.zeros_(self.mass_head.weight);nn.init.zeros_(self.mass_head.bias)

    def decode(self,z,mask,tick_norm,times):
        output=super().decode(z,mask,tick_norm,times)
        features=output.pop('quantity_features');frame=output.pop('quantity_frame')
        shape=self.shape_head(features).float().reshape_as(output['volume']) if self.use_shape else None
        mass=self.mass_head(frame).float() if self.use_mass else None
        original=output['volume']
        output['volume']=shape_mass_residual(original,mask,shape,mass)
        if self.use_shape or self.use_mass:
            output['base_volume']=original
            output['base_quantity_logit']=output.pop('quantity_logit')
        return output

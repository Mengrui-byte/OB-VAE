"""Batched regional affine heads with Linear-compatible autocast bias."""
import torch

from model.layers.local_phase import local_phase_residual
from model.layers.regional_phase import RegionalPhaseReadout


def regional_linear(seed,weight,bias):
    """One fused affine per region; preserve the matmul autocast dtype."""
    regions,features,outputs=weight.shape
    x=seed.movedim(-2,0).reshape(regions,-1,features)
    value=torch.baddbmm(bias[:,None,:],x,weight)
    return value.reshape(regions,*seed.shape[:-2],outputs).movedim(0,-2)


class RegionalLinearPhaseReadout(RegionalPhaseReadout):
    def forward(self,quantity,seed,mask,*,use_shape=False,use_mass=False):
        expected=(*quantity.shape[:-1],self.regions,self.feature_dim)
        if quantity.shape[-1]!=self.regions*self.region_width or seed.shape!=expected:
            raise ValueError('Unexpected regional grid or post-Z seed shape')
        shape=regional_linear(seed,self.shape_weight,self.shape_bias).float() if use_shape else None
        mass=regional_linear(seed,self.mass_weight[...,None],self.mass_bias[:,None]).float()[...,0] if use_mass else None
        return local_phase_residual(quantity,mask,shape,mass,self.region_width)

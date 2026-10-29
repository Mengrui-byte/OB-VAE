"""Post-latent phase selection within fixed sixteen-level regions."""
import torch
from torch import nn


def local_phase_residual(quantity,mask,shape=None,log_mass=None,region_width=16):
    """Preserve region mass for shape-only corrections and identity at zero.

    The residual form keeps exact identity even for empty/tiny regions, where a
    clamped normalizer would otherwise shrink the original quantity.
    """
    if quantity.shape!=mask.shape or quantity.shape[-1]%region_width:
        raise ValueError('Quantity/mask must match and levels must divide into regions')
    groups=quantity.shape[-1]//region_width
    grouped_shape=(*quantity.shape[:-1],groups,region_width)
    q=quantity.float().masked_fill(~mask,0).reshape(grouped_shape)
    valid=mask.reshape(grouped_shape)
    if shape is not None:
        if shape.shape!=q.shape:raise ValueError('Expected one shape logit per regional phase')
        raw=shape.float().masked_fill(~valid,0)
        center=raw-raw.sum(-1,keepdim=True)/valid.sum(-1,keepdim=True).clamp_min(1)
        increment=torch.expm1(8*torch.tanh(center/8))*valid
        mass=q.sum(-1,keepdim=True)
        change=(q*increment).sum(-1,keepdim=True)
        denominator=(q*(1+increment)).sum(-1,keepdim=True).clamp_min(1e-8)
        # Sum of the correction numerator is M*change-change*M == 0.
        correction=q*(mass*increment-change)/denominator
        q=(q+correction).clamp_min(0)
    if log_mass is not None:
        if log_mass.shape!=q.shape[:-1]:raise ValueError('Expected one mass logit per region')
        increment=torch.expm1(4*torch.tanh(log_mass.float()/4))
        q=(q+q*increment[...,None]).clamp_min(0)
    return q.reshape_as(quantity).masked_fill(~mask,0)


class LocalPhaseReadout(nn.Module):
    def __init__(self,feature_dim=16,region_width=16):
        super().__init__();self.feature_dim=feature_dim;self.region_width=region_width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.shape=nn.Linear(feature_dim,region_width)
            self.mass=nn.Linear(feature_dim,1)
        for layer in [self.shape,self.mass]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,quantity,seed,mask,*,use_shape=False,use_mass=False):
        if quantity.shape[-1]%self.region_width:raise ValueError('Incomplete regional grid')
        expected=(*quantity.shape[:-1],quantity.shape[-1]//self.region_width,self.feature_dim)
        if seed.shape!=expected:raise ValueError(f'Expected post-Z quantity seed {expected}')
        shape=self.shape(seed).float() if use_shape else None
        mass=self.mass(seed).float()[...,0] if use_mass else None
        return local_phase_residual(quantity,mask,shape,mass,self.region_width)

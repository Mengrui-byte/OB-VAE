"""Region-specific counterpart to the shared post-Z phase readout."""
import torch
from torch import nn

from model.layers.local_phase import local_phase_residual


class RegionalPhaseReadout(nn.Module):
    """Keep side/time sharing, but allow each fixed region a distinct basis."""
    def __init__(self,regions=128,feature_dim=16,region_width=16):
        super().__init__()
        self.regions,self.feature_dim,self.region_width=regions,feature_dim,region_width
        self.shape_weight=nn.Parameter(torch.zeros(regions,feature_dim,region_width))
        self.shape_bias=nn.Parameter(torch.zeros(regions,region_width))
        self.mass_weight=nn.Parameter(torch.zeros(regions,feature_dim))
        self.mass_bias=nn.Parameter(torch.zeros(regions))

    def forward(self,quantity,seed,mask,*,use_shape=False,use_mass=False):
        expected=(*quantity.shape[:-1],self.regions,self.feature_dim)
        if quantity.shape[-1]!=self.regions*self.region_width or seed.shape!=expected:
            raise ValueError('Unexpected regional grid or post-Z seed shape')
        shape=(torch.einsum('...rf,rfp->...rp',seed,self.shape_weight)+self.shape_bias).float() if use_shape else None
        mass=(torch.einsum('...rf,rf->...r',seed,self.mass_weight)+self.mass_bias).float() if use_mass else None
        return local_phase_residual(quantity,mask,shape,mass,self.region_width)

"""One Gaussian VAE with region-specific post-Z quantity readout."""
from model.autoencoder.local_phase_vae import LocalPhaseOrderBookVAE
from model.layers.regional_phase import RegionalPhaseReadout
from model.layers.regional_linear import RegionalLinearPhaseReadout


class RegionalPhaseOrderBookVAE(LocalPhaseOrderBookVAE):
    def __init__(self,*,use_shape=False,use_mass=False,readout_kernel='einsum',**kwargs):
        if readout_kernel not in ['einsum','linear']:raise ValueError('Unknown regional readout kernel')
        super().__init__(use_shape=use_shape,use_mass=use_mass,**kwargs)
        # Source Encoder/Decoder parameters keep their original names/order.
        # These are dataset-wide weights, not extra per-window latent values.
        self.readout_kernel=readout_kernel
        layer=RegionalLinearPhaseReadout if readout_kernel=='linear' else RegionalPhaseReadout
        self.local_readout=layer(regions=kwargs.get('levels',2048)//16)

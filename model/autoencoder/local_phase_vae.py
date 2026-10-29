"""Original single Gaussian with optional local quantity shape/mass readout."""
from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
from model.layers.local_phase import LocalPhaseReadout


class LocalPhaseOrderBookVAE(LocalHintOrderBookVAE):
    def __init__(self,*,use_shape=False,use_mass=False,**kwargs):
        super().__init__(**kwargs);self.use_shape,self.use_mass=use_shape,use_mass
        # Appended after the entire original parameter sequence/hint adapter.
        self.local_readout=LocalPhaseReadout(feature_dim=16,region_width=16)

    def decode(self,z,mask,tick_norm,times):
        output=super().decode(z,mask,tick_norm,times)
        original=output['volume']
        output['volume']=self.local_readout(original,output['quantity_seed'],mask,use_shape=self.use_shape,use_mass=self.use_mass)
        if self.use_shape or self.use_mass:
            output['base_volume']=original
            # The original logit belongs to the base Softplus quantity head.
            output['base_quantity_logit']=output.pop('quantity_logit')
        return output

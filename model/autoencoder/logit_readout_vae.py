"""One Gaussian, fixed price path, optional ordered quantity-logit correction."""
from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
from model.layers.logit_readout import QuantityLogitReadout,add_quantity_logit


class LogitReadoutOrderBookVAE(LocalHintOrderBookVAE):
    def __init__(self,*,feature_source='seed',**kwargs):
        super().__init__(**kwargs)
        self.feature_source=feature_source;self.volume_decoder.return_frame=True
        # Append after every source parameter; all features are decoded from Z.
        self.local_readout=QuantityLogitReadout(feature_source=feature_source,levels=kwargs.get('levels',2048))

    def decode(self,z,mask,tick_norm,times):
        output=super().decode(z,mask,tick_norm,times)
        delta=self.local_readout(output['quantity_seed'],output['quantity_frame'])
        output['base_volume']=output['volume'];output['base_quantity_logit']=output['quantity_logit']
        output['volume'],output['quantity_logit']=add_quantity_logit(output['quantity_logit'],delta,mask,self.volume_decoder.q0)
        return output

"""Original single Gaussian and CNN, with a 130-parameter paired logit head."""
from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
from model.layers.paired_logit import PairedQuantityLogitReadout,pair_features,log_quantity_residual
from model.layers.logit_readout import add_quantity_logit


class PairedLogitOrderBookVAE(LocalHintOrderBookVAE):
    def __init__(self,*,phase_specific=False,readout_space='logit',**kwargs):
        if type(phase_specific) is not bool:raise ValueError('Explicit Boolean phase flag required')
        if readout_space not in ['logit','log_quantity']:raise ValueError('Unknown paired response space')
        super().__init__(**kwargs);self.phase_specific=phase_specific
        self.readout_space=readout_space
        self.volume_decoder.return_features=True
        self.local_readout=PairedQuantityLogitReadout()
        self.return_readout_features=False

    def decode(self,z,mask,tick_norm,times):
        output=super().decode(z,mask,tick_norm,times)
        features=pair_features(output.pop('quantity_features'));output.pop('quantity_frame')
        delta=self.local_readout(features,phase_specific=self.phase_specific).reshape_as(output['volume'])
        output['base_volume']=output['volume'];output['base_quantity_logit']=output['quantity_logit']
        if self.readout_space=='logit':
            output['volume'],output['quantity_logit']=add_quantity_logit(output['quantity_logit'],delta,mask,self.volume_decoder.q0)
        else:
            output['volume']=log_quantity_residual(output['volume'],delta,mask)
            output.pop('quantity_logit')
        if self.return_readout_features:output['quantity_pair_features']=features
        return output

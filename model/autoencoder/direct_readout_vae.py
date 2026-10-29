"""One8192 Gaussian with a zero-initialized direct quantity-logit residual."""
from model.autoencoder.regional_phase_vae import RegionalPhaseOrderBookVAE
from model.layers.direct_quantity_readout import FrameQuantityLogitMLP
from model.layers.logit_readout import add_quantity_logit


class DirectReadoutOrderBookVAE(RegionalPhaseOrderBookVAE):
    def __init__(self, *, readout_activation='gelu', readout_hidden=2048, readout_seed=42, **kwargs):
        if kwargs.get('use_shape', False) or kwargs.get('use_mass', False):
            raise ValueError('Direct readout controls require inactive historical phase heads')
        super().__init__(**kwargs)
        self.volume_decoder.return_frame = True
        self.direct_readout_config = dict(readout_activation=readout_activation,
            readout_hidden=readout_hidden, readout_seed=readout_seed)
        # Append after all source parameters; no source name or order changes.
        self.direct_quantity_readout = FrameQuantityLogitMLP(levels=kwargs.get('levels', 2048),
            activation=readout_activation, hidden_dim=readout_hidden, seed=readout_seed)

    def decode(self, z, mask, tick_norm, times):
        output = super().decode(z, mask, tick_norm, times)
        delta = self.direct_quantity_readout(output['quantity_frame'])
        output['base_volume'] = output['volume']
        output['base_quantity_logit'] = output['quantity_logit']
        output['direct_quantity_logit_delta'] = delta.masked_fill(~mask, 0)
        output['volume'], output['quantity_logit'] = add_quantity_logit(
            output['quantity_logit'], delta, mask, self.volume_decoder.q0)
        return output

"""One8192-scalar posterior; auxiliary peak logits never gate quantity output."""
from model.autoencoder.unified_book_vae import UnifiedOrderBookVAE
from model.decoders.peak_volume import PeakAuxVolumeDecoder


class PeakAuxOrderBookVAE(UnifiedOrderBookVAE):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        old = self.volume_decoder
        new = PeakAuxVolumeDecoder(**kwargs)
        missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
        assert missing == ['peak_head.weight','peak_head.bias'] and not unexpected
        self.volume_decoder = new

    def decode(self, z, mask, tick_norm, times):
        if z.shape != (mask.shape[0],64,self.latent_channels):
            raise ValueError('Unique whole-window latent shape mismatch')
        price = self.price_decoder(z,mask,tick_norm,times)
        quantity = self.volume_decoder(z,price,mask,tick_norm,times)
        return dict(**price,**quantity)

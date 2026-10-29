"""Single-posterior book VAE with a direct within-region quantity residual."""
from model.autoencoder.unified_book_vae import UnifiedOrderBookVAE
from model.decoders.phase_volume import PhaseDetailVolumeDecoder


class PhaseDetailOrderBookVAE(UnifiedOrderBookVAE):
    def __init__(self, *, detail_mode='independent', **kwargs):
        super().__init__(**kwargs)
        self.phase_decoder_config = dict(detail_mode=detail_mode)
        self.volume_decoder = PhaseDetailVolumeDecoder(detail_mode=detail_mode, **kwargs)

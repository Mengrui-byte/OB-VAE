"""Preserve detailed quantities before the same single 8192-scalar posterior."""
import torch

from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
from model.encoders.fine_phase import FinePhaseFrameEncoder


class FinePhaseOrderBookVAE(LocalHintOrderBookVAE):
    def __init__(self, *, fine_enabled=False, fine_transform='log1p', fine_hidden=256,
                 layout='frame1', frame_ffn=1536, **kwargs):
        if layout != 'frame1':
            raise ValueError('Fine phase experiment requires frame1 layout')
        super().__init__(layout=layout, frame_ffn=frame_ffn, **kwargs)
        original = self.encoder
        # Construct an extended Encoder without perturbing the original model
        # initialization RNG. Keep every original Encoder tensor unchanged.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            extended = FinePhaseFrameEncoder(fine_enabled=fine_enabled, fine_transform=fine_transform,
                fine_hidden=fine_hidden, frame_ffn=frame_ffn, **kwargs)
        missing, extra = extended.load_state_dict(original.state_dict(), strict=False)
        expected = ['fine_phase.norm.weight', 'fine_phase.norm.bias', 'fine_phase.input.weight',
                    'fine_phase.input.bias', 'fine_phase.output.weight', 'fine_phase.output.bias']
        if missing != expected or extra:
            raise ValueError('Original Encoder transfer changed')
        self.encoder = extended

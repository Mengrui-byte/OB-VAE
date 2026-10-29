"""One unchanged Gaussian; quantity Decoder may read all serialized Z frames."""
import torch

from model.autoencoder.local_hint_vae import LocalHintOrderBookVAE
from model.decoders.window_context import WindowContextTemporalExpander


class QuantityContextOrderBookVAE(LocalHintOrderBookVAE):
    def __init__(self, *, context_frames=0, layout='frame1', frame_ffn=1536, **kwargs):
        if layout != 'frame1':
            raise ValueError('Window context requires the frame-aligned8192 posterior')
        super().__init__(layout=layout, frame_ffn=frame_ffn, **kwargs)
        old = self.volume_decoder.temporal
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            temporal = WindowContextTemporalExpander(context_frames)
        missing, unexpected = temporal.load_state_dict(old.state_dict(), strict=False)
        assert len(missing) == 6 and all(n.startswith('context_attention.') for n in missing)
        assert not unexpected
        self.volume_decoder.temporal = temporal

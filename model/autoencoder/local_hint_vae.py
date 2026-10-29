"""An auxiliary post-bottleneck hint head; one Gaussian and original outputs."""
import torch
from torch import nn
from model.autoencoder.time_layout_vae import TimeLayoutOrderBookVAE
from model.decoders.local_hint import LocalHintVolumeDecoder


class LocalHintOrderBookVAE(TimeLayoutOrderBookVAE):
    def __init__(self,*,layout='frame1',frame_ffn=1536,**kwargs):
        super().__init__(layout=layout,frame_ffn=frame_ffn,**kwargs)
        original=self.volume_decoder
        volume=LocalHintVolumeDecoder(**kwargs)
        volume.temporal=original.temporal
        volume.load_state_dict(original.state_dict(),strict=True)
        self.volume_decoder=volume
        # Append after original parameters; no effect on generation or its RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.hint_adapter=nn.Linear(16,16)
        with torch.no_grad():
            self.hint_adapter.weight.copy_(torch.eye(16));self.hint_adapter.bias.zero_()

    def decode(self,z,mask,tick_norm,times):
        out=super().decode(z,mask,tick_norm,times)
        out['hint_local']=self.hint_adapter(out['quantity_seed'])
        return out

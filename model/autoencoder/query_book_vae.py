"""One shared8192-scalar latent with a full nonlinear query quantity decoder."""
from model.autoencoder.unified_book_vae import UnifiedOrderBookVAE
from model.decoders.query_volume import QueryVolumeDecoder


class QueryOrderBookVAE(UnifiedOrderBookVAE):
    def __init__(self,*,frame_hidden=1408,**kwargs):
        super().__init__(**kwargs)
        old=self.volume_decoder
        new=QueryVolumeDecoder(frame_hidden=frame_hidden,**kwargs)
        for name in ['temporal','position','geometry']:
            getattr(new,name).load_state_dict(getattr(old,name).state_dict(),strict=True)
        self.volume_decoder=new

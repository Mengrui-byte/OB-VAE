"""Single8192-scalar posterior with patch4 or frame1 temporal organization."""
from model.autoencoder.unified_book_vae import UnifiedOrderBookVAE
from model.encoders.frame_aligned import FrameAlignedEncoder
from model.decoders.frame_aligned import FrameTemporalExpander
from model.decoders.logit_volume import LogitVolumeDecoder


class TimeLayoutOrderBookVAE(UnifiedOrderBookVAE):
    def __init__(self,*,layout='patch4',frame_ffn=1536,**kwargs):
        if layout not in ['patch4','frame1']:raise ValueError('Unknown temporal layout')
        super().__init__(**kwargs);self.layout=layout
        original=self.volume_decoder;volume=LogitVolumeDecoder(**kwargs)
        volume.load_state_dict(original.state_dict(),strict=True);self.volume_decoder=volume
        if layout=='frame1':
            old=self.encoder;new=FrameAlignedEncoder(frame_ffn=frame_ffn,**kwargs)
            state=new.state_dict();shared={k:v for k,v in old.state_dict().items() if k in state and v.shape==state[k].shape}
            new.load_state_dict(shared,strict=False);self.encoder=new
            for decoder in [self.price_decoder,self.volume_decoder]:
                temporal=FrameTemporalExpander()
                temporal.refine.load_state_dict(decoder.temporal.refine.state_dict(),strict=True)
                decoder.temporal=temporal

    def decode(self,z,mask,tick_norm,times):
        if z.shape!=(mask.shape[0],64,128):raise ValueError('Expected unique64x128 latent')
        price=self.price_decoder(z,mask,tick_norm,times)
        return dict(**price,**self.volume_decoder(z,price,mask,tick_norm,times))

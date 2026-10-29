"""Frame or regional organization under the same unique8192 Gaussian budget."""
from model.autoencoder.time_layout_vae import TimeLayoutOrderBookVAE
from model.encoders.region_grid import RegionGridEncoder
from model.decoders.region_grid import RegionSpatialDecoder,RegionPriceDecoder,RegionVolumeDecoder


def copy_compatible(destination, source):
    state = destination.state_dict()
    shared = {k:v for k,v in source.state_dict().items() if k in state and state[k].shape==v.shape}
    destination.load_state_dict(shared,strict=False)


class RegionGridOrderBookVAE(TimeLayoutOrderBookVAE):
    def __init__(self, *, organization='frame', frame_ffn=1536,
                 temporal_ffn=2048, regional_ffn=2048, seed_hidden=1024, **kwargs):
        if organization not in ('frame','region'):
            raise ValueError('Unknown spatial organization')
        super().__init__(layout='frame1',frame_ffn=frame_ffn,**kwargs)
        self.organization = organization
        if organization == 'frame':
            return
        encoder = RegionGridEncoder(temporal_ffn=temporal_ffn,regional_ffn=regional_ffn,**kwargs)
        copy_compatible(encoder,self.encoder)
        self.encoder = encoder
        for name,cls,conditional in [('price_decoder',RegionPriceDecoder,False),('volume_decoder',RegionVolumeDecoder,True)]:
            old = getattr(self,name)
            if conditional:
                decoder = cls(**kwargs)
            else:
                decoder = cls(**{k:v for k,v in kwargs.items() if k!='quantity_scale'})
            decoder.temporal = old.temporal
            decoder.spatial = RegionSpatialDecoder(old.spatial.levels,conditional,old.spatial.activation_checkpointing,seed_hidden)
            copy_compatible(decoder,old)
            setattr(self,name,decoder)

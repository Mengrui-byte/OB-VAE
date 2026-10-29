"""Hybrid sparse-address Gaussian AE; no values bypass final64x128 code."""
import torch
from model.autoencoder.time_layout_vae import TimeLayoutOrderBookVAE
from model.encoders.sparse_address import SparseAddressEncoder
from model.decoders.sparse_address import SparseAddressVolumeDecoder
from model.layers.sparse_address import SparsePriceExpander,price_dimension


class SparseAddressOrderBookVAE(TimeLayoutOrderBookVAE):
    def __init__(self,*,peak_slots=2048,use_addresses=True,frame_ffn=1536,**kwargs):
        super().__init__(layout='frame1',frame_ffn=frame_ffn,**kwargs)
        self.peak_slots=peak_slots;self.price_dim=price_dimension(peak_slots)
        self.encoder=SparseAddressEncoder(peak_slots=peak_slots,frame_ffn=frame_ffn,**kwargs)
        original=self.volume_decoder
        volume=SparseAddressVolumeDecoder(peak_slots=peak_slots,use_addresses=use_addresses,**kwargs)
        volume.temporal=original.temporal;self.volume_decoder=volume
        for decoder in [self.price_decoder,self.volume_decoder]:
            decoder.temporal=SparsePriceExpander(decoder.temporal,peak_slots)

    def load_dense_initial(self,state,basis,mean):
        if basis.shape!=(32,self.price_dim) or mean.shape!=(32,):raise ValueError('Wrong training-only price projection')
        destination=self.state_dict()
        shared={k:v for k,v in state.items() if k in destination and v.shape==destination[k].shape}
        self.load_state_dict(shared,strict=False)
        with torch.no_grad():
            self.encoder.mu.weight.copy_(basis.T@state['encoder.mu.weight'])
            self.encoder.mu.bias.copy_(basis.T@(state['encoder.mu.bias']-mean))
            for decoder in [self.price_decoder,self.volume_decoder]:
                decoder.temporal.project.weight.copy_(basis)
                decoder.temporal.project.bias.copy_(mean)

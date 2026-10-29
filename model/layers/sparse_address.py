"""All addresses/amplitudes and price coordinates live inside the8192 scalars."""
import torch
from torch import nn
from model.decoders.frame_aligned import FrameTemporalExpander


def price_dimension(peak_slots):
    if not isinstance(peak_slots,int) or peak_slots<0 or peak_slots%128 or peak_slots>3840:
        raise ValueError('peak_slots must be a multipleof128 from0to3840')
    return (8192-2*peak_slots)//256


def split_code(z,peak_slots):
    if z.ndim!=3 or z.shape[1:]!=(64,128):raise ValueError('Expected unique[B,64,128]')
    d=price_dimension(peak_slots);flat=z.reshape(z.shape[0],8192);end=256*d
    return flat[:,:end].reshape(z.shape[0],256,d),flat[:,end:end+peak_slots],flat[:,end+peak_slots:]


class SparsePriceExpander(FrameTemporalExpander):
    def __init__(self,original,peak_slots):
        super().__init__();self.peak_slots=peak_slots
        self.project=nn.Linear(price_dimension(peak_slots),32)
        self.mlp=original.mlp;self.refine=original.refine

    def forward(self,z,times):
        price,_,_=split_code(z,self.peak_slots)
        # Parent method's frame unflattening is fixed32; only pass it features
        # recovered fromthe price partofthesame finalZ, never Encoder state.
        expanded=self.project(price).reshape(z.shape[0],64,128)
        return super().forward(expanded,times)

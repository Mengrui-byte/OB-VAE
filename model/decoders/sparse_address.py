"""Neural background plus sparse values read only from the uniqueZ."""
import torch
from model.decoders.logit_volume import LogitVolumeDecoder
from model.layers.sparse_address import split_code


class SparseAddressVolumeDecoder(LogitVolumeDecoder):
    def __init__(self,*,peak_slots=2048,use_addresses=True,**kwargs):
        super().__init__(**kwargs);self.peak_slots=peak_slots;self.use_addresses=use_addresses

    def forward(self,z,price_output,mask,tick_norm,times):
        out=super().forward(z,price_output,mask,tick_norm,times)
        if self.peak_slots and self.use_addresses:
            _,positions,amplitudes=split_code(z,self.peak_slots)
            flat=out['volume'].reshape(z.shape[0],-1)
            indices=positions.round().clamp(0,flat.shape[-1]-1).long()
            quantity=flat.scatter(1,indices,amplitudes.clamp_min(0)).reshape_as(out['volume']).masked_fill(~mask,0)
            out['background_volume']=out['volume'];out['background_quantity_logit']=out.pop('quantity_logit')
            out['volume']=quantity
        return out

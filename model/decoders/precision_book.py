"""Optional FP32 price/quantity branches; state_dict and parameter IDs preserved."""
import torch
from torch import nn
from model.decoders.unified_book import PriceDecoder
from model.decoders.local_hint import LocalHintVolumeDecoder


class PrecisionPriceDecoder(PriceDecoder):
    force_float32=False

    def forward(self,z,mask,tick_norm,times):
        if not self.force_float32:return super().forward(z,mask,tick_norm,times)
        with torch.autocast(device_type=z.device.type,enabled=False):
            return super().forward(z.float(),mask,tick_norm.float(),times.float())


class PrecisionHintVolumeDecoder(LocalHintVolumeDecoder):
    force_float32=False

    def forward(self,z,price_output,mask,tick_norm,times):
        if not self.force_float32:return super().forward(z,price_output,mask,tick_norm,times)
        with torch.autocast(device_type=z.device.type,enabled=False):
            return super().forward(z.float(),price_output,mask,tick_norm.float(),times.float())


class PrecisionHintAdapter(nn.Linear):
    force_float32=False

    def forward(self,inputs):
        if not self.force_float32:return super().forward(inputs)
        with torch.autocast(device_type=inputs.device.type,enabled=False):return super().forward(inputs.float())

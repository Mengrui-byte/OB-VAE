"""Optional whole-frame-encoder FP32 arithmetic without new parameters."""
import torch
from model.encoders.frame_aligned import FrameAlignedEncoder


class PrecisionFrameEncoder(FrameAlignedEncoder):
    force_float32=False

    def forward(self,price,volume,mask,tick_norm,times):
        if not self.force_float32:return super().forward(price,volume,mask,tick_norm,times)
        with torch.autocast(device_type=price.device.type,enabled=False):
            return super().forward(price.float(),volume.float(),mask,tick_norm.float(),times.float())

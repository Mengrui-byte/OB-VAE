"""Fixed, invertible serialization of ordered time/side/region coordinates."""
import torch
from torch import nn


def pack_grid(value):
    if value.ndim != 5 or value.shape[1:] != (64, 2, 8, 8):
        raise ValueError('Expected [B,64,2,8,8]')
    return value.reshape(value.shape[0], 64, 128)


def unpack_grid(value):
    if value.ndim != 3 or value.shape[1:] != (64, 128):
        raise ValueError('Expected unique [B,64,128] latent')
    return value.reshape(value.shape[0], 64, 2, 8, 8)


class RegionSeed(nn.Module):
    """Each regional code emits four frames and its own ordered subregions."""
    def __init__(self, levels=2048, hidden=1024):
        super().__init__()
        if levels < 128 or levels % 128:
            raise ValueError('Eight regions per side require levels divisible by128')
        self.subregions = levels // 128
        self.mlp = nn.Sequential(nn.Linear(8, hidden), nn.GELU(),
                                 nn.Linear(hidden, 4*self.subregions*16))
        self.frame_condition = nn.Linear(256, 16)

    def forward(self, z, frame):
        batch = z.shape[0]
        h = self.mlp(unpack_grid(z)).reshape(batch,64,2,8,4,self.subregions,16)
        h = h.permute(0,1,4,2,3,5,6).reshape(batch,256,2,8*self.subregions,16)
        return h + self.frame_condition(frame)[:,:,None,None,:]

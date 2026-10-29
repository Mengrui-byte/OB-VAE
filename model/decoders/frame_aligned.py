"""Unpack one64x128 code into ordered256x32 frame states before decoding."""
import torch
from torch import nn
from model.dual.book_layers import Residual1d,sinusoidal


class FrameTemporalExpander(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp=nn.Sequential(nn.Linear(32,2048),nn.GELU(),nn.Linear(2048,256))
        self.refine=Residual1d(256)

    @staticmethod
    def unpack_frames(z):
        if z.ndim!=3 or z.shape[1:]!=(64,128):raise ValueError('Expected[B,64,128]')
        return z.reshape(z.shape[0],256,32)

    def forward(self,z,times):
        h=self.mlp(self.unpack_frames(z));B=z.shape[0]
        h=h+sinusoidal(times,256).to(h.dtype)
        return self.refine(h.transpose(1,2),h.new_ones(B,1,256)).transpose(1,2)

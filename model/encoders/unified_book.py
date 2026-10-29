"""Ordered spatial reduction and whole-window single Gaussian posterior."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from model.layers.book_features import BookFeatureExtractor
from model.dual.book_layers import PositionCode, Residual1d, sinusoidal


class UnifiedBookEncoder(nn.Module):
    def __init__(self, levels=2048, quantity_scale=1., activation_checkpointing=True, latent_channels=128):
        super().__init__()
        self.levels = levels
        self.activation_checkpointing = activation_checkpointing
        self.features = BookFeatureExtractor(quantity_scale)
        self.input_proj = nn.Conv1d(16, 32, 1)
        channels = (32, 64, 128, 192, 256)
        self.down = nn.ModuleList(nn.Conv1d(a, b, 4, 2, 1) for a,b in zip(channels[:-1], channels[1:]))
        self.spatial = nn.ModuleList(Residual1d(c, d) for c,d in zip(channels[1:], (1,1,2,2)))
        self.position = PositionCode(levels)
        self.position_proj = nn.Conv1d(32, 256, 1)
        self.cross_norm = nn.LayerNorm(256)
        self.cross = nn.MultiheadAttention(256, 8, batch_first=True)
        self.cross_ffn = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,1024), nn.GELU(), nn.Linear(1024,256))
        self.region = nn.Sequential(nn.LayerNorm(256), nn.Linear(256,16), nn.GELU())
        self.frame = nn.Sequential(nn.LayerNorm(2*(levels//16)*16), nn.Linear(2*(levels//16)*16,512), nn.GELU())
        self.patch = nn.Linear(2048,256)
        self.temporal = nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,1024,dropout=0.,activation='gelu',batch_first=True,norm_first=True), 2, enable_nested_tensor=False)
        self.mu = nn.Linear(256,latent_channels)
        self.logvar = nn.Linear(256,latent_channels)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, -4.)

    def _space(self, features, mask):
        h = self.input_proj(features)*mask
        for down, residual in zip(self.down, self.spatial):
            h = down(h)
            mask = F.max_pool1d(mask,2)
            h = residual(h*mask,mask)
        return h, mask

    def forward(self, price, volume, mask, tick_norm, times):
        B,T,S,L = price.shape
        if T != 256 or S != 2 or L != self.levels:
            raise ValueError('Expected complete [B,256,2,levels] window')
        features = self.features(price, volume, mask, tick_norm)
        m = mask.reshape(B*T*2,1,L).float()
        if self.training and self.activation_checkpointing:
            h,m = checkpoint(self._space, features, m, use_reentrant=False)
        else:
            h,m = self._space(features,m)
        K = L//16
        pos = self.position(B,T,K,h,times).reshape(B*T*2,32,K)
        h = ((h+self.position_proj(pos))*m).reshape(B*T,2,256,K).transpose(-1,-2)
        valid = m.reshape(B*T,2,K).bool()
        v = self.cross_norm(h)
        bid = self.cross(v[:,0],v[:,1],v[:,1],key_padding_mask=~valid[:,1],need_weights=False)[0]
        ask = self.cross(v[:,1],v[:,0],v[:,0],key_padding_mask=~valid[:,0],need_weights=False)[0]
        h = h+torch.stack((bid,ask),1)
        h = (h+self.cross_ffn(h))*valid[...,None]
        regional = self.region(h)*valid[...,None]
        frame = self.frame(regional.reshape(B,T,-1))
        h = self.patch(frame.reshape(B,64,2048))
        h = h+sinusoidal(times.reshape(B,64,4).mean(-1),256).to(h.dtype)
        h = self.temporal(h)
        return self.mu(h).float(), self.logvar(h).float().clamp(-12,8)

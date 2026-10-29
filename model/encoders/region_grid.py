"""Ordered regional posterior:64 time patches x2 sides x8 regions x8 channels."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.encoders.unified_book import UnifiedBookEncoder
from model.dual.book_layers import sinusoidal
from model.layers.region_grid import pack_grid


class RegionGridEncoder(UnifiedBookEncoder):
    def __init__(self, levels=2048, quantity_scale=1., activation_checkpointing=True,
                 latent_channels=128, temporal_ffn=2048, regional_ffn=2048):
        if latent_channels != 128 or levels < 128 or levels % 128:
            raise ValueError('Unique64x128 posterior and8 regions per side required')
        super().__init__(levels, quantity_scale, activation_checkpointing, latent_channels)
        del self.frame
        self.patch = nn.Linear(4*(levels//128)*16, 256)
        layer = nn.TransformerEncoderLayer(256,8,temporal_ffn,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.temporal = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.regional = nn.TransformerEncoderLayer(256,8,regional_ffn,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.mu = nn.Linear(256,8)
        self.logvar = nn.Linear(256,8)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias,-4.)

    @staticmethod
    def safe_padding(valid):
        # Transformer cannot attend an all-padding row. Dummy position is
        # subsequently masked out and never becomes an additional condition.
        safe = valid.clone()
        safe[~safe.any(-1),0] = True
        return ~safe

    def _context(self, h, valid):
        batch = h.shape[0]
        v = valid.permute(0,2,3,1).reshape(batch*16,64)
        h = h.permute(0,2,3,1,4).reshape(batch*16,64,256)
        h = self.temporal(h,src_key_padding_mask=self.safe_padding(v))*v[...,None]
        h = h.reshape(batch,2,8,64,256).permute(0,3,1,2,4).reshape(batch*64,16,256)
        v = valid.reshape(batch*64,16)
        h = self.regional(h,src_key_padding_mask=self.safe_padding(v))*v[...,None]
        return h.reshape(batch,64,2,8,256)

    def forward(self, price, volume, mask, tick_norm, times):
        B,T,S,L = price.shape
        if (T,S,L) != (256,2,self.levels):
            raise ValueError('Expected complete256-frame two-sided book')
        features = self.features(price,volume,mask,tick_norm)
        m = mask.reshape(B*T*2,1,L).float()
        if self.training and self.activation_checkpointing:
            h,m = checkpoint(self._space,features,m,use_reentrant=False)
        else:
            h,m = self._space(features,m)
        K = L//16
        pos = self.position(B,T,K,h,times).reshape(B*T*2,32,K)
        h = ((h+self.position_proj(pos))*m).reshape(B*T,2,256,K).transpose(-1,-2)
        valid = m.reshape(B*T,2,K).bool()
        v = self.cross_norm(h)
        bid = self.cross(v[:,0],v[:,1],v[:,1],key_padding_mask=self.safe_padding(valid[:,1]),need_weights=False)[0]
        ask = self.cross(v[:,1],v[:,0],v[:,0],key_padding_mask=self.safe_padding(valid[:,0]),need_weights=False)[0]
        h = h+torch.stack((bid,ask),1)
        h = (h+self.cross_ffn(h))*valid[...,None]
        h = (self.region(h)*valid[...,None]).reshape(B,64,4,2,8,K//8,16)
        h = h.permute(0,1,3,4,2,5,6).reshape(B,64,2,8,-1)
        valid = valid.reshape(B,64,4,2,8,K//8).any(2).any(-1)
        h = self.patch(h)
        patch_times = times.reshape(B,64,4).mean(-1)
        h = h+sinusoidal(patch_times,256).to(h.dtype)[:,:,None,None,:]
        if self.training and self.activation_checkpointing:
            h = checkpoint(self._context,h,valid,use_reentrant=False)
        else:
            h = self._context(h,valid)
        mu = self.mu(h).float()*valid[...,None]
        lv = self.logvar(h).float().clamp(-12,8)*valid[...,None]
        return pack_grid(mu),pack_grid(lv)

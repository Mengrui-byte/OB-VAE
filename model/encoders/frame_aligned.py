"""Frame-aligned posterior, serialized as the same single64x128 Gaussian."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from model.encoders.unified_book import UnifiedBookEncoder
from model.dual.book_layers import sinusoidal


class FrameAlignedEncoder(UnifiedBookEncoder):
    def __init__(self,levels=2048,quantity_scale=1.,activation_checkpointing=True,latent_channels=128,frame_ffn=1536):
        if latent_channels!=128:raise ValueError('External posterior must remain64x128')
        super().__init__(levels,quantity_scale,activation_checkpointing,latent_channels)
        self.patch=nn.Linear(512,256)
        self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,frame_ffn,dropout=0.,activation='gelu',batch_first=True,norm_first=True),2,enable_nested_tensor=False)
        self.mu=nn.Linear(256,32);self.logvar=nn.Linear(256,32)
        nn.init.zeros_(self.logvar.weight);nn.init.constant_(self.logvar.bias,-4.)

    @staticmethod
    def pack_frames(value):
        if value.ndim!=3 or value.shape[1:]!=(256,32):raise ValueError('Expected[B,256,32]')
        return value.reshape(value.shape[0],64,128)

    def encode_context(self,price,volume,mask,tick_norm,times):
        B,T,S,L=price.shape
        if T!=256 or S!=2 or L!=self.levels:raise ValueError('Expected complete256frame book')
        features=self.features(price,volume,mask,tick_norm);m=mask.reshape(B*T*2,1,L).float()
        if self.training and self.activation_checkpointing:h,m=checkpoint(self._space,features,m,use_reentrant=False)
        else:h,m=self._space(features,m)
        K=L//16;pos=self.position(B,T,K,h,times).reshape(B*T*2,32,K)
        h=((h+self.position_proj(pos))*m).reshape(B*T,2,256,K).transpose(-1,-2)
        valid=m.reshape(B*T,2,K).bool();v=self.cross_norm(h)
        bid=self.cross(v[:,0],v[:,1],v[:,1],key_padding_mask=~valid[:,1],need_weights=False)[0]
        ask=self.cross(v[:,1],v[:,0],v[:,0],key_padding_mask=~valid[:,0],need_weights=False)[0]
        h=h+torch.stack((bid,ask),1);h=(h+self.cross_ffn(h))*valid[...,None]
        frame=self.frame((self.region(h)*valid[...,None]).reshape(B,T,-1))
        h=self.temporal(self.patch(frame)+sinusoidal(times,256).to(frame.dtype))
        return h

    def forward(self,price,volume,mask,tick_norm,times):
        h=self.encode_context(price,volume,mask,tick_norm,times)
        return self.pack_frames(self.mu(h).float()),self.pack_frames(self.logvar(h).float().clamp(-12,8))

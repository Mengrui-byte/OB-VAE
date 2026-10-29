"""8192-scalar whole-window VAE: one posterior, one draw, two decoders."""
import torch
from torch import nn
from model.encoders.unified_book import UnifiedBookEncoder
from model.decoders.unified_book import PriceDecoder, VolumeDecoder


class UnifiedOrderBookVAE(nn.Module):
    def __init__(self, levels=2048, quantity_scale=1., activation_checkpointing=True, latent_channels=128):
        super().__init__()
        if levels % 16 or levels < 16:
            raise ValueError('levels must be a positive multiple of16')
        self.latent_channels = latent_channels
        self.encoder = UnifiedBookEncoder(levels,quantity_scale,activation_checkpointing,latent_channels)
        self.price_decoder = PriceDecoder(levels,activation_checkpointing,latent_channels)
        self.volume_decoder = VolumeDecoder(levels,quantity_scale,activation_checkpointing,latent_channels)

    def decode(self, z, mask, tick_norm, times):
        if z.shape != (mask.shape[0],64,self.latent_channels):
            raise ValueError('Unique whole-window latent shape does not match configured channels')
        price = self.price_decoder(z,mask,tick_norm,times)
        volume = self.volume_decoder(z,price,mask,tick_norm,times)
        return dict(**price,volume=volume)

    def forward(self, price, volume, mask, tick_norm, times=None, sample=None):
        if times is None:
            times = torch.arange(256,device=price.device)[None].expand(price.shape[0],-1)
        mu,logvar = self.encoder(price,volume,mask,tick_norm,times)
        if sample is None:
            sample = self.training
        z = mu+torch.exp(.5*logvar)*torch.randn_like(mu) if sample else mu
        return dict(**self.decode(z,mask,tick_norm,times),mu=mu,logvar=logvar,z=z)

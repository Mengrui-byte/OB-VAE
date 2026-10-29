"""PCA window state with an unchanged pretrained cross generator interface."""

import torch
from torch.nn import functional as F

from model.experimental.dual_fusion import FusedOrderBookVAE
from model.layers.grouped_pca import GroupedPCACodec


class PCAWindowVAE(FusedOrderBookVAE):
    def __init__(self, *, state_dim, global_components, window_size=16,
                 pca_seed=42, **core_config):
        super().__init__(fusion_variant='cross', fusion_seed=pca_seed, **core_config)
        self.pca_state_config = dict(state_dim=state_dim, global_components=global_components,
                                     window_size=window_size, pca_seed=pca_seed)
        self.state_codec = GroupedPCACodec(state_dim=state_dim, global_components=global_components,
                window_size=window_size, levels=core_config.get('levels',2048),
                global_dim=core_config.get('global_dim',64), local_dim=core_config.get('local_dim',16))
        self.freeze_source()

    def freeze_source(self):
        self.encoder.requires_grad_(False);self.fusion.requires_grad_(False)
        self.encoder.eval();self.fusion.eval()

    def load_cross_initial(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith('state_codec.') for key in missing):
            raise ValueError(f'Cross warm start mismatch: {missing}, {unexpected}')

    def extract_market_state(self, price, volume, mask, tick_norm, times=None):
        self.freeze_source()
        with torch.no_grad():
            p = self.encoder(price,volume,mask,tick_norm,times=times,sample=False)
            f = self.fusion([(p['z_global'],p['z_local'])],p['latent_mask'],times)
            return self.state_codec.encode(f['global_latent'],f['local_latent'])

    def decode_state(self, state, mask, tick_norm, times=None):
        B,T,_,L = mask.shape
        latent_mask = F.adaptive_max_pool1d(mask.reshape(B*T*2,1,L).float(),L//16).reshape(B,T,2,L//16).bool()
        zg,zl = self.state_codec.decode(state,latent_mask)
        geometry = self.price_decoder(zg,zl,mask,tick_norm,times=times)
        quantity = self.volume_decoder(zg,zl,geometry,mask,tick_norm,times=times)
        quantity = self.volume_refinement(zg,zl,quantity,geometry,mask,tick_norm,times)
        return {**geometry,'volume':quantity,'market_state':state}

    def forward(self, price, volume, mask, tick_norm, times=None, sample=None):
        # The source and PCA are deterministic. Only the generation modules are
        # adapted; the frozen source posterior KL is not optimized in this stage.
        state = self.extract_market_state(price,volume,mask,tick_norm,times)
        return self.decode_state(state,mask,tick_norm,times)

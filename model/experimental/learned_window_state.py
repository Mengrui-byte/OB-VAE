"""One task-adapted window vector with a frozen pretrained cross source."""

import torch

from model.encoders.window_residual import WindowEncodingResidual
from model.experimental.pca_residual_state import ResidualPCAWindowVAE


class LearnedWindowStateVAE(ResidualPCAWindowVAE):
    def __init__(self, *, encoding_kind='nonlinear', encoding_hidden_dim=256,
                 attention_dim=64, encoding_seed=42, **config):
        super().__init__(**config)
        self.learned_state_config = dict(encoding_kind=encoding_kind,
            encoding_hidden_dim=encoding_hidden_dim, attention_dim=attention_dim,
            encoding_seed=encoding_seed)
        codec = self.state_codec
        self.encoding_residual = WindowEncodingResidual(kind=encoding_kind,
            state_dim=codec.state_dim, window_size=codec.window_size,
            spatial_tokens=codec.levels // 16, global_dim=codec.global_dim,
            local_dim=codec.local_dim, hidden_dim=encoding_hidden_dim,
            attention_dim=attention_dim, seed=encoding_seed)

    def load_residual_initial(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith('encoding_residual.') for key in missing):
            raise ValueError(f'Learned-state warm start mismatch: {missing}, {unexpected}')

    def encode_latents(self, zg, zl, latent_mask):
        codec = self.state_codec
        base = codec.encode(zg, zl)
        g = ((zg.float().flatten(1) - codec.global_mean) / codec.global_scale).reshape_as(zg)
        l = ((zl.float().flatten(1) - codec.local_mean) / codec.local_scale).reshape_as(zl)
        return base + self.encoding_residual(g, l, latent_mask).float()

    def extract_market_state(self, price, volume, mask, tick_norm, times=None):
        self.freeze_source()
        with torch.no_grad():
            p = self.encoder(price, volume, mask, tick_norm, times=times, sample=False)
            f = self.fusion([(p['z_global'], p['z_local'])], p['latent_mask'], times)
        # Compression remains outside no_grad, even though the source is frozen.
        return self.encode_latents(f['global_latent'], f['local_latent'], p['latent_mask'])

"""Window-level market state aggregation with reconstruction through one vector."""

import torch
from torch.nn import functional as F

from model.experimental.dual_fusion import FusedOrderBookVAE
from model.layers.window_state import WindowStateCodec


class WindowStateVAE(FusedOrderBookVAE):
    def __init__(self, *, state_dim=1024, state_slots=16, state_width=64,
                 state_seed=42, **core_config):
        super().__init__(fusion_variant="cross", fusion_seed=state_seed, **core_config)
        self.state_config = dict(state_dim=state_dim, state_slots=state_slots,
                                 state_width=state_width, state_seed=state_seed)
        self.window_codec = WindowStateCodec(
            core_config.get("levels", 2048), core_config.get("global_dim", 64),
            core_config.get("local_dim", 16), state_dim=state_dim, slots=state_slots,
            width=state_width, seed=state_seed)

    def load_cross_initial(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not name.startswith("window_codec.") for name in missing):
            raise ValueError(f"Cross warm start differs: missing={missing}, unexpected={unexpected}")

    def encode_window(self, price, volume, mask, tick_norm, times=None, sample=False):
        posterior, sources = self._posterior_sources(price, volume, mask, tick_norm, times, sample)
        fused = self.fusion(sources, posterior["latent_mask"], times)
        compact = self.window_codec.compress(fused["unified"], posterior["latent_mask"], times)
        return dict(market_state=compact["state"], aggregation_tokens=compact["aggregation_tokens"],
                    posterior=posterior)

    def decode_window(self, market_state, mask, tick_norm, times=None):
        B, T, _, L = mask.shape
        K = L // 16
        latent_mask = F.adaptive_max_pool1d(mask.reshape(B * T * 2, 1, L).float(), K)
        latent_mask = latent_mask.reshape(B, T, 2, K).bool()
        zg, zl = self.window_codec.expand(market_state, latent_mask, times)
        geometry = self.price_decoder(zg, zl, mask, tick_norm, times=times)
        quantity = self.volume_decoder(zg, zl, geometry, mask, tick_norm, times=times)
        quantity = self.volume_refinement(zg, zl, quantity, geometry, mask, tick_norm, times)
        return {**geometry, "volume": quantity}

    def extract_market_state(self, price, volume, mask, tick_norm, times=None):
        """One deterministic vector for the complete observed input window.

        The vector is available at the window endpoint. No per-minute state or
        causal interpretation of the earlier rows is implied.
        """
        return self.encode_window(price, volume, mask, tick_norm, times, sample=False)["market_state"]

    def forward(self, price, volume, mask, tick_norm, times=None, sample=None):
        sample = self.training if sample is None else sample
        encoded = self.encode_window(price, volume, mask, tick_norm, times, sample)
        decoded = self.decode_window(encoded["market_state"], mask, tick_norm, times)
        return {**decoded, **encoded}

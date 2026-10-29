"""Shared/two-tower encoders with a single cross-attention generation memory."""

import torch

from model.encoders.dual_tower import ModalityOrderBookEncoder
from model.experimental.dual_long import ResearchOrderBookVAE
from model.layers.cross_latent import UnifiedCrossLatent


FUSION_VARIANTS = ("control", "cross", "tower_mean", "tower_cross")


class FusedOrderBookVAE(ResearchOrderBookVAE):
    def __init__(self, *, fusion_variant="cross", fusion_seed=42, **core_config):
        if fusion_variant not in FUSION_VARIANTS:
            raise ValueError(f"Unknown fusion variant: {fusion_variant}")
        # Carry forward the quantity-amplitude head for all four arms.
        super().__init__(variant="volume", refinement_seed=fusion_seed, **core_config)
        self.fusion_variant, self.fusion_seed = fusion_variant, fusion_seed
        self.two_towers = fusion_variant in ("tower_mean", "tower_cross")
        encoder_config = {key: core_config[key] for key in ("levels", "quantity_scale", "global_dim", "local_dim", "causal") if key in core_config}
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(fusion_seed + 101)
            if self.two_towers:
                self.encoder = ModalityOrderBookEncoder(modality="price", **encoder_config)
                self.quantity_encoder = ModalityOrderBookEncoder(modality="quantity", **encoder_config)
            if fusion_variant != "control":
                self.fusion = UnifiedCrossLatent(
                    core_config.get("levels", 2048), core_config.get("global_dim", 64), core_config.get("local_dim", 16),
                    sources=2 if self.two_towers else 1, causal=core_config.get("causal", False),
                    attention=fusion_variant != "tower_mean")

    def load_volume_initial(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any(not key.startswith(("fusion.", "quantity_encoder.")) for key in missing):
            raise ValueError(f"Warm start differs: missing={missing}, unexpected={unexpected}")
        if self.two_towers:
            self.quantity_encoder.load_state_dict(self.encoder.state_dict(), strict=True)

    def _posterior_sources(self, price, volume, mask, tick_norm, times, sample):
        if not self.two_towers:
            posterior = self.encoder(price, volume, mask, tick_norm, times=times, sample=sample)
            return posterior, [(posterior["z_global"], posterior["z_local"])]
        p = self.encoder(price, volume, mask, tick_norm, times=times, sample=False)
        q = self.quantity_encoder(price, volume, mask, tick_norm, times=times, sample=False)
        # Shared Gaussian noise aligns the two streams and the paired baseline;
        # KL is explicitly averaged over the two marginal posteriors in the loss.
        sample = self.training if sample is None else sample
        noise_g = torch.randn_like(p["mu_global"]) if sample else None
        noise_l = torch.randn_like(p["mu_local"]) if sample else None
        sources = []
        for branch in (p, q):
            zg = branch["mu_global"]
            zl = branch["mu_local"]
            if sample:
                zg = zg + torch.exp(.5 * branch["logvar_global"]) * noise_g
                zl = zl + torch.exp(.5 * branch["logvar_local"]) * noise_l
            sources.append((zg, zl * branch["latent_mask"][..., None]))
        posterior = dict(p, towers=[p, q], z_global=(sources[0][0] + sources[1][0]) * .5,
                         z_local=(sources[0][1] + sources[1][1]) * .5)
        return posterior, sources

    def forward(self, price, volume, mask, tick_norm, times=None, sample=None):
        if self.fusion_variant == "control":
            return super().forward(price, volume, mask, tick_norm, times=times, sample=sample)
        posterior, sources = self._posterior_sources(price, volume, mask, tick_norm, times, sample)
        fused = self.fusion(sources, posterior["latent_mask"], times)
        zg, zl = fused["global_latent"], fused["local_latent"]
        geometry = self.price_decoder(zg, zl, mask, tick_norm, times=times)
        quantity = self.volume_decoder(zg, zl, geometry, mask, tick_norm, times=times)
        quantity = self.volume_refinement(zg, zl, quantity, geometry, mask, tick_norm, times)
        return {**geometry, "volume": quantity, "posterior": posterior,
                "unified_latent": fused["unified"], "unified_mask": fused["unified_mask"]}

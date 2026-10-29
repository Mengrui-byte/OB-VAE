"""Controlled decoder variants for the twelve-hour dual-VAE campaign."""

import torch

from model.dual import OrderBookVAE
from model.decoders.dual_refinement import PriceGapRefinement, VolumeAmplitudeRefinement
from model.layers.latent_refinement import TemporalLatentRefinement


VARIANTS = ("control","spatial","volume","temporal","joint")


class ResearchOrderBookVAE(OrderBookVAE):
    def __init__(self, *, variant="control", refinement_seed=42, **core_config):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown research variant: {variant}")
        super().__init__(**core_config)
        self.variant = variant
        self.refinement_seed = refinement_seed
        levels=core_config.get("levels",2048)
        dg=core_config.get("global_dim",64)
        dl=core_config.get("local_dim",16)
        # Keep shared pretrained parameters and per-module initialization aligned
        # across single-module and joint variants.
        with torch.random.fork_rng(devices=[]):
            if variant in ("spatial","joint"):
                torch.manual_seed(refinement_seed+11)
                self.price_refinement=PriceGapRefinement(levels,dg,dl)
            if variant in ("volume","joint"):
                torch.manual_seed(refinement_seed+17)
                self.volume_refinement=VolumeAmplitudeRefinement(levels,dg,dl,core_config.get("quantity_scale",1.))
            if variant in ("temporal","joint"):
                torch.manual_seed(refinement_seed+23)
                self.temporal_refinement=TemporalLatentRefinement(dg,dl,core_config.get("causal",False))

    def load_common_initial(self, state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        prefixes=("price_refinement.","volume_refinement.","temporal_refinement.")
        if unexpected or any(not name.startswith(prefixes) for name in missing):
            raise ValueError(f"Shared initial state mismatch: missing={missing}, unexpected={unexpected}")

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        if self.variant=="control":
            return super().forward(price,volume,mask,tick_norm,times=times,sample=sample)
        posterior=self.encoder(price,volume,mask,tick_norm,times=times,sample=sample)
        zg,zl=posterior["z_global"],posterior["z_local"]
        if hasattr(self,"temporal_refinement"):
            zg,zl=self.temporal_refinement(zg,zl,posterior["latent_mask"])
        geometry=self.price_decoder(zg,zl,mask,tick_norm,times=times)
        if hasattr(self,"price_refinement"):
            geometry=self.price_refinement(zg,zl,geometry,mask,tick_norm,times)
        quantity=self.volume_decoder(zg,zl,geometry,mask,tick_norm,times=times)
        if hasattr(self,"volume_refinement"):
            quantity=self.volume_refinement(zg,zl,quantity,geometry,mask,tick_norm,times)
        return {**geometry,"volume":quantity,"posterior":posterior}

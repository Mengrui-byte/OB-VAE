"""Masked reconstruction and diagonal Gaussian KL for the dual-decoder VAE."""

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn


@dataclass(frozen=True)
class DualLossConfig:
    price: float = 1.0
    volume_log: float = 1.0
    volume_linear: float = 0.0
    gap_log: float = 0.1
    spread_log: float = 0.1
    mid: float = 0.1
    price_slope: float = 0.0
    price_temporal: float = 0.0
    volume_tail: float = 0.0
    beta_global: float = 1e-4
    beta_local: float = 1e-4

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not any(v > 0 for k, v in asdict(self).items() if not k.startswith("beta")):
            raise ValueError("At least one reconstruction weight must be positive")


def window_mean(value, mask):
    """Equal weight per window; average only its valid elements."""
    dims = tuple(range(1, value.ndim))
    return value.masked_fill(~mask, 0).sum(dims) / mask.sum(dims).clamp_min(1)


class DualVAELoss(nn.Module):
    def __init__(self, quantity_scale=1.0, config=None):
        super().__init__()
        if not math.isfinite(quantity_scale) or quantity_scale <= 0:
            raise ValueError("quantity_scale must be finite and positive")
        self.quantity_scale = float(quantity_scale)
        self.config = config or DualLossConfig()

    def forward(self, output, batch, kl_factor=1.0):
        if not math.isfinite(kl_factor) or not 0 <= kl_factor <= 1:
            raise ValueError("kl_factor must be in [0, 1]")
        price, volume, mask = batch["price"], batch["volume"], batch["mask"]
        pred_price, pred_volume = output["price_norm"].float(), output["volume"].float()
        if pred_price.shape != price.shape or pred_volume.shape != volume.shape:
            raise ValueError("Prediction and target shapes differ")
        q0 = self.quantity_scale
        adjacent = mask[..., 1:] & mask[..., :-1]
        sign = price.new_tensor([-1, 1])[None, None, :, None]
        gap = (price[..., 1:] - price[..., :-1]) * sign
        tick = batch["tick_norm"][:, None, None, None]
        spread = price[:, :, 1, 0] - price[:, :, 0, 0]
        mid = (price[:, :, 1, 0] + price[:, :, 0, 0]) * 0.5
        frames = mask[..., 0].all(-1)
        terms = {
            "price": window_mean((pred_price - price).square(), mask),
            "volume_log": window_mean(
                (torch.log1p(pred_volume / q0) - torch.log1p(volume / q0)).square(), mask),
            "volume_linear": window_mean(((pred_volume - volume) / q0).square(), mask),
            "gap_log": window_mean(
                (torch.log((output["gap_norm"].float() / tick).clamp_min(1e-6))
                 - torch.log((gap / tick).clamp_min(1e-6))).square(), adjacent),
            "spread_log": window_mean(
                (torch.log((output["spread_norm"].float() / tick[:, :, 0, 0]).clamp_min(1e-6))
                 - torch.log((spread / tick[:, :, 0, 0]).clamp_min(1e-6))).square(), frames),
            "mid": window_mean((output["mid_norm"].float() - mid).square(), frames),
        }
        terms["volume_tail"] = window_mean(
            (torch.log1p(pred_volume/q0)-torch.log1p(volume/q0)).square(),
            mask & (volume>=100)) if self.config.volume_tail else pred_volume.new_zeros(price.shape[0])
        slope_terms = []
        for lag in ((16,64,256) if self.config.price_slope else ()):
            if lag >= price.shape[-1]:
                continue
            valid = mask[...,lag:] & mask[...,:-lag]
            # A multi-level derivative in ticks per rank; tick is side metadata.
            difference = ((pred_price[...,lag:] - pred_price[...,:-lag])
                          - (price[...,lag:] - price[...,:-lag])) / (lag * tick)
            slope_terms.append(window_mean(difference.square(),valid))
        zero = pred_price.new_zeros(price.shape[0])
        if self.config.price_slope and not slope_terms:
            raise ValueError("price_slope requires more than 16 levels")
        terms["price_slope"] = sum(slope_terms) / len(slope_terms) if slope_terms else zero
        temporal_terms = []
        for lag in ((1,4,8) if self.config.price_temporal else ()):
            if lag >= price.shape[1]:
                continue
            valid = mask[:,lag:] & mask[:,:-lag]
            difference = ((pred_price[:,lag:] - pred_price[:,:-lag])
                          - (price[:,lag:] - price[:,:-lag])) / (lag * tick)
            temporal_terms.append(window_mean(difference.square(),valid))
        if self.config.price_temporal and not temporal_terms:
            raise ValueError("price_temporal requires at least two frames")
        terms["price_temporal"] = sum(temporal_terms) / len(temporal_terms) if temporal_terms else zero
        posterior = output.get("posterior")
        if posterior is None:
            if self.config.beta_global or self.config.beta_local:
                raise ValueError("A posterior is required when KL weights are nonzero")
            terms["kl_global"] = zero
            terms["kl_local"] = zero
        for kind, valid in (() if posterior is None else (("global", frames), ("local", posterior["latent_mask"]))):
            branches = posterior.get("towers", [posterior])
            branch_kl = []
            for branch in branches:
                mu = branch[f"mu_{kind}"].float()
                logvar = branch[f"logvar_{kind}"].float()
                # Each source retains its own KL; an averaged mean could hide
                # two divergent posteriors whose means cancel each other.
                kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)
                branch_kl.append(window_mean(kl, valid))
            terms[f"kl_{kind}"] = sum(branch_kl) / len(branch_kl)
        cfg = self.config
        reconstruction = sum(getattr(cfg, name) * terms[name] for name in (
            "price", "volume_log", "volume_linear", "gap_log", "spread_log", "mid",
            "price_slope", "price_temporal", "volume_tail"))
        total = reconstruction + kl_factor * (
            cfg.beta_global * terms["kl_global"] + cfg.beta_local * terms["kl_local"])
        return {"loss": total.mean(), "reconstruction": reconstruction.mean(),
                **{name: value.mean() for name, value in terms.items()}}

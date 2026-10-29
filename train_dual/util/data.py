"""Adapt the project's minute NPZ windows to the dual VAE input contract."""

import math

import torch
from torch.utils.data import Dataset

from util.dataset import OrderBookDataset


class DualWindowDataset(Dataset):
    def __init__(self, *, levels=2048, time_unit_seconds=60.0, **kwargs):
        if levels < 16 or levels > 2048 or levels % 16:
            raise ValueError("levels must be a multiple of 16 in [16, 2048]")
        if not math.isfinite(time_unit_seconds) or time_unit_seconds <= 0:
            raise ValueError("time_unit_seconds must be finite and positive")
        self.base = OrderBookDataset(return_counts=True, **kwargs)
        self.levels = levels
        self.time_unit_seconds = time_unit_seconds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        data = item["data"][..., :self.levels, :]
        counts = item["level_counts"].clamp(max=self.levels)
        # Base reader guarantees consecutive minute windows, including across days.
        times = torch.arange(data.shape[0], dtype=torch.float32) * (60 / self.time_unit_seconds)
        return {"data": data, "level_counts": counts, "times": times}


class DualNormalizer:
    def __init__(self, *, tick_size, price_scale=50.0):
        if any(not math.isfinite(v) or v <= 0 for v in (tick_size, price_scale)):
            raise ValueError("tick_size and price_scale must be finite and positive")
        self.tick_size = float(tick_size)
        self.price_scale = float(price_scale)  # bps RMS divisor, not absolute price

    def prepare(self, batch, device):
        device = torch.device(device)
        raw = batch["data"].to(device=device, dtype=torch.float32)
        if raw.ndim != 5 or raw.shape[2] != 2 or raw.shape[-1] != 2:
            raise ValueError("data must have shape [B,T,2,L,2]")
        counts = batch["level_counts"].to(device)
        if counts.shape != raw.shape[:3] or counts.is_floating_point() or counts.dtype == torch.bool:
            raise ValueError("level_counts must be integer [B,T,2]")
        if not bool(((counts > 0) & (counts <= raw.shape[-2])).all()):
            raise ValueError("Every frame needs at least one valid level on both sides")
        mask = torch.arange(raw.shape[-2], device=device) < counts[..., None]
        # Ignore arbitrary padding, including NaNs, before arithmetic.
        price = raw[..., 0].masked_fill(~mask, 0)
        volume = raw[..., 1].masked_fill(~mask, 0)
        if not bool(torch.isfinite(price).all() and torch.isfinite(volume).all()):
            raise ValueError("Valid prices and quantities must be finite")
        if bool(((price <= 0) & mask).any() or (volume < 0).any()):
            raise ValueError("Valid prices must be positive; quantities must be nonnegative")
        adjacent = mask[..., 1:] & mask[..., :-1]
        sign = price.new_tensor([-1, 1])[None, None, :, None]
        if bool((((price[..., 1:] - price[..., :-1]) * sign <= 0) & adjacent).any()):
            raise ValueError("Prices must be strictly ordered from best to deepest level")
        if bool((price[:, :, 0, 0] >= price[:, :, 1, 0]).any()):
            raise ValueError("Best bid must be below best ask")
        # Float64 anchor avoids rounding away a half-tick at the first midpoint.
        anchor = (price[:, 0, 0, 0].double() + price[:, 0, 1, 0].double()) * 0.5
        factor = anchor * (self.price_scale / 10000)
        normalized = ((price.double() - anchor[:, None, None, None])
                      / factor[:, None, None, None]).float().masked_fill(~mask, 0)
        tick_norm = (self.tick_size / factor).float()
        times = batch.get("times")
        if times is None:
            times = torch.arange(raw.shape[1]).expand(raw.shape[0], -1)
        times = times.to(device=device, dtype=torch.float32)
        if times.shape != raw.shape[:2] or not bool(torch.isfinite(times).all()):
            raise ValueError("times must be finite [B,T]")
        if not bool((times[:, 0] == 0).all()) or bool((times[:, 1:] <= times[:, :-1]).any()):
            raise ValueError("times must start at zero and strictly increase")
        if not bool(torch.isfinite(normalized).all() and torch.isfinite(tick_norm).all()
                    and (tick_norm > 0).all()):
            raise ValueError("Normalized prices / ticks exceed float32 range")
        return {"price": normalized, "volume": volume, "mask": mask,
                "tick_norm": tick_norm, "times": times, "anchor": anchor,
                "price_factor": factor}

    def inverse_price(self, price_norm, batch):
        price = (price_norm.double() * batch["price_factor"][:, None, None, None]
                 + batch["anchor"][:, None, None, None])
        return price.masked_fill(~batch["mask"], 0)

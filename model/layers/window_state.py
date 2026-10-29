"""Aggregate a complete latent window into one fixed-dimensional state vector."""

import math

import torch
from torch import nn

from model.dual.book_layers import PositionCode, sinusoidal


class WindowStateCodec(nn.Module):
    """Sixteen learned aggregation slots, then a strict window-vector bottleneck.

    The expansion path receives only [B,state_dim] and structural metadata.
    Neither the aggregation tokens nor the original latent grid bypass state.
    """

    def __init__(self, levels, global_dim=64, local_dim=16, *, state_dim=1024,
                 slots=16, width=64, heads=4, seed=42):
        super().__init__()
        if width % heads or width % 2 or state_dim <= 0 or state_dim > slots * width:
            raise ValueError("Invalid state dimension, attention width or head count")
        self.levels, self.global_dim, self.local_dim = levels, global_dim, local_dim
        self.state_dim, self.slots, self.width = state_dim, slots, width
        memory_width = max(global_dim, local_dim)
        # Common modules have identical initialization across bottleneck sizes.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 211)
            self.position = PositionCode(levels)
            self.position_proj = nn.Linear(32, width)
            self.global_type = nn.Parameter(torch.zeros(1, 1, 1, width))
            self.memory_proj = nn.Linear(memory_width, width)
            self.memory_norm = nn.LayerNorm(width)
            self.queries = nn.Parameter(torch.randn(slots, width) * .02)
            self.pool_norm = nn.LayerNorm(width)
            self.pool_cross = nn.MultiheadAttention(width, heads, batch_first=True)
            self.pool_ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                          nn.GELU(), nn.Linear(4 * width, width))
            self.read_norm = nn.LayerNorm(width)
            self.read_cross = nn.MultiheadAttention(width, heads, batch_first=True)
            self.read_ffn = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 4 * width),
                                          nn.GELU(), nn.Linear(4 * width, width))
            self.global_head = nn.Linear(width, global_dim)
            self.local_head = nn.Linear(width, local_dim)
            # Calibration sets train-only output centers/scales. Initially the
            # output heads predict those centers; warmup learns their residuals.
            nn.init.zeros_(self.global_head.weight)
            nn.init.zeros_(self.global_head.bias)
            nn.init.zeros_(self.local_head.weight)
            nn.init.zeros_(self.local_head.bias)
        total = slots * width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 223)
            # A shared orthonormal basis makes the dimensions nested at start.
            basis = torch.empty(total, total)
            nn.init.orthogonal_(basis)
            self.to_state = nn.Linear(total, state_dim, bias=False)
            self.from_state = nn.Linear(state_dim, total, bias=False)
            with torch.no_grad():
                self.to_state.weight.copy_(basis[:state_dim])
                self.from_state.weight.copy_(basis[:state_dim].T)
        self.register_buffer("global_mean", torch.zeros(global_dim))
        self.register_buffer("global_scale", torch.ones(global_dim))
        self.register_buffer("local_mean", torch.zeros(local_dim))
        self.register_buffer("local_scale", torch.ones(local_dim))

    def calibrate(self, statistics):
        with torch.no_grad():
            for name in ("global_mean", "global_scale", "local_mean", "local_scale"):
                value = torch.as_tensor(statistics[name], dtype=getattr(self, name).dtype,
                                        device=getattr(self, name).device)
                getattr(self, name).copy_(value)

    def positions(self, latent_mask, ref, times=None):
        B, T, _, K = latent_mask.shape
        if times is None:
            times = torch.arange(T, device=ref.device).expand(B, T)
        local = self.position(B, T, K, ref, times).permute(0, 1, 2, 4, 3).reshape(B, T, 2 * K, 32)
        local = self.position_proj(local)
        global_pos = sinusoidal(times, self.width).to(local.dtype).unsqueeze(2) + self.global_type
        return torch.cat([global_pos, local], 2)

    def compress(self, unified, latent_mask, times=None):
        B, T, _, K = latent_mask.shape
        valid = latent_mask.reshape(B, T, 2 * K)
        valid = torch.cat([valid.any(-1, keepdim=True), valid], -1)
        if not bool(valid.reshape(B, -1).any(-1).all()):
            raise ValueError("Each window must contain valid latent tokens")
        # Values retain their amplitudes; positional features distinguish side,
        # rank and minute when the entire window is flattened into one memory.
        memory = self.memory_proj(unified)
        keys = self.memory_norm(memory + self.positions(latent_mask, unified, times))
        query_pos = sinusoidal(torch.arange(self.slots, device=unified.device), self.width)
        queries = (self.queries + query_pos).to(memory.dtype)[None].expand(B, -1, -1)
        update = self.pool_cross(self.pool_norm(queries), keys.reshape(B, -1, self.width),
                                 memory.reshape(B, -1, self.width),
                                 key_padding_mask=~valid.reshape(B, -1), need_weights=False)[0]
        tokens = queries + update
        tokens = tokens + self.pool_ffn(tokens)
        state = self.to_state(tokens.reshape(B, -1)) / math.sqrt(self.slots)
        return dict(state=state, aggregation_tokens=tokens)

    def expand(self, state, latent_mask, times=None):
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"Expected one [B,{self.state_dim}] window vector")
        B, T, _, K = latent_mask.shape
        if state.shape[0] != B:
            raise ValueError("State and metadata batch sizes differ")
        memory = self.from_state(state * math.sqrt(self.slots)).reshape(B, self.slots, self.width)
        query = self.positions(latent_mask, state, times).reshape(B, -1, self.width)
        update = self.read_cross(self.read_norm(query), memory, memory, need_weights=False)[0]
        decoded = query + update
        decoded = decoded + self.read_ffn(decoded)
        decoded = decoded.reshape(B, T, 1 + 2 * K, self.width)
        zg = self.global_head(decoded[:, :, 0]) * self.global_scale + self.global_mean
        zl = self.local_head(decoded[:, :, 1:]) * self.local_scale + self.local_mean
        zl = zl.reshape(B, T, 2, K, self.local_dim) * latent_mask[..., None]
        return zg, zl

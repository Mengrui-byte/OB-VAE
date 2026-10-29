"""Euclidean residual vector quantization with independent EMA codebooks."""

import torch
from torch import nn
from torch.nn import functional as F


class ResidualEMAQuantizer(nn.Module):
    """Quantize one vector per window, returning progressively summed codes.

    Index zero is a fixed zero vector at every level. This explicit skip code
    ensures that nearest-code residual energy cannot grow at a later level.
    It is a documented variant of standard residual VQ, not a padding symbol.
    """

    def __init__(self, dimension, levels=8, size=256, decay=.99):
        super().__init__()
        if dimension < 1 or levels < 1 or size < 2 or not 0 <= decay < 1:
            raise ValueError('Invalid residual quantizer configuration')
        self.dimension, self.levels, self.size, self.decay = dimension, levels, size, decay
        self.register_buffer('embedding', torch.zeros(levels, size, dimension))
        self.register_buffer('ema_count', torch.ones(levels, size))
        self.register_buffer('ema_sum', torch.zeros(levels, size, dimension))
        self.register_buffer('initialized', torch.tensor(False))
        self.register_buffer('updates', torch.tensor(0, dtype=torch.long))

    @staticmethod
    def nearest(x, centers):
        distances = x.square().sum(-1, keepdim=True) + centers.square().sum(-1)[None] - 2 * x @ centers.T
        return distances.argmin(-1)

    @torch.no_grad()
    def initialize(self, features, iterations=20, seed=42):
        x = features.detach().float()
        if x.ndim != 2 or x.shape[-1] != self.dimension or len(x) < self.size:
            raise ValueError('RQ initialization needs at least one full codebook of training vectors')
        generator = torch.Generator(device=x.device).manual_seed(seed)
        residual = x.clone()
        history = []
        for level in range(self.levels):
            order = torch.randperm(len(x), generator=generator, device=x.device)
            centers = torch.cat([torch.zeros_like(x[:1]), residual[order[:self.size-1]]], 0)
            for _ in range(iterations):
                indices = self.nearest(residual, centers)
                counts = torch.bincount(indices, minlength=self.size).float()
                sums = torch.zeros_like(centers).index_add_(0, indices, residual)
                live = counts > 0
                centers[live] = sums[live] / counts[live, None]
                dead = (~live).nonzero().flatten()
                dead = dead[dead != 0]
                if len(dead):
                    error = (residual - centers[indices]).square().sum(-1)
                    replacement = error.topk(len(dead)).indices
                    centers[dead] = residual[replacement]
                centers[0].zero_()
            self.embedding[level].copy_(centers)
            self.ema_sum[level].copy_(centers)
            self.ema_count[level].fill_(1)
            indices = self.nearest(residual, centers)
            residual = residual - centers[indices]
            history.append(float(residual.square().mean()))
        self.initialized.fill_(True)
        self.updates.zero_()
        return history

    @torch.no_grad()
    def _update(self, level, residual, indices):
        counts = torch.bincount(indices, minlength=self.size).float()
        sums = torch.zeros_like(self.ema_sum[level]).index_add_(0, indices, residual)
        self.ema_count[level].mul_(self.decay).add_(counts, alpha=1-self.decay)
        self.ema_sum[level].mul_(self.decay).add_(sums, alpha=1-self.decay)
        centers = self.ema_sum[level] / self.ema_count[level].clamp_min(1e-5)[:, None]
        if int(self.updates) % 128 == 127:
            dead = (self.ema_count[level] < .1).nonzero().flatten()
            dead = dead[dead != 0]
            if len(dead):
                replace = torch.arange(len(dead), device=residual.device) % len(residual)
                centers[dead] = residual[replace]
                self.ema_count[level, dead] = 1.
                self.ema_sum[level, dead] = centers[dead]
        centers[0].zero_()
        self.embedding[level].copy_(centers)
        self.ema_sum[level, 0].zero_()

    def forward(self, features, depth=None):
        if not bool(self.initialized):
            raise RuntimeError('Initialize codebooks from training features before quantization')
        if features.ndim != 2 or features.shape[-1] != self.dimension:
            raise ValueError('Expected a batch of complete-window vectors')
        depth = self.levels if depth is None else depth
        if not 1 <= depth <= self.levels:
            raise ValueError('Invalid quantization depth')
        x = features.float()
        residual = x.detach().clone()
        total = torch.zeros_like(residual)
        partials, codes, energies = [], [], [residual.square().mean(-1)]
        for level in range(depth):
            indices = self.nearest(residual, self.embedding[level])
            quantized = F.embedding(indices, self.embedding[level]).detach()
            if self.training:
                self._update(level, residual, indices)
            residual = residual - quantized
            total = total + quantized
            partials.append(x + (total-x).detach())
            codes.append(indices)
            energies.append(residual.square().mean(-1))
        if self.training:
            self.updates.add_(1)
        commitment = torch.stack([(x-p.detach()).square().mean() for p in partials]).mean()
        return dict(quantized=partials[-1], partials=partials, codes=torch.stack(codes,-1),
                    commitment=commitment, residual_energy=torch.stack(energies,-1))

    def decode_codes(self, codes):
        if codes.ndim != 2 or not 1 <= codes.shape[-1] <= self.levels:
            raise ValueError('Expected [B,residual_depth] integer codes')
        if bool(((codes < 0) | (codes >= self.size)).any()):
            raise ValueError('Code outside codebook')
        return sum(F.embedding(codes[:,level], self.embedding[level]) for level in range(codes.shape[-1]))

"""Quantity decoder with a zero-initialized, direct within-region detail head.

The detail path reads only decoder features generated from the shared Z and
predicted price geometry. It does not access Encoder features or true books.
"""
import torch
from torch import nn
from torch.nn import functional as F

from model.decoders.unified_book import VolumeDecoder
from model.dual.book_layers import mask_at, mean_at


class PhaseDetailVolumeDecoder(VolumeDecoder):
    def __init__(self, *, detail_mode='independent', **kwargs):
        super().__init__(**kwargs)
        if detail_mode not in ('off', 'shared', 'independent'):
            raise ValueError('detail_mode must be off, shared or independent')
        self.detail_mode = detail_mode
        # Appended after all original modules: original parameter names/order and
        # initialization sequence remain intact. One region emits 16 ordered levels.
        self.phase_head = nn.Conv1d(256, 16, 1)
        nn.init.zeros_(self.phase_head.weight)
        nn.init.zeros_(self.phase_head.bias)

    @staticmethod
    def phase_to_levels(values):
        """[N,16,K] -> [N,K*16], level index = region*16 + offset."""
        if values.ndim != 3 or values.shape[1] != 16:
            raise ValueError('Expected [N,16,K] within-region outputs')
        return values.transpose(1, 2).reshape(values.shape[0], -1)

    def _detail_seed(self, frame, mask, times, condition):
        """Reproduce the existing conditioned seed without changing old modules.

        The prototype recomputes this small seed path, sharing its parameters
        with the original spatial decoder, to leave its historical code intact.
        """
        B, T, _ = frame.shape
        K = self.spatial.levels // 16
        seed = self.spatial.seed(frame).reshape(B, T, 2, K, 16).transpose(-1, -2)
        pos = self.spatial.position(B, T, K, frame, times)
        h = self.spatial.proj(torch.cat((seed, pos), 3).reshape(B*T*2, 48, K))
        m = mask.reshape(B*T*2, 1, self.spatial.levels).float()
        regional_mask = mask_at(m, K)
        h = h * regional_mask
        h = self.spatial.fuse(torch.cat((h, mean_at(condition, m, K)), 1)) * regional_mask
        return h

    def forward(self, z, price_output, mask, tick_norm, times):
        if self.detail_mode == 'off':
            return super().forward(z, price_output, mask, tick_norm, times)
        B, T, _, L = mask.shape
        p, gap = price_output['price_norm'], price_output['gap_norm']
        loggap = F.pad(torch.log((gap / tick_norm[:, None, None, None]).clamp_min(1e-6)), (1, 0))
        adjacent = F.pad(mask[..., 1:] & mask[..., :-1], (1, 0))
        loggap = loggap * adjacent
        pos = self.position(B, T, L, z, times)
        geometry = torch.cat((p.unsqueeze(-2), loggap.unsqueeze(-2), pos), 3).reshape(B*T*2, 34, L)
        m = mask.reshape(B*T*2, 1, L).float()
        condition = self.geometry(geometry) * m
        frame = self.temporal(z, times)
        coarse_feature = self.spatial(frame, mask, times, condition)
        coarse_logit = self.quantity(coarse_feature).float().reshape(B, T, 2, L)
        detail = self.phase_head(self._detail_seed(frame, mask, times, condition))
        if self.detail_mode == 'shared':
            # Same allocated parameter count, but a region receives one shared
            # correction. This intentionally restricts within-region freedom.
            detail = detail.mean(1, keepdim=True).expand(-1, 16, -1)
        detail_logit = self.phase_to_levels(detail).float().reshape(B, T, 2, L)
        return (self.q0 * F.softplus(coarse_logit + detail_logit)).masked_fill(~mask, 0)

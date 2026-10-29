"""An ordered quantity route INSIDE the single frame-aligned Encoder."""
import torch
from torch import nn

from model.encoders.frame_aligned import FrameAlignedEncoder


class OrderedQuantityResidual(nn.Module):
    def __init__(self, levels, quantity_scale, hidden=256):
        super().__init__()
        if levels < 1 or quantity_scale <= 0 or type(hidden) is not int or hidden <= 0:
            raise ValueError('Invalid ordered quantity residual dimensions/scale')
        self.levels = levels
        self.quantity_scale = float(quantity_scale)
        # No additional global RNG consumption relative to the original model.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.norm = nn.LayerNorm(2 * levels)
            self.input = nn.Linear(2 * levels, hidden)
            self.output = nn.Linear(hidden, 32)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, volume, mask, feature_transform):
        if volume.shape != mask.shape or volume.ndim != 4 or volume.shape[2:] != (2, self.levels):
            raise ValueError('Expected ordered [B,T,2,levels] quantities/mask')
        value = volume.float().masked_fill(~mask, 0) / self.quantity_scale
        if feature_transform == 'log1p':
            value = torch.log1p(value)
        elif feature_transform != 'linear':
            raise ValueError('Unknown fine quantity feature transform')
        # Side/rank order is retained. The result is only a posterior-mean
        # correction; no feature from this route reaches either Decoder.
        return self.output(torch.nn.functional.gelu(self.input(self.norm(value.flatten(2)))))


class FinePhaseFrameEncoder(FrameAlignedEncoder):
    def __init__(self, *, fine_enabled=False, fine_transform='log1p', fine_hidden=256, **kwargs):
        if type(fine_enabled) is not bool or fine_transform not in ['linear', 'log1p']:
            raise ValueError('Explicit valid fine feature switches required')
        super().__init__(**kwargs)
        self.fine_enabled, self.fine_transform = fine_enabled, fine_transform
        self.fine_phase = OrderedQuantityResidual(self.levels, kwargs.get('quantity_scale', 1.), fine_hidden)

    def forward(self, price, volume, mask, tick_norm, times):
        mu, logvar = super().forward(price, volume, mask, tick_norm, times)
        if self.fine_enabled:
            correction = self.fine_phase(volume, mask, self.fine_transform).float()
            mu = mu + self.pack_frames(correction)
        return mu, logvar

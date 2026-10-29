"""Matched affine/nonlinear, full-resolution quantity logits from post-Z frames."""
import torch
from torch import nn


class FrameQuantityLogitMLP(nn.Module):
    def __init__(self, *, levels=2048, frame_dim=256, hidden_dim=2048, activation='gelu', seed=42):
        super().__init__()
        if any(type(v) is not int or v <= 0 for v in [levels, frame_dim, hidden_dim]):
            raise ValueError('Positive integer dimensions required')
        if activation not in ['identity', 'gelu'] or type(seed) is not int:
            raise ValueError('Explicit identity/GELU activation and integer initialization seed required')
        self.levels, self.frame_dim = levels, frame_dim
        self.hidden_dim, self.activation_name, self.seed = hidden_dim, activation, seed
        # CPU-only initialization must preserve the caller CPU and CUDA RNGs.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.input_projection = nn.Linear(frame_dim, hidden_dim)
            self.output_projection = nn.Linear(hidden_dim, 2 * levels)
        self.activation = nn.Identity() if activation == 'identity' else nn.GELU()
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, frame):
        if frame.ndim != 3 or frame.shape[-1] != self.frame_dim:
            raise ValueError('Expected [B,T,frame_dim] post-Z features')
        # Identical FP32 policy in both arms, including under caller autocast.
        with torch.autocast(frame.device.type, enabled=False):
            h = self.activation(self.input_projection(frame.float()))
            return self.output_projection(h).reshape(*frame.shape[:-1], 2, self.levels)

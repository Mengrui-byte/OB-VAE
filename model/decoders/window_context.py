"""Post-Z quantity context with matched local and full-window attention."""
import torch
from torch import nn

from model.decoders.frame_aligned import FrameTemporalExpander
from model.dual.book_layers import sinusoidal


class WindowContextTemporalExpander(FrameTemporalExpander):
    def __init__(self, context_frames=0):
        if type(context_frames) is not int or context_frames not in (0, 5, 17, 256):
            raise ValueError('Context must be disabled,5,17 or256 frames')
        super().__init__()
        self.context_frames = context_frames
        # No extra consumption of the source model RNG stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            self.context_attention = nn.MultiheadAttention(
                256, 8, kdim=32, vdim=32, dropout=0., batch_first=True)
        nn.init.zeros_(self.context_attention.out_proj.weight)
        nn.init.zeros_(self.context_attention.out_proj.bias)
        positions = torch.arange(256)
        mask = (positions[:, None] - positions[None, :]).abs() > context_frames // 2
        self.register_buffer('context_mask', mask, persistent=False)

    def forward(self, z, times):
        frame = super().forward(z, times)
        if self.context_frames == 0:
            return frame
        tokens = self.unpack_frames(z)
        keys = tokens + sinusoidal(times, 32).to(tokens.dtype)
        mask = None if self.context_frames == 256 else self.context_mask
        residual, _ = self.context_attention(frame, keys, tokens, attn_mask=mask, need_weights=False)
        return frame + residual

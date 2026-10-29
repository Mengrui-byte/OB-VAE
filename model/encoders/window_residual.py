"""Task-adapted window encoders initialized as an exact PCA baseline."""

import torch
from torch import nn


class WindowEncodingResidual(nn.Module):
    """Read the complete source features, never just the retained PCA code.

    The attention slots have fixed coverage groups; none denotes one minute.
    The output is a correction to a single window vector, in PCA code units.
    """

    def __init__(self, *, kind, state_dim, window_size, spatial_tokens,
                 global_dim, local_dim, hidden_dim=256, attention_dim=64, seed=42):
        super().__init__()
        if kind not in ('fixed', 'linear', 'nonlinear', 'attention'):
            raise ValueError('Unknown encoding residual')
        self.kind = kind
        self.register_buffer('state_scale', torch.ones(state_dim))
        full_dim = window_size * (global_dim + 2 * spatial_tokens * local_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 719)
            if kind == 'linear':
                self.output = nn.Linear(full_dim, state_dim, bias=False)
            elif kind == 'nonlinear':
                self.input = nn.Sequential(nn.Linear(full_dim, hidden_dim), nn.GELU())
                self.output = nn.Linear(hidden_dim, state_dim)
            elif kind == 'attention':
                if attention_dim % 4:
                    raise ValueError('Attention width must be divisible by four')
                self.global_input = nn.Linear(global_dim, attention_dim)
                self.local_input = nn.Linear(local_dim, attention_dim)
                self.time_position = nn.Parameter(torch.randn(window_size, attention_dim) * .02)
                self.rank_position = nn.Parameter(torch.randn(spatial_tokens, attention_dim) * .02)
                self.side_position = nn.Parameter(torch.randn(2, attention_dim) * .02)
                self.queries = nn.Parameter(torch.randn(16, attention_dim) * .02)
                self.attention = nn.MultiheadAttention(attention_dim, 4, batch_first=True)
                self.norm = nn.LayerNorm(attention_dim)
                self.output = nn.Linear(16 * attention_dim, state_dim)
            if kind != 'fixed':
                nn.init.zeros_(self.output.weight)
                if self.output.bias is not None:
                    nn.init.zeros_(self.output.bias)

    def forward(self, global_features, local_features, latent_mask):
        if self.kind == 'fixed':
            return global_features.new_zeros((len(global_features), len(self.state_scale)))
        if self.kind in ('linear', 'nonlinear'):
            x = torch.cat([global_features.flatten(1), local_features.flatten(1)], -1)
            if self.kind == 'nonlinear':
                x = self.input(x)
            else:
                # With 66k inputs, an unscaled zero-initialized dense residual
                # takes an excessively large first Adam step. This fixed gain
                # preserves linear expressivity while conditioning its updates.
                x = x / (x.shape[-1] ** .5)
        else:
            b, t, _, n, _ = local_features.shape
            g = self.global_input(global_features) + self.time_position[None]
            l = (self.local_input(local_features)
                 + self.time_position[None, :, None, None]
                 + self.rank_position[None, None, None]
                 + self.side_position[None, None, :, None])
            q = self.queries[None].expand(b, -1, -1)
            groups = [self.attention(q[:, :4], g, g, need_weights=False)[0]]
            # Near/middle/far depth coverage, proportional to the source ranks.
            edges = [0, max(1, n // 8), max(2, n // 2), n]
            offset = 4
            for side in range(2):
                for start, end in zip(edges[:-1], edges[1:]):
                    memory = l[:, :, side, start:end].flatten(1, 2)
                    valid = latent_mask[:, :, side, start:end].flatten(1, 2)
                    if not memory.shape[1]:
                        groups.append(q[:, offset:offset + 2] * 0)
                    else:
                        # Empty groups use a zero sentinel and contribute zero.
                        present = valid.any(-1)
                        memory = memory.masked_fill(~valid[..., None], 0)
                        safe = valid.clone()
                        safe[:, 0] |= ~present
                        pooled = self.attention(q[:, offset:offset + 2], memory, memory,
                                                key_padding_mask=~safe, need_weights=False)[0]
                        groups.append(pooled * present[:, None, None])
                    offset += 2
            x = self.norm(torch.cat(groups, 1)).flatten(1)
        return self.output(x) * self.state_scale

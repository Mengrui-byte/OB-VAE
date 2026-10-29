"""Cross-attend global/local sources into one masked latent token sequence."""

import torch
from torch import nn
from torch.nn import functional as F

from model.dual.book_layers import PositionCode, sinusoidal


class UnifiedCrossLatent(nn.Module):
    """All decoder inputs are views of U; no raw-input bypass is introduced.

    U has one global slot plus two ordered grids of local slots. A lossless
    embedding initializes U; learned position queries read global memories over
    time and local memories within the frame. The zero output projection makes
    a shared-encoder warm start exactly preserve the pretrained generation.
    """

    def __init__(self, levels, global_dim, local_dim, *, sources=1, heads=4, causal=False, attention=True):
        super().__init__()
        self.global_dim, self.local_dim = global_dim, local_dim
        self.width = max(global_dim, local_dim)
        if self.width % heads:
            raise ValueError("Unified width must be divisible by heads")
        if sources not in (1, 2):
            raise ValueError("Expected one shared source or two modality towers")
        self.sources, self.causal, self.attention = sources, causal, attention
        if attention:
            self.position = PositionCode(levels)
            self.position_proj = nn.Linear(32, self.width)
            self.global_query = nn.Parameter(torch.zeros(1, 1, 1, self.width))
            self.query_norm = nn.LayerNorm(self.width)
            self.global_proj = nn.ModuleList([nn.Linear(global_dim, self.width) for _ in range(sources)])
            self.local_proj = nn.ModuleList([nn.Linear(local_dim, self.width) for _ in range(sources)])
            self.global_norm = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(sources)])
            self.local_norm = nn.ModuleList([nn.LayerNorm(self.width) for _ in range(sources)])
            self.global_cross = nn.ModuleList([nn.MultiheadAttention(self.width, heads, batch_first=True) for _ in range(sources)])
            self.local_cross = nn.ModuleList([nn.MultiheadAttention(self.width, heads, batch_first=True) for _ in range(sources)])
            self.output = nn.Linear(self.width, self.width)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, latents, latent_mask, times=None):
        if len(latents) != self.sources:
            raise ValueError("Latent source count differs from fusion configuration")
        B, T, _, K = latent_mask.shape
        valid = latent_mask.reshape(B, T, 2 * K)
        unified_mask = torch.cat([valid.any(-1, keepdim=True), valid], -1)
        embedded = []
        for zg, zl in latents:
            global_slot = F.pad(zg, (0, self.width - self.global_dim)).unsqueeze(2)
            local_slots = F.pad(zl.reshape(B, T, 2 * K, self.local_dim), (0, self.width - self.local_dim))
            embedded.append(torch.cat([global_slot, local_slots], 2))
        unified = torch.stack(embedded).mean(0)
        if self.attention:
            zg = latents[0][0]
            if times is None:
                times = torch.arange(T, device=zg.device).expand(B, T)
            pos = self.position(B, T, K, zg, times).permute(0, 1, 2, 4, 3).reshape(B, T, 2 * K, 32)
            local_queries = self.position_proj(pos)
            global_queries = self.global_query + sinusoidal(times, self.width).to(local_queries.dtype).unsqueeze(2)
            query = self.query_norm(torch.cat([global_queries, local_queries], 2)).reshape(B * T, 1 + 2 * K, self.width)
            updates = []
            for i, (global_latent, local_latent) in enumerate(latents):
                glob = self.global_norm[i](self.global_proj[i](global_latent))
                glob = glob[:, None].expand(B, T, T, self.width).reshape(B * T, T, self.width)
                local = self.local_norm[i](self.local_proj[i](local_latent.reshape(B * T, 2 * K, self.local_dim)))
                # Every frame has a valid bid and ask. Causal mode also masks
                # future global memory, independently of the encoder's TCN.
                if self.causal:
                    frames = []
                    for t in range(T):
                        rows = torch.arange(B, device=zg.device) * T + t
                        frames.append(self.global_cross[i](query[rows], glob[rows, :t + 1], glob[rows, :t + 1], need_weights=False)[0])
                    global_update = torch.stack(frames, 1).reshape(B * T, 1 + 2 * K, self.width)
                else:
                    global_update = self.global_cross[i](query, glob, glob, need_weights=False)[0]
                local_update = self.local_cross[i](query, local, local, key_padding_mask=~valid.reshape(B * T, 2 * K), need_weights=False)[0]
                updates.append((global_update + local_update) * .5)
            delta = self.output(torch.stack(updates).mean(0)).reshape_as(unified)
            unified = unified + delta
        unified = unified * unified_mask[..., None]
        return dict(unified=unified, unified_mask=unified_mask,
                    global_latent=unified[:, :, 0, :self.global_dim],
                    local_latent=unified[:, :, 1:, :self.local_dim].reshape(B, T, 2, K, self.local_dim))

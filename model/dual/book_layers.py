import math

import torch
from torch import nn
from torch.nn import functional as F


def sinusoidal(values, dim):
    freq = torch.exp(
        -math.log(10000.0)
        * torch.arange(0, dim, 2, device=values.device).float()
        / dim
    )
    phase = values.float()[..., None] * freq
    return torch.stack(
        (phase.sin(), phase.cos()), dim=-1
    ).flatten(-2)


def mask_at(mask, length):
    """mask: [N,1,L]，聚合到更低分辨率。"""
    return F.adaptive_max_pool1d(
        mask.float(), length
    ).to(mask.dtype)


def mean_at(x, mask, length):
    """仅对有效位置求区域均值。"""
    numerator = F.adaptive_avg_pool1d(
        x * mask, length
    )
    denominator = F.adaptive_avg_pool1d(
        mask, length
    )
    return numerator / denominator.clamp_min(1e-6)


class PositionCode(nn.Module):
    def __init__(self, levels, dim=32):
        super().__init__()
        self.levels = levels
        self.dim = dim
        self.side = nn.Embedding(2, dim)

    def forward(self, B, T, length, ref, times=None):
        # times: [B,T]，相对窗口起点、使用固定时间单位。
        if times is None:
            times = torch.arange(
                T, device=ref.device
            )[None].expand(B, T)

        # 不同分辨率对应统一的原始档位坐标。
        rank = (
            torch.arange(length, device=ref.device) + 0.5
        )
        rank = rank * self.levels / length

        p = sinusoidal(rank, self.dim)[None, None, None]
        t = sinusoidal(times, self.dim)[:, :, None, None]
        s = self.side.weight[None, None, :, None]

        # [B,T,2,C,length]
        return (
            (p + t + s)
            .to(ref.dtype)
            .permute(0, 1, 2, 4, 3)
        )


class Residual1d(nn.Module):
    def __init__(self, channels, dilation=1, causal=False):
        super().__init__()
        self.dilation = dilation
        self.causal = causal

        self.norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(2)
        ])
        self.convs = nn.ModuleList([
            nn.Conv1d(
                channels, channels,
                kernel_size=3,
                dilation=dilation,
            )
            for _ in range(2)
        ])

    def forward(self, x, mask):
        h = x
        d = self.dilation
        padding = (2 * d, 0) if self.causal else (d, d)

        for norm, conv in zip(self.norms, self.convs):
            # 每个位置沿通道维做 LayerNorm。
            h = F.gelu(
                norm(h.transpose(1, 2))
            ).transpose(1, 2)

            h = conv(
                F.pad(h * mask, padding)
            ) * mask

        return (x + h) * mask


class LatentStem(nn.Module):
    def __init__(self, levels, global_dim=64, local_dim=16):
        super().__init__()
        self.position = PositionCode(levels)
        self.proj = nn.Conv1d(
            global_dim + local_dim + 32,
            256,
            kernel_size=1,
        )

    def forward(self, z_global, z_local, mask, times=None):
        # z_global: [B,T,Dg]
        # z_local:  [B,T,2,K,Dl]
        B, T, _, K, _ = z_local.shape

        local = z_local.permute(0, 1, 2, 4, 3)
        glob = z_global[:, :, None, :, None].expand(
            B, T, 2, -1, K
        )
        pos = self.position(B, T, K, z_global, times)

        h = torch.cat(
            (local, glob, pos), dim=3
        ).reshape(B * T * 2, -1, K)

        m = mask.reshape(
            B * T * 2, 1, -1
        ).to(h.dtype)

        return self.proj(h) * mask_at(m, K)


class UpBlock(nn.Module):
    def __init__(
        self, cin, cout, condition_dim=0, dilation=2
    ):
        super().__init__()

        self.conv = nn.Conv1d(
            cin, cout, kernel_size=3, padding=1
        )
        self.fuse = (
            nn.Conv1d(
                cout + condition_dim, cout, kernel_size=1
            )
            if condition_dim else None
        )
        self.res1 = Residual1d(cout, dilation=1)
        self.res2 = Residual1d(cout, dilation=dilation)

    def forward(self, h, full_mask, condition=None):
        length = 2 * h.shape[-1]
        mask = mask_at(full_mask, length)

        h = F.interpolate(
            h, size=length, mode="nearest"
        ) * mask
        h = self.conv(h) * mask

        if self.fuse is not None:
            cond = mean_at(condition, full_mask, length)
            h = self.fuse(
                torch.cat((h, cond), dim=1)
            ) * mask

        h = self.res1(h, mask)
        return self.res2(h, mask)


if __name__ == "__main__":
    torch.manual_seed(7)

    B, T, L = 2, 3, 2048
    K = L // 16  # 128 个区域 token

    zg = torch.randn(B, T, 64)
    zl = torch.randn(B, T, 2, K, 16)
    mask = torch.ones(B, T, 2, L, dtype=torch.bool)

    stem = LatentStem(levels=L).eval()
    up = UpBlock(256, 192).eval()

    with torch.no_grad():
        h = stem(zg, zl, mask)
        print("latent 投影:", h.shape)
        # [12,256,128]

        full_mask = mask.reshape(B * T * 2, 1, L).float()
        h = up(h, full_mask)

        print("一级上采样:", h.shape)
        # [12,192,256]
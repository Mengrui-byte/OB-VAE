import torch
from torch import nn
from torch.nn import functional as F

from .book_layers import PositionCode, Residual1d


class OrderBookEncoder(nn.Module):
    def __init__(
        self,
        levels=2048,
        quantity_scale=1.0,
        global_dim=64,
        local_dim=16,
        causal=False,
    ):
        super().__init__()
        assert levels % 16 == 0

        self.levels = levels
        self.register_buffer(
            "q0", torch.tensor(float(quantity_scale))
        )
        self.scales = (1, 2, 4, 8, 16, 32)

        # 负的高斯二阶导：正尖峰对应中心正响应。
        for r in self.scales:
            grid = torch.arange(
                -3 * r, 3 * r + 1
            ).float()

            kernel = (
                (1 - (grid / r) ** 2)
                * torch.exp(-0.5 * (grid / r) ** 2)
            )
            kernel = kernel - kernel.mean()
            kernel = kernel / kernel.abs().sum()

            self.register_buffer(
                f"log_kernel_{r}",
                kernel[None, None],
            )

        self.input_proj = nn.Conv1d(16, 32, 1)

        channels = (32, 64, 128, 192, 256)

        self.down = nn.ModuleList([
            nn.Conv1d(a, b, 4, stride=2, padding=1)
            for a, b in zip(channels[:-1], channels[1:])
        ])
        self.spatial = nn.ModuleList([
            Residual1d(
                c, dilation=1 if c <= 128 else 2
            )
            for c in channels[1:]
        ])

        self.position = PositionCode(levels)
        self.position_proj = nn.Conv1d(32, 256, 1)

        self.cross_norm = nn.LayerNorm(256)
        self.cross = nn.MultiheadAttention(
            256, 8, batch_first=True
        )
        self.cross_ffn = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )

        self.temporal = nn.ModuleList([
            Residual1d(
                256, dilation=d, causal=causal
            )
            for d in (1, 2)
        ])

        self.local_stats = nn.Linear(
            256, 2 * local_dim
        )
        self.global_stats = nn.Linear(
            256, 2 * global_dim
        )

    def build_features(
        self, price, volume, mask, tick_norm
    ):
        B, T, _, L = price.shape

        eps = tick_norm[:, None, None, None]
        sign = price.new_tensor(
            [-1, 1]
        )[None, None, :, None]

        # bid、ask 的相邻价差都转成正方向。
        gaps = (
            price[..., 1:] - price[..., :-1]
        ) * sign

        adjacent = (
            mask[..., 1:] & mask[..., :-1]
        )

        log_gap = F.pad(
            torch.log(
                (gaps / eps).clamp_min(1e-6)
            ) * adjacent,
            (1, 0),
        )

        amount = volume / self.q0
        u = torch.log1p(amount)

        base = torch.stack(
            (price, amount, u, log_gap), dim=-2
        ).reshape(B * T * 2, 4, L)

        m = mask.reshape(
            B * T * 2, 1, L
        ).to(price.dtype)

        u = u.reshape(B * T * 2, 1, L) * m

        peaks, logs = [], []

        for r in self.scales:
            left = F.pad(u, (r, 0))[..., :L]
            right = F.pad(u, (0, r))[..., r:r + L]

            ml = F.pad(m, (r, 0))[..., :L]
            mr = F.pad(m, (0, r))[..., r:r + L]

            peak = F.relu(
                torch.minimum(u - left, u - right) / r
            )
            peaks.append(peak * m * ml * mr)

            kernel = getattr(
                self, f"log_kernel_{r}"
            ).to(u.dtype)
            width = kernel.shape[-1]

            response = F.conv1d(
                u, kernel, padding=width // 2
            )
            support = F.conv1d(
                m,
                torch.ones_like(kernel),
                padding=width // 2,
            )

            # 屏蔽邻域不完整的位置。
            logs.append(
                response * (support >= width - 0.5)
            )

        # 基础4通道 + 双侧差分6通道 + LoG6通道。
        return torch.cat(
            [base, *peaks, *logs], dim=1
        ) * m

    def forward(
        self,
        price,
        volume,
        mask,
        tick_norm,
        times=None,
        sample=None,
    ):
        B, T, _, L = price.shape
        assert L == self.levels

        m = mask.reshape(
            B * T * 2, 1, L
        ).to(price.dtype)

        features = self.build_features(
            price, volume, mask, tick_norm
        )
        h = self.input_proj(features) * m

        # L → L/2 → L/4 → L/8 → L/16。
        for down, res in zip(self.down, self.spatial):
            h = down(h)
            m = F.max_pool1d(m, 2)
            h = res(h * m, m)

        K = h.shape[-1]

        pos = self.position(
            B, T, K, h, times
        ).reshape(B * T * 2, 32, K)

        h = (h + self.position_proj(pos)) * m
        h = h.reshape(
            B * T, 2, 256, K
        ).permute(0, 1, 3, 2)

        valid = m.reshape(B * T, 2, K).bool()

        # 同一时刻的 bid ↔ ask 双向交互。
        v = self.cross_norm(h)

        bid = self.cross(
            v[:, 0], v[:, 1], v[:, 1],
            key_padding_mask=~valid[:, 1],
            need_weights=False,
        )[0]

        ask = self.cross(
            v[:, 1], v[:, 0], v[:, 0],
            key_padding_mask=~valid[:, 0],
            need_weights=False,
        )[0]

        h = h + torch.stack((bid, ask), dim=1)
        h = (
            h + self.cross_ffn(h)
        ) * valid[..., None]

        # 同一区域 token 沿时间轴运行 TCN。
        h = h.reshape(B, T, 2, K, 256)
        latent_mask = valid.reshape(B, T, 2, K)

        temporal_mask = (
            latent_mask.permute(0, 2, 3, 1)
            .reshape(B * 2 * K, 1, T)
            .to(h.dtype)
        )

        h = h.permute(
            0, 2, 3, 4, 1
        ).reshape(B * 2 * K, 256, T)

        for block in self.temporal:
            h = block(h, temporal_mask)

        h = h.reshape(
            B, 2, K, 256, T
        ).permute(0, 4, 1, 2, 3)

        # 局部 posterior。
        mu_l, logvar_l = self.local_stats(
            h
        ).chunk(2, dim=-1)

        mu_l = mu_l * latent_mask[..., None]
        logvar_l = (
            logvar_l.clamp(-12, 8)
            * latent_mask[..., None]
        )

        # 全局 posterior。
        pooled = (
            h * latent_mask[..., None]
        ).sum(dim=(2, 3))

        pooled = pooled / (
            latent_mask.sum(dim=(2, 3))
            [..., None].clamp_min(1)
        )

        mu_g, logvar_g = self.global_stats(
            pooled
        ).chunk(2, dim=-1)
        logvar_g = logvar_g.clamp(-12, 8)

        sample = (
            self.training if sample is None else sample
        )

        def draw(mu, logvar):
            if not sample:
                return mu
            return (
                mu
                + torch.exp(0.5 * logvar)
                * torch.randn_like(mu)
            )

        return {
            "z_global": draw(mu_g, logvar_g),
            "z_local": (
                draw(mu_l, logvar_l)
                * latent_mask[..., None]
            ),
            "mu_global": mu_g,
            "logvar_global": logvar_g,
            "mu_local": mu_l,
            "logvar_local": logvar_l,
            "latent_mask": latent_mask,
        }


if __name__ == "__main__":
    torch.manual_seed(7)

    B, T, L = 2, 3, 2048

    tick_norm = torch.full((B,), 0.001)
    rank = torch.arange(L).float()[None, None, None]
    sign = torch.tensor([-1.0, 1.0])[None, None, :, None]

    # 构造有序价格：bid 递减，ask 递增。
    price = sign * (0.002 + rank * 0.001)
    price = price.expand(B, T, 2, L).clone()

    volume = torch.rand(B, T, 2, L) * 10
    volume[..., 1000] = 100

    mask = torch.ones(B, T, 2, L, dtype=torch.bool)

    model = OrderBookEncoder(
        levels=L,
        quantity_scale=2.0,
        causal=False,
    ).eval()

    with torch.no_grad():
        out = model(
            price,
            volume,
            mask,
            tick_norm,
            sample=False,
        )

    print("输入价格:", price.shape)               # [2,3,2,2048]
    print("全局 latent:", out["z_global"].shape)  # [2,3,64]
    print("局部 latent:", out["z_local"].shape)   # [2,3,2,128,16]
    print("区域 mask:", out["latent_mask"].shape) # [2,3,2,128]

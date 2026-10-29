import torch
from torch import nn
from torch.nn import functional as F

from .book_layers import LatentStem, UpBlock, mask_at


class PriceDecoder(nn.Module):
    def __init__(
        self,
        levels=2048,
        global_dim=64,
        local_dim=16,
        gap_activation="softplus",
        fix_first_mid=True,
    ):
        super().__init__()
        assert gap_activation in ("softplus", "relu")

        self.gap_activation = gap_activation
        self.fix_first_mid = fix_first_mid

        self.stem = LatentStem(
            levels, global_dim, local_dim
        )

        channels = (256, 192, 128, 64, 32)
        self.up = nn.ModuleList([
            UpBlock(
                a, b,
                dilation=1 if b == 32 else 2,
            )
            for a, b in zip(channels[:-1], channels[1:])
        ])

        self.gap_head = nn.Conv1d(32, 1, 1)
        self.global_head = nn.Sequential(
            nn.Linear(global_dim + local_dim, 128),
            nn.GELU(),
            nn.Linear(128, 2),
        )

    def forward(
        self,
        z_global,
        z_local,
        mask,
        tick_norm,
        times=None,
    ):
        B, T, _, L = mask.shape
        K = z_local.shape[-2]

        m = mask.reshape(
            B * T * 2, 1, L
        ).to(z_global.dtype)

        h = self.stem(
            z_global, z_local, mask, times
        )
        for block in self.up:
            h = block(h, m)

        lm = mask_at(m, K).reshape(B, T, 2, K)
        pooled = (
            z_local * lm[..., None]
        ).sum(dim=(2, 3))
        pooled = pooled / (
            lm.sum(dim=(2, 3))
            [..., None].clamp_min(1)
        )

        raw = self.global_head(
            torch.cat((z_global, pooled), dim=-1)
        ).float()

        # mid 相对固定 anchor 的偏移可以正也可以负。
        mid = raw[..., 0]
        if self.fix_first_mid:
            mid = torch.cat(
                (
                    torch.zeros_like(mid[:, :1]),
                    mid[:, 1:],
                ),
                dim=1,
            )

        positive = (
            F.softplus
            if self.gap_activation == "softplus"
            else F.relu
        )

        eps = tick_norm.float()[:, None]
        spread = eps * (
            1 + positive(raw[..., 1])
        )

        # L-1 个相邻价差，至少为一个 tick。
        raw_gap = (
            self.gap_head(h)
            .reshape(B, T, 2, L)
            [..., 1:]
            .float()
        )
        gap = eps[:, :, None, None] * (
            1 + positive(raw_gap)
        )
        gap = gap * (
            mask[..., 1:] & mask[..., :-1]
        )

        distance = F.pad(
            torch.cumsum(
                gap, dim=-1, dtype=torch.float32
            ),
            (1, 0),
        )

        sign = distance.new_tensor(
            [-1, 1]
        )[None, None, :, None]

        x = mid[:, :, None, None] + sign * (
            0.5 * spread[:, :, None, None]
            + distance
        )

        return {
            "price_norm": x * mask,
            "mid_norm": mid,
            "spread_norm": spread,
            "gap_norm": gap,
        }


if __name__ == "__main__":
    torch.manual_seed(7)

    B, T, L = 2, 3, 2048
    K = L // 16

    zg = torch.randn(B, T, 64)
    zl = torch.randn(B, T, 2, K, 16)

    mask = torch.ones(B, T, 2, L, dtype=torch.bool)
    tick_norm = torch.full((B,), 0.001)

    model = PriceDecoder(
        levels=L,
        gap_activation="softplus",
        fix_first_mid=True,
    ).eval()

    with torch.no_grad():
        out = model(zg, zl, mask, tick_norm)

    price = out["price_norm"]
    bid = price[:, :, 0]
    ask = price[:, :, 1]

    print("价格:", price.shape)                  # [2,3,2,2048]
    print("价差:", out["gap_norm"].shape)        # [2,3,2,2047]
    print("mid:", out["mid_norm"].shape)         # [2,3]
    print("spread:", out["spread_norm"].shape)   # [2,3]

    print("bid 递减:", bool((bid[..., 1:] < bid[..., :-1]).all()))
    print("ask 递增:", bool((ask[..., 1:] > ask[..., :-1]).all()))
    print("买卖价不交叉:", bool((bid[..., 0] < ask[..., 0]).all()))

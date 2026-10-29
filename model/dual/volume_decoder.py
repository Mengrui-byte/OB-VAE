import torch
from torch import nn
from torch.nn import functional as F

from .book_layers import (
    LatentStem,
    UpBlock,
    mask_at,
    mean_at,
)


class VolumeDecoder(nn.Module):
    def __init__(
        self,
        levels=2048,
        quantity_scale=1.0,
        global_dim=64,
        local_dim=16,
        detach_price=False,
    ):
        super().__init__()

        self.detach_price = detach_price
        self.register_buffer(
            "q0", torch.tensor(float(quantity_scale))
        )

        self.stem = LatentStem(
            levels, global_dim, local_dim
        )

        # 价格1维 + log gap 1维 + 位置编码32维。
        self.geometry = nn.Sequential(
            nn.Conv1d(34, 32, 1),
            nn.GELU(),
            nn.Conv1d(32, 32, 1),
        )
        self.initial_fuse = nn.Conv1d(
            256 + 32, 256, 1
        )

        channels = (256, 192, 128, 64, 32)
        self.up = nn.ModuleList([
            UpBlock(
                a,
                b,
                condition_dim=32,
                dilation=1 if b == 32 else 2,
            )
            for a, b in zip(channels[:-1], channels[1:])
        ])
        self.quantity_head = nn.Conv1d(32, 1, 1)

    def forward(
        self,
        z_global,
        z_local,
        price_output,
        mask,
        tick_norm,
        times=None,
    ):
        B, T, _, L = mask.shape

        price = price_output["price_norm"]
        gap = price_output["gap_norm"]

        if self.detach_price:
            price = price.detach()
            gap = gap.detach()

        eps = tick_norm[:, None, None, None]

        # 第一档没有前驱 gap，使用中性特征0。
        log_gap = F.pad(
            torch.log(
                (gap / eps).clamp_min(1e-6)
            ),
            (1, 0),
        )

        pos = self.stem.position(
            B, T, L, z_global, times
        )

        geometry = torch.cat(
            (
                price.unsqueeze(-2),
                log_gap.unsqueeze(-2),
                pos,
            ),
            dim=-2,
        ).reshape(B * T * 2, 34, L)

        m = mask.reshape(
            B * T * 2, 1, L
        ).to(geometry.dtype)

        geometry = self.geometry(
            geometry * m
        ) * m

        h = self.stem(
            z_global, z_local, mask, times
        )

        condition = mean_at(
            geometry, m, h.shape[-1]
        )
        h = self.initial_fuse(
            torch.cat((h, condition), dim=1)
        )
        h = h * mask_at(m, h.shape[-1])

        # 在所有上采样尺度注入预测价格几何。
        for block in self.up:
            h = block(h, m, geometry)

        raw = (
            self.quantity_head(h)
            .reshape(B, T, 2, L)
            .float()
        )

        return self.q0 * F.softplus(raw) * mask


if __name__ == "__main__":
    from .price_decoder import PriceDecoder

    torch.manual_seed(7)

    B, T, L = 2, 3, 2048
    K = L // 16

    zg = torch.randn(B, T, 64)
    zl = torch.randn(B, T, 2, K, 16)

    mask = torch.ones(B, T, 2, L, dtype=torch.bool)
    tick_norm = torch.full((B,), 0.001)

    price_model = PriceDecoder(
        levels=L,
    ).eval()

    volume_model = VolumeDecoder(
        levels=L,
        quantity_scale=2.0,
        detach_price=False,
    ).eval()

    with torch.no_grad():
        price_out = price_model(
            zg, zl, mask, tick_norm
        )
        volume_hat = volume_model(
            zg, zl, price_out, mask, tick_norm
        )

    print("条件价格:", price_out["price_norm"].shape)
    # [2,3,2,2048]

    print("重建挂单量:", volume_hat.shape)
    # [2,3,2,2048]

    print("有效档数量为正:", bool((volume_hat[mask] > 0).all()))

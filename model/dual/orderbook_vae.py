import torch
from torch import nn

from .encoder import OrderBookEncoder
from .price_decoder import PriceDecoder
from .volume_decoder import VolumeDecoder


class OrderBookVAE(nn.Module):
    def __init__(
        self,
        levels=2048,
        quantity_scale=1.0,
        global_dim=64,
        local_dim=16,
        causal=False,
        gap_activation="softplus",
        detach_price=False,
        fix_first_mid=True,
    ):
        super().__init__()

        self.encoder = OrderBookEncoder(
            levels=levels,
            quantity_scale=quantity_scale,
            global_dim=global_dim,
            local_dim=local_dim,
            causal=causal,
        )

        self.price_decoder = PriceDecoder(
            levels=levels,
            global_dim=global_dim,
            local_dim=local_dim,
            gap_activation=gap_activation,
            fix_first_mid=fix_first_mid,
        )

        self.volume_decoder = VolumeDecoder(
            levels=levels,
            quantity_scale=quantity_scale,
            global_dim=global_dim,
            local_dim=local_dim,
            detach_price=detach_price,
        )

    def forward(
        self,
        price,
        volume,
        mask,
        tick_norm,
        times=None,
        sample=None,
    ):
        posterior = self.encoder(
            price,
            volume,
            mask,
            tick_norm,
            times=times,
            sample=sample,
        )

        # 两个 decoder 共用同一次 posterior 采样。
        zg = posterior["z_global"]
        zl = posterior["z_local"]

        price_out = self.price_decoder(
            zg,
            zl,
            mask,
            tick_norm,
            times=times,
        )

        volume_hat = self.volume_decoder(
            zg,
            zl,
            price_out,
            mask,
            tick_norm,
            times=times,
        )

        return {
            **price_out,
            "volume": volume_hat,
            "posterior": posterior,
        }


if __name__ == "__main__":
    torch.manual_seed(7)

    B, T, L = 2, 3, 2048

    # 示例归一化参数。
    anchor = torch.tensor([60000.0, 65000.0])
    tick_size = torch.tensor([0.1, 0.1])
    price_rms = 50.0
    quantity_scale = 2.0

    tick_norm = (
        1e4 * tick_size / (anchor * price_rms)
    )

    # 每侧2048档，价格间隔为一个tick。
    rank = torch.arange(L).float()[None, None, None]
    sign = torch.tensor([-1.0, 1.0])[None, None, :, None]

    price = (
        sign
        * (rank + 0.5)
        * tick_norm[:, None, None, None]
    )
    price = price.expand(B, T, 2, L).clone()

    volume = torch.rand(B, T, 2, L) * 10
    volume[..., 1000] = 100

    # 张量容量是2048档；示例中某帧bid只有1024个有效档。
    counts = torch.full(
        (B, T, 2), L, dtype=torch.long
    )
    counts[0, -1, 0] = 1024

    mask = (
        torch.arange(L)[None, None, None]
        < counts[..., None]
    )

    price = price * mask
    volume = volume * mask

    model = OrderBookVAE(
        levels=L,
        quantity_scale=quantity_scale,
        causal=False,
        gap_activation="softplus",
        detach_price=False,
        fix_first_mid=True,
    ).eval()

    with torch.no_grad():
        out = model(
            price,
            volume,
            mask,
            tick_norm,
            sample=False,
        )

        # 逆变换回原始价格单位。
        price_hat = (
            anchor[:, None, None, None]
            * (1 + price_rms * out["price_norm"] / 1e4)
        ) * mask

    print("输入价格:", price.shape)              # [2,3,2,2048]
    print("重建价格:", price_hat.shape)          # [2,3,2,2048]
    print("重建数量:", out["volume"].shape)      # [2,3,2,2048]
    print("相邻价差:", out["gap_norm"].shape)    # [2,3,2,2047]

    print(
        "全局 latent:",
        out["posterior"]["z_global"].shape,
    )  # [2,3,64]

    print(
        "局部 latent:",
        out["posterior"]["z_local"].shape,
    )  # [2,3,2,128,16]

    print(
        "padding 数量为零:",
        bool((out["volume"][~mask] == 0).all()),
    )

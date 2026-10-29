import torch

def raw_batch(size=2, frames=3, levels=32):
    rank = torch.arange(levels)[None, None, None, :]
    sign = torch.tensor([-1, 1])[None, None, :, None]
    anchor = torch.arange(size)[:, None, None, None] * 10 + 1000.0
    drift = torch.arange(frames)[None, :, None, None] * 0.25
    price = anchor + drift + sign * (rank + 0.5) * 0.25
    volume = torch.linspace(0.1, 4, price.numel()).reshape(price.shape)
    counts = torch.full((size, frames, 2), levels, dtype=torch.long)
    counts[0, -1, 0] = max(1, levels // 2)
    mask = torch.arange(levels) < counts[..., None]
    data = torch.stack((price, volume), -1).masked_fill(~mask[..., None], 0)
    return {"data": data, "level_counts": counts}

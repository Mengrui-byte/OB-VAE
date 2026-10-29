"""Fixed, masked multiscale order-book features (rank axis only)."""
import torch
from torch import nn
from torch.nn import functional as F


class BookFeatureExtractor(nn.Module):
    scales = (1, 2, 4, 8, 16, 32)

    def __init__(self, quantity_scale):
        super().__init__()
        self.register_buffer('q0', torch.tensor(float(quantity_scale)))
        for r in self.scales:
            j = torch.arange(-3*r, 3*r+1).float()
            k = (1-(j/r)**2)*torch.exp(-.5*(j/r)**2)
            k = k-k.mean()
            self.register_buffer(f'log_kernel_{r}', (k/k.abs().sum())[None, None])

    def forward(self, price, volume, mask, tick_norm):
        # FP32 is required for signed LoG and exact full-neighborhood counts.
        with torch.autocast(price.device.type, enabled=False):
            price = price.float().masked_fill(~mask, 0)
            amount = volume.float().masked_fill(~mask, 0)/self.q0
            u = torch.log1p(amount)
            B, T, _, L = price.shape
            direction = price.new_tensor([-1, 1])[None, None, :, None]
            gaps = (price[..., 1:]-price[..., :-1])*direction
            adjacent = mask[..., 1:] & mask[..., :-1]
            loggap = F.pad(torch.log((gaps/tick_norm[:, None, None, None]).clamp_min(1e-6))*adjacent, (1, 0))
            base = torch.stack((price, amount, u, loggap), -2).reshape(B*T*2, 4, L)
            m = mask.reshape(B*T*2, 1, L).float()
            u = u.reshape(B*T*2, 1, L)*m
            peaks, logs = [], []
            for r in self.scales:
                left = F.pad(u, (r, 0))[..., :L]
                right = F.pad(u, (0, r))[..., r:r+L]
                ml = F.pad(m, (r, 0))[..., :L]
                mr = F.pad(m, (0, r))[..., r:r+L]
                peaks.append(F.relu(torch.minimum(u-left, u-right)/r)*m*ml*mr)
                kernel = getattr(self, f'log_kernel_{r}')
                width = kernel.shape[-1]
                response = F.conv1d(u, kernel, padding=width//2)
                support = F.conv1d(m, torch.ones_like(kernel), padding=width//2)
                logs.append(response*(support >= width-.5))
            return torch.cat([base, *peaks, *logs], 1)*m

"""One window vector from fixed, train-fitted global/local PCA projections."""

import torch
from torch import nn


class GroupedPCACodec(nn.Module):
    def __init__(self, *, state_dim, global_components, window_size=16,
                 levels=2048, global_dim=64, local_dim=16):
        super().__init__()
        self.state_dim, self.global_components = state_dim, global_components
        self.local_components = state_dim - global_components
        self.window_size, self.levels = window_size, levels
        self.global_dim, self.local_dim = global_dim, local_dim
        self.local_length = window_size * 2 * (levels // 16) * local_dim
        self.global_length = window_size * global_dim
        if not (0 < global_components <= self.global_length and 0 < self.local_components <= self.local_length):
            raise ValueError('Invalid PCA component allocation')
        for group, size, components in [('global', self.global_length, global_components),
                                         ('local', self.local_length, self.local_components)]:
            self.register_buffer(group+'_mean', torch.zeros(size))
            self.register_buffer(group+'_scale', torch.ones(size))
            self.register_buffer(group+'_basis', torch.zeros(components, size))

    def load_basis(self, data):
        with torch.no_grad():
            for group, count in [('global', self.global_components), ('local', self.local_components)]:
                for field in ('mean', 'scale', 'basis'):
                    source = data[group+'_'+field]
                    if field == 'basis': source = source[:count]
                    destination = getattr(self, group+'_'+field)
                    if source.shape != destination.shape:
                        raise ValueError(f'PCA {group}_{field} shape mismatch')
                    destination.copy_(source.to(destination))
                if not bool((getattr(self, group+'_scale') > 0).all()):
                    raise ValueError('PCA scales must be positive')

    def encode(self, zg, zl):
        if zg.shape[1:] != (self.window_size, self.global_dim) or zl.shape[1:] != (self.window_size, 2, self.levels//16, self.local_dim):
            raise ValueError('PCA codec requires the configured complete window')
        # PCA uses strict FP32 arithmetic, including when the generator trains
        # under BF16 autocast. Both groups become one [B,state_dim] vector.
        with torch.autocast(device_type=zg.device.type, enabled=False):
            g = (zg.float().reshape(len(zg), -1) - self.global_mean) / self.global_scale
            l = (zl.float().reshape(len(zl), -1) - self.local_mean) / self.local_scale
            return torch.cat([g @ self.global_basis.T, l @ self.local_basis.T], -1)

    def decode(self, state, latent_mask):
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f'Expected [B,{self.state_dim}] window state')
        with torch.autocast(device_type=state.device.type, enabled=False):
            g = state[:, :self.global_components].float() @ self.global_basis
            l = state[:, self.global_components:].float() @ self.local_basis
            g = (g * self.global_scale + self.global_mean).reshape(-1, self.window_size, self.global_dim)
            l = (l * self.local_scale + self.local_mean).reshape(-1, self.window_size, 2, self.levels//16, self.local_dim)
            return g, l * latent_mask[..., None]

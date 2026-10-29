"""RQ-VAE over a complete continuous window latent, without return targets."""

import torch
from torch import nn

from model.layers.residual_quantizer import ResidualEMAQuantizer


class LatentRQVAE(nn.Module):
    def __init__(self, input_dim=1024, code_dim=128, hidden_dim=512,
                 levels=8, codebook_size=256, kind='ae', seed=42, normalization='per_dimension'):
        super().__init__()
        if kind not in ('ae','direct') or (kind == 'direct' and code_dim != input_dim):
            raise ValueError('Direct RVQ preserves the input dimension')
        self.kind, self.input_dim, self.code_dim = kind, input_dim, code_dim
        if normalization not in ('per_dimension','global'):
            raise ValueError('Unknown latent normalization')
        self.normalization=normalization
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_scale', torch.ones(input_dim))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+1351)
            if kind == 'ae':
                self.encoder = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.GELU(),
                    nn.Linear(hidden_dim,code_dim),nn.LayerNorm(code_dim,elementwise_affine=False))
                self.decoder = nn.Sequential(nn.Linear(code_dim,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,input_dim))
            else:
                self.encoder, self.decoder = nn.Identity(), nn.Identity()
        self.quantizer = ResidualEMAQuantizer(code_dim,levels,codebook_size)

    @torch.no_grad()
    def fit_normalization(self, training_states):
        self.input_mean.copy_(training_states.float().mean(0))
        if self.normalization=='per_dimension':
            self.input_scale.copy_(training_states.float().std(0).clamp_min(1e-4))
        else:
            rms=(training_states.float()-self.input_mean).square().mean().sqrt().clamp_min(1e-4)
            self.input_scale.fill_(float(rms))

    def normalize(self, states):
        return (states.float()-self.input_mean)/self.input_scale

    def encode(self, states):
        return self.encoder(self.normalize(states))

    def decode(self, features):
        return self.decoder(features)*self.input_scale+self.input_mean

    def continuous(self, states):
        return self.decode(self.encode(states))

    def forward(self, states, depth=None):
        features=self.encode(states)
        result=self.quantizer(features,depth)
        result.update(reconstruction=self.decode(result['quantized']),features=features)
        return result

    def decode_codes(self, codes):
        return self.decode(self.quantizer.decode_codes(codes))

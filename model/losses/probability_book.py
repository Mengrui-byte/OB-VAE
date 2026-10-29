"""Ordered quantity-distribution supervision, with the original output intact."""
import math

import torch
from torch.nn import functional as F

from model.losses.controlled_book import ControlledBookLoss


def quantity_distributions(prediction, target, mask, quantity_scale):
    """Same normalization/floor as ControlledBookLoss's existing fine KL."""
    target = target.float().masked_fill(~mask, 0) / quantity_scale
    prediction = prediction.float().masked_fill(~mask, 0) / quantity_scale
    mass = target.sum(-1, keepdim=True)
    p = target / mass.clamp_min(1e-8)
    stable = prediction + 1e-6 * mask
    ph = stable / stable.sum(-1, keepdim=True).clamp_min(1e-8)
    valid = mask.any(-1) & (mass[..., 0] > 0)
    return p, ph, valid


def rank_wasserstein(p, ph):
    """Per-distribution W1 at fixed coordinates l/(L-1), using FP32 CDFs."""
    if p.shape[-1] < 2:
        return p.sum(-1) * 0
    return (p.float() - ph.float()).cumsum(-1)[..., :-1].abs().sum(-1) / (p.shape[-1] - 1)


def pooled_kl(p, ph, group):
    padding = (-p.shape[-1]) % group
    a = F.pad(p, (0, padding)).reshape(*p.shape[:-1], -1, group).sum(-1)
    b = F.pad(ph, (0, padding)).reshape(*ph.shape[:-1], -1, group).sum(-1)
    return (a * (a.clamp_min(1e-12).log() - b.clamp_min(1e-12).log())).sum(-1)


def valid_mean(value, valid):
    return (value * valid).sum() / valid.sum().clamp_min(1)


class ProbabilityBookLoss(ControlledBookLoss):
    def __init__(self, quantity_scale, add_wasserstein=False, add_multiscale=False,
                 w1_weight=1., multiscale_weight=1., multiscale_groups=(8, 64), **kwargs):
        super().__init__(quantity_scale, **kwargs)
        if self.mode != 'shape_mass':
            raise ValueError('Probability controls retain the existing shape_mass objective')
        if any(not math.isfinite(w) or w < 0 for w in (w1_weight, multiscale_weight)):
            raise ValueError('Probability weights must be finite and nonnegative')
        self.multiscale_groups = tuple(multiscale_groups)
        if not self.multiscale_groups or any(not isinstance(g, int) or g < 2 for g in self.multiscale_groups):
            raise ValueError('Expected integer grouping sizes >=2')
        self.add_wasserstein, self.add_multiscale = add_wasserstein, add_multiscale
        self.w1_weight, self.multiscale_weight = w1_weight, multiscale_weight

    def forward(self, out, batch, beta=0.):
        values = super().forward(out, batch, beta)
        values['pre_probability_loss'] = values['loss']
        p, ph, valid = quantity_distributions(out['volume'], batch['volume'], batch['mask'], self.quantity_scale)
        values['quantity_rank_w1'] = valid_mean(rank_wasserstein(p, ph), valid)
        terms = []
        for group in self.multiscale_groups:
            term = valid_mean(pooled_kl(p, ph, group), valid)
            values[f'quantity_kl_group{group}'] = term
            terms.append(term)
        values['quantity_multiscale_kl'] = torch.stack(terms).mean()
        if self.add_wasserstein:
            values['loss'] = values['loss'] + self.w1_weight * values['quantity_rank_w1']
        if self.add_multiscale:
            values['loss'] = values['loss'] + self.multiscale_weight * values['quantity_multiscale_kl']
        values['reconstruction'] = values['loss'] - beta * values['kl']
        return values

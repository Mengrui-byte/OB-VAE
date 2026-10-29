"""Protect absolute reconstruction across fixed physical quantity ranges."""
import math

import torch
from torch.nn import functional as F

from model.losses.recall_guard import RecallGuardBookLoss


QUANTITY_LOWER_BOUNDS = (0., .01, .1, 1., 10., 100.)


def stratified_quantity_huber(prediction, target, mask, scales):
    if prediction.shape != target.shape or prediction.shape != mask.shape:
        raise ValueError('Quantity and validity shapes must match')
    if mask.dtype != torch.bool:
        raise ValueError('Boolean validity mask required')
    scales = torch.as_tensor(scales, device=prediction.device, dtype=torch.float32)
    if scales.shape != (6,) or not bool(torch.isfinite(scales).all() and (scales > 0).all()):
        raise ValueError('Six finite positive training quantity scales required')
    with torch.autocast(prediction.device.type, enabled=False):
        pred = prediction.float().masked_fill(~mask, 0)
        truth = target.float().masked_fill(~mask, 0)
        dims = tuple(range(1, pred.ndim))
        losses, counts = [], []
        for i, lower in enumerate(QUANTITY_LOWER_BOUNDS):
            selected = mask & (truth >= lower)
            if i < 5: selected &= truth < QUANTITY_LOWER_BOUNDS[i+1]
            count = selected.sum(dims)
            error = (pred-truth) / scales[i]
            point = F.huber_loss(error, torch.zeros_like(error), reduction='none')
            losses.append(point.masked_fill(~selected, 0).sum(dims) / count.clamp_min(1))
            counts.append(count)
        losses, counts = torch.stack(losses, -1), torch.stack(counts, -1)
        present = counts > 0
        per_window = losses.sum(-1) / present.sum(-1).clamp_min(1)
        total = per_window.sum() / present.any(-1).sum().clamp_min(1)
        per_bin = losses.sum(0) / present.sum(0).clamp_min(1)
    return total, per_bin, counts.sum(0)


class StratifiedQuantityBookLoss(RecallGuardBookLoss):
    def __init__(self, quantity_scale, *, bin_scales, stratified_weight=0., **kwargs):
        super().__init__(quantity_scale, **kwargs)
        scales = torch.as_tensor(bin_scales, dtype=torch.float32)
        if scales.shape != (6,) or not bool(torch.isfinite(scales).all() and (scales > 0).all()):
            raise ValueError('Invalid training-only quantity-bin scales')
        if not math.isfinite(stratified_weight) or stratified_weight < 0:
            raise ValueError('Nonnegative finite stratified weight required')
        self.register_buffer('quantity_bin_scales', scales)
        self.stratified_weight = stratified_weight

    def forward(self, out, batch, beta=0., teacher_local=None):
        values = super().forward(out, batch, beta, teacher_local)
        values['pre_stratified_loss'] = values['loss']
        loss, parts, counts = stratified_quantity_huber(out['volume'], batch['volume'], batch['mask'], self.quantity_bin_scales)
        values['stratified_quantity'] = loss
        for i in range(6):
            values[f'quantity_bin{i}_loss'] = parts[i]
            values[f'quantity_bin{i}_count'] = counts[i]
        if self.stratified_weight:
            values['loss'] = values['loss'] + self.stratified_weight * loss
        values['reconstruction'] = values['loss'] - beta * values['kl']
        return values

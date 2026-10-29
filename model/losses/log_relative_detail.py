"""Relative log quantity and phase-sensitive local detail, with inherited loss.

Both objectives default to equal present-bin and nonempty-window weights.
The relative point term optionally weights bins by count**power; the signed
detail term retains the historical equal-bin reduction.
They change no decoder, activation, bottleneck, or existing objective weights.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

from model.losses.local_detail import checked_radii,signed_spatial_detail
from model.losses.quantity_strata import QUANTITY_LOWER_BOUNDS


def checked_log_quantities(prediction,target,mask,epsilon):
    if not math.isfinite(epsilon) or epsilon<=0:
        raise ValueError('Finite positive physical BTC epsilon required')
    if prediction.shape!=target.shape or prediction.shape!=mask.shape or mask.dtype!=torch.bool or prediction.ndim<2:
        raise ValueError('Matching quantity/mask arrays with a batch dimension required')
    p=prediction.float().masked_fill(~mask,0);t=target.float().masked_fill(~mask,0)
    if not bool(torch.isfinite(p).all() and torch.isfinite(t).all() and (p>=0).all() and (t>=0).all()):
        raise ValueError('Valid quantities must be finite and nonnegative')
    return torch.log(p+epsilon),torch.log(t+epsilon),t


def checked_count_power(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
        raise ValueError('Quantity-bin count power must be finite and between0 and1')
    return float(value)


def balanced_huber_error(error,truth,valid,*,count_power=0.):
    count_power=checked_count_power(count_power)
    dims=tuple(range(1,error.ndim));losses=[];counts=[]
    point=F.huber_loss(error,torch.zeros_like(error),reduction='none')
    for i,lower in enumerate(QUANTITY_LOWER_BOUNDS):
        selected=valid&(truth>=lower)
        if i<5:selected &= truth<QUANTITY_LOWER_BOUNDS[i+1]
        number=selected.sum(dims);counts.append(number)
        losses.append(point.masked_fill(~selected,0).sum(dims)/number.clamp_min(1))
    values=torch.stack(losses,-1);numbers=torch.stack(counts,-1);present=numbers>0
    if count_power==0:
        # Historical/default path is retained bit for bit, including gradients.
        per_window=values.sum(-1)/present.sum(-1).clamp_min(1)
    else:
        weights=numbers.to(values.dtype).pow(count_power)
        per_window=(values*weights).sum(-1)/weights.sum(-1).clamp_min(1)
    total=per_window.sum()/present.any(-1).sum().clamp_min(1)
    parts=values.sum(0)/present.sum(0).clamp_min(1)
    return total,parts,numbers.sum(0)


def relative_log_quantity_huber(prediction,target,mask,*,epsilon=1e-4,count_power=0.):
    with torch.autocast(prediction.device.type,enabled=False):
        p,t,truth=checked_log_quantities(prediction,target,mask,epsilon)
        return balanced_huber_error(p-t,truth,mask,count_power=count_power)


def signed_log_quantity_detail_huber(prediction,target,mask,scales,*,epsilon=1e-4,radii=(1,2,4,8)):
    radii=checked_radii(radii)
    scales=torch.as_tensor(scales,dtype=torch.float32,device=prediction.device)
    if scales.shape!=(len(radii),6) or not bool(torch.isfinite(scales).all() and (scales>0).all()):
        raise ValueError('Positive train-only detail RMS per radius and bin required')
    with torch.autocast(prediction.device.type,enabled=False):
        p,t,truth=checked_log_quantities(prediction,target,mask,epsilon)
        totals=[];parts=[];counts=[];active=[]
        for row,radius in enumerate(radii):
            hp,valid=signed_spatial_detail(p,mask,radius);ht,_=signed_spatial_detail(t,mask,radius)
            # A position's scale is chosen only by its true FP32 quantity bin.
            scale=torch.ones_like(truth)
            for i,lower in enumerate(QUANTITY_LOWER_BOUNDS):
                selected=truth>=lower
                if i<5:selected &= truth<QUANTITY_LOWER_BOUNDS[i+1]
                scale=torch.where(selected,scales[row,i],scale)
            loss,per_bin,number=balanced_huber_error((hp-ht)/scale,truth,valid)
            totals.append(loss);parts.append(per_bin);counts.append(number);active.append(valid.any().float())
        active=torch.stack(active)
        total=(torch.stack(totals)*active).sum()/active.sum().clamp_min(1)
        return total,torch.stack(parts),torch.stack(counts)


class RelativeDetailBookLoss(nn.Module):
    def __init__(self,base,*,detail_scales,relative_weight=0.,detail_weight=0.,epsilon=1e-4,radii=(1,2,4,8),relative_count_power=0.):
        super().__init__()
        if not isinstance(base,nn.Module):raise ValueError('Explicit inherited loss required')
        if any(not math.isfinite(w) or w<0 for w in [relative_weight,detail_weight]):raise ValueError('Finite nonnegative auxiliary weights required')
        if not math.isfinite(epsilon) or epsilon<=0:raise ValueError('Finite positive epsilon required')
        self.radii=checked_radii(radii);scale=torch.as_tensor(detail_scales,dtype=torch.float32)
        if scale.shape!=(len(self.radii),6) or not bool(torch.isfinite(scale).all() and (scale>0).all()):raise ValueError('Invalid detail scales')
        self.base=base;self.relative_weight=relative_weight;self.detail_weight=detail_weight;self.epsilon=epsilon
        self.relative_count_power=checked_count_power(relative_count_power)
        self.register_buffer('detail_scales',scale)

    def forward(self,out,batch,beta=0.,teacher_local=None):
        values=self.base(out,batch,beta,teacher_local=teacher_local)
        values['pre_relative_detail_loss']=values['loss']
        relative,relative_parts,counts=relative_log_quantity_huber(out['volume'],batch['volume'],batch['mask'],epsilon=self.epsilon,count_power=self.relative_count_power)
        detail,detail_parts,detail_counts=signed_log_quantity_detail_huber(out['volume'],batch['volume'],batch['mask'],self.detail_scales,epsilon=self.epsilon,radii=self.radii)
        values['relative_quantity_log']=relative;values['signed_log_quantity_detail']=detail
        for i in range(6):
            values[f'relative_log_bin{i}']=relative_parts[i]
            values[f'relative_log_bin{i}_count']=counts[i]
        for row,radius in enumerate(self.radii):
            values[f'signed_log_detail_radius{radius}']=detail_parts[row].sum()/(detail_counts[row]>0).sum().clamp_min(1)
        if self.relative_weight:values['loss']=values['loss']+self.relative_weight*relative
        if self.detail_weight:values['loss']=values['loss']+self.detail_weight*detail
        values['reconstruction']=values['loss']-beta*values['kl']
        return values

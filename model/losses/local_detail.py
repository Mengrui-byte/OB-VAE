"""Paired signed quantity detail matching, separate from global variance gain."""
import math

import torch
from torch.nn import functional as F

from model.losses.quantity_strata import QUANTITY_LOWER_BOUNDS, StratifiedQuantityBookLoss


def checked_radii(radii):
    values=tuple(radii)
    if not values or any(isinstance(r,bool) or not isinstance(r,int) or r<=0 for r in values):
        raise ValueError('Positive integer detail radii required')
    if len(set(values))!=len(values):raise ValueError('Duplicate detail radii')
    return values


def signed_spatial_detail(value, mask, radius):
    """Center minus endpoint average; only a complete valid neighborhood counts.

    This loss feature is computed along ordered levels, not along physical-price
    distance or time. Both signed lobes are retained. No ReLU/no peak labels.
    """
    checked_radii((radius,))
    if value.shape!=mask.shape or mask.dtype!=torch.bool or value.ndim<2:
        raise ValueError('Expected batch/level arrays and matching boolean mask')
    value=value.masked_fill(~mask,0)
    length=value.shape[-1]
    if length<=2*radius:return value*0,torch.zeros_like(mask)
    difference=value[...,radius:-radius]-.5*(value[...,:-2*radius]+value[...,2*radius:])
    response=F.pad(difference,(radius,radius))
    # Padding is invalid; mask holes anywhere in the neighborhood invalidate it.
    valid=F.avg_pool1d(mask.reshape(-1,1,length).float(),2*radius+1,stride=1,
                       padding=radius,count_include_pad=True).reshape_as(mask)>=1-1e-6
    return response.masked_fill(~valid,0),valid


def quantity_detail_huber(prediction,target,mask,scales,*,radii=(1,2,4,8)):
    """Average within true quantity bins/windows/radii, using train-only RMS.

    A displaced peak incurs error even if prediction has exactly the same global
    variance and histogram. Constant log-quantity offsets remain the job of the
    original reconstruction objective, which must be kept alongside this term.
    """
    radii=checked_radii(radii)
    if prediction.shape!=target.shape or prediction.shape!=mask.shape or mask.dtype!=torch.bool or prediction.ndim<2:
        raise ValueError('Quantity/mask shapes must agree, with a batch dimension')
    scales=torch.as_tensor(scales,device=prediction.device,dtype=torch.float32)
    if scales.shape!=(len(radii),6) or not bool(torch.isfinite(scales).all() and (scales>0).all()):
        raise ValueError('Positive train RMS per radius and quantity bin required')
    with torch.autocast(prediction.device.type,enabled=False):
        pred=prediction.float().masked_fill(~mask,0);truth=target.float().masked_fill(~mask,0)
        if not bool(torch.isfinite(pred).all() and torch.isfinite(truth).all() and (pred>=0).all() and (truth>=0).all()):
            raise ValueError('Valid quantities must be finite and nonnegative')
        up,ut=torch.log1p(pred),torch.log1p(truth)
        dims=tuple(range(1,pred.ndim));totals=[];parts=[];counts=[];nonempty=[]
        for n,radius in enumerate(radii):
            hp,valid=signed_spatial_detail(up,mask,radius)
            ht,_=signed_spatial_detail(ut,mask,radius)
            losses=[];numbers=[]
            for i,lower in enumerate(QUANTITY_LOWER_BOUNDS):
                selected=valid&(truth>=lower)
                if i<5:selected &= truth<QUANTITY_LOWER_BOUNDS[i+1]
                number=selected.sum(dims)
                err=(hp-ht)/scales[n,i]
                point=F.huber_loss(err,torch.zeros_like(err),reduction='none')
                losses.append(point.masked_fill(~selected,0).sum(dims)/number.clamp_min(1))
                numbers.append(number)
            losses=torch.stack(losses,-1);numbers=torch.stack(numbers,-1);present=numbers>0
            per_window=losses.sum(-1)/present.sum(-1).clamp_min(1)
            totals.append(per_window.sum()/present.any(-1).sum().clamp_min(1))
            parts.append(losses.sum(0)/present.sum(0).clamp_min(1));counts.append(numbers.sum(0))
            nonempty.append(present.any().to(torch.float32))
        totals=torch.stack(totals);active=torch.stack(nonempty)
        loss=(totals*active).sum()/active.sum().clamp_min(1)
    return loss,torch.stack(parts),torch.stack(counts)


class LocalDetailBookLoss(StratifiedQuantityBookLoss):
    def __init__(self,quantity_scale,*,detail_scales,detail_weight=0.,detail_radii=(1,2,4,8),**kwargs):
        super().__init__(quantity_scale,**kwargs)
        self.detail_radii=checked_radii(detail_radii)
        scales=torch.as_tensor(detail_scales,dtype=torch.float32)
        if scales.shape!=(len(self.detail_radii),6) or not bool(torch.isfinite(scales).all() and (scales>0).all()):
            raise ValueError('Invalid detail RMS scale matrix')
        if not math.isfinite(detail_weight) or detail_weight<0:raise ValueError('Finite nonnegative detail weight required')
        self.register_buffer('detail_scales',scales);self.detail_weight=detail_weight

    def forward(self,out,batch,beta=0.,teacher_local=None):
        values=super().forward(out,batch,beta,teacher_local)
        values['pre_detail_loss']=values['loss']
        loss,parts,counts=quantity_detail_huber(out['volume'],batch['volume'],batch['mask'],self.detail_scales,radii=self.detail_radii)
        values['local_signed_detail']=loss
        for n,radius in enumerate(self.detail_radii):
            valid=counts[n]>0
            values[f'detail_radius{radius}']=parts[n].sum()/valid.sum().clamp_min(1)
        for i in range(6):
            valid=counts[:,i]>0
            values[f'detail_bin{i}']=parts[:,i].sum()/valid.sum().clamp_min(1)
        if self.detail_weight:values['loss']=values['loss']+self.detail_weight*loss
        values['reconstruction']=values['loss']-beta*values['kl']
        return values

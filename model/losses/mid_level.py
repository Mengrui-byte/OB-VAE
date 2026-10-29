"""Protect low-frequency absolute midpoint levels without suppressing true moves."""
import math
import torch
from model.losses.quantity_strata import StratifiedQuantityBookLoss


def midpoint_lowpass_mse(prediction,target,valid,*,chunks=(16,64,256)):
    if prediction.shape!=target.shape or prediction.shape!=valid.shape or prediction.ndim!=2 or valid.dtype!=torch.bool:
        raise ValueError('Expected matching [batch,time] midpoint and boolean validity')
    chunks=tuple(chunks);length=prediction.shape[1]
    if not chunks or len(set(chunks))!=len(chunks) or any(isinstance(k,bool) or not isinstance(k,int) or k<=0 or length%k for k in chunks):
        raise ValueError('Unique positive block lengths must divide the full window')
    with torch.autocast(prediction.device.type,enabled=False):
        p=prediction.float().masked_fill(~valid,0);t=target.float().masked_fill(~valid,0)
        if not bool(torch.isfinite(p).all() and torch.isfinite(t).all()):raise ValueError('Valid midpoints must be finite')
        error=p-t;values=[]
        for k in chunks:
            counts=valid.reshape(valid.shape[0],-1,k).sum(-1)
            mean=error.reshape(error.shape[0],-1,k).sum(-1)/counts.clamp_min(1)
            nonempty=counts>0
            per_window=(mean.square()*nonempty).sum(-1)/nonempty.sum(-1).clamp_min(1)
            values.append(per_window.sum()/nonempty.any(-1).sum().clamp_min(1))
        parts=torch.stack(values);loss=parts.mean()
    return loss,parts


class MidLevelBookLoss(StratifiedQuantityBookLoss):
    def __init__(self,quantity_scale,*,mid_lowpass_weight=0.,mid_chunks=(16,64,256),**kwargs):
        super().__init__(quantity_scale,**kwargs)
        if not math.isfinite(mid_lowpass_weight) or mid_lowpass_weight<0:raise ValueError('Finite nonnegative lowpass weight required')
        self.mid_lowpass_weight=mid_lowpass_weight;self.mid_chunks=tuple(mid_chunks)

    def forward(self,out,batch,beta=0.,teacher_local=None):
        values=super().forward(out,batch,beta,teacher_local);values['pre_mid_lowpass_loss']=values['loss']
        target=batch['price'].float()[...,0].mean(-1);valid=batch['mask'][...,0].all(-1)
        loss,parts=midpoint_lowpass_mse(out['mid_norm'],target,valid,chunks=self.mid_chunks)
        values['mid_lowpass_mse']=loss
        for i,k in enumerate(self.mid_chunks):values[f'mid_level_block{k}']=parts[i]
        if self.mid_lowpass_weight:values['loss']=values['loss']+self.mid_lowpass_weight*loss
        values['reconstruction']=values['loss']-beta*values['kl']
        return values

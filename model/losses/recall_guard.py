"""Same-index large-quantity fitting, false-positive control and mid supervision."""
import math
import torch
from torch.nn import functional as F
from model.losses.local_hint import LocalHintBookLoss


def masked_mean(value,valid):
    return (value*valid).sum()/valid.sum().clamp_min(1)


class RecallGuardBookLoss(LocalHintBookLoss):
    def __init__(self,quantity_scale,*,use_tail=False,protect_mid=False,
                 tail_weight=0.,mid_weight=0.,large_threshold=100.,mid_delta_weight=1.,
                 fp_normalization='active',fp_error='full',fp_weight=1.,**kwargs):
        super().__init__(quantity_scale,**kwargs)
        if any(not math.isfinite(x) or x<0 for x in [tail_weight,mid_weight,mid_delta_weight,fp_weight]):
            raise ValueError('Nonnegative finite auxiliary weights required')
        if not math.isfinite(large_threshold) or large_threshold<=0:raise ValueError('Invalid large threshold')
        self.use_tail,self.protect_mid=use_tail,protect_mid
        self.tail_weight,self.mid_weight,self.large_threshold=tail_weight,mid_weight,large_threshold
        self.mid_delta_weight=mid_delta_weight
        if fp_normalization not in ['active','positive']:raise ValueError('Invalid false-positive normalizer')
        if fp_error not in ['full','margin_excess']:raise ValueError('Invalid false-positive penalty')
        if fp_error=='margin_excess' and fp_normalization!='positive':raise ValueError('Smooth margin requires a target-only denominator')
        self.fp_normalization,self.fp_error,self.fp_weight=fp_normalization,fp_error,fp_weight

    def forward(self,out,batch,beta=0.,teacher_local=None):
        values=super().forward(out,batch,beta,teacher_local)
        values['pre_guard_loss']=values['loss']
        q,pred=batch['volume'].float(),out['volume'].float();mask=batch['mask']
        error=(pred-q)/self.quantity_scale
        point=F.huber_loss(error,torch.zeros_like(error),reduction='none')
        positive=mask&(q>=self.large_threshold)
        false_positive=mask&(q<self.large_threshold)&(pred.detach()>=self.large_threshold)
        positive_loss=masked_mean(point,positive)
        if self.fp_error=='full' and self.fp_normalization=='active':
            false_positive_loss=masked_mean(point,false_positive) # Original rule, unchanged.
        else:
            if self.fp_error=='full':fp_point,fp_mask=point,false_positive
            else:
                # One training quantity-scale below the decision threshold:
                # continuous at100, with nonzero cost at100 (Huber(1)=0.5).
                margin=min(self.quantity_scale,self.large_threshold)
                excess=F.relu((pred-self.large_threshold+margin)/self.quantity_scale)
                fp_point=F.huber_loss(excess,torch.zeros_like(excess),reduction='none')
                fp_mask=mask&(q<self.large_threshold)
            false_positive_loss=(fp_point*fp_mask).sum()/positive.sum().clamp_min(1)
        tail=positive_loss+false_positive_loss if self.fp_weight==1. else positive_loss+self.fp_weight*false_positive_loss
        frames=mask[...,0].all(-1)
        target_mid=batch['price'].float()[...,0].mean(-1)
        predicted_mid=out['mid_norm'].float()
        mid_level=masked_mean((predicted_mid-target_mid).square(),frames)
        adjacent=frames[:,1:]&frames[:,:-1]
        delta_error=(predicted_mid[:,1:]-predicted_mid[:,:-1])-(target_mid[:,1:]-target_mid[:,:-1])
        mid_delta=masked_mean(delta_error.square(),adjacent)
        protection=mid_level+mid_delta if self.mid_delta_weight==1. else mid_level+self.mid_delta_weight*mid_delta
        values.update(tail_positive_huber=positive_loss,tail_false_positive_huber=false_positive_loss,
                      tail_balanced=tail,mid_level_mse=mid_level,mid_delta_mse=mid_delta,mid_protection=protection)
        if self.use_tail:values['loss']=values['loss']+self.tail_weight*tail
        if self.protect_mid:
            # Preserve the original unit-weight graph for the replay control.
            term=mid_level+mid_delta if self.mid_delta_weight==1. else protection
            values['loss']=values['loss']+self.mid_weight*term
        values['reconstruction']=values['loss']-beta*values['kl']
        return values

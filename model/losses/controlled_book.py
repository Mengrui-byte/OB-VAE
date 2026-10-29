"""Single-factor quantity-loss ablations; architecture and price loss unchanged."""
import torch
from torch.nn import functional as F
from model.losses.unified_book import UnifiedBookLoss


class ControlledBookLoss(UnifiedBookLoss):
    def __init__(self,quantity_scale,mode='baseline',spatial_weight=1.,temporal_weight=1.,shape_weight=1.,mass_weight=1.,**kwargs):
        super().__init__(quantity_scale,**kwargs)
        if mode not in ('baseline','quantity_mse','spatial_difference','temporal_difference','shape_kl','mass','shape_mass'):raise ValueError(mode)
        self.mode,self.spatial_weight,self.temporal_weight=mode,spatial_weight,temporal_weight
        self.shape_weight,self.mass_weight=shape_weight,mass_weight

    @staticmethod
    def masked_huber(error,mask):
        return (F.huber_loss(error,torch.zeros_like(error),reduction='none')*mask).sum()/mask.sum().clamp_min(1)

    def forward(self,out,batch,beta=0.):
        values=super().forward(out,batch,beta)
        values['base_loss']=values['loss']
        qh,q=out['volume'].float()/self.quantity_scale,batch['volume'].float()/self.quantity_scale
        mask=batch['mask'];zero=qh.new_zeros(())
        values.update(spatial_difference=zero,temporal_difference=zero,quantity_shape_kl=zero,quantity_mass=zero)
        if self.mode=='quantity_mse':
            error=qh-q;point=.5*error.square();mean=(point*mask).sum()/mask.sum().clamp_min(1)
            idx=batch['volume'].masked_fill(~mask,-torch.inf).topk(min(self.topk,q.shape[-1]),-1).indices;chosen=mask.gather(-1,idx)
            old_point=F.huber_loss(error,torch.zeros_like(error),reduction='none')
            pair_delta=((point-old_point).gather(-1,idx)*chosen).sum()/chosen.sum().clamp_min(1)
            values['loss']=values['loss']+self.quantity_weight*(mean-values['quantity'])+self.pair_weight*pair_delta
            values['quantity']=mean;values['pair']=values['pair']+pair_delta
        elif self.mode=='spatial_difference':
            terms=[]
            for lag in (1,4,16):
                if lag>=q.shape[-1]:continue
                error=(qh[...,lag:]-qh[...,:-lag])-(q[...,lag:]-q[...,:-lag])
                terms.append(self.masked_huber(error,mask[...,lag:]&mask[...,:-lag]))
            extra=torch.stack(terms).mean();values['spatial_difference']=extra;values['loss']=values['loss']+self.spatial_weight*extra
        elif self.mode=='temporal_difference':
            terms=[]
            for lag in (1,4,16):
                if lag>=q.shape[1]:continue
                error=(qh[:,lag:]-qh[:,:-lag])-(q[:,lag:]-q[:,:-lag])
                terms.append(self.masked_huber(error,mask[:,lag:]&mask[:,:-lag]))
            extra=torch.stack(terms).mean();values['temporal_difference']=extra;values['loss']=values['loss']+self.temporal_weight*extra
        elif self.mode in ('shape_kl','mass','shape_mass'):
            target=q*mask;pred=qh*mask
            tm=target.sum(-1);pm=pred.sum(-1);valid_side=mask.any(-1)
            if self.mode in ('shape_kl','shape_mass'):
                p=target/tm[...,None].clamp_min(1e-8)
                # Add a small mass floor only at valid levels for stable log probabilities.
                stable=pred+1e-6*mask;ph=stable/stable.sum(-1,keepdim=True).clamp_min(1e-8)
                per_side=(p*(p.clamp_min(1e-12).log()-ph.clamp_min(1e-12).log())).sum(-1)
                valid=valid_side&(tm>0);shape=(per_side*valid).sum()/valid.sum().clamp_min(1)
                values['quantity_shape_kl']=shape;values['loss']=values['loss']+self.shape_weight*shape
            if self.mode in ('mass','shape_mass'):
                error=torch.log1p(pm)-torch.log1p(tm)
                mass=self.masked_huber(error,valid_side)
                values['quantity_mass']=mass;values['loss']=values['loss']+self.mass_weight*mass
        values['reconstruction']=values['loss']-beta*values['kl']
        return values

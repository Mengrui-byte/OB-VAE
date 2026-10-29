"""Training-only normalized Local hints; preserve the original book loss."""
import math
import torch
from torch.nn import functional as F
from model.losses.probability_book import ProbabilityBookLoss


class LocalHintBookLoss(ProbabilityBookLoss):
    def __init__(self,quantity_scale,*,use_hint=False,hint_weight=0.,hint_mean=None,hint_scale=None,**kwargs):
        super().__init__(quantity_scale,**kwargs)
        if not math.isfinite(hint_weight) or hint_weight<0:raise ValueError('Invalid hint weight')
        self.use_hint,self.hint_weight=use_hint,hint_weight
        self.register_buffer('hint_mean',torch.zeros(16) if hint_mean is None else torch.as_tensor(hint_mean,dtype=torch.float32))
        self.register_buffer('hint_scale',torch.ones(16) if hint_scale is None else torch.as_tensor(hint_scale,dtype=torch.float32))
        if self.hint_mean.shape!=(16,) or self.hint_scale.shape!=(16,) or not bool(torch.isfinite(self.hint_mean).all() and torch.isfinite(self.hint_scale).all() and (self.hint_scale>0).all()):
            raise ValueError('Expected16 finite channel means andpositive scales')

    def forward(self,out,batch,beta=0.,teacher_local=None):
        values=super().forward(out,batch,beta)
        values['pre_hint_loss']=values['loss']
        hint=values['loss'].new_zeros(())
        if teacher_local is not None:
            prediction=out['hint_local'].float()
            target=(teacher_local.detach().float()-self.hint_mean)/self.hint_scale
            if target.shape!=prediction.shape:raise ValueError('Teacher/student Local coordinates differ')
            B,T,S,L=batch['mask'].shape
            valid=F.max_pool1d(batch['mask'].reshape(B*T*S,1,L).float(),16).reshape(B,T,S,L//16).bool()
            hint=((prediction-target).square()*valid[...,None]).sum()/(valid.sum().clamp_min(1)*16)
        if self.use_hint:
            if teacher_local is None:raise ValueError('Enabled hint requires target fromthe SAME raw input')
            values['loss']=values['loss']+self.hint_weight*hint
        values['local_hint_mse']=hint
        values['reconstruction']=values['loss']-beta*values['kl']
        return values

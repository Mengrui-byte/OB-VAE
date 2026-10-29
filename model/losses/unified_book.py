"""Masked same-index reconstruction with ONE summed window KL."""
import torch
from torch import nn
from torch.nn import functional as F


class UnifiedBookLoss(nn.Module):
    def __init__(self, quantity_scale=1., price_weight=1., quantity_weight=1., log_weight=1., pair_weight=.25, topk=16):
        super().__init__()
        self.quantity_scale = float(quantity_scale)
        self.price_weight, self.quantity_weight = price_weight, quantity_weight
        self.log_weight, self.pair_weight, self.topk = log_weight,pair_weight,topk

    def forward(self, out, batch, beta=0.):
        mask = batch['mask']
        count = mask.sum().clamp_min(1)
        dp = out['price_norm'].float()-batch['price'].float()
        dq = (out['volume'].float()-batch['volume'].float())/self.quantity_scale
        dl = torch.log1p(out['volume'].float()/self.quantity_scale)-torch.log1p(batch['volume'].float()/self.quantity_scale)
        p = F.huber_loss(dp,torch.zeros_like(dp),reduction='none')
        q = F.huber_loss(dq,torch.zeros_like(dq),reduction='none')
        log = F.huber_loss(dl,torch.zeros_like(dl),reduction='none')
        lp,lq,ll = [(x*mask).sum()/count for x in (p,q,log)]
        k = min(self.topk,mask.shape[-1])
        idx = batch['volume'].masked_fill(~mask,-torch.inf).topk(k,dim=-1).indices
        chosen = mask.gather(-1,idx)
        pair = ((p+q).gather(-1,idx)*chosen).sum()/chosen.sum().clamp_min(1)
        kl = .5*(out['mu'].square()+out['logvar'].exp()-1-out['logvar']).sum((1,2)).mean()
        reconstruction = self.price_weight*lp+self.quantity_weight*(lq+self.log_weight*ll)+self.pair_weight*pair
        return dict(loss=reconstruction+beta*kl,reconstruction=reconstruction,price=lp,quantity=lq,quantity_log=ll,pair=pair,kl=kl)

"""Mass-preserving log-shape residual, exactly identity atzero initialization."""
import torch


def shape_mass_residual(quantity,mask,shape=None,log_mass=None):
    q=quantity.float().masked_fill(~mask,0)
    if shape is None and log_mass is None:return q
    mass=q.sum(-1,keepdim=True)
    if shape is not None:
        raw=shape.float().masked_fill(~mask,0)
        centered=raw-(raw.sum(-1,keepdim=True)/mask.sum(-1,keepdim=True).clamp_min(1))
        delta=8*torch.tanh(centered/8)
        increment=torch.expm1(delta)*mask
        # Avoid round-trip log/softmax errors atthe common initialfunction.
        # delta==0 -> increment==0,ratio==1, outputbit-exactlyq.
        ratio=mass/(mass+(q*increment).sum(-1,keepdim=True)).clamp_min(1e-8)
        q=q*(1+increment)*ratio
    if log_mass is not None:
        scale=torch.exp(4*torch.tanh(log_mass.float()/4))
        q=q*scale[...,None]
    return q.masked_fill(~mask,0)

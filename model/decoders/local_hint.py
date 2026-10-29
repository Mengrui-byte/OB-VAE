"""Expose the existing quantity seed; no extra input or reconstruction path."""
import torch
from torch.nn import functional as F
from model.decoders.logit_volume import LogitVolumeDecoder


class LocalHintVolumeDecoder(LogitVolumeDecoder):
    def forward(self,z,price_output,mask,tick_norm,times):
        B,T,_,L=mask.shape
        p=price_output['price_norm'];gap=price_output['gap_norm']
        if getattr(self,'detach_price_geometry',False):p,gap=p.detach(),gap.detach()
        loggap=F.pad(torch.log((gap/tick_norm[:,None,None,None]).clamp_min(1e-6)),(1,0))
        loggap=loggap*F.pad(mask[...,1:]&mask[...,:-1],(1,0))
        pos=self.position(B,T,L,z,times)
        geometry=torch.cat((p.unsqueeze(-2),loggap.unsqueeze(-2),pos),3).reshape(B*T*2,34,L)
        m=mask.reshape(B*T*2,1,L).float();condition=self.geometry(geometry)*m
        frame=self.temporal(z,times)
        # Same linear operation as spatial.forward; explicitly retained on the
        # autograd path so an auxiliary target can supervise this post-Z seed.
        seed=self.spatial.seed(frame).reshape(B,T,2,L//16,16)
        h=self.spatial(frame,mask,times,condition)
        r=self.quantity(h).float().reshape(B,T,2,L)
        output=dict(volume=(self.q0*F.softplus(r)).masked_fill(~mask,0),
                    quantity_logit=r.masked_fill(~mask,0),quantity_seed=seed)
        if getattr(self,'return_features',False):
            output.update(quantity_features=h,quantity_frame=frame)
        elif getattr(self,'return_frame',False):
            output['quantity_frame']=frame
        return output

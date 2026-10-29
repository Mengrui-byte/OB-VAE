"""Residual codewords are ordered pairs of heterogeneous matrices.

Distances are literal unweighted product-space distances: squared l2 uses
Eg+El; squared linfinity uses max(Eg,El). No per-branch rescaling or division
by matrix size is performed. Training means are common translation offsets.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class PairedMatrixQuantizer(nn.Module):
    def __init__(self, global_shape, local_shape, levels=8, size=256, metric='l2', mode='joint'):
        super().__init__()
        if metric not in ('l2','max') or mode not in ('joint','independent'):
            raise ValueError('Unknown product metric or code allocation')
        if levels<1 or size<2:raise ValueError('Invalid matrix codebook')
        self.global_shape,self.local_shape=tuple(global_shape),tuple(local_shape)
        self.levels,self.size,self.metric,self.mode=levels,size,metric,mode
        self.register_buffer('global_mean',torch.zeros(self.global_shape))
        self.register_buffer('local_mean',torch.zeros(self.local_shape))
        self.register_buffer('global_codebook',torch.zeros(levels,size,*self.global_shape))
        self.register_buffer('local_codebook',torch.zeros(levels,size,*self.local_shape))
        self.register_buffer('initialized',torch.tensor(False))

    @staticmethod
    def squared_distances(x,centers):
        return (x.square().sum(-1,keepdim=True)+centers.square().sum(-1)[None]-2*x@centers.T).clamp_min(0)

    def select(self,global_residual,local_residual,global_centers,local_centers):
        eg=self.squared_distances(global_residual,global_centers)
        el=self.squared_distances(local_residual,local_centers)
        if self.mode=='independent':return eg.argmin(-1),el.argmin(-1)
        cost=eg+el if self.metric=='l2' else torch.maximum(eg,el)
        index=cost.argmin(-1)
        return index,index

    @staticmethod
    def _means(x,indices,size):
        membership=F.one_hot(indices,size).T.to(x.dtype)
        counts=membership.sum(-1)
        means=membership@x/counts.clamp_min(1)[:,None]
        return means,counts,membership

    def _assigned(self,g,l,cg,cl,indices):
        eg=self.squared_distances(g,cg).gather(1,indices[:,None]).squeeze(1)
        el=self.squared_distances(l,cl).gather(1,indices[:,None]).squeeze(1)
        return eg,el

    def _max_update(self,g,l,cg,cl,indices,inner_steps):
        """Convex fixed-assignment prototype optimization with checked descent.

        Arithmetic means alone need not minimize a maximum-of-two-quadratics.
        Try means and then exact active-branch subgradients, accepting changes
        separately per cluster only when its assigned objective does not grow.
        """
        mg,counts,membership=self._means(g,indices,self.size)
        ml=membership@l/counts.clamp_min(1)[:,None]
        eg,el=self._assigned(g,l,cg,cl,indices)
        old=membership@torch.maximum(eg,el)
        ng,nl=self._assigned(g,l,mg,ml,indices)
        candidate=membership@torch.maximum(ng,nl)
        accept=(candidate<=old)&(counts>0)
        cg=torch.where(accept[:,None],mg,cg);cl=torch.where(accept[:,None],ml,cl)
        eg=torch.where(accept[indices],ng,eg);el=torch.where(accept[indices],nl,el)
        for _ in range(inner_steps):
            active_g=(eg>=el).to(g.dtype);active_l=1-active_g
            ag=membership*active_g[None];al=membership*active_l[None]
            dg=2*(ag.sum(-1)[:,None]*cg-ag@g)/counts.clamp_min(1)[:,None]
            dl=2*(al.sum(-1)[:,None]*cl-al@l)/counts.clamp_min(1)[:,None]
            old=membership@torch.maximum(eg,el)
            pending=counts>0
            for step in (.25,.125,.0625):
                pg,pl=cg-step*dg,cl-step*dl
                ng,nl=self._assigned(g,l,pg,pl,indices)
                candidate=membership@torch.maximum(ng,nl)
                accept=pending&(candidate<=old)
                cg=torch.where(accept[:,None],pg,cg);cl=torch.where(accept[:,None],pl,cl)
                eg=torch.where(accept[indices],ng,eg);el=torch.where(accept[indices],nl,el)
                pending=pending&~accept
                if not bool(pending.any()):break
        return cg,cl

    @torch.no_grad()
    def fit(self,global_matrices,local_matrices,iterations=12,max_inner_steps=2,seed=42,callback=None):
        g=global_matrices.float().reshape(len(global_matrices),-1)
        l=local_matrices.float().reshape(len(local_matrices),-1)
        if len(g)!=len(l) or len(g)<self.size:raise ValueError('Need enough paired training windows')
        self.global_mean.copy_(g.mean(0).reshape(self.global_shape))
        self.local_mean.copy_(l.mean(0).reshape(self.local_shape))
        rg=g-self.global_mean.flatten();rl=l-self.local_mean.flatten()
        generator=torch.Generator(device=g.device).manual_seed(seed)
        history=[]
        for level in range(self.levels):
            initial=torch.randperm(len(g),generator=generator,device=g.device)[:self.size]
            cg,cl=rg[initial].clone(),rl[initial].clone()
            for iteration in range(iterations):
                ig,il=self.select(rg,rl,cg,cl)
                if self.mode=='joint' and self.metric=='max':
                    cg,cl=self._max_update(rg,rl,cg,cl,ig,max_inner_steps)
                    counts=torch.bincount(ig,minlength=self.size);dead=(counts==0).nonzero().flatten()
                    if len(dead):
                        eg,el=self._assigned(rg,rl,cg,cl,ig)
                        replacements=torch.maximum(eg,el).topk(len(dead)).indices
                        cg[dead]=rg[replacements];cl[dead]=rl[replacements]
                else:
                    cg,cg_count,_=self._means(rg,ig,self.size)
                    cl,cl_count,_=self._means(rl,il,self.size)
                    if self.mode=='joint':
                        dead=(cg_count==0).nonzero().flatten()
                        if len(dead):
                            eg,el=self._assigned(rg,rl,cg,cl,ig)
                            replacement=(eg+el).topk(len(dead)).indices
                            cg[dead]=rg[replacement];cl[dead]=rl[replacement]
                    else:
                        for values,centers,index,count in [(rg,cg,ig,cg_count),(rl,cl,il,cl_count)]:
                            dead=(count==0).nonzero().flatten()
                            if len(dead):
                                error=(values-centers[index]).square().sum(-1)
                                centers[dead]=values[error.topk(len(dead)).indices]
                if callback and (iteration==0 or iteration==iterations-1):
                    callback(dict(level=level+1,iteration=iteration+1,iterations=iterations))
            self.global_codebook[level].copy_(cg.reshape_as(self.global_codebook[level]))
            self.local_codebook[level].copy_(cl.reshape_as(self.local_codebook[level]))
            ig,il=self.select(rg,rl,cg,cl)
            rg=rg-cg[ig];rl=rl-cl[il]
            value=dict(level=level+1,global_residual_energy=float(rg.square().sum(-1).mean()),
                local_residual_energy=float(rl.square().sum(-1).mean()),
                used_global=int(ig.unique().numel()),used_local=int(il.unique().numel()))
            history.append(value)
            if callback:callback(value)
        self.initialized.fill_(True)
        return history

    def forward(self,global_matrices,local_matrices,depth=None):
        if not bool(self.initialized):raise RuntimeError('Fit matrix codebooks on training windows first')
        depth=self.levels if depth is None else depth
        if not 1<=depth<=self.levels:raise ValueError('Invalid matrix residual depth')
        b=len(global_matrices)
        if tuple(global_matrices.shape[1:])!=self.global_shape or tuple(local_matrices.shape[1:])!=self.local_shape:
            raise ValueError('Matrix shapes differ from the codeword shapes')
        rg=global_matrices.float().reshape(b,-1)-self.global_mean.flatten()
        rl=local_matrices.float().reshape(b,-1)-self.local_mean.flatten()
        reconstructed_g=self.global_mean.flatten()[None].expand(b,-1).clone()
        reconstructed_l=self.local_mean.flatten()[None].expand(b,-1).clone()
        codes=[];energies=[torch.stack([rg.square().sum(-1),rl.square().sum(-1)],-1)]
        for level in range(depth):
            cg=self.global_codebook[level].flatten(1);cl=self.local_codebook[level].flatten(1)
            ig,il=self.select(rg,rl,cg,cl)
            qg,ql=cg[ig],cl[il]
            rg=rg-qg;rl=rl-ql;reconstructed_g=reconstructed_g+qg;reconstructed_l=reconstructed_l+ql
            codes.append(ig if self.mode=='joint' else torch.stack([ig,il],-1))
            energies.append(torch.stack([rg.square().sum(-1),rl.square().sum(-1)],-1))
        return dict(global_latent=reconstructed_g.reshape(b,*self.global_shape),
            local_latent=reconstructed_l.reshape(b,*self.local_shape),codes=torch.stack(codes,1),residual_energy=torch.stack(energies,1))

    def decode_codes(self,codes):
        if self.mode=='joint':
            if codes.ndim!=2:raise ValueError('Joint pair needs one index per residual level')
        elif codes.ndim!=3 or codes.shape[-1]!=2:raise ValueError('Independent pair needs two indices per level')
        if not 1<=codes.shape[1]<=self.levels or bool(((codes<0)|(codes>=self.size)).any()):raise ValueError('Invalid matrix codes')
        g=self.global_mean[None].expand(len(codes),*self.global_shape).clone()
        l=self.local_mean[None].expand(len(codes),*self.local_shape).clone()
        for level in range(codes.shape[1]):
            ig=codes[:,level] if self.mode=='joint' else codes[:,level,0]
            il=codes[:,level] if self.mode=='joint' else codes[:,level,1]
            g=g+self.global_codebook[level,ig];l=l+self.local_codebook[level,il]
        return g,l

    def logical_bits(self,depth=None):
        return (self.levels if depth is None else depth)*math.ceil(math.log2(self.size))*(1 if self.mode=='joint' else 2)

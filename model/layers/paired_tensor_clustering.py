"""Whole-tensor paired prototypes: Lloyd initialization and PAM swap controls."""

import torch
from torch.nn import functional as F

from model.layers.paired_matrix_quantizer import PairedMatrixQuantizer


class PairedTensorClustering(PairedMatrixQuantizer):
    METHODS=('kmeans_random','kmeans_pp','pam_squared','pam_distance')

    def __init__(self,global_shape,local_shape,levels=8,size=16,method='kmeans_pp',max_swaps=12):
        super().__init__(global_shape,local_shape,levels,size,metric='l2',mode='joint')
        if method not in self.METHODS:raise ValueError(method)
        self.method,self.max_swaps=method,max_swaps
        self.register_buffer('medoid_indices',torch.full((levels,size),-1,dtype=torch.long))

    def cost(self,g,l,cg,cl):
        return self.squared_distances(g,cg)+self.squared_distances(l,cl)

    def seed_plus_plus(self,g,l,generator):
        first=int(torch.randint(len(g),(1,),device=g.device,generator=generator))
        selected=[first];nearest=self.cost(g,l,g[first:first+1],l[first:first+1]).squeeze(1)
        for _ in range(1,self.size):
            weights=nearest.clone();weights[selected]=0
            if float(weights.sum())>0:
                index=int(torch.multinomial(weights,1,generator=generator))
            else:
                available=torch.ones(len(g),device=g.device,dtype=torch.bool);available[selected]=False
                candidates=available.nonzero().flatten();index=int(candidates[torch.randint(len(candidates),(1,),device=g.device,generator=generator)])
            selected.append(index)
            nearest=torch.minimum(nearest,self.cost(g,l,g[index:index+1],l[index:index+1]).squeeze(1))
        return torch.tensor(selected,device=g.device)

    @staticmethod
    def swap_deltas(distance,medoids):
        """Exact best-SWAP cost decomposition; tested against exhaustive swaps.

        Each row removes one medoid; each column inserts a training observation.
        This is a vectorized PAM best-swap search, not alternating cluster medoids.
        """
        values,assignment=distance[:,medoids].topk(2,dim=1,largest=False,sorted=True)
        first,second=values[:,0],values[:,1]
        nearest_with_candidate=torch.minimum(distance,first[:,None])
        base=(nearest_with_candidate-first[:,None]).sum(0,dtype=torch.float64)
        correction=torch.minimum(distance,second[:,None])-nearest_with_candidate
        membership=F.one_hot(assignment[:,0],len(medoids)).T.to(distance.dtype)
        delta=(membership@correction).double()+base[None]
        delta[:,medoids]=float('inf')
        return delta

    def pam(self,g,l,initial,level,callback):
        distance=self.cost(g,l,g,l)
        if self.method=='pam_distance':distance.sqrt_()
        distance.fill_diagonal_(0)
        medoids=initial.clone();history=[]
        objective=float(distance[:,medoids].min(1).values.sum(dtype=torch.float64))
        for swap in range(self.max_swaps):
            delta=self.swap_deltas(distance,medoids)
            index=int(delta.argmin());slot,candidate=divmod(index,len(g))
            proposal=medoids.clone();proposal[slot]=candidate
            proposed=float(distance[:,proposal].min(1).values.sum(dtype=torch.float64))
            if proposed>=objective-1e-10*max(objective,1.):break
            medoids=proposal;history.append(dict(swap=swap+1,before=objective,after=proposed,
                removed_slot=slot,inserted_training_index=candidate));objective=proposed
            if callback:callback(dict(level=level+1,swap=swap+1,max_swaps=self.max_swaps,objective=objective))
        del distance
        return medoids,history

    @torch.no_grad()
    def fit(self,global_matrices,local_matrices,iterations=12,max_inner_steps=2,seed=42,callback=None):
        if self.method=='kmeans_random':
            return super().fit(global_matrices,local_matrices,iterations,max_inner_steps,seed,callback)
        g=global_matrices.float().reshape(len(global_matrices),-1)
        l=local_matrices.float().reshape(len(local_matrices),-1)
        if len(g)!=len(l) or len(g)<self.size:raise ValueError('Not enough paired training windows')
        self.global_mean.copy_(g.mean(0).reshape(self.global_shape));self.local_mean.copy_(l.mean(0).reshape(self.local_shape))
        rg,rl=g-self.global_mean.flatten(),l-self.local_mean.flatten()
        generator=torch.Generator(device=g.device).manual_seed(seed);history=[]
        for level in range(self.levels):
            initial=self.seed_plus_plus(rg,rl,generator)
            swaps=[]
            if self.method.startswith('pam'):
                medoids,swaps=self.pam(rg,rl,initial,level,callback)
                cg,cl=rg[medoids].clone(),rl[medoids].clone();self.medoid_indices[level].copy_(medoids)
            else:
                cg,cl=rg[initial].clone(),rl[initial].clone()
                for iteration in range(iterations):
                    index=self.cost(rg,rl,cg,cl).argmin(1)
                    cg,count,membership=self._means(rg,index,self.size)
                    cl=membership@rl/count.clamp_min(1)[:,None]
                    dead=(count==0).nonzero().flatten()
                    if len(dead):
                        eg,el=self._assigned(rg,rl,cg,cl,index)
                        replacement=(eg+el).topk(len(dead)).indices;cg[dead]=rg[replacement];cl[dead]=rl[replacement]
                    if callback and iteration in (0,iterations-1):callback(dict(level=level+1,iteration=iteration+1,iterations=iterations))
            self.global_codebook[level].copy_(cg.reshape_as(self.global_codebook[level]))
            self.local_codebook[level].copy_(cl.reshape_as(self.local_codebook[level]))
            index=self.cost(rg,rl,cg,cl).argmin(1);rg=rg-cg[index];rl=rl-cl[index]
            value=dict(level=level+1,global_residual_energy=float(rg.square().sum(1).mean()),
                local_residual_energy=float(rl.square().sum(1).mean()),used_global=int(index.unique().numel()),
                used_local=int(index.unique().numel()),swaps=swaps,
                objective='sum raw product distance' if self.method=='pam_distance' else 'sum raw squared product distance')
            history.append(value)
            if callback:callback({k:v for k,v in value.items() if k!='swaps'})
        self.initialized.fill_(True)
        return history

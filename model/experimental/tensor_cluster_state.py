"""Full-latent tensor-pair codewords, one shared index per complete window."""

from torch import nn

from model.experimental.paired_matrix_state import PairedMatrixOrderBookVAE
from model.layers.paired_tensor_clustering import PairedTensorClustering


class TensorClusterOrderBookVAE(PairedMatrixOrderBookVAE):
    def __init__(self,*,cluster_method='kmeans_pp',cluster_swaps=12,matrix_window=16,matrix_levels=8,matrix_size=16,trainable_codebooks=False,**core):
        super().__init__(matrix_window=matrix_window,matrix_levels=matrix_levels,matrix_size=matrix_size,**core)
        self.tensor_cluster_config=dict(cluster_method=cluster_method,cluster_swaps=cluster_swaps,
            matrix_window=matrix_window,matrix_levels=matrix_levels,matrix_size=matrix_size)
        if trainable_codebooks:self.tensor_cluster_config['trainable_codebooks']=True
        self.matrix_quantizer=PairedTensorClustering((matrix_window,core.get('global_dim',64)),
            (matrix_window,2,core.get('levels',2048)//16,core.get('local_dim',16)),
            levels=matrix_levels,size=matrix_size,method=cluster_method,max_swaps=cluster_swaps)
        if trainable_codebooks:
            for name in ('global_codebook','local_codebook'):
                value=self.matrix_quantizer._buffers.pop(name)
                self.matrix_quantizer.register_parameter(name,nn.Parameter(value))
        self.straight_through=False

    def configure_training(self,mode):
        if mode not in ('books','decoder','joint','all'):raise ValueError(mode)
        self.requires_grad_(False)
        if mode!='decoder':
            if not isinstance(self.matrix_quantizer.global_codebook,nn.Parameter):raise ValueError('Trainable codebooks required')
            self.matrix_quantizer.requires_grad_(True)
        if mode!='books':
            for module in (self.price_decoder,self.volume_decoder,self.volume_refinement):module.requires_grad_(True)
        if mode=='all':self.encoder.requires_grad_(True);self.fusion.requires_grad_(True)
        self.straight_through=mode=='all'

    def forward_tensor_latents(self,g,l,mask,tick_norm,times=None):
        q=self.matrix_quantizer(g,l,self.matrix_depth)
        gh,lh=q['global_latent'],q['local_latent']
        error=((g-gh).square().sum()+(l-lh).square().sum())/(g.numel()+l.numel())
        if self.straight_through:
            gh=gh+(g-g.detach());lh=lh+(l-l.detach())
        if self.evaluation_mode=='global_only':lh=l
        elif self.evaluation_mode=='local_only':gh=g
        elif self.evaluation_mode!='quantized':raise ValueError(self.evaluation_mode)
        return dict(self.decode_latents(gh,lh,mask,tick_norm,times),matrix_codes=q['codes'],
                    matrix_residual_energy=q['residual_energy'],latent_mse=error)

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        if self.evaluation_mode=='original':return super().forward(price,volume,mask,tick_norm,times,sample=False)
        p=self.encoder(price,volume,mask,tick_norm,times=times,sample=False)
        f=self.fusion([(p['z_global'],p['z_local'])],p['latent_mask'],times)
        return self.forward_tensor_latents(f['global_latent'],f['local_latent'],mask,tick_norm,times)

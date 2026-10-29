"""Frozen best cross model with a common-index global/local matrix codeword."""

from torch.nn import functional as F

from model.experimental.dual_fusion import FusedOrderBookVAE
from model.layers.paired_matrix_quantizer import PairedMatrixQuantizer


class PairedMatrixOrderBookVAE(FusedOrderBookVAE):
    def __init__(self,*,matrix_window=16,matrix_levels=8,matrix_size=256,matrix_metric='l2',matrix_mode='joint',**core):
        super().__init__(fusion_variant='cross',fusion_seed=42,**core)
        self.paired_matrix_config=dict(matrix_window=matrix_window,matrix_levels=matrix_levels,
            matrix_size=matrix_size,matrix_metric=matrix_metric,matrix_mode=matrix_mode)
        self.matrix_quantizer=PairedMatrixQuantizer((matrix_window,core.get('global_dim',64)),
            (matrix_window,2*(core.get('levels',2048)//16)*core.get('local_dim',16)),
            levels=matrix_levels,size=matrix_size,metric=matrix_metric,mode=matrix_mode)
        self.requires_grad_(False)
        self.matrix_depth=matrix_levels
        self.evaluation_mode='quantized'

    def load_cross_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not key.startswith('matrix_quantizer.') for key in missing):
            raise ValueError(f'Paired matrix source mismatch: {missing}, {unexpected}')

    def decode_latents(self,g,l,mask,tick_norm,times=None):
        b,t,_,levels=mask.shape
        latent_mask=F.adaptive_max_pool1d(mask.reshape(b*t*2,1,levels).float(),levels//16).reshape(b,t,2,levels//16).bool()
        l=l.reshape(b,t,2,levels//16,-1)*latent_mask[...,None]
        geometry=self.price_decoder(g,l,mask,tick_norm,times=times)
        quantity=self.volume_decoder(g,l,geometry,mask,tick_norm,times=times)
        quantity=self.volume_refinement(g,l,quantity,geometry,mask,tick_norm,times)
        return dict(geometry,volume=quantity)

    def decode_codes(self,codes,mask,tick_norm,times=None):
        g,l=self.matrix_quantizer.decode_codes(codes)
        return self.decode_latents(g,l,mask,tick_norm,times)

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        if self.evaluation_mode=='original':
            # Preserve the exact source path, including its tensor layouts.
            return super().forward(price,volume,mask,tick_norm,times=times,sample=False)
        posterior=self.encoder(price,volume,mask,tick_norm,times=times,sample=False)
        fused=self.fusion([(posterior['z_global'],posterior['z_local'])],posterior['latent_mask'],times)
        g=fused['global_latent'];l=fused['local_latent'].flatten(2)
        extra={}
        quantized=self.matrix_quantizer(g,l,self.matrix_depth)
        gh,lh=quantized['global_latent'],quantized['local_latent']
        if self.evaluation_mode=='global_only':lh=l
        elif self.evaluation_mode=='local_only':gh=g
        elif self.evaluation_mode!='quantized':raise ValueError('Unknown paired-matrix evaluation mode')
        extra=dict(matrix_codes=quantized['codes'],matrix_residual_energy=quantized['residual_energy'])
        return dict(self.decode_latents(gh,lh,mask,tick_norm,times),**extra)

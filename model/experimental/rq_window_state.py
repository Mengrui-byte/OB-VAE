"""Quantize the existing single window vector and reuse its frozen book decoder."""

from model.autoencoder.latent_rqvae import LatentRQVAE
from model.experimental.learned_window_state import LearnedWindowStateVAE


class RQWindowStateVAE(LearnedWindowStateVAE):
    def __init__(self, *, rq_code_dim=128, rq_hidden_dim=512, rq_levels=8,
                 rq_codebook_size=256, rq_kind='ae', rq_seed=42, rq_normalization='per_dimension', **config):
        super().__init__(**config)
        self.rq_config=dict(rq_code_dim=rq_code_dim,rq_hidden_dim=rq_hidden_dim,rq_levels=rq_levels,
            rq_codebook_size=rq_codebook_size,rq_kind=rq_kind,rq_seed=rq_seed,rq_normalization=rq_normalization)
        self.latent_codec=LatentRQVAE(input_dim=self.state_codec.state_dim,code_dim=rq_code_dim,
            hidden_dim=rq_hidden_dim,levels=rq_levels,codebook_size=rq_codebook_size,kind=rq_kind,seed=rq_seed,normalization=rq_normalization)
        self.requires_grad_(False)
        self.latent_codec.requires_grad_(True)
        self.evaluation_mode='quantized'
        self.quantization_depth=rq_levels

    def load_window_initial(self,state):
        missing,unexpected=self.load_state_dict(state,strict=False)
        if unexpected or any(not key.startswith('latent_codec.') for key in missing):
            raise ValueError(f'RQ warm start mismatch: {missing}, {unexpected}')

    def encode_codes(self,price,volume,mask,tick_norm,times=None,depth=None):
        state=super().extract_market_state(price,volume,mask,tick_norm,times)
        return self.latent_codec(state,depth)['codes']

    def decode_codes(self,codes,mask,tick_norm,times=None):
        state=self.latent_codec.decode_codes(codes)
        return super().decode_state(state,mask,tick_norm,times)

    def forward(self,price,volume,mask,tick_norm,times=None,sample=None):
        state=super().extract_market_state(price,volume,mask,tick_norm,times)
        extra={}
        if self.evaluation_mode=='original':
            reconstructed=state
        elif self.evaluation_mode=='continuous':
            reconstructed=self.latent_codec.continuous(state)
        elif self.evaluation_mode=='quantized':
            rq=self.latent_codec(state,self.quantization_depth)
            reconstructed=rq['reconstruction']
            extra=dict(rq_codes=rq['codes'],rq_features=rq['features'])
        else:
            raise ValueError('Unknown RQ evaluation mode')
        out=super().decode_state(reconstructed,mask,tick_norm,times)
        return dict(out,source_market_state=state,**extra)

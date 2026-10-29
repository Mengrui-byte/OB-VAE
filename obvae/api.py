"""Trusted-checkpoint loading for the audited baseline and its fixed teacher."""
from pathlib import Path
import sys
import torch
from model.autoencoder.direct_readout_vae import DirectReadoutOrderBookVAE
from model.experimental.full_window_matrix import FullWindowMatrixOrderBookVAE
from model.layers.branch_precision import apply_branch_precision_
from train_dual.util.data import DualNormalizer

CODE_ROOT = Path(__file__).resolve().parents[1]
ROOT = CODE_ROOT if (CODE_ROOT / 'weights/manifest.json').exists() else Path(sys.prefix) / 'share/ob-vae'
DEFAULT_CHECKPOINT = ROOT / 'weights/gelu512.pt'

def load_checkpoint(path=DEFAULT_CHECKPOINT, device='cpu', *, teacher=False):
    saved=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
    if not teacher:admit_neural_payload(saved)
    if teacher:
        model=FullWindowMatrixOrderBookVAE(**saved['window_matrix_config'],**saved['model_config'])
    else:
        for key in ['direct_readout_config','regional_phase_config','unified_vae_config','branch_precision_config']:
            if key not in saved:raise ValueError('Missing baseline checkpoint field: '+key)
        model=DirectReadoutOrderBookVAE(**saved['direct_readout_config'],**saved['regional_phase_config'],**saved['unified_vae_config'])
    model.load_state_dict(saved['model'],strict=True)
    if not teacher:apply_branch_precision_(model,**saved['branch_precision_config'])
    model.to(device).eval()
    normalization=saved['config']['normalization']
    norm=DualNormalizer(**{k:normalization[k] for k in ['tick_size','price_scale']})
    return model,norm,saved


def reconstruct(model, normalizer, raw, device='cpu'):
    if raw['data'].shape[1:] != (256,2,2048,2):
        raise ValueError('Production baseline requires [B,256,2,2048,2]')
    batch=normalizer.prepare(raw,device)
    model.eval()
    with torch.inference_mode():
        out=model(*[batch[k] for k in ['price','volume','mask','tick_norm','times']],sample=False)
    return dict(price=normalizer.inverse_price(out['price_norm'],batch),quantity=out['volume'],z=out['z'],mask=batch['mask'])


def admit_neural_payload(payload):
    """Validate only this migrated production family; not a quality verdict."""
    if payload.get('format_version')!=1 or payload.get('unified_vae_config',{}).get('levels')!=2048:
        raise ValueError('Need full 2048-level format1 baseline')
    if any(k in payload for k in ['paired_logit_config','log_quantity_input_config','spectral_quantity_config','window_matrix_config','sparse_address_config']):
        raise ValueError('Unaudited model combination')
    config=payload.get('direct_readout_config',{})
    if config.get('readout_activation') not in ['identity','gelu'] or config.get('readout_hidden')!=2048 or config.get('readout_seed')!=42:
        raise ValueError('Unexpected direct readout configuration')
    regional=payload.get('regional_phase_config',{})
    if regional.get('layout')!='frame1' or regional.get('use_shape') or regional.get('use_mass'):
        raise ValueError('Unexpected regional baseline configuration')
    if payload.get('branch_precision_config')!=dict(encoder_fp32=True,decoders_fp32=True):
        raise ValueError('Need fullFP32 audited baseline')
    return 'direct_readout'

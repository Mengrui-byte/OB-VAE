"""Parameter-preserving whole-encoder x post-latent-decoder precision policy."""
import torch
from torch import nn
from model.encoders.frame_aligned import FrameAlignedEncoder
from model.encoders.precision_frame import PrecisionFrameEncoder
from model.decoders.unified_book import PriceDecoder
from model.decoders.local_hint import LocalHintVolumeDecoder
from model.decoders.precision_book import PrecisionPriceDecoder,PrecisionHintVolumeDecoder,PrecisionHintAdapter
from model.layers.sensitive_precision import sensitive_precision_policy


def apply_branch_precision_(model,*,encoder_fp32=False,decoders_fp32=False):
    if type(encoder_fp32) is not bool or type(decoders_fp32) is not bool:
        raise ValueError('Whole-branch precision switches must be booleans')
    if any(sensitive_precision_policy(model).values()):
        raise ValueError('Mixing layer and branch policies requires a separate audit')
    if getattr(model,'use_shape',False) or getattr(model,'use_mass',False):
        raise ValueError('Active external phase readouts need a separate precision audit')
    modules=[(model.encoder,FrameAlignedEncoder,PrecisionFrameEncoder,encoder_fp32),
             (model.price_decoder,PriceDecoder,PrecisionPriceDecoder,decoders_fp32),
             (model.volume_decoder,LocalHintVolumeDecoder,PrecisionHintVolumeDecoder,decoders_fp32),
             (model.hint_adapter,nn.Linear,PrecisionHintAdapter,decoders_fp32)]
    # Validate every module before mutating any class or policy.
    for module,original,replacement,enabled in modules:
        if type(module) not in [original,replacement]:raise ValueError('Unsupported original module family for branch precision')
        if enabled and any(p.dtype!=torch.float32 for p in module.parameters()):raise ValueError('FP32 master parameters required')
    for module,original,replacement,enabled in modules:
        module.__class__=replacement;module.force_float32=enabled
    return model


def branch_precision_policy(model):
    encoder=bool(getattr(model.encoder,'force_float32',False))
    decoders=[bool(getattr(module,'force_float32',False)) for module in [model.price_decoder,model.volume_decoder,model.hint_adapter]]
    if len(set(decoders))!=1:raise ValueError('Post-latent branches must share one declared precision flag')
    return dict(encoder_fp32=encoder,decoders_fp32=decoders[0])

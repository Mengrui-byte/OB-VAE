"""Parameter-preserving precision switches for two existing Conv1d layers.

The policy must be stored in checkpoint metadata: it is not a tensor parameter
or a latent, and it does not change the state_dict layout.
"""
import torch
from torch import nn


class SensitiveConv1d(nn.Conv1d):
    force_float32 = False

    def forward(self, inputs):
        if not self.force_float32:
            return super().forward(inputs)
        if self.weight.dtype != torch.float32:
            raise ValueError('Sensitive FP32 convolution requires FP32 parameters')
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return super().forward(inputs.float())


def apply_sensitive_precision_(model, *, encoder_input_fp32=False, quantity_head_fp32=False):
    """Change arithmetic without replacing any parameter or consuming RNG.

    Conv1d is a Python nn.Module; changing its class to this subclass preserves
    the module object, hooks, registered Parameter objects and optimizer links.
    The disabled path calls the original Conv1d.forward unchanged.
    """
    for flag in [encoder_input_fp32, quantity_head_fp32]:
        if type(flag) is not bool:
            raise ValueError('Precision switches must be explicit booleans')
    layers=[model.encoder.input_proj,model.volume_decoder.quantity]
    if any(type(layer) not in [nn.Conv1d,SensitiveConv1d] for layer in layers):
        raise ValueError('Precision policy expects the original Conv1d layers')
    for layer,flag in zip(layers,[encoder_input_fp32,quantity_head_fp32]):
        layer.__class__=SensitiveConv1d
        layer.force_float32=flag
    return model


def sensitive_precision_policy(model):
    return dict(encoder_input_fp32=bool(getattr(model.encoder.input_proj,'force_float32',False)),
                quantity_head_fp32=bool(getattr(model.volume_decoder.quantity,'force_float32',False)))

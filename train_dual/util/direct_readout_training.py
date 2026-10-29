"""Preserve original three Adam groups when appending a direct quantity head."""
from copy import deepcopy

import torch


def restore_direct_readout_adam(model, saved, original_names):
    named = dict(model.named_parameters())
    original = [n for n in named if not n.startswith(('hint_adapter.', 'local_readout.', 'direct_quantity_readout.'))]
    hint = [n for n in named if n.startswith('hint_adapter.')]
    regional = [n for n in named if n.startswith('local_readout.')]
    added = [n for n in named if n.startswith('direct_quantity_readout.')]
    if original != list(original_names) or hint != ['hint_adapter.weight', 'hint_adapter.bias']:
        raise ValueError('Original source/hint parameter mapping changed')
    if regional != ['local_readout.' + n for n in ['shape_weight', 'shape_bias', 'mass_weight', 'mass_bias']]:
        raise ValueError('Expected four historical regional tensors')
    if added != ['direct_quantity_readout.' + n for n in [
            'input_projection.weight', 'input_projection.bias', 'output_projection.weight', 'output_projection.bias']]:
        raise ValueError('Expected four direct-head tensors')
    names = [original, hint, regional, added]
    groups = saved['param_groups']
    if len(groups) not in [3, 4]:
        raise ValueError('Need the three source Adam groups or four resumed groups')
    for group, group_names in zip(groups, names):
        if len(group['params']) != len(group_names): raise ValueError('Adam group size changed')
        for index, name in zip(group['params'], group_names):
            for key in ['exp_avg', 'exp_avg_sq', 'max_exp_avg_sq']:
                value = saved['state'].get(index, {}).get(key)
                if value is not None and value.shape != named[name].shape:
                    raise ValueError('Adam moment shape changed: ' + name)
    optimizer = torch.optim.AdamW([dict(params=[named[n] for n in group]) for group in names[:len(groups)]])
    optimizer.load_state_dict(deepcopy(saved))
    if len(groups) == 3:
        settings = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
        optimizer.add_param_group(dict(settings, params=[named[n] for n in added]))
    return optimizer, names

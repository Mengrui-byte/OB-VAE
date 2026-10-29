"""Reconstruct a source checkpoint's actual objective for controlled continuation."""
from model.losses.quantity_strata import StratifiedQuantityBookLoss
from model.losses.mid_level import MidLevelBookLoss
from model.losses.local_detail import LocalDetailBookLoss
from model.losses.log_relative_detail import RelativeDetailBookLoss


def inherited_book_loss(payload,plan,lane,calibration,quantity_calibration,detail_calibration=None):
    if 'regional_phase_config' not in payload:
        raise ValueError('This continuation only audits the existing regional model family')
    if 'relative_detail_loss_config' in payload:
        config=payload['relative_detail_loss_config']
        required={'relative_enabled','detail_enabled','relative_weight','detail_weight','epsilon_BTC','radii','detail_RMS'}
        if not isinstance(config,dict) or not required<=set(config):
            raise ValueError('Repeated relative/detail inheritance requires complete audited settings')
        for flag,weight in [('relative_enabled','relative_weight'),('detail_enabled','detail_weight')]:
            if type(config[flag]) is not bool or config[weight] is None or (not config[flag] and config[weight]!=0):
                raise ValueError('Inherited relative/detail flags and coefficients differ')
        original={key:value for key,value in payload.items() if key!='relative_detail_loss_config'}
        base=inherited_book_loss(original,plan,lane,calibration,quantity_calibration,detail_calibration)
        return RelativeDetailBookLoss(base,detail_scales=config['detail_RMS'],relative_weight=config['relative_weight'],detail_weight=config['detail_weight'],epsilon=config['epsilon_BTC'],radii=tuple(config['radii']),relative_count_power=config.get('relative_count_power',0.))
    spec=plan['lanes'][lane]
    strata=payload.get('stratified_quantity_loss_config',dict(weight=0.))
    scales=strata.get('scales_BTC',quantity_calibration['RMS_scales_BTC'])
    if list(scales)!=list(quantity_calibration['RMS_scales_BTC']):
        raise ValueError('Source loss scales differ from the inherited calibration capsule')
    kw=dict(bin_scales=scales,stratified_weight=strata['weight'],mode='shape_mass',
        use_hint=spec['hint'],hint_weight=calibration['hint_weight'],hint_mean=calibration['channel_mean'],hint_scale=calibration['channel_scale'],
        use_tail=spec['use_tail'],tail_weight=calibration['tail_weight'],protect_mid=spec['protect_mid'],mid_weight=calibration['mid_weight'],
        mid_delta_weight=spec['mid_delta_weight'],add_wasserstein=False,add_multiscale=False,
        **plan['source_FP_rule'],**plan['objective'])
    mid=payload.get('mid_lowpass_loss_config');detail=payload.get('local_detail_loss_config')
    if mid is not None and detail is not None:raise ValueError('Combined inherited auxiliaries require a separate audit')
    q0=plan['normalization']['quantity_scale']
    if mid is not None:return MidLevelBookLoss(q0,mid_lowpass_weight=mid['weight'],mid_chunks=tuple(mid['chunks']),**kw)
    if detail is not None:
        scales=detail.get('RMS_scales')
        if scales is None:
            if detail_calibration is None:raise ValueError('Missing inherited detail scales')
            scales=detail_calibration['RMS_scales']
        return LocalDetailBookLoss(q0,detail_scales=scales,detail_weight=detail['weight'],detail_radii=tuple(detail['radii']),**kw)
    return StratifiedQuantityBookLoss(q0,**kw)

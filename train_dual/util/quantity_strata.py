"""Streaming quantity-scale diagnostics alongside the original evaluator."""
import math
import numpy as np

from model.losses.quantity_strata import QUANTITY_LOWER_BOUNDS


QUANTITY_STRATA_SCHEMA_VERSION = 2


def quantity_bin_mask_numpy(values, index):
    """Membership follows training's float32 target comparisons.

    Metrics still accumulate values/errors in float64. Only bin assignment uses
    the stored float32 coordinates, avoiding0.01 moving to a lower bin when its
    float32 representation is widened to float64. This is not data smoothing.
    """
    if index not in range(6):raise ValueError('Invalid quantity bin')
    stored=np.asarray(values,dtype=np.float32)
    selected=stored>=np.float32(QUANTITY_LOWER_BOUNDS[index])
    if index<5:selected &= stored<np.float32(QUANTITY_LOWER_BOUNDS[index+1])
    return selected


class QuantityStrataMetrics:
    def __init__(self):
        names = ['count','positive_count','absolute_error','squared_error','target_squared',
                 'log_squared_error','target_sum','prediction_sum','within20','within2']
        self.accumulators = [{key:0 for key in names} for _ in range(6)]

    def update(self, target, prediction, mask):
        q, p, valid = np.asarray(target,dtype=np.float64), np.asarray(prediction,dtype=np.float64), np.asarray(mask,dtype=bool)
        if q.shape != p.shape or q.shape != valid.shape:
            raise ValueError('Mismatched stratified evaluation shapes')
        if not np.isfinite(q[valid]).all() or not np.isfinite(p[valid]).all() or (q[valid]<0).any() or (p[valid]<0).any():
            raise ValueError('Expected finite nonnegative valid quantities')
        for i, lower in enumerate(QUANTITY_LOWER_BOUNDS):
            selected = valid & quantity_bin_mask_numpy(q, i)
            truth, pred = q[selected], p[selected]
            if not truth.size: continue
            error, positive = pred-truth, truth>0
            a = self.accumulators[i]
            a['count'] += int(truth.size); a['positive_count'] += int(positive.sum())
            a['absolute_error'] += float(np.abs(error).sum()); a['squared_error'] += float((error**2).sum())
            a['target_squared'] += float((truth**2).sum())
            a['log_squared_error'] += float(((np.log1p(pred)-np.log1p(truth))**2).sum())
            a['target_sum'] += float(truth.sum()); a['prediction_sum'] += float(pred.sum())
            a['within20'] += int((np.abs(error[positive]) <= .2*truth[positive]).sum())
            a['within2'] += int(((pred[positive]>=.5*truth[positive])&(pred[positive]<=2*truth[positive])).sum())

    def finish(self):
        rows = []
        for i,a in enumerate(self.accumulators):
            n, positive = a['count'], a['positive_count']
            rmse = math.sqrt(a['squared_error']/n) if n else None
            target_rms = math.sqrt(a['target_squared']/n) if n else None
            rows.append(dict(lower_BTC=QUANTITY_LOWER_BOUNDS[i], upper_BTC=QUANTITY_LOWER_BOUNDS[i+1] if i<5 else None,
                count=n, MAE_BTC=a['absolute_error']/n if n else None, RMSE_BTC=rmse,
                target_RMS_BTC=target_rms, normalized_RMSE=rmse/target_rms if target_rms else None,
                log1p_RMSE=math.sqrt(a['log_squared_error']/n) if n else None,
                target_mean_BTC=a['target_sum']/n if n else None, prediction_mean_BTC=a['prediction_sum']/n if n else None,
                quantity_mass_ratio=a['prediction_sum']/a['target_sum'] if a['target_sum'] else None,
                within20percent=a['within20']/positive if positive else None,
                within_factor2=a['within2']/positive if positive else None))
        return rows


class _CaptureStrata:
    def __init__(self, model):
        self.model, self.metrics = model, QuantityStrataMetrics()

    def eval(self):
        self.model.eval(); return self

    def train(self):
        self.model.train(); return self

    def __call__(self, *args, **kwargs):
        output = self.model(*args, **kwargs)
        self.metrics.update(args[1].detach().cpu().numpy(), output['volume'].detach().float().cpu().numpy(), args[2].detach().cpu().numpy())
        return output


def evaluate_with_quantity_strata(model, normalizer, dataset, windows, device):
    from train_dual.experiments.long_worker import evaluate
    adapter = _CaptureStrata(model)
    result = evaluate(adapter, normalizer, dataset, windows, device)
    result['quantity_strata'] = adapter.metrics.finish()
    result['quantity_strata_schema_version'] = QUANTITY_STRATA_SCHEMA_VERSION
    result['quantity_strata_membership'] = 'float32 target boundaries, identical to Torch training; error accumulation float64'
    return result

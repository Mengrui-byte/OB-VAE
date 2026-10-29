"""Spatial slope and within-window temporal reconstruction statistics."""

import math
import numpy as np


class PairStatistics:
    def __init__(self):
        self.values = np.zeros(8, dtype=np.float64)

    def update(self, target, prediction, mask):
        t, p = np.asarray(target)[mask].astype(np.float64), np.asarray(prediction)[mask].astype(np.float64)
        self.values += [len(t), t.sum(), p.sum(), np.square(t).sum(), np.square(p).sum(),
                        (t*p).sum(), np.square(p-t).sum(), np.abs(p-t).sum()]

    def finish(self):
        n, st, sp, st2, sp2, cross, se, ae = self.values
        if not n:
            return None
        vt, vp = max(0.,st2-st*st/n), max(0.,sp2-sp*sp/n)
        return dict(count=int(n), rmse=math.sqrt(se/n), mae=ae/n,
                    target_rms=math.sqrt(st2/n), prediction_rms=math.sqrt(sp2/n),
                    correlation=(cross-st*sp/n)/math.sqrt(vt*vp) if vt>1e-16 and vp>1e-16 else None,
                    centered_amplitude_ratio=math.sqrt(vp/vt) if vt>1e-16 else None,
                    skill_vs_zero=1-se/st2 if st2>1e-16 else None)


class DynamicsMetrics:
    def __init__(self, tick_size=0.01):
        self.tick_size = tick_size
        self.stats = {}
        self.direction = {}
        self.windows = 0

    def add(self, name, truth, prediction, valid):
        self.stats.setdefault(name,PairStatistics()).update(truth,prediction,valid)

    def update(self, truth, prediction, mask):
        """Arrays [T,2,L,2] in raw units; comparisons never cross windows."""
        self.windows += 1
        p, ph = truth[...,0],prediction[...,0]
        q, qh = truth[...,1],prediction[...,1]
        self.add("price_usdt",p,ph,mask)
        self.add("quantity_btc",q,qh,mask)
        sign=np.array([-1.,1.])[None,:,None]
        for lag in (1,16,64,256):
            if lag>=p.shape[-1]:
                continue
            valid=mask[...,lag:]&mask[...,:-lag]
            target=(p[...,lag:]-p[...,:-lag])*sign/lag
            estimated=(ph[...,lag:]-ph[...,:-lag])*sign/lag
            self.add(f"spatial_slope_{lag}_levels_usdt_per_level",target,estimated,valid)
            count=valid.sum(-1,keepdims=True).clip(1)
            target_mean=np.where(valid,target,0).sum(-1,keepdims=True)/count
            estimated_mean=np.where(valid,estimated,0).sum(-1,keepdims=True)/count
            self.add(f"spatial_slope_{lag}_within_frame_variation",target-target_mean,
                     estimated-estimated_mean,valid)
        mid=p[...,0].mean(1);midh=ph[...,0].mean(1)
        for lag in (1,4,8):
            if lag>=len(p):
                continue
            valid=mask[lag:]&mask[:-lag]
            dt=(p[lag:]-p[:-lag])/lag
            dh=(ph[lag:]-ph[:-lag])/lag
            self.add(f"temporal_price_{lag}_minutes_usdt_per_minute",dt,dh,valid)
            dm=(mid[lag:]-mid[:-lag])/lag
            dmh=(midh[lag:]-midh[:-lag])/lag
            self.add(f"temporal_mid_{lag}_minutes_usdt_per_minute",dm,dmh,np.ones(len(dm),bool))
            self.add(f"temporal_shape_{lag}_minutes_after_mid_removal",dt-dm[:,None,None],
                     dh-dmh[:,None,None],valid)
            logq=np.log1p(q);logqh=np.log1p(qh)
            self.add(f"temporal_log_quantity_{lag}_minutes",(logq[lag:]-logq[:-lag])/lag,
                     (logqh[lag:]-logqh[:-lag])/lag,valid)
            active=np.abs(dm*lag)>=self.tick_size
            hits=((np.sign(dm)==np.sign(dmh))&active).sum()
            old=self.direction.get(lag,(0,0))
            self.direction[lag]=(old[0]+int(hits),old[1]+int(active.sum()))
        # Static, rigidly translated initial profile is a separate benchmark for
        # spatial-shape evolution; it uses each observed frame's midpoint.
        rigid=p[:1]+(mid-mid[0])[:,None,None]
        self.add("rigid_initial_book_price_baseline",p,rigid,mask&mask[:1])

    def finish(self):
        return dict(windows=self.windows,metrics={name:stat.finish() for name,stat in self.stats.items()},
                    mid_direction={str(lag):dict(correct=hit,nonzero_changes=n,accuracy=hit/n if n else None)
                                   for lag,(hit,n) in self.direction.items()})

"""Build versioned model-family results documentation from existing local evidence."""
import argparse,ast,collections,csv,datetime,hashlib,html,json,os,re,shutil
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
SOURCE=Path(os.environ.get('OBVAE_SOURCE','/home/mengrui/e2e_strategy/OBreconstuction')).resolve()
DOC=REPO/'docs/branches';AUDIT=REPO/'recons/branch_atlas_20261029'
LINEAGE=json.loads((REPO/'docs/model_lineage.json').read_text())
NODES=LINEAGE['nodes'];NOW=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
MAIN='recons/recall80_goal_20261027_235344'

def read(p):
    try:return json.loads(p.read_text())
    except (ValueError,OSError):return {}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rel(p):return str(p.relative_to(SOURCE))
def walk(g):
    for base,dirs,files in os.walk(g):
        dirs[:]=[d for d in dirs if d not in ['source_snapshot','data','public','viewer','logs','seen','unseen','__pycache__','figures','verification']]
        for f in files:yield Path(base)/f

def normalized(a):
    m=a.get('metrics',{})
    if not isinstance(m,dict) or not isinstance(m.get('quantity_btc'),dict):return None
    q=m['quantity_btc'].get('rmse');p=m.get('price_usdt',{}).get('rmse')
    if q is None or p is None:return None
    t=a.get('thresholds',{}).get('100',a.get('threshold100',{}));rec=t.get('recall');prec=t.get('precision')
    f1=2*t['tp']/(2*t['tp']+t.get('fp',0)+t.get('fn',0)) if t.get('tp') is not None and 2*t['tp']+t.get('fp',0)+t.get('fn',0)>0 else None
    return dict(price=p,quantity=q,recall=rec,precision=prec,F1=f1,dynamics=m.get('temporal_log_quantity_1_minutes',{}).get('correlation'),
                windows=a.get('windows'),count=m['quantity_btc'].get('count'),target_rms=m['quantity_btc'].get('target_rms'),indices=a.get('window_indices'),threshold100=t)

def profiles_for(p,key,step):
    parent=p.parent;name=p.name;paths=[]
    if name in ['results.json','comparison.json'] and re.fullmatch(r'k(4|16)_depth8',key):paths.append(parent/key.split('_')[0]/'profiles.npz')
    if name in ['results.json','comparison.json']:
        paths.extend([parent/key/'profiles.npz',parent/key/'test_profiles.npz'])
    if name=='depth'+str(step)+'_evaluation.json':paths.append(parent/f'depth{step}/profiles.npz')
    if step is not None:paths.append(parent/f'step{step}_profiles.npz')
    if (parent/'final_evaluation.json').exists() and (parent/'profiles.npz').exists():
        aa=read(p).get('result',read(p));bb=read(parent/'final_evaluation.json').get('result',{})
        if normalized(aa)==normalized(bb):paths.append(parent/'profiles.npz')
    if 'dual_12h_' in str(p) and 'volume_s42' in str(p):paths.append(SOURCE/'checkpoints/dual_12h_20261024_004600/reconstruction_20261024_111849/profiles.npz')
    if name.startswith('test_') or name=='final_test_evaluation.json':paths.append(parent/'test_profiles.npz')
    if 'final_evaluation' in name:
        paths.extend([parent/'profiles.npz',parent/f'development_step{step}_128windows_profiles.npz'])
    if name=='latest_evaluation.json':paths.extend([parent/'profiles.npz'])
    if name=='frozen_evaluation.json':paths.append(parent/'frozen_profiles.npz')
    if '/strict_fp32/' in str(p):paths.append(Path(str(parent).replace('/strict_fp32',''))/'profiles.npz')
    return next((rel(x) for x in paths if x.is_file()),None)

def weight_for(p,key,step):
    parent=p.parent;paths=[]
    if 'sparse_address_diagnostic' in str(p) and (parent/'model.pt').is_file():return rel(parent/'model.pt')
    if p.name=='results.json' and re.fullmatch(r'k(4|16)_depth8',key):paths.append(parent/key.split('_')[0]/'final.pt')
    if 'sparse_address_diagnostic' in str(p):paths.append(parent/'model.pt')
    if p.name=='metrics.jsonl':paths.extend([parent/'best_quantity.pt',parent/'best_val.pt',parent/'last.pt'])
    elif p.name in ['results.json','comparison.json']:paths.extend([parent/key/'final.pt'])
    elif p.name.startswith('depth'):paths.append(parent/'final.pt')
    else:
        if step is not None:paths.append(parent/f'step{step}.pt')
        if p.name=='latest_evaluation.json':paths.extend([parent/f'evaluation_step_{step:07d}.pt'] if step is not None else []);paths.extend([parent/'latest.pt',parent/'last.pt'])
        is_step=bool(re.fullmatch(r'step\d+_(evaluation|development)\.json',p.name))
        if is_step:
            exact=parent/f'step{step}.pt'
            if exact.is_file():return rel(exact)
            current=normalized(read(p).get('result',read(p)))
            final=normalized(read(parent/'final_evaluation.json').get('result',{}))
            if current!=final:return None
        paths.extend([parent/'final.pt',parent/'latest.pt',parent/'last.pt'])
        if '/strict_fp32/' in str(p):paths.extend([Path(str(parent).replace('/strict_fp32',''))/'final.pt'])
    return next((rel(x) for x in paths if x.is_file()),None)

EXCLUDED=[]
def scan(node):
    paths=list(node['experiments'])
    extensions={
        'gaussian8192':['recons/loss_control_20261026_120833','recons/distribution_control_20261026_132026','recons/loss_allocation_20261026_150351','recons/data_scale_20261026_162127','recons/probability_shape_20261026_181628'],
        'hint':[MAIN+'/round_001_mid_tail',MAIN+'/round_002_fp_rule',MAIN+'/round_003_mid_delta'],
        'local_phase':[MAIN+'/round_007_quantity_factorial',MAIN+'/round_008_auxiliary',MAIN+'/round_009_relative_detail',MAIN+'/round_010_branch_precision',MAIN+'/round_011_relative_aggregation',MAIN+'/relative_aggregation_early_pilot_20261028'],
        'direct':[MAIN+'/relative_strength_direct_pilot_20261029'],
    }
    paths.extend(p for p in extensions.get(node['id'],[]) if (SOURCE/p).exists())
    if node['id']=='matrix16':paths.append('checkpoints/frame_matrix_e2e_resume_20261025_234027')
    if node['id']=='paired_readout':
        paths.extend(MAIN+'/'+x for x in ['paired_response_space_v3','paired_logit_readout','price_safe_logit_readout','price_safe_readout'] if (SOURCE/MAIN/x).exists())
    candidates=[]
    for root in paths:
        g=SOURCE/root
        for p in walk(g):
            sub=p.relative_to(g).parts
            if p.suffix not in ['.json','.jsonl']:continue
            if 'pilot' in sub:
                EXCLUDED.append(dict(path=rel(p),reason='training-only synthetic pilot, never Q1 test'))
                continue
            if any(w in str(p.relative_to(g)) for w in ['seen_evaluation','_seen.json','_unseen.json','train_step','small_fit','train_cache','/moving/','/real/']):continue
            if node['id'] in ['legacy','local_queries','peak_geometry'] and p.name=='metrics.jsonl':
                cfg=read(p.parent/'run_config.json').get('args',{});period=[cfg.get('val_start'),cfg.get('val_end')]
                for line_number,line in enumerate(p.read_text().splitlines(),1):
                    d=json.loads(line);a=d.get('validation',{});q=a.get('quantity_rmse');price=a.get('price_rmse_usdt')
                    if q is None or price is None:continue
                    candidates.append(dict(family=node['id'],cohort=root,lane=p.parent.name,path=rel(p),selector=f'line:{line_number}:validation',step=d.get('optimizer_steps'),epoch=d.get('epoch'),
                        price=price,quantity=q,recall=None,precision=None,F1=None,dynamics=None,windows=a.get('samples'),count=a.get('valid_levels'),target_rms=None,indices=None,threshold100={},
                        window_minutes=cfg.get('window_size',256),period=period,kind='test_history',profile=None,weight=weight_for(p,'',None),raw=d))
                continue
            if p.suffix!='.json':continue
            name=p.name
            valid=(name in ['final_evaluation.json','latest_evaluation.json','evaluation.json','final_test_evaluation.json','results.json'] or bool(re.fullmatch(r'(step\d+_(evaluation|development)|depth\d+_evaluation|test_step\d+_\d+windows)\.json',name)))
            if not valid:continue
            if name=='results.json' and node['id']!='vq':continue
            if any(x in sub for x in ['reference','frozen_source','uncompressed','original_source','common']):
                if node['id'] in ['continuous','matrix16','joint4096','vq']:continue
            if node['id']=='region_task' or node['id']=='peak_region':
                if root.endswith('region_task_20261027_145807') and 'development' not in sub:continue
            if node['id']=='dual' and '/runs/' in str(p) and '_s42/' not in str(p):continue
            if node['id']=='continuous' and p.parent.name in ['fixed']:continue
            d=read(p);items=[('',d.get('result',d))]
            if name=='results.json':items=[(k,v) for k,v in d.items() if isinstance(v,dict) and not any(x in k for x in ['original','prefix','previous'])]
            for key,a in items:
                m=normalized(a)
                if not m:continue
                if a.get('split') in ['train','seen','training']:continue
                parsed=re.search(r'(?:step|depth)(\d+)',name);step=d.get('step',a.get('step',int(parsed[1]) if parsed else None))
                if step==0 and node['id']!='sparse':continue
                lane=key or p.parent.name
                if node['id']=='vq' and lane in ['continuous','original','reference']:continue
                window=16 if node['id'] in ['dual','cross','continuous','vq','matrix16','joint4096'] else 256
                if node['id']=='gaussian8192' and (m.get('windows')!=16 or abs((m.get('target_rms') or 0)-2.2170200567880882)>1e-7):
                    EXCLUDED.append(dict(path=rel(p),reason='Gaussian8192 candidate is not the canonical16 Q1 windows; not ranked as that test protocol'))
                    continue
                c=dict(family=node['id'],cohort=root,lane=lane,path=rel(p),selector=key or ('result' if 'result' in d else ''),step=step,epoch=None,**m,
                       window_minutes=window,period=['2025-01-01','2025-04-01'],kind='diagnostic' if node['id']=='sparse' else 'test',
                       profile=profiles_for(p,key,step),weight=weight_for(p,key,step),raw=d if not key else a)
                candidates.append(c)
    # Canonical native largest-common positive endpoints per cohort. Old final and depth records remain valid endpoints.
    final=[]
    for cohort in sorted(set(c['cohort'] for c in candidates)):
        rows=[c for c in candidates if c['cohort']==cohort]
        steps=[c for c in rows if re.fullmatch(r'step\d+_(evaluation|development)\.json',Path(c['path']).name)]
        if steps:
            sets=[{c['step'] for c in steps if c['lane']==lane} for lane in sorted({c['lane'] for c in steps})];common=set.intersection(*sets)
            if common:
                endpoint=max(common)
                for c in rows:
                    c['common_endpoint_step']=endpoint
                    c['is_common_endpoint']=(c['step']==endpoint) if c['step'] is not None else Path(c['path']).name=='final_evaluation.json'
                    final.append(c)
            else:final.extend(rows)
        else:final.extend(rows)
    # Deduplicate identical aliases, but keep evaluation precision and population differences explicit.
    unique={}
    for c in final:
        signature=(c['cohort'],c['lane'],c['step'],c['epoch'],c['quantity'],c['price'],c['windows'],c['count'])
        if signature not in unique or Path(c['path']).name.startswith('final'):unique[signature]=c
    return list(unique.values())


def inventory():
    AUDIT.mkdir(parents=True,exist_ok=True)
    allrows=[]
    for node in NODES:
        rows=scan(node)
        # Protocol grouping prevents ranking January694 windows against full-Q1 or 64 against128 windows.
        for c in rows:
            c['protocol']=json.dumps([c['window_minutes'],c['period'],c['windows'],c['count'],round(c['target_rms'],7) if c['target_rms'] is not None else None],ensure_ascii=False)
        groups=collections.defaultdict(list)
        for c in rows:groups[c['protocol']].append(c)
        winners=[min(v,key=lambda x:(x['quantity'],x['price'])) for v in groups.values()]
        # Main representative uses the documented largest population, with a saved exact weight preferred; all protocol winners remain reported.
        winners.sort(key=lambda c:(-(c['windows'] or 0),c['quantity'],c['price']))
        print(node['id'],len(rows),'protocols',len(groups))
        for c in winners[:4]:print(' ',c['lane'],c['step'],c['epoch'],'q',round(c['quantity'],5),'p',round(c['price'],3),'n',c['windows'],'profile',c['profile'],'weight',c['weight'])
        allrows.extend(rows)
    (AUDIT/'candidates.json').write_text(json.dumps(allrows,ensure_ascii=False,indent=2))
    (AUDIT/'excluded_records.json').write_text(json.dumps(EXCLUDED,ensure_ascii=False,indent=2))
    return allrows



ARCH={
'legacy': [('Observed book','B x 256 x 2 x 2048 x 2'),('Anchor / RMS scaling','window-anchor coordinates'),('Spatial CNN + temporal stages','spatial pooling at each compressed time'),('Attention pool / latent','B x 16 x 64'),('Shared latent decoder','price and quantity reconstruction')],
'local_queries':[('Observed book','256 minutes / both sides'),('Spatial-temporal encoder','local region features'),('Local queries','8 queries x 8 features'),('Compressed state','B x 16 x 64'),('Main decoder + V5 patch residual','latent-expanded features only'),('Reconstructed book','price / quantity')],
'peak_geometry':[('Observed book','256 x 2 x 2048 price/quantity'),('Background + native peak descriptors','4 local peak slots; rank/price/quantity'),('Alternating spatial / temporal encoder','price-aligned past feature cells'),('Local-region bottleneck','B x 16 x 64'),('Reverse temporal/spatial + patch decoder','32 peak-latent features; no input bypass'),('Reconstructed book','price / quantity')],
'dual':[('Observed book','16 minutes / 2048 levels per side'),('Shared encoder','price + log-gap + quantity + differences / LoG'),('Two Gaussian fields','G: 16x64; L: 16x2x128x16'),('Price decoder','ordered mid / spread / gap'),('Quantity decoder + volume refinement','quantity conditioned on predicted price')],
'cross':[('Observed book','16 minutes'),('Shared encoder or price/quantity towers','control / cross / tower_mean / tower_cross'),('G/L fusion memory','time / side / depth Cross-Attention'),('Fused Global / Local slots','ordered local information retained'),('Dual decoders','price + quantity / amplitude head')],
'continuous':[('Observed book + frozen Cross','16 minutes'),('Global / Local fields','original continuous coordinates'),('PCA / learned state compressor','selected checkpoint state_dim below'),('Single continuous state','selected dimensions, not discrete codes'),('Inverse PCA / learned inverse','restore Global / Local'),('Original dual decoders','same reconstruction task')],
'vq':[('Observed book + frozen Cross','16 minutes'),('Global continuous + Local matrix','Local quantization control'),('Residual quantization','R=8 / K=16 for selected k16_depth8'),('Quantized Local','sum of eight selected codeword tensors'),('Global + reconstructed Local decoders','no concatenation of residual codes')],
'matrix16':[('Observed book / Cross encoder','16 minutes'),('Per-frame packed G/L','16 x 4160'),('Frame compression','4160 -> 1024; linear / residual MLP'),('Window matrix','B x 16 x 1024'),('Frame inverse','1024 -> 4160 -> G/L'),('Price / quantity decoders','selected E2E state below')],
'joint4096':[('Observed book / frozen Cross','16 minutes'),('Flatten original Global / Local','66560 scalar coordinates'),('Fixed joint PCA','66560 -> 4096'),('One continuous state','B x 4096'),('Inverse PCA','restore Global / Local'),('Adapted price / quantity decoders','original / Dice / shape-loss controls')],
'full256':[('Observed full window','B x 256 x 2 x 2048 x 2'),('Shared encoder + Cross','whole-window temporal context'),('Uncompressed G/L reference','1064960 content scalars'),('Optional channel/time codec','compressed alternatives: 16 x 1024'),('Price / quantity decoders','selected reference bypasses the codec')],
'merge':[('Full256 encoder + Cross','G / Local fields'),('Selected local-only bottleneck','B x 256 x 2 x 128 x 16'),('GlobalReadoutFromLocal','no independent Global decoder input'),('Original dual decoders','derived Global + same Local'),('Reconstructed full book','price / quantity')],
'gaussian8192':[('Observed full window','B x 256 x 2 x 2048 x 2'),('16 feature channels + spatial encoder','4 stride-2 stages; 2048 -> 128'),('Bid/Ask attention + frame reduction','4-frame packing / temporal attention'),('Single Gaussian posterior','mu / logvar / Z: B x 64 x 128'),('Two independent temporal/spatial decoders','nearest upsample x4 + predicted geometry'),('Ordered price + nonnegative quantity','same Z / no Encoder bypass')],
'phase':[('Full256 encoder','original 8192 posterior'),('Single shared Z','B x 64 x 128'),('Temporal expansion','256 frames x 256 features'),('Regional spatial seed','128 regions / side'),('Original upsample + offset residual','Conv1d 256->16; independent offsets'),('Softplus quantity + price decoder','no extra per-window latent')],
'peak_region':[('Observed full256','one Gaussian family'),('Spatial-temporal encoder','baseline / regional / peak controls'),('Single Z','B x 64 x 128'),('Selected native decoder','original_warm / query / peak / region variants'),('Price + quantity','selected configuration determines active heads')],
'frame1':[('Observed full256','spatial features / Bid-Ask attention'),('Frame-aligned transformer','256 tokens / FFN1536'),('Per-frame posterior coordinates','256 x 32 -> reshape 64 x 128'),('Single shared Z','8192 scalars'),('Frame temporal expansion','reshape 256 x 32; MLP -> 256 features'),('Price + quantity spatial decoders','Huber / pre-Softplus logit controls')],
'hint':[('Observed full256','same raw window for student and teacher'),('frame1 encoder -> single Z','B x 64 x 128'),('Two original decoders','post-Z quantity seed: 256x2x128x16'),('Training-only hint adapter','Linear16->16; match frozen teacher Local'),('Book losses + optional quantity augmentation','teacher not required for inference')],
'local_phase':[('Full256 frame1 encoder','single Gaussian 64 x 128'),('Shared Z -> quantity seed','256 x 2 x 128 x 16'),('LocalPhase / RegionalPhase heads','16-offset shape + one regional mass'),('Shape redistribution + mass scaling','on/off switches; zero-initialized residuals'),('Quantity + original price decoder','best selected switches listed below')],
'fine_encoder':[('Observed full256','price / quantity / spatial details'),('FinePhase input feature path','linear or log1p; before spatial reduction'),('Original encoder + temporal transformer','new path still passes through Gaussian'),('Single Z','B x 64 x 128'),('Original dual decoders','no Encoder-to-Decoder skip')],
'paired_readout':[('Full256 encoder','frozen source in matched pilot'),('Single Z','B x 64 x 128'),('Post-Z frame or seed features','same source price / quantity paths'),('Small additive/multiplicative readout','130-parameter paired / logit / scale controls'),('Reconstructed price / quantity','native pilot; cached fit separately labelled')],
'context':[('Full256 frame1 encoder','same 8192 Gaussian'),('Z reshaped to 256 x 32','only content input to both decoders'),('Quantity-frame queries','256-dimensional original expansion'),('Residual Cross-Attention','read local5 / local17 / global256 Z'),('Quantity spatial decoder + price decoder','zero output projection; no quality result')],
'direct':[('Full256 frame1 encoder','price/quantity observed; frozen in this pilot'),('Single shared Z','B x 64 x 128'),('Quantity frame features','B x 256 x 256'),('Direct per-level residual','Linear256->2048 -> GELU -> Linear2048->4096'),('Original logit + delta','Softplus once; price branch unchanged'),('Reconstructed full book','20691748 model parameters')],
'log_input':[('Observed full256','physical quantity / log1p contrast'),('Original16 + contrast13 channels','zero-initialized input adapter'),('frame1 encoder','same original temporal layout'),('Single Gaussian Z','B x 64 x 128'),('GELU residual + dual decoders','loss curve/strength is a separate factor')],
 'temporal_input':[('Observed full256','current and past window frames'),('Temporal quantity context','lag1/4; exact-price or rank alignment'),('29 original + 6 context channels','zero-initialized 32x6x1 input adapter'),('Single encoder / Gaussian','B x 64 x 128'),('Unchanged GELU and dual decoders','implementation / preflight only; no result')],
'fourier':[('Observed full256','standalone Fourier package'),('Log-quantity input + frame1 encoder','single Gaussian 64 x 128'),('Dual decoders + original GELU path','same shared Z; no external runtime imports'),('Spectral quantity residual','off / local / 64-bin / 1153-bin FNO'),('Quantity logit correction','complex spectral multipliers + local path'),('Ordered price / nonnegative quantity','ongoing common-checkpoint comparison')],
'sparse':[('Observed full256','diagnostic content encoder'),('Price latent + explicit sparse content','per-window quantity addresses / values'),('Fixed 8192 scalar allocation','m3072 leaves 8 price channels'),('Neural price + sparse reconstruction','engineered route; not pure Gaussian'),('Diagnostic book','do not compare as native model success')],
}


def json_small(value):
    import torch
    if torch.is_tensor(value):return dict(tensor_shape=list(value.shape),dtype=str(value.dtype))
    if isinstance(value,dict):return {str(k):json_small(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_small(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(type(value).__name__)


WEIGHT_CACHE={}

def weight_metadata(c):
    if c.get('weight') in WEIGHT_CACHE:return WEIGHT_CACHE[c['weight']]
    if not c.get('weight'):return dict(available=False,reason='No exact saved checkpoint identified; no later checkpoint substituted')
    import torch
    p=SOURCE/c['weight'];s=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
    configs={k:json_small(v) for k,v in s.items() if k.endswith('_config') or k in ['architecture','variant','fusion_variant','state_dim','residual_levels','codebook_size','step','epoch','lane']}
    states=s.get('model',s.get('model_state_dict'))
    if isinstance(states,dict):configs['state_tensor_count']=len(states);configs['state_scalars']=sum(v.numel() for v in states.values() if torch.is_tensor(v))
    check=dict(available=True,path=c['weight'],sha256=sha(p),bytes=p.stat().st_size,configs=configs)
    if c.get('epoch') is not None:
        assert s.get('epoch')==c['epoch'],(c['path'],c['epoch'],s.get('epoch'))
        q=s.get('val_metrics',{}).get('quantity_rmse')
        if q is not None:assert abs(q-c['quantity'])<1e-10
        check['epoch_and_saved_quantity_metric_match']=True
    elif c.get('step') is not None and not Path(c['path']).name.startswith('depth'):
        if s.get('step') is not None:
            assert s['step']==c['step'],(c['path'],c['step'],s['step'])
            check['saved_step_matches']=True
    WEIGHT_CACHE[c['weight']]=check
    return check


def arch_svg(node,dest):
    from xml.sax.saxutils import escape
    stages=ARCH[node['id']];W=850;H=len(stages)*100+30
    colors=['#e6f2f9','#e9f4ef','#fdf2d9']
    out=f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-label="{escape(node["name"])}"><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#41677d"/></marker></defs><rect width="100%" height="100%" fill="white"/>'
    for i,(name,shape) in enumerate(stages):
        y=15+i*100;fill=colors[i%3]
        out+=f'<rect x="35" y="{y}" width="780" height="74" rx="10" fill="{fill}" stroke="#a8c0cc"/><text x="425" y="{y+29}" text-anchor="middle" font-family="sans-serif" font-size="19" font-weight="600" fill="#183345">{escape(name)}</text><text x="425" y="{y+54}" text-anchor="middle" font-family="sans-serif" font-size="15" fill="#416278">{escape(shape)}</text>'
        if i<len(stages)-1:out+=f'<path d="M425,{y+75} L425,{y+98}" stroke="#41677d" stroke-width="2" marker-end="url(#a)"/>'
    dest.write_text(out+'</svg>')


def figure_candidates(node,rows,dest):
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    if not rows:
        for ax in axs:ax.axis('off')
        axs[0].text(.5,.6,'Implementation / preflight only',ha='center',va='center',fontsize=16)
        axs[1].text(.5,.6,'No trained reconstruction score\nNo best checkpoint claimed',ha='center',va='center',fontsize=15)
    else:
        # At most10 smallest-error records for readability; full rows retained in JSON/CSV.
        selected=sorted(rows,key=lambda c:(c['quantity'],c['price']))[:10]
        labels=[(Path(c['cohort']).name[:16]+' / '+c['lane']+' / '+str(c['epoch'] or c['step'] or 'fixed')) for c in selected]
        for ax,key,title in zip(axs,['quantity','price'],['Quantity RMSE / BTC (lower better)','Price RMSE / USDT (lower better)']):
            ax.barh(range(len(selected)),[c[key] for c in selected],color='#168375' if key=='quantity' else '#267ca3');ax.set_yticks(range(len(selected)),labels,fontsize=7);ax.invert_yaxis();ax.set_title(title);ax.grid(axis='x',alpha=.15)
    fig.suptitle(node['id']+' / recorded candidates; compare within protocol')
    fig.savefig(dest,dpi=125);plt.close(fig)


def profile_figure(c,node,directory,tag):
    import numpy as np
    import matplotlib.pyplot as plt
    if not c.get('profile'):return None
    path=SOURCE/c['profile'];a=np.load(path)
    prediction_key='prediction'
    if node['id']=='dual':prediction_key=c['lane'].split('_')[0]
    if prediction_key not in a or 'truth' not in a:return None
    truth=a['truth'];pred=a[prediction_key]
    assert truth.shape==pred.shape and truth.ndim==5 and truth.shape[2:]==(2,2048,2),(path,truth.shape,pred.shape)
    T=truth.shape[1];assert T==c['window_minutes'],(path,T,c['window_minutes'])
    ts=list(dict.fromkeys([0,T//2,T-1]));mask=a['mask'][0] if 'mask' in a else truth[0,...,0]>0
    figures=[]
    for logarithmic in [False,True]:
        fig,axs=plt.subplots(len(ts),2,figsize=(12,8),layout='constrained',squeeze=False)
        for ri,t in enumerate(ts):
            for side in [0,1]:
                ax=axs[ri,side]
                for name,y,color in [('Truth',truth[0,t,side,:,1],'#172e3b'),('Prediction',pred[0,t,side,:,1],'#12836d')]:
                    y=np.asarray(y,dtype=float).copy();y[~mask[t,side]]=np.nan
                    ax.plot(np.arange(2048),np.log1p(y) if logarithmic else y,color=color,lw=.75,label=name)
                ax.set(title=f'minute {t} / '+('Bid' if side==0 else 'Ask'),xlabel='ranked level (complete 2048)',ylabel='log1p(quantity BTC)' if logarithmic else 'Quantity BTC');ax.grid(alpha=.16)
        axs[0,0].legend(fontsize=8);fig.suptitle(node['id']+' / '+c['lane']+' / first saved case only')
        name=tag+('_log.png' if logarithmic else '_btc.png');fig.savefig(directory/name,dpi=115);plt.close(fig);figures.append(name)
    # Price curves use the very same stored example and full-depth coordinates.
    fig,axs=plt.subplots(len(ts),2,figsize=(12,8),layout='constrained',squeeze=False)
    for ri,t in enumerate(ts):
        for side in [0,1]:
            ax=axs[ri,side]
            for name,y,color in [('Truth',truth[0,t,side,:,0],'#172e3b'),('Prediction',pred[0,t,side,:,0],'#267ca3')]:
                y=np.asarray(y,dtype=float).copy();y[~mask[t,side]]=np.nan
                ax.plot(np.arange(2048),y,color=color,lw=.9,label=name)
            ax.set(title=f'minute {t} / '+('Bid' if side==0 else 'Ask'),xlabel='ranked level (complete 2048)',ylabel='Price / USDT');ax.grid(alpha=.16)
    axs[0,0].legend(fontsize=8);fig.suptitle(node['id']+' / '+c['lane']+' / saved price reconstruction')
    price_name=tag+'_price.png';fig.savefig(directory/price_name,dpi=115);plt.close(fig);figures.append(price_name)
    if node['id']=='dual':
        comparison=read(SOURCE/'checkpoints/dual_12h_20261024_004600/reconstruction_20261024_111849/comparison.json')
        assert comparison['variants'][prediction_key]['sha256']==sha(SOURCE/c['weight'])
    # Small exact plotted arrays are versioned; full source predictions remain untouched.
    np.savez_compressed(directory/(tag+'_display_points.npz'),truth=truth[0,ts],prediction=pred[0,ts],mask=mask[ts],minutes=ts,source_case=0)
    info=dict(source=c['profile'],source_sha256=sha(path),prediction_key=prediction_key,source_shape=list(truth.shape),selected_source_case=0,minutes=ts,indices=a['indices'].tolist() if 'indices' in a else None,images=figures,full2048depth=True,not_population_evaluation=True)
    return info


def artifact(p,dest):
    p=SOURCE/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);return dict(source=rel(p),sha256=sha(p),copy=str(dest.relative_to(DOC)))


def render():
    os.environ['MPLCONFIGDIR']=str(AUDIT/'.mpl-cache')
    import matplotlib
    matplotlib.use('Agg')
    rows=inventory();DOC.mkdir(parents=True,exist_ok=True)
    records=[]
    for node in NODES:
        print('BUILD',node['id'],flush=True)
        directory=DOC/node['id'];directory.mkdir(parents=True,exist_ok=True)
        for old in directory.glob('best_*'):
            if old.is_file():old.unlink()
        candidates=[c for c in rows if c['family']==node['id']]
        protocols=collections.defaultdict(list)
        for c in candidates:protocols[c['protocol']].append(c)
        winners=[min(v,key=lambda c:(c['quantity'],c['price'])) for v in protocols.values()]
        winners.sort(key=lambda c:(-(c['windows'] or 0),c['quantity'],c['price']))
        # Render the selected model, while retaining the broader family variants in text/configs.
        primary=winners[0] if winners else None
        config=weight_metadata(primary).get('configs',{}) if primary else {}
        if node['id']=='full256':ARCH['full256']=[('Observed full256','B x 256 x 2 x 2048 x 2'),('Shared encoder + Cross','whole-window temporal context'),('Uncompressed G/L reference','1064960 content scalars; no matrix compression'),('Independent price / quantity decoders','ordered price; quantity amplitude refinement')]
        if node['id']=='cross':ARCH['cross'][1]=('Selected shared encoder','single source; cross fusion enabled')
        if node['id']=='continuous':
            d=config.get('pca_state_config',{});ARCH['continuous'][2]=('Grouped PCA state compression',f"state_dim={d.get('state_dim')}; global_components={d.get('global_components')}")
        if node['id']=='local_phase':
            d=config.get('local_phase_config',config.get('regional_phase_config',{}));ARCH['local_phase'][2]=('Selected local phase switches',f"use_shape={d.get('use_shape')}; use_mass={d.get('use_mass')}")
            if not d.get('use_shape') and not d.get('use_mass'):ARCH['local_phase'][3]=('Original quantity path','shape/mass correction bypassed')
        if node['id']=='paired_readout' and 'shape_mass_config' in config:
            ARCH['paired_readout']=[('Observed full256','frame1 encoder; only Encoder frozen here'),('Single Z','frame1; B x 64 x 128'),('Original quantity decoder','32-channel full-depth features + frame256'),('Shape/mass neural readout','Conv1d32->1 + Linear256->2'),('Quantity shape redistribution + mass scaling','use_shape=True, use_mass=True; detached price geometry')]
        if node['id']=='fourier':
            d=config.get('spectral_quantity_config',{});ARCH['fourier'][3]=('Selected spectral mode: '+str(d.get('spectral_mode')),f"width={d.get('spectral_width')}, layers={d.get('spectral_layers')}; off means control")
            if d.get('spectral_mode')=='off':ARCH['fourier'][4]=('Spectral correction bypassed','delta=0; original GELU path only')
        arch_svg(node,directory/'architecture.svg');figure_candidates(node,candidates,directory/'metrics.png')
        copied=[];winner_details=[]
        for i,c in enumerate(winners):
            evidence_name=f'best_{i}_evidence.json';raw=dict(source_path=c['path'],selector=c['selector'],source_sha256=sha(SOURCE/c['path']),record=c['raw'])
            (directory/evidence_name).write_text(json.dumps(json_small(raw),ensure_ascii=False,indent=2))
            plan_path=next((SOURCE/c['cohort']/name for name in ['plan.json','experiment_plan.json'] if (SOURCE/c['cohort']/name).exists()),None)
            if plan_path:artifact(rel(plan_path),directory/f'best_{i}_plan.json')
            meta=weight_metadata(c);(directory/f'best_{i}_checkpoint.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
            profile=profile_figure(c,node,directory,f'best_{i}')
            winner_details.append(dict(candidate={k:v for k,v in c.items() if k!='raw'},checkpoint=meta,profile=profile,evidence=evidence_name))
        # Documentation includes every candidate and exact criteria, not just a winner.
        compact=[{k:v for k,v in c.items() if k!='raw'} for c in candidates]
        (directory/'candidates.json').write_text(json.dumps(compact,ensure_ascii=False,indent=2))
        for title,paths in [('source',node['code'])]:
            (directory/'source_hashes.json').write_text(json.dumps({p:sha(SOURCE/p) for p in paths},indent=2))
        record=dict(id=node['id'],name=node['name'],candidates=len(candidates),protocols=len(protocols),winners=winner_details,status='recorded_results' if winners else 'no_trained_test_result')
        records.append(record)
        md=f"# {node['name']}\n\n更新：{NOW}。输入范围：{node['window']}。\n\n[返回总表](../README.md) · [模型谱系](../../MODEL_LINEAGE.md)\n\n## 模型结构\n\n以下图对应主要展示点的结构；分支其他可选结构见谱系说明，实际开关和精度以检查点配置为准。\n\n![结构](architecture.svg)\n\n"
        md+='```mermaid\nflowchart TD\n'
        for j,(name,shape) in enumerate(ARCH[node['id']]):md+=f'  N{j}["{name}<br/>{shape}"]\n'+(f'  N{j-1} --> N{j}\n' if j else '')
        md+='```\n\n'
        md+=f"- 瓶颈：{node['latent']}。\n- 架构变化：{node['change']}。\n- 结论边界：{node['conclusion']}\n- 原模型代码（相对OBreconstuction）："+', '.join('`'+x+'`' for x in node['code'])+'。\n\n'
        if not winners:
            md+='## 最佳结果：尚不存在\n\n已找到实现/预检或只读诊断，但没有可排名的已训练测试重建结果。不能把源模型的0步重放、单窗CPU预检或数据匹配审计当作本分支的新模型成绩。\n\n'
            audit_paths=[]
            for exp in node['experiments']:
                for name in ['status.json','audit.json','feature_probe_result.json','execution_readiness_v1.json']:
                    p=SOURCE/exp/name
                    if p.exists():
                        dest=directory/('audit_'+str(len(audit_paths))+'.json');copied.append(artifact(rel(p),dest));audit_paths.append(dest.name)
            md+='现有证据：'+', '.join(f'[{x}]({x})' for x in audit_paths)+'。\n\n'
        else:
            md+='## 各评估口径的最佳已记录检查点\n\n“最佳”定义为本分支、相同窗口长度/测试时段/窗口数/有效计数/目标RMS下所有已保存正步数评估中挂量RMSE最低；这是回顾性选点，包含中间检查点，不冒充独立测试确认；价格、≥100BTC指标和动态相关同时报告，不声称所有指标同时最优。不同口径分别保留最佳点，不做跨口径总冠军。下表第一项只是较大测试样本口径的展示代表。\n\n'
            md+='| 组别 | 步数/epoch | 口径 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | ≥100精确率 | F1 | 动态相关 |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n'
            f=lambda v:'未记录' if v is None else f'{v:.5f}'
            pct=lambda v:'未记录' if v is None else f'{v:.2%}'
            for w in winner_details:
                c=w['candidate'];md+=f"| {c['lane']} | {c['step']} / {c['epoch']} | {c['window_minutes']}分钟×{c['windows']}窗，{c['period']} | {f(c['price'])} | {f(c['quantity'])} | {pct(c['recall'])} | {pct(c['precision'])} | {pct(c['F1'])} | {f(c['dynamics'])} |\n"
            for i,w in enumerate(winner_details):
                c=w['candidate'];meta=w['checkpoint'];md+=f"\n### 最佳点{i+1}：{c['lane']}\n\n- 实验：`{c['cohort']}`。\n- 原始指标：`{c['path']}`，字段/行：`{c['selector']}`。\n- [固定指标证据]({w['evidence']}) · [权重元数据与哈希](best_{i}_checkpoint.json) · [原实验计划](best_{i}_plan.json)。\n"
                if c['lane'] in ['off','control','baseline','reference','original_warm','original_fresh','control_linear','control_log']:
                    md+='- **该点是控制/参考路径，不代表新增模块改善。**\n'
                if node['id']=='local_phase':
                    active=meta.get('configs',{}).get('local_phase_config',meta.get('configs',{}).get('regional_phase_config',{}))
                    if active and not active.get('use_shape') and not active.get('use_mass'):md+='- **该点的形状/总量头均关闭，收益来自其后续训练/损失配置，不能归因于这些架构头。**\n'
                if node['id']=='fourier':
                    enabled=[x for x in candidates if x['lane']!='off' and x['protocol']==c['protocol']]
                    if enabled:
                        e=min(enabled,key=lambda x:x['quantity']);md+=f"- FNO开启组中最低挂量RMSE为`{e['lane']}`，{e['quantity']:.5f} BTC；价格RMSE {e['price']:.4f} USDT。若off更好，不能宣布FNO获益。\n"
                if meta['available']:md+=f"- 权重：`{meta['path']}`；SHA256 `{meta['sha256']}`。权重留在原项目，本次不向Git批量上传大检查点。\n"
                else:md+='- 没有找到可直接匹配的单一权重文件；不能视为已具备完整复现资产。\n'
                if meta.get('configs'):md+='\n<details><summary>实际检查点结构配置</summary>\n\n```json\n'+json.dumps(meta['configs'],ensure_ascii=False,indent=2)+'\n```\n\n</details>\n'
                p=w['profile']
                if p:
                    md+=f"\n原始预测：`{p['source']}`，SHA256 `{p['source_sha256']}`。固定第一个保存案例，分钟{p['minutes']}，完整2048档。它是案例可视化，不替代全测试统计；没有新增模型推理。\n\n"
                    for name in p['images']:md+=f'![真实黑色，预测绿色；完整档位]({name})\n\n'
                    md+=f'[图中精确数据](best_{i}_display_points.npz)；未做深度平均、峰值截断或按误差挑样本。\n'
                else:md+='\n未找到与这个最佳点明确匹配的保存订单簿预测。本页展示真实指标/训练历史图，不借用其他检查点的曲线，不伪造订单簿预览。\n'
            # Best F1 alternative is explicit and can differ from minimum-RMSE winner.
            scored=[c for c in candidates if c['F1'] is not None]
            if scored:
                c=max(scored,key=lambda c:c['F1']);md+=f"\n### 若优先大单F1\n\n保留候选中F1最高点：`{c['cohort']}/{c['lane']}`，step={c['step']}，F1={c['F1']:.2%}，挂量RMSE={c['quantity']:.5f} BTC，价格RMSE={c['price']:.4f} USDT。它可能属于不同测试口径，不自动替代上面的最小挂量误差点。完整候选与口径见JSON。\n"
        if winners:
            terminal=[x for x in candidates if x.get('is_common_endpoint')]
            if terminal:
                e=min(terminal,key=lambda x:x['quantity'])
                md+=f"\n## 同预算共同终点参考\n\n该分支保留的共同终点评估中，最低挂量误差点为`{e['cohort']}/{e['lane']}`，step={e['step']}，挂量RMSE={e['quantity']:.5f} BTC、价格RMSE={e['price']:.4f} USDT。它与回顾性最低误差检查点是两个选型口径；不同来源或测试协议不能据此直接作因果比较。\n"
        md+='\n## 所有保留候选与比较边界\n\n![误差可视化](metrics.png)\n\n图中最多展示10个低挂量误差点，只能在相同口径内比较；所有候选均保留在[candidates.json](candidates.json)。0步初始化不参与已训练分支排名；显式稀疏地址诊断单独标明。历史参考/未压缩模型不冒充压缩方案结果；训练拟合、合成移动峰和已见窗口不混入测试排名。历史2025Q1已重复查看，不称独立确认。\n'
        (directory/'README.md').write_text(md)
    (DOC/'summary.json').write_text(json.dumps(dict(as_of=NOW,selection_policy='Minimum quantity RMSE per protocol; all positive recorded checkpoints, with maximum-common endpoints reported separately; price/F1/dynamics and all candidates retained; no global capacity ranking.',records=records),ensure_ascii=False,indent=2))
    index=f'''# 各模型分支最佳已记录结果与结构\n\n截至{NOW}。共{len(NODES)}个研究分支。每个分支给出模型结构SVG/Mermaid、实际检查点配置、各测试口径最佳点、原始指标、权重SHA256及可用的完整2048档重建图。\n\n## 如何理解“最佳”\n\n- 主规则是**同分支、同评估口径下最低挂量RMSE**，并列价格、大单召回/精确率/F1与挂量动态相关；这与当前main为综合F1选型的规则不同。不能把低挂量RMSE理解为整簿已经恢复好。\n- 测试时段、窗口长度、窗口数、有效计数和目标RMS不同的实验分组报告，不构造跨容量/跨口径总冠军。\n- 每个分支查找所有已保存正步数检查点的最低挂量误差，包含中间点；这是回顾性选型，不代表同预算结构改进或独立确认。最大共同端点另外标记并报告；早期epoch历史按真实测试记录选点并核对权重epoch。\n- 若控制组更好，赢家仍明确标成控制，不将其说成新增结构的收益。分支仍在训练时，结果为共同阶段点。\n- **未压缩完整256分钟Cross**与**唯一Gaussian8192**是不同容量；历史16分钟又是不同任务。稀疏地址诊断不是原生Gaussian神经模型。\n- 没有正式测试结果的分支写明“暂无”，只有结构/预检证据；缺少最佳点对应预测时仅绘制指标，不挪用其他权重的订单簿。\n\n## 分支总览\n\n| 分支 | 最小挂量误差代表 | 窗口/测试数 | 价格RMSE USDT | 挂量RMSE BTC | ≥100召回 | 结构/结果 |\n|---|---|---|---:|---:|---:|---|\n'''
    for n,r in zip(NODES,records):
        if r['winners']:
            c=r['winners'][0]['candidate'];index+=f"| {n['name']} | {c['lane']} | {c['window_minutes']}分钟×{c['windows']} | {c['price']:.4f} | {c['quantity']:.5f} | {('未记录' if c['recall'] is None else format(c['recall'],'.2%'))} | [详细页]({n['id']}/README.md) |\n"
        else:index+=f"| {n['name']} | 尚无已训练测试结果 | — | — | — | — | [结构与审计]({n['id']}/README.md) |\n"
    index+='\n候选范围以谱系中的代表实验为起点，并补入相同模型家族的后续损失/数据/精度控制；例如LocalHint的round001–003、RegionalPhase的round007–011、Direct的强度短对照，以及Unified8192的损失/数据量控制。只有原始测试口径清楚的结果进入排名。\n'
    index+='\n## 资料与复现\n\n[完整谱系](../MODEL_LINEAGE.md) · [机器可读结果索引](summary.json)。所有源指标与模型代码历史文件保持不变；图表直接读取已保存预测，本次不训练、不读取保留集、不重新做全量测试。原项目目录在本机为`/home/mengrui/e2e_strategy/OBreconstuction`。\n'
    index+='\n## 重建文档\n\n在可访问原实验的机器上，安装`pip install -e \".[reports]\"`，设置`OBVAE_SOURCE=/path/to/OBreconstuction`后运行`python tools/build_branch_atlas.py --render`。脚本只读取保存结果和权重元数据，不训练或重新评价测试集。\n'
    index += '\n## 当前8192主线共享骨干\n\n这部分适用于Unified/frame1及其后续紧凑模型，不套用到历史Global/Local、V4/V8或稀疏地址诊断：\n\n- 输入特征通常为基础4通道、双侧差分6通道、六尺度负LoG6通道；新增输入适配另列。空间卷积以四次stride2把2048档降到128个粗区域，主通道64/128/192/256；同帧买卖侧Cross-Attention后汇总。\n- patch4按4帧打包后在64个时间token上处理；frame1在256帧上做时间注意力，每帧32个Gaussian坐标重排为64×128。两者只有一个Gaussian；β=0/不采样的实验是确定性重建，不等于已验证先验生成质量。\n- 两个Decoder读取同一个Z，先还原每分钟256维特征，再产生双侧128区域×16维种子。空间Decoder四级nearest上采样与卷积，主通道192/128/64/32。\n- 价格使用mid、正spread和正gap累加；挂量经q0·Softplus保持非负，并可使用预测价格几何条件。首帧mid锚点、mask/tick/相对时间是元数据，不是Encoder特征跳接。\n- 细档位残差、形状/总量、直接MLP、全窗注意力及FNO都是这个共同骨干上的独立研究方向；实际启用开关、数值策略和历史配置见各分支权重JSON。\n'
    (DOC/'README.md').write_text(index)
    enrich_markdown(records)
    write_html(records)
    print('COMPLETE',len(records),sum(bool(r['winners']) for r in records),'branches with result',flush=True)


def write_html(records):
    esc=lambda x:html.escape(str(x))
    parts=[]
    for n,r in zip(NODES,records):
        winners=r['winners'];summary='暂无已训练测试结果'
        if winners:
            c=winners[0]['candidate'];summary=f"{c['lane']} · {c['window_minutes']}分钟×{c['windows']}窗 · Q-RMSE {c['quantity']:.5f} BTC"
        detail=f'<details class="branch" data-search="{esc(n["id"]+n["name"])}" id="{n["id"]}"><summary>{esc(n["name"])}<small>{esc(summary)}</small></summary><p>{esc(n["conclusion"])}</p><div class="grid"><div><h3>结构示意</h3><img loading="lazy" src="{n["id"]}/architecture.svg" alt="模型结构"></div><div><h3>记录误差对照</h3><img loading="lazy" src="{n["id"]}/metrics.png" alt="价格与挂量误差"><p>{esc(n["latent"])}</p><p><a href="{n["id"]}/README.md">完整Markdown说明</a> · <a href="{n["id"]}/candidates.json">全部候选JSON</a></p></div></div>'
        for i,w in enumerate(winners):
            c=w['candidate'];detail+=f'<h3>评估口径{i+1}的最低挂量误差点</h3><p>{esc(c["cohort"])} / {esc(c["lane"])}；step={c["step"]}，epoch={c["epoch"]}；测试{c["windows"]}窗。</p><div class="scroll"><table><tr><th>价格RMSE USDT</th><th>挂量RMSE BTC</th><th>≥100召回</th><th>≥100精确率</th><th>挂量动态相关</th></tr><tr>'
            for key in ['price','quantity','recall','precision','dynamics']:
                v=c[key];detail+='<td>'+('未记录' if v is None else f'{v:.2%}' if key in ['recall','precision'] else f'{v:.5f}')+'</td>'
            detail+='</tr></table></div>'
            if c['lane'] in ['off','control','baseline','reference','original_warm','original_fresh','control_linear','control_log']:detail+='<p class="callout">这个结果来自控制/参考路径，不是新增模块的增益。</p>'
            if w['profile']:
                for image in w['profile']['images']:detail+=f'<img class="reconstruction" loading="lazy" src="{n["id"]}/{image}" alt="匹配保存检查点的完整2048档重建">'
                detail+='<p class="note">固定第一个保存案例的首/中/末分钟，完整2048档；原始挂量、log挂量和价格分别展示。不做峰值截断，不替代全测试统计。</p>'
            else:detail+='<p>未找到与该点明确匹配的订单簿预测；仅展示已记录指标。</p>'
            detail+=f'<p><a href="{n["id"]}/{w["evidence"]}">原始指标证据</a> · <a href="{n["id"]}/best_{i}_checkpoint.json">实际检查点配置与权重哈希</a></p>'
        detail+='</details>';parts.append(detail)
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>各模型分支最佳结果与结构</title><style>*{box-sizing:border-box}body{margin:0;background:#f1f6f9;color:#183345;font:15px/1.8 system-ui,sans-serif}main{max-width:1350px;margin:auto;padding:30px 20px}h1{font-size:31px}h2{font-size:22px}a{color:#146e98}.grid{display:grid;grid-template-columns:1fr 1.2fr;gap:25px}img{width:100%;height:auto}.reconstruction{margin-top:20px}.callout{background:#fff3da;padding:16px;border-left:4px solid #be862b}.branch{padding:22px;background:white;border:1px solid #dce6ed;border-radius:12px;margin:20px 0}summary{cursor:pointer;font-size:19px;font-weight:650}summary small{display:block;font-size:13px;font-weight:400;color:#5c7483}input{padding:10px;font:inherit;width:100%;border:1px solid #bdd0dc;border-radius:6px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}td,th{padding:8px;text-align:right;border-bottom:1px solid #dce6ed}.note{font-size:12px;color:#5c7483}@media(max-width:800px){.grid{grid-template-columns:1fr}main{padding:15px 10px}.branch{padding:15px}}</style></head><body><main><h1>每个模型分支的最佳已记录结果与结构</h1>'''
    page+=f'<p>截至{esc(NOW)} · 25个研究分支 · 23个有历史重建结果 · 2个仅预检/审计。</p><p class="callout">最佳按同评估口径内挂量RMSE定义，回顾性选取已保存检查点；价格、大单和动态指标同时保留。16分钟/256分钟、不同测试窗口数及不同容量不能混排。稀疏地址属于工程诊断。当前main的综合F1基线选择是另一种口径。</p><p><a href="README.md">完整Markdown总表</a> · <a href="summary.json">机器可读汇总</a></p><input id="search" placeholder="搜索模型分支">'+''.join(parts)
    page+='''</main><script>document.getElementById('search').addEventListener('input',function(){let q=this.value.toLowerCase();document.querySelectorAll('.branch').forEach(e=>e.hidden=!e.dataset.search.toLowerCase().includes(q));});</script></body></html>'''
    (DOC/'index.html').write_text(page)

def enrich_markdown(records):
    """Expose price/quantity tradeoffs instead of presenting a scalar winner as universal."""
    rows=[]
    for record in records:
        directory=DOC/record['id'];candidates=read(directory/'candidates.json')
        p=directory/'README.md';text=p.read_text();text=text.split('<!-- PARETO -->')[0]
        groups=collections.defaultdict(list)
        for c in candidates:groups[c['protocol']].append(c)
        text+='\n<!-- PARETO -->\n## 价格与挂量的取舍\n\n最小挂量误差可能以明显价格退化为代价；下面按每个评估口径分别列出价格/挂量二维非支配点（最多10个）。这是多目标取舍，不是因果或显著性结论。完整点仍在候选JSON中。\n'
        for protocol,cs in groups.items():
            front=[c for c in cs if not any(o['quantity']<=c['quantity'] and o['price']<=c['price'] and (o['quantity']<c['quantity'] or o['price']<c['price']) for o in cs)]
            seen=set();front=[c for c in sorted(front,key=lambda v:(v['quantity'],v['price'])) if not ((c['price'],c['quantity']) in seen or seen.add((c['price'],c['quantity'])))]
            text+='\n口径：`'+protocol+'`\n\n| 组别 | step / epoch | 价格RMSE | 挂量RMSE |\n|---|---|---:|---:|\n'
            for c in front[:10]:text+=f"| {c['lane']} | {c['step']} / {c['epoch']} | {c['price']:.4f} | {c['quantity']:.5f} |\n"
        p.write_text(text)
        for w in record['winners']:
            c=w['candidate'];rows.append({k:c[k] for k in ['family','cohort','lane','step','epoch','window_minutes','windows','price','quantity','recall','precision','F1','dynamics','path']})
    if rows:
        with (DOC/'best_results.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--render',action='store_true');parser.add_argument('--html-only',action='store_true');args=parser.parse_args()
    if args.html_only:
        NOW=read(DOC/'summary.json')['as_of']
        records=read(DOC/'summary.json')['records'];enrich_markdown(records);write_html(records)
    elif args.render:render()
    else:inventory()

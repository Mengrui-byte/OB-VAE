"""One GPU / one model / one seed, with exact periodic restart state."""

import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader,Subset

from model.experimental.dual_long import ResearchOrderBookVAE
from model.losses.dual import DualLossConfig,DualVAELoss
from train_dual.util.checkpoint import atomic_save,seed_everything,source_hashes
from train_dual.util.data import DualNormalizer,DualWindowDataset
from train_dual.util.dynamics import DynamicsMetrics


def atomic_json(path,value):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def datasets(root,initial,split):
    cfg=initial['config']['arguments']
    limits={'train':('2022-01-01','2025-01-01'),
            'val':('2025-01-01','2025-04-01'),
            'test':('2025-04-01','2025-07-01')}
    start,end=limits[split]
    return DualWindowDataset(asset='BTCUSDT',npz_dir=root/'data/BTCUSDT.npz',start=start,end=end,
                             levels=cfg['levels'],window_size=16,stride=16,
                             time_unit_seconds=cfg.get('time_unit_seconds',60))


def build_model(initial,variant,seed,device):
    seed_everything(seed)
    model=ResearchOrderBookVAE(variant=variant,refinement_seed=seed,**initial['model_config'])
    model.load_common_initial(initial['model'])
    return model.to(device)


def forward(model,batch,sample):
    return model(batch['price'],batch['volume'],batch['mask'],batch['tick_norm'],
                 times=batch['times'],sample=sample)


def evaluate(model,normalizer,dataset,windows,device):
    """FP32 common evaluation regardless of training autocast settings."""
    model.eval()
    indices=np.unique(np.linspace(0,len(dataset)-1,windows,dtype=int))
    metrics=DynamicsMetrics(normalizer.tick_size)
    tail_n=0;tail_log_sse=tail_raw_sse=tail_target_sum=tail_pred_sum=0.
    target_max=prediction_max=0.
    counts={q:np.zeros(3,dtype=np.int64) for q in (10,100)}
    with torch.inference_mode():
        for index in indices:
            raw={k:v.unsqueeze(0) for k,v in dataset[int(index)].items()}
            batch=normalizer.prepare(raw,device)
            output=forward(model,batch,False)
            truth=raw['data'][0].numpy().astype(np.float64)
            prediction=np.stack([normalizer.inverse_price(output['price_norm'],batch)[0].cpu().numpy(),
                                 output['volume'][0].float().cpu().numpy()],-1)
            mask=batch['mask'][0].cpu().numpy()
            if not np.isfinite(prediction[mask]).all():
                raise FloatingPointError('Nonfinite evaluation output')
            metrics.update(truth,prediction,mask)
            q,qh=truth[...,1],prediction[...,1]
            large=mask&(q>=100)
            tail_n+=int(large.sum())
            tail_log_sse+=float(np.square(np.log1p(qh[large])-np.log1p(q[large])).sum())
            tail_raw_sse+=float(np.square(qh[large]-q[large]).sum())
            tail_target_sum+=float(q[large].sum());tail_pred_sum+=float(qh[large].sum())
            target_max=max(target_max,float(q[mask].max()));prediction_max=max(prediction_max,float(qh[mask].max()))
            for threshold in counts:
                a,b=(q>=threshold)&mask,(qh>=threshold)&mask
                counts[threshold]+=[int((a&b).sum()),int((~a&b&mask).sum()),int((a&~b).sum())]
    result=metrics.finish()
    result.update(tail_levels=tail_n,tail_log_rmse=math.sqrt(tail_log_sse/tail_n) if tail_n else None,
                  tail_quantity_rmse=math.sqrt(tail_raw_sse/tail_n) if tail_n else None,
                  tail_target_mean=tail_target_sum/tail_n if tail_n else None,
                  tail_prediction_mean=tail_pred_sum/tail_n if tail_n else None,
                  target_max=target_max,prediction_max=prediction_max,
                  thresholds={str(q):dict(tp=int(v[0]),fp=int(v[1]),fn=int(v[2]),
                      precision=float(v[0]/(v[0]+v[1])) if v[0]+v[1] else None,
                      recall=float(v[0]/(v[0]+v[2])) if v[0]+v[2] else None) for q,v in counts.items()},
                  evaluation_precision='fp32',window_indices=indices.tolist())
    model.train()
    return result


def score(result,reference):
    def vector(item):
        m=item['metrics']
        return [m['quantity_btc']['rmse'],item['tail_log_rmse'],
                m['spatial_slope_64_levels_usdt_per_level']['rmse'],
                m['temporal_mid_1_minutes_usdt_per_minute']['rmse']]
    a,b=vector(result),vector(reference)
    return float(sum(w*math.log(max(x,1e-10)/max(y,1e-10))
                     for w,x,y in zip((.35,.25,.25,.15),a,b)))


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state())


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu());torch.cuda.set_rng_state(state['cuda'].cpu())


def metadata(initial,variant,seed,loss_config,root):
    cfg=json.loads(json.dumps(initial['config']))
    cfg['loss']=asdict(loss_config)
    cfg['arguments'].update(npz_dir=str(root/'data/BTCUSDT.npz'),seed=seed,precision='bf16')
    return dict(format_version=1,model_config=initial['model_config'],research_variant=variant,
                refinement_seed=seed,config=cfg,source_hashes=source_hashes(),epoch_completed=False)


def worker(root,variant,seed,stop_at,physical_batch=32,effective_batch=256,workers=2):
    torch.set_num_threads(4)
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    if torch.cuda.get_device_capability(device)[0]<8:
        raise RuntimeError('This campaign requires hardware BF16 support')
    if effective_batch%physical_batch:
        raise ValueError('Effective batch must divide into physical microbatches')
    folder=root/'runs'/f'{variant}_s{seed}'
    folder.mkdir(parents=True,exist_ok=True)
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    common=json.loads((root/'common_config.json').read_text())
    reference=json.loads((root/'reference.json').read_text())
    if source_hashes()!=common['source_hashes']:
        raise RuntimeError('Frozen source identity changed')
    model=build_model(initial,variant,seed,device)
    normalizer=DualNormalizer(tick_size=.01,price_scale=50)
    criterion=DualVAELoss(1.,DualLossConfig(**common['loss']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5,weight_decay=.01)
    train=datasets(root,initial,'train');validation=datasets(root,initial,'val')
    step=epoch=cursor=samples=0;best=float('inf')
    seed_everything(seed)
    if (folder/'last.pt').exists():
        saved=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        for state in optimizer.state.values():
            for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                if name in state:state[name]=state[name].to(device)
        step,epoch,cursor,samples,best=[saved[k] for k in ('global_step','data_epoch','data_cursor','samples_seen','best_score')]
        restore_rng(saved['rng'])
    base_metadata=metadata(initial,variant,seed,criterion.config,root)
    atomic_json(folder/'run_config.json',dict(variant=variant,seed=seed,physical_batch=physical_batch,
        effective_batch=effective_batch,stop_at=stop_at,lr=3e-5,loss=common['loss'],initial_sha256=common['initial_sha256'],
        train_windows=len(train),validation_windows=common['evaluation_windows'],source_hashes=common['source_hashes']))
    model.train();optimizer.zero_grad(set_to_none=True)
    micro=group_n=0;loss_sum=0.;started=time.monotonic();last_report=started
    group_start=(epoch,cursor,rng_state())
    torch.cuda.reset_peak_memory_stats()

    def payload(full):
        result={**base_metadata,'model':model.state_dict(),'epoch':epoch+1,'global_step':step,
                'data_epoch':epoch,'data_cursor':cursor,'samples_seen':samples,'best_score':best}
        if full:result.update(optimizer=optimizer.state_dict(),rng=rng_state())
        else:result['inference_only']=True
        return result

    def validate():
        nonlocal best
        before=time.monotonic()
        values=evaluate(model,normalizer,validation,common['evaluation_windows'],device)
        value=score(values,reference)
        row=dict(utc=utc(),variant=variant,seed=seed,step=step,samples_seen=samples,
                 elapsed_seconds=time.monotonic()-started,score=value,evaluation_seconds=time.monotonic()-before,
                 result=values)
        if value<best:
            best=value;atomic_save(payload(False),folder/'best.pt')
        atomic_save(payload(False),folder/f'evaluation_step_{step:07d}.pt')
        with (folder/'evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        atomic_json(folder/'latest_evaluation.json',row)
        return value

    try:
        while time.time()<stop_at-30:
            order=np.random.default_rng(seed+epoch*1009).permutation(len(train))
            generator=torch.Generator().manual_seed(seed+epoch*13)
            loader=DataLoader(Subset(train,order[cursor:].tolist()),batch_size=physical_batch,
                              shuffle=False,num_workers=workers,pin_memory=True,generator=generator)
            reached_end=True
            for raw in loader:
                if time.time()>=stop_at-30:
                    reached_end=False;break
                if micro==0:group_start=(epoch,cursor,rng_state())
                batch=normalizer.prepare(raw,device);size=len(batch['price'])
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    output=forward(model,batch,True)
                terms=criterion(output,batch,kl_factor=1.)
                if not bool(torch.stack([x.detach() for x in terms.values()]).isfinite().all()):
                    raise FloatingPointError('Nonfinite training loss')
                (terms['loss']*size).backward()
                micro+=1;group_n+=size;cursor+=size;loss_sum+=terms['loss'].item()*size
                if micro==effective_batch//physical_batch:
                    for parameter in model.parameters():
                        if parameter.grad is not None:parameter.grad.div_(group_n)
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
                    step+=1;samples+=group_n
                    mean_loss=loss_sum/group_n;micro=group_n=0;loss_sum=0.
                    if step%10==0:
                        now=time.monotonic()
                        row=dict(state='training',utc=utc(),variant=variant,seed=seed,step=step,
                            data_epoch=epoch,data_cursor=cursor,samples_seen=samples,loss=mean_loss,
                            gradient_norm=float(norm),elapsed_seconds=now-started,
                            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                            reserved_gib=torch.cuda.memory_reserved()/2**30)
                        atomic_json(folder/'progress.json',row)
                        with (folder/'train.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                        print(json.dumps(row),flush=True);last_report=now
                    if step%50==0:atomic_save(payload(True),folder/'last.pt')
                    if step%100==0 and time.time()<stop_at-90:
                        validate();atomic_save(payload(True),folder/'last.pt')
            if reached_end:
                epoch+=1;cursor=0
            else:break
        if micro:
            epoch,cursor,state=group_start;restore_rng(state);optimizer.zero_grad(set_to_none=True)
        atomic_save(payload(True),folder/'last.pt')
        atomic_save(payload(False),folder/'final.pt')
        atomic_json(folder/'exit_status.json',dict(state='time_budget_complete',utc=utc(),step=step,samples_seen=samples))
    except BaseException as error:
        atomic_json(folder/'exit_status.json',dict(state='failed',utc=utc(),step=step,error=repr(error)))
        raise


def prepare(root):
    """Validate data/source identities and calibrate only on training windows."""
    import zipfile
    torch.set_num_threads(4);device=torch.device('cuda:0')
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    identity=json.loads((root/'data_identity.json').read_text())
    path=root/'data/BTCUSDT.npz'
    if path.stat().st_size!=identity['npz_bytes']:raise ValueError('NPZ size mismatch')
    with zipfile.ZipFile(path) as archive:
        members={i.filename:dict(bytes=i.file_size,crc32=i.CRC) for i in archive.infolist()}
    if members!=identity['members']:raise ValueError('Archive identities differ')
    model=build_model(initial,'control',42,device).eval()
    normalizer=DualNormalizer(tick_size=.01,price_scale=50)
    train=datasets(root,initial,'train')
    loss=dict(initial['config']['loss'],price_slope=.012255924722983856,
              price_temporal=5.531006151533404e-7,volume_tail=0.)
    base=DualVAELoss(1.,DualLossConfig(**loss))
    measure=DualVAELoss(1.,DualLossConfig(**dict(loss,volume_tail=1.)))
    records=[]
    with torch.inference_mode():
        for index in np.unique(np.linspace(0,len(train)-1,32,dtype=int)):
            raw={k:v.unsqueeze(0) for k,v in train[int(index)].items()}
            batch=normalizer.prepare(raw,device);out=forward(model,batch,False)
            a,b=base(out,batch),measure(out,batch)
            records.append(dict(index=int(index),base=a['reconstruction'].item(),tail=b['volume_tail'].item()))
    positive=[r['tail'] for r in records if r['tail']>0]
    if not positive:raise ValueError('No large training quantities for calibration')
    loss['volume_tail']=.25*float(np.median([r['base'] for r in records]))/float(np.median(positive))
    common=dict(loss=loss,records=records,evaluation_windows=256,source_hashes=source_hashes(),
                initial_sha256=hashlib.sha256((root/'initial.pt').read_bytes()).hexdigest(),
                tail_calibration='25 percent of median reconstruction loss, training windows only')
    atomic_json(root/'common_config.json',common)
    reference=evaluate(model,normalizer,datasets(root,initial,'val'),256,device)
    atomic_json(root/'reference.json',reference)
    print(json.dumps(dict(prepared=True,loss=loss,reference_quantity_rmse=reference['metrics']['quantity_btc']['rmse'])),flush=True)


def resource_probe(root,variant,physical_batch):
    torch.set_num_threads(4);device=torch.device('cuda:0')
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    model=build_model(initial,variant,42,device)
    ds=datasets(root,initial,'train')
    normalizer=DualNormalizer(tick_size=.01,price_scale=50)
    raw=next(iter(DataLoader(ds,batch_size=physical_batch)))
    batch=normalizer.prepare(raw,device)
    common=json.loads((root/'common_config.json').read_text())
    criterion=DualVAELoss(1.,DualLossConfig(**common['loss']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-5)
    torch.cuda.reset_peak_memory_stats();started=time.monotonic()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):out=forward(model,batch,True)
        value=criterion(out,batch)['loss']
        if not torch.isfinite(value):raise FloatingPointError('Nonfinite resource probe loss')
        value.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    torch.cuda.synchronize()
    result=dict(variant=variant,physical_batch=physical_batch,precision='bf16',loss=float(value),
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,seconds=time.monotonic()-started)
    atomic_json(root/f'resource_{variant}.json',result);print(json.dumps(result),flush=True)


def final_evaluation(root,variant,seed,step,extended=False):
    """Use the untouched quarter only after the development winner is frozen."""
    torch.set_num_threads(4)
    initial=torch.load(root/'initial.pt',map_location='cpu',weights_only=False)
    data=datasets(root,initial,'test')
    normalizer=DualNormalizer(tick_size=.01,price_scale=50)
    result={}
    for name in (tuple([variant]) if extended else dict.fromkeys(('control',variant))):
        path=root/'runs'/f'{name}_s{seed}'/f'evaluation_step_{step:07d}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=build_model(initial,name,seed,'cuda:0')
        model.load_state_dict(saved['model'],strict=True)
        values=evaluate(model,normalizer,data,512 if extended else 256,'cuda:0')
        result[name]=dict(checkpoint=str(path),result=values)
        del model
        torch.cuda.empty_cache()
    folder=root/'holdout';folder.mkdir(exist_ok=True)
    filename=f'extended_seed_{seed}.json' if extended else f'seed_{seed}.json'
    atomic_json(folder/filename,dict(seed=seed,step=step,variant=variant,
        split=['2025-04-01','2025-07-01'],selection_already_frozen=True,results=result))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    parser.add_argument('--mode',choices=('prepare','probe','train','final','final-extra'),default='train')
    parser.add_argument('--variant',choices=('control','spatial','volume','temporal'),default='control')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--stop-at',type=float)
    parser.add_argument('--physical-batch',type=int,default=32)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--step',type=int)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.root)
    elif args.mode=='probe':resource_probe(args.root,args.variant,args.physical_batch)
    elif args.mode in ('final','final-extra'):
        if args.step is None:parser.error('--step required for final evaluation')
        final_evaluation(args.root,args.variant,args.seed,args.step,args.mode=='final-extra')
    else:
        if args.stop_at is None:parser.error('--stop-at is required')
        worker(args.root,args.variant,args.seed,args.stop_at,args.physical_batch,256,args.workers)


if __name__=='__main__':main()

"""Portable inference/evaluation and explicit controlled continuation."""
import argparse,hashlib,json,os,random,shutil,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Subset, Dataset
from obvae.api import ROOT,CODE_ROOT,DEFAULT_CHECKPOINT,load_checkpoint
from train_dual.util.data import DualWindowDataset
from train_dual.util.quantity_strata import evaluate_with_quantity_strata


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')


def dataset(path,split):
    start,end=('2022-01-01','2025-01-01') if split=='train' else ('2025-01-01','2025-04-01')
    return DualWindowDataset(asset='BTCUSDT',npz_dir=path,start=start,end=end,levels=2048,window_size=256,stride=256,time_unit_seconds=60)


def evaluate(args):
    torch.set_num_threads(args.threads)
    model,norm,_=load_checkpoint(args.checkpoint,args.device)
    data=dataset(args.data,'test');indices=np.load(ROOT/'research/val_indices.npy')
    selected=Subset(data,list(map(int,indices)))
    result=evaluate_with_quantity_strata(model,norm,selected,len(selected),args.device)
    result.update(evaluation_role='test',period=['2025-01-01','2025-04-01'],window_indices=indices.tolist())
    write(args.output,dict(result=result,checkpoint=str(args.checkpoint),weight_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()))
    print(json.dumps(dict(output=str(args.output),windows=len(indices)),ensure_ascii=False))


def inspect(args):
    torch.set_num_threads(args.threads);model,_,saved=load_checkpoint(args.checkpoint,'cpu')
    print(json.dumps(dict(checkpoint=str(args.checkpoint),parameters=sum(p.numel() for p in model.parameters()),step=saved['step'],source_steps=saved['source_steps'],input_shape=[1,256,2,2048,2],latent_shape=[1,64,128],branch_precision=saved['branch_precision_config']),indent=2))


class IndexedWindows(Dataset):
    def __init__(self, dataset, indices):self.dataset,self.indices=dataset,indices
    def __len__(self):return len(self.indices)
    def __getitem__(self, index):return dict(self.dataset[int(self.indices[index])],cache_position=torch.tensor(index))


def train(args):
    """Continue a full four-group checkpoint, never silently reset historical Adam."""
    from train_dual.util.inherited_book_loss import inherited_book_loss
    from train_dual.util.direct_readout_training import restore_direct_readout_adam
    from train_dual.util.resume_data import boundary_loader
    from util.model_sources import freeze_model_sources
    if args.steps<=0:raise ValueError('steps must be positive')
    torch.set_num_threads(args.threads)
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(args.output);root.mkdir(parents=True,exist_ok=False)
    model,norm,saved=load_checkpoint(args.resume,args.device)
    if 'optimizer' not in saved or not saved.get('epoch_boundary'):
        raise ValueError('Continuation needs a full epoch-boundary training checkpoint; inference weights cannot restore Adam/order')
    plan=json.loads((ROOT/'research/plan.json').read_text());lane=saved['lane']
    model.requires_grad_(True);model.encoder.requires_grad_(False);model.price_decoder.requires_grad_(False)
    # Historical disabled region-specific heads remain frozen.
    for name,flag in plan['lanes'][lane].get('gradient_parameter_flags',{}).items():model.get_parameter(name).requires_grad_(flag)
    optimizer,_=restore_direct_readout_adam(model,saved['optimizer'],plan['original_parameter_names'])
    cal=json.loads((ROOT/'research/calibration.json').read_text());qc=json.loads((ROOT/'research/quantity_calibration.json').read_text());dc=json.loads((ROOT/'research/detail_calibration.json').read_text())
    loss=inherited_book_loss(saved,plan,lane,cal,qc,dc).to(args.device)
    teacher,teacher_norm,_=load_checkpoint(args.teacher,args.device,teacher=True);teacher.requires_grad_(False)
    original=dataset(args.data,'train');indices=np.load(ROOT/'research/train_indices.npy');selected=IndexedWindows(original,indices)
    loader,sampler=boundary_loader(selected,saved['sampler_generator_state'],workers=args.workers)
    state=saved['rng_state'];random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'].cpu())
    if str(args.device).startswith('cuda') and state['cuda'] is not None:torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])
    iterator=iter(loader);accumulation=plan['accumulation'];exposures=np.asarray(saved['training_data']['exposure_counts']).copy()
    write(root/'plan.json',dict(seed=42,updates=args.steps,accumulation=accumulation,resume=str(args.resume),teacher=str(args.teacher),evaluation_policy='No test access inside training; explicit evaluate command after completion.',source_checkpoint_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()))
    hashes=freeze_model_sources(root/'source_snapshot',CODE_ROOT)
    for name in ['obvae','train_dual','util']:
        for p in (CODE_ROOT/name).rglob('*.py'):
            rel=p.relative_to(CODE_ROOT);dest=root/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest);hashes[str(rel)]=hashlib.sha256(dest.read_bytes()).hexdigest()
    write(root/'source_snapshot/sha256.json',hashes)
    frozen={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith(('encoder.','price_decoder.'))}
    started=time.monotonic()
    for step in range(1,args.steps+1):
        model.train();model.encoder.eval();model.price_decoder.eval();optimizer.zero_grad(set_to_none=True)
        last=None
        for _ in range(accumulation):
            try:raw=next(iterator)
            except StopIteration:iterator=iter(loader);raw=next(iterator)
            position=int(raw.pop('cache_position')[0]);exposures[position]+=1
            b=norm.prepare(raw,args.device);tb=teacher_norm.prepare(raw,args.device)
            with torch.no_grad():_,target=teacher.encode_fused(tb['price'],tb['volume'],tb['mask'],tb['tick_norm'],tb['times'])
            out=model(*[b[k] for k in ['price','volume','mask','tick_norm','times']],sample=False)
            values=loss(out,b,teacher_local=target)
            if not bool(torch.isfinite(values['loss'])):raise FloatingPointError('Nonfinite continuation loss')
            (values['loss']/accumulation).backward();last=values
            if step==1:
                assert list(raw['data'].shape)==[1,256,2,2048,2] and list(out['z'].shape)==[1,64,128]
                write(root/'actual_shape_contract.json',dict(input=list(raw['data'].shape),latent=list(out['z'].shape),price=list(out['price_norm'].shape),quantity=list(out['volume'].shape)))
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),plan['clip_grad'],error_if_nonfinite=True);optimizer.step()
        record=dict(step=step,total=args.steps,loss=float(last['loss'].detach()),gradient_norm=float(grad),elapsed_seconds=time.monotonic()-started)
        write(root/'progress.json',record)
        with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    for k,v in frozen.items():assert torch.equal(v,model.state_dict()[k].detach().cpu()),k
    # A terminal checkpoint is an explicit new research result. No claim of bitwise historical CUDA replay.
    result={k:v for k,v in saved.items() if k not in ['optimizer','rng_state','training_data']}
    result.update(model=model.state_dict(),optimizer=optimizer.state_dict(),step=saved['step']+args.steps,source_hashes=hashes,
        epoch_boundary=(args.steps*accumulation)%len(selected)==0,
        sampler_generator_state=sampler.get_state(),rng_state=dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
        training_data=dict(exposure_counts=exposures,windows=len(selected)),continuation_metadata=dict(updates=args.steps,exact_historical_trajectory_claim=False))
    torch.save(result,root/'final_training.pt');write(root/'status.json',dict(state='complete',steps=args.steps,frozen_encoder_and_price_exact=True,test_accesses=0))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    for name,fn in [('inspect',inspect),('evaluate',evaluate),('train',train)]:
        s=sub.add_parser(name);s.set_defaults(func=fn);s.add_argument('--threads',type=int,default=4)
        if name=='train':
            s.add_argument('--resume',type=Path,required=True);s.add_argument('--teacher',type=Path,required=True);s.add_argument('--steps',type=int,default=128);s.add_argument('--workers',type=int,default=2)
        else:s.add_argument('--checkpoint',type=Path,default=DEFAULT_CHECKPOINT)
        if name!='inspect':s.add_argument('--device',default='cuda:0');s.add_argument('--data',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.func(args)

if __name__=='__main__':main()

"""Standalone deterministic RGB-D post-training, with fresh Adam state."""
from __future__ import annotations
import os
import argparse,json,math,time
from pathlib import Path
import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs,ProjectConfiguration
from torch.optim.lr_scheduler import CosineAnnealingLR,LinearLR,SequentialLR
from torch.utils.data import DataLoader
from .builder import build_model,build_dataset,read_config
from .checkpoint import public_contract
from .gradient_norm_order import REFERENCE_GRADIENT_NORM_ORDER
from .data.schema import atomic_json,fingerprint
from .utils.pytorch_utils import set_global_seed
from .utils.optimizer_precision import prepare_optimizer_precision
from .utils.training_metrics import reduce_training_metrics


def freeze_encoders(model):
    model.eval().requires_grad_(False)
    for name,child in model.named_children():
        if name not in model.FROZEN_MODULES:child.train().requires_grad_(True)
    for p in model.parameters(recurse=False):p.requires_grad_(True)


def clip_reference_norm(model,max_norm,order):
    """Keep the reference zero-norm slots after deleting unused parameters."""
    named=dict(model.named_parameters())
    active=[p for p in model.parameters() if p.grad is not None]
    if not active:raise RuntimeError('No gradients')
    if any(p.grad.dtype!=torch.float32 for p in active):raise RuntimeError('Expected FP32 master gradients')
    pairs=[(n,p.grad) for n,p in named.items() if p.grad is not None]
    values=torch._foreach_norm([g for _,g in pairs],2)
    norms=dict(zip([n for n,_ in pairs],values))
    zero=active[0].grad.new_zeros(())
    ordered=[norms.pop(n,zero) for n in order]
    if norms:raise RuntimeError('Gradient parameters absent from initialization contract')
    total=torch.linalg.vector_norm(torch.stack(ordered),2)
    if not torch.isfinite(total):raise FloatingPointError('Nonfinite gradient norm')
    coef=torch.clamp(max_norm/(total+1e-6),max=1.)
    torch._foreach_mul_([p.grad for p in active],coef)
    return total


def run(args):
    cfg=read_config('training')
    if cfg['mixed_precision']!='bf16' or cfg['gradient_accumulation_steps']!=1:
        raise ValueError('This fixed reproduction recipe requires BF16 and one microbatch per Adam update')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.fp32_precision='ieee'
    out=Path(args.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    os.environ['CRWM_ASSETS_ROOT']=str(Path(args.assets_root).expanduser().resolve())
    accelerator=Accelerator(mixed_precision='bf16',step_scheduler_with_optimizer=False,
        project_config=ProjectConfiguration(save_on_each_node=True),
        kwargs_handlers=[DistributedDataParallelKwargs(bucket_cap_mb=cfg['ddp_bucket_cap_mb'],gradient_as_bucket_view=cfg['ddp_gradient_as_bucket_view'])])
    mode = accelerator.distributed_type.value
    if mode != 'NO' and not mode.startswith('MULTI_'):
        raise ValueError('This recipe requires native data-parallel execution')
    if accelerator.num_processes!=cfg['reference_world_size'] and args.stop_after_steps is None:
        raise ValueError('Full reproduction requires the reference world size; use --stop-after-steps for a labelled finite smoke run')
    set_global_seed(cfg['seed'])
    model=build_model(str(accelerator.device),torch.bfloat16)
    dataset=build_dataset(args.data_root,args.text_cache_root,args.sidecar_root)
    model.bind_multisource_contract(dataset.checkpoint_contract)
    model.mot.checkpoint_full_layers=cfg['full_layer_checkpointing']
    freeze_encoders(model)
    prepare_optimizer_precision(model)
    # Same second seed point as the reference trainer before DataLoader creation.
    worker_init=set_global_seed(cfg['seed'],get_worker_init_fn=True)
    sampler=dataset.make_sampler(seed=cfg['seed'],batch_size=cfg['batch_size'],num_processes=accelerator.num_processes,accumulation_steps=1)
    loader_kw=dict(batch_size=cfg['batch_size'],sampler=sampler,num_workers=cfg['num_workers'],pin_memory=True,worker_init_fn=worker_init)
    if cfg['num_workers']>0:loader_kw.update(persistent_workers=True,prefetch_factor=4)
    loader=DataLoader(dataset,**loader_kw)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['adam_betas']))
    scheduler=SequentialLR(optimizer,[LinearLR(optimizer,start_factor=1/cfg['warmup_steps'],end_factor=1.,total_iters=cfg['warmup_steps']),CosineAnnealingLR(optimizer,T_max=cfg['max_steps']-cfg['warmup_steps'],eta_min=cfg['learning_rate']*.01)],milestones=[cfg['warmup_steps']])
    model,optimizer,loader,scheduler=accelerator.prepare(model,optimizer,loader,scheduler)
    raw=accelerator.unwrap_model(model)
    raw.load_checkpoint(args.initial_weights)
    norm_order=REFERENCE_GRADIENT_NORM_ORDER
    optimizer.zero_grad(set_to_none=True)
    step=0
    if args.resume:
        meta=json.loads((Path(args.resume)/'trainer_state.json').read_text())
        if meta['contract']!=public_contract(dataset.checkpoint_contract):raise ValueError('Resume data contract mismatch')
        if meta['training']!=cfg:raise ValueError('Resume optimizer/schedule contract mismatch')
        sampler.load_state_dict(meta['sampler'])
        accelerator.load_state(args.resume);step=int(meta['step'])
    stop=cfg['max_steps'] if args.stop_after_steps is None else min(args.stop_after_steps,cfg['max_steps'])
    if stop<=step:raise ValueError('Stop step must exceed restored step')
    def save():
        state=out/'checkpoints/state'/f'step_{step:06d}'
        weights=out/'checkpoints/weights'/f'step_{step:06d}.pt'
        weights.parent.mkdir(parents=True,exist_ok=True)
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            raw.save_checkpoint(str(weights),step=step)
            # Full resume preserves model/Adam/scheduler, per-rank RNG and the committed sampler.
        accelerator.save_state(str(state),safe_serialization=False)
        if accelerator.is_local_main_process:
            atomic_json(state/'trainer_state.json',dict(step=step,sampler=sampler.state_dict(),contract=public_contract(dataset.checkpoint_contract),training=cfg))
        accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        atomic_json(out/'run_contract.json',dict(training=cfg,world_size=accelerator.num_processes,finite=stop<cfg['max_steps'],contract=public_contract(dataset.checkpoint_contract)))
    iterator=iter(loader);started=time.monotonic()
    try:
        while step<stop:
            try:sample=next(iterator)
            except StopIteration:
                iterator=iter(loader);sample=next(iterator)
            ids=torch.stack((sample['window_index'],sample['sample_position']),1)
            observed=accelerator.gather(ids).cpu().tolist()
            with accelerator.autocast():loss,metrics=model(sample)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
            accelerator.backward(loss)
            grad_norm=clip_reference_norm(raw,cfg['max_grad_norm'],norm_order)
            optimizer.step()
            if accelerator.optimizer_step_was_skipped:raise RuntimeError('Unexpected skipped BF16 Adam step')
            scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
            sampler.commit(cfg['batch_size']*accelerator.num_processes,observed=observed)
            value,logs,norm=reduce_training_metrics(accelerator,loss,metrics,grad_norm)
            if accelerator.is_main_process:
                row=dict(step=step,loss=value,grad_norm=norm,lr=optimizer.param_groups[0]['lr'],elapsed_s=time.monotonic()-started,windows_sha256=fingerprint(sorted(observed,key=lambda x:x[1])),**logs)
                with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
            if step%cfg['save_every']==0 or step==stop:save()
    finally:
        native=getattr(loader,'base_dataloader',loader)
        workers=getattr(native,'_iterator',None)
        if workers is not None and hasattr(workers,'_shutdown_workers'):workers._shutdown_workers()
        accelerator.end_training()


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('data-root','text-cache-root','sidecar-root','assets-root','initial-weights','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--stop-after-steps',type=int)
    p.add_argument('--resume')
    args=p.parse_args();run(args)

if __name__=='__main__':main()

"""Standalone eager SFT: VGen + GT action + two-step selected cotrain."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time
import math
import fcntl
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from torch.utils.data import DataLoader

from policy_core.model.WAM_policy.build_models import build_WAM_policy_task_flow
from policy_core.settings import settings
from go3_nano_preview.sft.checkpoint import save_checkpoint
from go3_nano_preview.sft.dataset import make_dataset
from go3_nano_preview.sft.objectives import Cotrain
from go3_nano_preview.sft.helpers.combined_cotrain import parse_combined_cotrain, build_micro_steps
from go3_nano_preview.sft.fixed_val_loss import FixedValLoss
from go3_nano_preview.sft.distributed import initialize, on_rank_zero, on_all_ranks, mean_losses
from go3_nano_preview.sft.sampler import DistributedWeightedSampler, check_dataset_agreement


def validate_recipe(cfg):
    combined = parse_combined_cotrain(cfg)
    if (not combined.enabled or combined.order != ('vgen','ae','cotrain')
            or not combined.share_batch or not combined.share_features or combined.warmup_steps):
        raise ValueError('Use the reference shared vgen/ae/cotrain recipe without warmup')
    if cfg.train.gradient_accumulation_steps != combined.grad_accum:
        raise ValueError('gradient_accumulation_steps must equal the three objectives')
    if not cfg.train.drop_last:
        raise ValueError('Distributed SFT requires drop_last=True and equal full batches on all ranks')
    ts = cfg.cotrain.two_step_cotrain
    if not ts['enabled'] or ts['ref_mode'] != 'gt' or ts['backward_topk'] < 1:
        raise ValueError('This recipe requires two-step GT reference and backward_topk >= 1')
    if (cfg.goal_mode.enabled or cfg.data.use_color_jitter or cfg.data.multi_view_schedule['enabled']
        or cfg.cotrain.cotrain_freeze_vgen['enabled'] or cfg.cotrain.cotrain_video_loss
        or cfg.cotrain.video_force_mixed_path or cfg.cotrain.video_sparse_downsample_factor != 1
        or cfg.cotrain.one_step_start_t['enabled'] or ts.get('sep_loss_enabled',False)
        or cfg.model.gradient_vlm2vgen or cfg.model.frame_concat_conditioning
        or cfg.data.repeat_in_batch != 1 or cfg.data.video_repeat_in_batch != 1
        or cfg.cotrain.expert['enabled']):
        raise ValueError('Configuration enables an objective outside the supplied standalone recipe')
    return combined


@contextlib.contextmanager
def preserve_rng():
    py, np_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def run_validation(model, validator, step, output):
    if validator is None:
        return
    cfg = model.cfg
    inferrer = SimpleNamespace(cfg=cfg, video_diffusion_model=model.model.vgen,
        device=next(model.parameters()).device, dtype=torch.bfloat16,
        SPATIAL_DOWN_RATIO=model.spatial, TEMPORAL_DOWN_RATIO=model.temporal,
        timestep_scale=cfg.cotrain.timestep_scale, vlm_text_only=cfg.model.vlm_text_only,
        vae=model.vae)
    inferrer._vae_encode_mem_future = lambda *args: model.model._vae_encode_mem_future(
        *args, cfg=cfg, vae=model.vae, stats_tracker=model.stats,
        use_video_mixed_path=False, video_sparse_downsample_factor=1)
    was_training = model.training
    model.eval()
    try:
        with preserve_rng():
            metrics = validator.evaluate(inferrer,step,video_frame_indices=model.frame_indices)
        if not all(np.isfinite(v) for v in metrics.values()):
            raise FloatingPointError('Non-finite fixed validation loss')
        record = dict(event='validation',step=step,metrics=metrics)
        with (output/'validation.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-link', help='Publish each complete inference checkpoint at this path')
    parser.add_argument('--steps', type=int, help='Final global step; defaults to config train_steps')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-every', type=int)
    parser.add_argument('--resume', help='DeepSpeed training-state directory')
    parser.add_argument('--data-only', action='store_true')
    parser.add_argument('--skip-validation', action='store_true', help='Explicit smoke-test override')
    parser.add_argument('--gradient-checkpointing', action='store_true', help='Explicit memory-saving override for smoke tests')
    parser.add_argument('--skip-save', action='store_true', help='Explicit smoke-test override; no output checkpoint')
    parser.add_argument('--synchronize-timing', action='store_true', help='Synchronize CUDA for measured training and data-plus-training times')
    parser.add_argument('--overwrite', action='store_true', help='Archive an existing output before starting a new run')
    args = parser.parse_args()
    if args.steps is not None and args.steps < 1:
        raise ValueError('steps must be positive')
    output = Path(args.output).expanduser().resolve()
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Training output must be outside the policy checkout')
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world = int(os.environ.get('WORLD_SIZE', '1'))
    settings.local_rank = local_rank
    torch.cuda.set_device(local_rank)
    # Model initialization is rank-identical; sampling and training noise use
    # separate rank-specific generators below, as in the reference trainer.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    spec = importlib.util.spec_from_file_location('robotwin_sft_config', args.config)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = module.make_config()
    combined = validate_recipe(cfg)
    control_group = initialize(cfg.train.nccl_timeout)
    torch.backends.cuda.matmul.allow_tf32 = cfg.train.allow_tf32
    torch.backends.cudnn.allow_tf32 = cfg.train.allow_tf32
    initial_step = int(cfg.train.initial_global_step)
    if args.gradient_checkpointing:
        cfg.model.gradient_checkpointing = True
    save_every = args.save_every or cfg.train.steps_to_save
    if save_every < 1 or cfg.validation.steps_to_val < 1:
        raise ValueError('Save and validation intervals must be positive')
    dataset = on_all_ranks(lambda: make_dataset(cfg), control_group)
    batch_size = args.batch_size if args.batch_size is not None else cfg.data.batch_size
    workers = args.workers if args.workers is not None else cfg.data.dataloader_num_workers
    if batch_size < 1 or workers < 0:
        raise ValueError('batch-size must be positive and workers nonnegative')
    weights = on_all_ranks(dataset.get_sampling_weights, control_group)
    check_dataset_agreement(weights, control_group)
    sampler = on_all_ranks(lambda: DistributedWeightedSampler(weights, batch_size,
        rank=rank, world=world, seed=args.seed,
        balanced=cfg.data.balanced_local_weighted_sampling), control_group)
    loader = DataLoader(dataset,batch_size=batch_size,sampler=sampler,num_workers=workers,
        pin_memory=cfg.data.pin_memory, drop_last=cfg.train.drop_last,
        generator=torch.Generator().manual_seed(args.seed + rank),
        **({'multiprocessing_context':cfg.data.multiprocessing_context,
            'prefetch_factor':cfg.data.dataloader_prefetch_factor,
            'persistent_workers':cfg.data.persistent_workers,
            'timeout':cfg.data.dataloader_timeout} if workers else {}))
    if args.steps is None:
        args.steps = cfg.train.train_steps
    if args.steps is None:
        # Match the original trainer's epoch-derived step limit. Explicit
        # train_steps takes precedence, including with combined objectives.
        args.steps = cfg.train.train_epochs * math.ceil(len(loader) / combined.grad_accum)
    if initial_step < 0 or (not args.resume and args.steps <= initial_step):
        raise ValueError('Final step must exceed the nonnegative initial_global_step')
    def first_batch():
        begin = time.monotonic()
        iterator = iter(loader)
        first = next(iterator)
        print(json.dumps({'event':'first_batch_local_ready','rank':rank,
            'seconds':time.monotonic()-begin}), flush=True)
        return iterator, first
    iterator, first = on_all_ranks(first_batch, control_group)
    print(json.dumps({'event':'data_ready','physical_episodes':dataset.episode_metadata.physical_episode_count,
        'video_shape':list(first['video'].shape),'action_shape':list(first['actions'].shape),
        'body_emb_id':first['body_emb_id'].tolist(),'world_size':world}),flush=True)
    if args.data_only:
        dist.destroy_process_group()
        return
    cfg.train.train_steps = args.steps
    flow = build_WAM_policy_task_flow(cfg=cfg, cotrain_schedule=None)
    model = Cotrain(flow,cfg)
    gradient_probes = {}
    def probe(name):
        def capture(gradient):
            gradient_probes[name] = float(gradient.detach().float().norm())
        return capture
    model.model.vgen.proj_out.weight.register_hook(probe('vgen_head_grad_norm'))
    model.model.ae.action_proj_out.weight.register_hook(probe('action_head_grad_norm'))
    import deepspeed
    groups = [dict(params=[p for p in model.model.vgen.parameters() if p.requires_grad], lr=cfg.train.vgen_lr),
              dict(params=[p for p in model.model.ae.parameters() if p.requires_grad], lr=cfg.train.idm_lr)]
    optimizer = torch.optim.AdamW(groups,lr=cfg.train.lr,betas=(cfg.train.beta1,cfg.train.beta2),
        eps=cfg.train.epsilon,weight_decay=cfg.train.weight_decay)
    scheduler = get_scheduler(cfg.train.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=cfg.train.lr_warmup_steps,num_training_steps=args.steps)
    ds_config = dict(cfg.train.deepspeed)
    ds_config.update(train_micro_batch_size_per_gpu=batch_size,
        gradient_accumulation_steps=combined.grad_accum,
        train_batch_size=batch_size*world*combined.grad_accum,
        zero_allow_untested_optimizer=True,steps_per_print=1000)
    engine, _, _, _ = deepspeed.initialize(model=model,optimizer=optimizer,
        lr_scheduler=scheduler,config=ds_config,dist_init_required=False)
    output_lock = []
    def create_output():
        output.parent.mkdir(parents=True, exist_ok=True)
        handle = (output.parent / f'.{output.name}.lock').open('a')
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output_lock.append(handle)
        if output.exists() and args.overwrite:
            if args.resume and Path(args.resume).resolve().is_relative_to(output):
                raise ValueError('Resume into a new output, not the directory containing the source checkpoint')
            output.rename(output.with_name(output.name + '.previous-' + datetime.now().strftime('%Y%m%d-%H%M%S-%f')))
        output.mkdir(exist_ok=False)
    on_rank_zero(create_output, control_group)
    on_all_ranks(lambda: output.stat(), control_group)
    # Weight-only initialization keeps the configured global numbering while
    # starting a fresh optimizer/warmup, matching the reference trainer.
    start = initial_step
    if args.resume:
        loaded, state = engine.load_checkpoint(args.resume)
        if loaded is None: raise RuntimeError('Could not load requested training checkpoint')
        start = int(state['step'])
        if start >= args.steps:
            raise ValueError('--steps must exceed the restored optimizer step')
    generator = torch.Generator(device=f'cuda:{local_rank}').manual_seed(args.seed+rank)
    validator = None if args.skip_validation else FixedValLoss.from_config(cfg)
    def save_run_settings():
        # Paths remain in external runtime files; record only effective non-path overrides.
        (output/'run_settings.json').write_text(json.dumps(dict(batch_size=batch_size,world_size=world,
            config_name=Path(args.config).name,config_sha256=hashlib.sha256(Path(args.config).read_bytes()).hexdigest(),
            objectives=list(combined.order), weights=combined.weights, gradient_accumulation_steps=combined.grad_accum,
            seed=args.seed,skip_validation=args.skip_validation,skip_save=args.skip_save,
            initial_global_step=initial_step,start_global_step=start,final_global_step=args.steps,
            synchronized_timing=args.synchronize_timing,
            nccl_timeout=cfg.train.nccl_timeout,train_epochs=cfg.train.train_epochs,
            local_batches_per_epoch=len(loader),balanced_local_weighted_sampling=cfg.data.balanced_local_weighted_sampling,
            gradient_checkpointing=cfg.model.gradient_checkpointing),indent=2))
    on_rank_zero(save_run_settings, control_group)
    if cfg.validation.run_validation_at_start:
        on_rank_zero(lambda: run_validation(model,validator,start,output), control_group)
    # Optimizer/scheduler/weights resume; sampling and augmentation restart.
    log = open(output/f'train_rank{rank}.jsonl','w')
    try:
        for index in range(start+1,args.steps+1):
            if args.synchronize_timing:
                torch.cuda.synchronize()
            iteration_begin=time.monotonic()
            batch = first if index == start+1 else next(iterator)
            begin=time.monotonic()
            gradient_probes.clear()
            model.clear_window()
            losses={}
            for micro in build_micro_steps(index-1,combined):
                loss=engine(batch,generator,index-1,micro.name)
                losses[micro.name]=loss.item()
                engine.backward(loss*micro.weight)
                engine.step()
            model.clear_window()
            if args.synchronize_timing:
                torch.cuda.synchronize()
            end=time.monotonic()
            record={'step':index,'losses':losses,'seconds':end-begin,
                'data_seconds':begin-iteration_begin,'iteration_seconds':end-iteration_begin,
                'grad_norm':float(engine.get_global_grad_norm()),'learning_rates':engine.get_lr(),
                'gpu_peak_bytes':torch.cuda.max_memory_allocated()}
            record.update(gradient_probes)
            record.update({k:float(v) for k,v in model.diagnostics.items()})
            record['losses_global_mean'] = mean_losses(losses, torch.device('cuda',local_rank))
            if not np.isfinite(record['grad_norm']):
                raise FloatingPointError('Non-finite gradient norm')
            log.write(json.dumps(record)+'\n');log.flush()
            if rank==0: print(json.dumps(record),flush=True)
            if not args.skip_save and (index % save_every == 0 or index == args.steps):
                on_rank_zero(lambda: save_checkpoint(model,output,index,args.checkpoint_link), control_group)
                engine.save_checkpoint(str(output/'training_state'),tag=f'step_{index}',client_state={'step':index})
            if index % cfg.validation.steps_to_val == 0:
                on_rank_zero(lambda: run_validation(model,validator,index,output), control_group)
        if rank == 0:
            print(json.dumps({'event':'training_complete','step':index}),flush=True)
    finally:
        log.close()
        dist.destroy_process_group()
        for handle in output_lock:
            handle.close()


if __name__ == '__main__':
    main()

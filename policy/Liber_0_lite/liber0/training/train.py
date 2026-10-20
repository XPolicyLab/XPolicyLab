import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from ..backbone import _configure_ieee_float32_matmul
from ..runtime import Policy
from .data import BatchStream, RoboDojoDataset
from .objective import TrainingModel
from .state import build_optimizer, build_scheduler, capture_rng, restore_checkpoint, save_checkpoint


def configure_training(model, checkpointing, compile_dit):
    model.dit.requires_grad_(True)
    model.backbone.lm_head.requires_grad_(False)
    if checkpointing:
        model.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.backbone.gradient_checkpointing_disable()
    if compile_dit:
        torch._inductor.config.emulate_precision_casts = True
        for layer in model.backbone.model.language_model.layers:
            layer.forward = torch.compile(layer.forward, fullgraph=False, dynamic=True, mode='default')
    model.train()
    model.backbone.lm_head.eval()


def parser():
    result = argparse.ArgumentParser()
    result.add_argument('--config', type=Path, required=True)
    result.add_argument('--seed', type=int)
    result.add_argument('--dataset', type=Path, required=True)
    result.add_argument('--checkpoint', type=Path, required=True)
    result.add_argument('--init-weights', type=Path)
    result.add_argument('--assets', type=Path, required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--batch-size', type=int, required=True)
    result.add_argument('--accumulation', type=int, required=True)
    result.add_argument('--num-workers', type=int, required=True)
    result.add_argument('--resume', type=Path)
    result.add_argument('--stop-after', type=int)
    return result


def backward_batch(training, inputs, accumulation, synchronize, autocast=False):
    context = training.no_sync() if not synchronize else nullcontext()
    with context:
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
            loss, values = training(inputs)
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite loss.')
        (loss / accumulation).backward()
    return values


def run(args):
    _configure_ieee_float32_matmul()
    print(sys.executable, torch.__version__, torch.version.cuda,
          os.environ['TORCH_ALLOW_TF32_CUBLAS_OVERRIDE'], torch.get_float32_matmul_precision(),
          torch.backends.cuda.matmul.allow_tf32, flush=True)
    if tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])) < (2, 11):
        raise RuntimeError('PyTorch 2.11 or newer is required.')
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires CUDA.')
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group('nccl')
    cfg = yaml.safe_load(args.config.read_text())
    if args.seed is not None:
        cfg['seed'] = args.seed
    if min(args.batch_size, args.accumulation) <= 0 or args.num_workers < 0:
        raise ValueError('Invalid batch size, accumulation or worker count.')
    effective_batch = world * args.batch_size * args.accumulation
    if effective_batch != cfg['global_batch_size']:
        raise ValueError(f'Global batch must be {cfg["global_batch_size"]}, got {effective_batch}.')
    random.seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    torch.manual_seed(cfg['seed'] + rank)
    checkpoint, dataset_root = args.checkpoint.resolve(), args.dataset.resolve()
    text = (checkpoint / 'config.yaml').read_text()
    stats = (checkpoint / 'dataset_stats.json').read_text()
    inference = yaml.safe_load(text)['inference']
    metadata = ['info.json', 'episodes.jsonl', 'tasks.jsonl']
    identity = {name: hashlib.sha256((dataset_root / 'meta' / name).read_bytes()).hexdigest() for name in metadata}
    cfg = dict(cfg, batch_size=args.batch_size, accumulation=args.accumulation, world_size=world,
               dataset=str(dataset_root), dataset_metadata=identity, assets=str(args.assets.resolve()),
               initializer=str(checkpoint), inference_config_sha256=hashlib.sha256(text.encode()).hexdigest(),
               stats_sha256=hashlib.sha256(stats.encode()).hexdigest(),
               initialization_weights=None if args.init_weights is None else str(args.init_weights.resolve()))
    if args.resume is not None:
        if json.loads((args.output / 'run.json').read_text()) != cfg:
            raise ValueError('Output run configuration differs from the resume request.')
    elif rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / 'run.json').write_text(json.dumps(cfg, indent=2) + '\n')
    if world > 1:
        dist.barrier()
    policy = Policy(checkpoint, args.assets, f'cuda:{local_rank}', 'model.pt', inference['num_inference_steps'],
                    initialization_weights=args.init_weights)
    model = policy.model
    configure_training(model, cfg['gradient_checkpointing'], cfg['compile_dit'])
    training = TrainingModel(model)
    optimizer = build_optimizer([p for p in training.parameters() if p.requires_grad], cfg)
    scheduler = build_scheduler(optimizer, cfg)
    progress = dict(step=0, epoch=0, batch=0)
    if args.resume is not None:
        progress = restore_checkpoint(args.resume, model, optimizer, scheduler,
                                      expected_config=cfg, rank=rank, world_size=world)
    if world > 1:
        training = DistributedDataParallel(training, device_ids=[local_rank], find_unused_parameters=True,
                                          bucket_cap_mb=200, gradient_as_bucket_view=True)
    data = RoboDojoDataset(dataset_root, json.loads(stats), inference,
                          seed=cfg['seed'], training=True, color=cfg['color'])
    stream = BatchStream(data, batch_size=args.batch_size, rank=rank, world_size=world,
                         seed=cfg['seed'], workers=args.num_workers, progress=progress, pin_memory=True)
    stop = cfg['max_steps'] if args.stop_after is None else args.stop_after
    if not progress['step'] < stop <= cfg['max_steps']:
        raise ValueError('stop-after must be after the resumed step and no greater than max_steps.')
    while progress['step'] < stop:
        optimizer.zero_grad(set_to_none=True)
        metrics = torch.zeros(2, device=model.device)
        for micro in range(args.accumulation):
            values = backward_batch(training, next(stream), args.accumulation,
                                    synchronize=(world == 1 or micro == args.accumulation - 1), autocast=True)
            metrics += torch.stack([values['loss_video'], values['loss_action']]) / args.accumulation
        grad_norm = torch.nn.utils.clip_grad_norm_(training.parameters(), cfg['max_grad_norm'], error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        progress['step'] += 1
        if world > 1:
            dist.all_reduce(metrics)
            metrics /= world
        if rank == 0 and (progress['step'] % cfg['log_every'] == 0 or progress['step'] == stop):
            print(json.dumps(dict(progress, loss_video=metrics[0].item(), loss_action=metrics[1].item(),
                                  grad_norm=float(grad_norm), lr=optimizer.param_groups[0]['lr'])), flush=True)
        if progress['step'] % cfg['save_every'] == 0 or progress['step'] == stop:
            states = [None] * world
            if world > 1:
                dist.all_gather_object(states, capture_rng(model.device))
            else:
                states[0] = capture_rng(model.device)
            if rank == 0:
                save_checkpoint(args.output / f'step_{progress["step"]:06d}', model, optimizer, scheduler,
                                progress=progress, config=cfg, rng_states=states, inference_config=text, stats=stats)
            if world > 1:
                dist.barrier()
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    run(parser().parse_args())

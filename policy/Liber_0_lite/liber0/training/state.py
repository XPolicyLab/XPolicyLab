import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from ..weights import CONTRACT


def build_optimizer(parameters, config):
    return torch.optim.AdamW(parameters, lr=config['learning_rate'],
                            weight_decay=config['weight_decay'], betas=(.9, .95), fused=False)


def build_scheduler(optimizer, config):
    total, warmup = config['max_steps'], config['warmup_steps']
    if not 0 <= warmup < total:
        raise ValueError('Expected 0 <= warmup_steps < max_steps.')
    minimum = config['min_learning_rate']
    if minimum is None:
        minimum = config['learning_rate'] * .01
    cosine = CosineAnnealingLR(optimizer, T_max=total - warmup, eta_min=minimum)
    if warmup == 0:
        return cosine
    return SequentialLR(optimizer, [LinearLR(optimizer, start_factor=1 / warmup,
                                            end_factor=1., total_iters=warmup), cosine], [warmup])


def capture_rng(device=torch.device('cpu')):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if torch.device(device).type == 'cuda' else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def save_checkpoint(destination, model, optimizer, scheduler, *, progress, config,
                    rng_states, inference_config, stats):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + '.pending')
    temporary.mkdir(parents=True, exist_ok=False)
    if destination.exists():
        raise FileExistsError(destination)
    weights = {key: value.detach().cpu() for key, value in model.dit.state_dict().items()}
    torch.save(dict(dit=weights, step=progress['step'], **CONTRACT), temporary / 'model.pt')
    torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    progress=progress, config=config, rng_states=rng_states), temporary / 'state.pt')
    (temporary / 'config.yaml').write_text(inference_config)
    (temporary / 'dataset_stats.json').write_text(stats)
    (temporary / 'training_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (temporary / 'complete.json').write_text(json.dumps(progress) + '\n')
    temporary.rename(destination)


def restore_checkpoint(source, model, optimizer, scheduler, *, expected_config, rank, world_size):
    source = Path(source)
    completed = json.loads((source / 'complete.json').read_text())
    # Resume state is a trusted local training artifact, not a downloaded model.
    state = torch.load(source / 'state.pt', map_location='cpu', weights_only=False)
    if state['config'] != expected_config or len(state['rng_states']) != world_size:
        raise ValueError('Resume configuration or world size differs from the saved run.')
    if state['progress'] != completed:
        raise ValueError('Incomplete training checkpoint.')
    weights = torch.load(source / 'model.pt', map_location='cpu', mmap=True, weights_only=True)
    if weights['step'] != completed['step'] or any(weights[key] != value for key, value in CONTRACT.items()):
        raise ValueError('Checkpoint weights and state do not match.')
    model.dit.load_state_dict(weights['dit'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    restore_rng(state['rng_states'][rank])
    return state['progress']

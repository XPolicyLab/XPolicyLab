from typing import Dict, Callable, Optional
import os
import random
import torch
import numpy as np

def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', os.environ.get('LOCAL_RANK', '0'))))

def set_global_seed(seed: int, get_worker_init_fn: bool=False) -> Optional[Callable[[int], None]]:
    assert np.iinfo(np.uint32).min < seed < np.iinfo(np.uint32).max, 'Seed outside the np.uint32 bounds!'
    os.environ['EXPERIMENT_GLOBAL_SEED'] = str(seed)
    global_rank = _resolve_global_rank()
    process_seed = seed + global_rank
    random.seed(process_seed)
    np.random.seed(process_seed)
    torch.manual_seed(process_seed)
    return worker_init_function if get_worker_init_fn else None

def worker_init_function(worker_id: int) -> None:
    process_seed = torch.initial_seed()
    global_rank = _resolve_global_rank()
    base_seed = process_seed - worker_id
    seed_seq = np.random.SeedSequence([base_seed, worker_id, global_rank])
    np.random.seed(seed_seq.generate_state(4))
    torch_seed_seq, random_seed_seq, tf_seed_seq = seed_seq.spawn(3)
    torch.manual_seed(torch_seed_seq.generate_state(1, dtype=np.uint64)[0])
    random_seed = (random_seed_seq.generate_state(2, dtype=np.uint64).astype(list) * [1 << 64, 1]).sum()
    random.seed(random_seed)

def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result

def _staged_randn(shape, generator, rand_device, out_device, out_dtype):
    pin = str(rand_device) == 'cpu' and torch.accelerator.is_available()
    t = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32, pin_memory=pin)
    return t.to(device=out_device, dtype=out_dtype, non_blocking=pin)

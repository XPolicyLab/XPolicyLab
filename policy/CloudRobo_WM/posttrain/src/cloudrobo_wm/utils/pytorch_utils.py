import os,random
import numpy as np
import torch

def _resolve_global_rank() -> int:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return int(torch.distributed.get_rank())
    return int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', os.environ.get('LOCAL_RANK', '0'))))

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

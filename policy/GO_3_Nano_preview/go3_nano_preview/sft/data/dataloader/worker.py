import os
import torch
import torch.distributed as dist
from torch.utils.data import get_worker_info
from policy_core.settings import settings

def accv_worker_init_fn(worker_id):
    if dist.is_available() and dist.is_initialized():
        local_rank = dist.get_rank() % torch.cuda.device_count()
    else:
        local_rank = settings.local_rank
    torch.cuda.set_device(local_rank)
    info = get_worker_info()
    if info is not None and hasattr(info, 'dataset'):
        dataset = info.dataset
        while dataset is not None and (not hasattr(dataset, 'gpuid')) and hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        if dataset is not None and hasattr(dataset, 'gpuid'):
            dataset.gpuid = local_rank
            if hasattr(dataset, 'decoder_manager'):
                del dataset.decoder_manager
    print(f'[INFO] DataLoader worker {worker_id} PID {os.getpid()} -> GPU {local_rank}', flush=True)

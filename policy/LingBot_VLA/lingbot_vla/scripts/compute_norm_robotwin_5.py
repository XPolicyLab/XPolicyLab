import json
import numpy as np
import os
import re
import time
from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import torch
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from lingbotvla.models import build_processor
from lingbotvla.utils import helper
from lingbotvla.utils.arguments import DataArguments, ModelArguments, TrainingArguments, parse_args
import lingbotvla.utils.normalize as normalize
from lingbotvla.data.vla_data.base_dataset import VlaDataset


if TYPE_CHECKING:
    from transformers import ProcessorMixin

    from lingbotvla.data.chat_template import ChatTemplate

logger = helper.create_logger(__name__)


@dataclass
class MyDataArguments(DataArguments):
    norm_path: str = field(
        default=None,
        metadata={"help": "Path to save norm stats."},
    )
    chunk_size: int = field(
        default=50,
        metadata={"help": "Chunk size of action."},
    )


@dataclass
class Arguments:
    model: "ModelArguments" = field(default_factory=ModelArguments)
    data: "MyDataArguments" = field(default_factory=MyDataArguments)
    train: "TrainingArguments" = field(default_factory=TrainingArguments)

def compute_norm(dataset, task_id, batch_size, stats, state_norm_keys, acton_norm_keys, delta_norm):
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=16, shuffle=False, drop_last=True)
    success = True
    try:
        for batch in tqdm(data_loader, desc=f"Computing stats of {task_id}"):
            for key in state_norm_keys:
                values = np.asarray(batch[key])
                # values = batch[key]
                stats[key].update(values.reshape(-1, values.shape[-1]))
            for key in acton_norm_keys:
                values = np.asarray(batch[key][:,0]) if not delta_norm[key] else np.asarray(batch[key].reshape(batch[key].shape[0], -1))
                stats[key].update(values.reshape(-1, values.shape[-1]))
    except: success = False
    return success


def main():
    args = parse_args(Arguments)
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")
    logger.info_rank0(json.dumps(asdict(args), indent=2))
    
    logger.info_rank0("Prepare data")
    stats = None
    
    assert args.data.datasets_type == 'vla'
    dataset = VlaDataset(repo_id=args.data.train_path, action_name='action')

    state_norm_keys = ['observation.state']
    acton_norm_keys = ['action']
    delta_norm = {'action': False} # all action dims do not need to minus state in Robotwin
    stats = {key: normalize.RunningStats() for key in acton_norm_keys+state_norm_keys}
    
    chunk_size = args.data.chunk_size

    try:
        success = compute_norm(dataset, args.data.train_path, args.train.global_batch_size, stats, state_norm_keys, acton_norm_keys, delta_norm)
    except Exception as e:
        fail_info = f"{args.data.train_path} {e}"
        print(fail_info)
        


    if success:
        norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}
        norm_stats = {}
        for key, stats in stats.items():
            if key in delta_norm and delta_norm[key]==True:
                norm_stats[key] = stats.get_statistics(chunk_size=chunk_size)
            else:
                norm_stats[key] = stats.get_statistics()

        output_path = Path(args.data.norm_path)
        print(f"Writing stats to: {output_path}")
        normalize.save(output_path, norm_stats, stats._count)
        


if __name__ == "__main__":
    main()
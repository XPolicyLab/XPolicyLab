import logging
import math
import os
from pathlib import Path
from typing import Dict, Optional
from datetime import timedelta

import hydra
import numpy as np

from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration

import torch
from torch.utils.data import DataLoader
from transformers.utils.versions import require_version

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from galaxea_fm.processors.base_processor import BaseProcessor
from galaxea_fm.data.base_lerobot_dataset import BaseLerobotDataset
from galaxea_fm.models.base_policy import BasePolicy
from galaxea_fm.utils.logging_config import (
    log_allocated_gpu_memory,
    log_amp_config,
    setup_logging,
)
from galaxea_fm.utils.pytorch_utils import dict_apply, set_global_seed
from galaxea_fm.utils.train_utils import init_experiment_tracker
from galaxea_fm.utils.load_pretrained_resumed import load_checkpoint_for_eval
from galaxea_fm.utils.config_resolvers import register_default_resolvers
from galaxea_fm.utils.tqdm import tqdm

register_default_resolvers()
logger = get_logger(__name__)
require_version("datasets==3.6.0", "To fix: uv pip install datasets==3.6.0")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from galaxea_fm.utils.visualize import plot_result

def dict_to_array(x):
    data = np.concatenate([item for _, item in x.items()], axis=-1)
    return data


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    OmegaConf.resolve(cfg)
    output_dir = Path(cfg.output_dir)

    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    project_config = ProjectConfiguration(project_dir=str(output_dir))
    init_process_group_kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=2))
    accelerator = Accelerator(
        mixed_precision="bf16" if cfg.model.enable_bf16_training else "no",
        project_config=project_config,
        kwargs_handlers=[init_process_group_kwargs],
        log_with=cfg.logger.type,
    )
    torch.cuda.set_device(device_id := accelerator.local_process_index)
    torch.cuda.empty_cache()

    setup_logging(log_level=logging.INFO, is_main_process=accelerator.is_main_process)
    logger.info(f"Output directory: {output_dir}")
    log_amp_config(logger, accelerator)
    init_experiment_tracker(cfg, accelerator, output_dir)

    set_global_seed(cfg.seed, get_worker_init_fn=False)

    # Load model (supports both legacy .pt and new directory formats)
    model: BasePolicy = instantiate(cfg.model.model_arch)
    model, dataset_stats = load_checkpoint_for_eval(cfg.ckpt_path, model, device="cpu")
    policy = model.cuda().eval()
    log_allocated_gpu_memory(logger, stage="loading model", device=0)

    processor: BaseProcessor = instantiate(cfg.data.processor)
    processor.set_normalizer_from_stats(dataset_stats)
    processor.eval()

    # Set tokenizer for Pi0FastPolicy (autoregressive models need tokenizer for action decoding)
    if hasattr(policy, 'set_tokenizer') and hasattr(processor, 'tokenizer'):
        policy.set_tokenizer(processor.tokenizer)
    
    dataset_val: BaseLerobotDataset = instantiate(cfg.data.dataset, is_training_set=False)
    dataset_val.set_processor(processor)
    
    dataloader = DataLoader(
        dataset_val, 
        shuffle=False, 
        batch_size=cfg.batch_size_val, 
        num_workers=cfg.model.num_workers, 
        pin_memory=cfg.model.pin_memory, 
        persistent_workers=cfg.model.persistent_workers, 
        worker_init_fn=None, 
    )
    episode_from = dataset_val.episode_data_index["from"]
    episode_to = dataset_val.episode_data_index["to"]
    num_episodes = len(episode_from)

    if cfg.get("eval_episodes_num"):
        eval_episodes_num = cfg.eval_episodes_num
    else:
        eval_episodes_num = num_episodes
    eval_end_frame = episode_to[eval_episodes_num - 1]

    gt_actions = []
    pd_actions = []
    for i, batch in tqdm.tqdm(enumerate(dataloader), desc="inferencing", total=len(dataloader)):
        batch = dict_apply(batch, lambda x: x.cuda() if isinstance(x, torch.Tensor) else x)
        with torch.no_grad():
            batch = policy.predict_action(batch)

        batch = dict_apply(batch, lambda x: x.cpu() if isinstance(x, torch.Tensor) else x)
        batch = processor.postprocess(batch)
        cur_pd_action = dict_apply(batch["action"], lambda x: x.cpu().numpy())
        cur_gt_action = dict_apply(batch["gt_action"], lambda x: x.cpu().numpy())

        pd_actions.append(dict_to_array(cur_pd_action))
        gt_actions.append(dict_to_array(cur_gt_action))
        if i * cfg.batch_size_val >= eval_end_frame:
            break

    pd_actions = np.concatenate(pd_actions, axis=0)
    gt_actions = np.concatenate(gt_actions, axis=0)[:, 0, :]


    for idx in range(eval_episodes_num):
        cur_path = output_dir / f"{idx:06}"
        cur_path.mkdir(exist_ok=True)
        cur_pd_action = pd_actions[episode_from[idx]: episode_to[idx]]
        cur_gt_action = gt_actions[episode_from[idx]: episode_to[idx]]
        plot_result(cur_path, cur_gt_action, cur_pd_action)


if __name__ == "__main__":
    main()

import logging, os
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from .models.wan22.helpers.io import PRETRAINED_ROOT_ENV
from .trainer import Wan22Trainer
from .utils.logging_config import get_logger, setup_logging
from .utils.config_io import save_training_config
from .utils.pytorch_utils import set_global_seed

logger = get_logger(__name__)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_train_device() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    device_count = torch.cuda.device_count()
    if device_count <= 1:
        return "cuda:0"
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank < 0 or local_rank >= device_count:
        return "cuda:0"
    return f"cuda:{local_rank}"


def build_datasets(data_cfg):
    train = instantiate(data_cfg.train)
    val_cfg = data_cfg.get("val")
    return (train, None if val_cfg is None else instantiate(val_cfg))


def run_training(cfg, *, source_config_path=None, trainer_cls=Wan22Trainer):
    if cfg.get("paths") is not None:
        # Pretrained video weights are looked up under <weights_root>/<model_id>.
        os.environ[PRETRAINED_ROOT_ENV] = str(cfg.paths.weights_root)
    setup_logging(
        log_level=logging.INFO, is_main_process=int(os.environ.get("RANK", "0")) == 0
    )
    set_global_seed(int(cfg.seed), rank_specific=False)
    os.makedirs(cfg.output_dir, exist_ok=True)
    save_training_config(
        OmegaConf.to_container(cfg, resolve=True),
        cfg.output_dir,
        source_config_path=source_config_path,
        rank=int(os.environ.get("RANK", "0")),
    )
    model = instantiate(
        cfg.model,
        model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision),
        device=_resolve_train_device(),
    )
    (train, val) = build_datasets(cfg.data)
    trainer = trainer_cls(cfg=cfg, model=model, train_dataset=train, val_dataset=val)
    trainer.train()
    return trainer

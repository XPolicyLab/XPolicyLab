from flexwam.config import resolve_config
import logging
import os
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from PIL import Image
import numpy as np
from einops import repeat
from omegaconf import OmegaConf

from flexwam.training.trainer import FlexWAMTrainer
from flexwam.utils.logging_config import get_logger, setup_logging
from flexwam.utils.tmpdir import configure_tmpdir
from flexwam.utils.camera_metrics import configure_camera_metrics
from flexwam.utils.video_io import save_mp4
from flexwam.utils import misc

logger = get_logger(__name__)

configure_tmpdir()


def _normalize_mixed_precision(mixed_precision: str) -> str:
    if not isinstance(mixed_precision, str):
        raise ValueError(f"`mixed_precision` must be str, got {type(mixed_precision)}")
    key = mixed_precision.strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _get_rank_info_from_env() -> tuple[int, int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def _log_local_cuda_diagnostics(device: str, mixed_precision: str) -> None:
    rank, local_rank, world_size = _get_rank_info_from_env()
    logger.info(
        "Runtime precision diag: rank=%d local_rank=%d world_size=%d requested_mixed_precision=%s resolved_device=%s torch=%s cuda=%s cuda_available=%s",
        rank,
        local_rank,
        world_size,
        mixed_precision,
        device,
        torch.__version__,
        torch.version.cuda,
        torch.cuda.is_available(),
    )

    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return

    cuda_device = torch.device(device)
    device_index = cuda_device.index
    if device_index is None:
        device_index = torch.cuda.current_device()

    with torch.cuda.device(device_index):
        device_name = torch.cuda.get_device_name(device_index)
        capability = torch.cuda.get_device_capability(device_index)
        bf16_supported = torch.cuda.is_bf16_supported()

    logger.info(
        "Runtime CUDA device diag: rank=%d device=%s device_index=%d name=%s capability=%s bf16_supported=%s",
        rank,
        device,
        device_index,
        device_name,
        capability,
        bf16_supported,
    )


def _run_bf16_smoke_test_if_enabled(device: str, mixed_precision: str) -> None:
    if _normalize_mixed_precision(mixed_precision) != "bf16":
        return
    if os.environ.get("FLEXWAM_RUN_BF16_SMOKE", "0").strip() != "1":
        return
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return

    rank, _, _ = _get_rank_info_from_env()
    cuda_device = torch.device(device)

    logger.info("Starting bf16 smoke test on rank=%d device=%s", rank, device)
    with torch.cuda.device(cuda_device):
        q = torch.randn((1, 4, 16, 32), device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn((1, 4, 16, 32), device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn((1, 4, 16, 32), device=cuda_device, dtype=torch.bfloat16, requires_grad=True)
        attn_mask = torch.ones((1, 1, 16, 16), device=cuda_device, dtype=torch.bool)

        mm_a = torch.randn((64, 64), device=cuda_device, dtype=torch.bfloat16)
        mm_b = torch.randn((64, 64), device=cuda_device, dtype=torch.bfloat16)
        mm_out = mm_a @ mm_b

        sdpa_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        loss = mm_out.float().mean() + sdpa_out.float().mean()
        loss.backward()
        torch.cuda.synchronize(device=cuda_device)

    logger.info("bf16 smoke test passed on rank=%d device=%s", rank, device)


def build_datasets(data_cfg: DictConfig):
    from flexwam.datasets.lerobot.utils.normalizer import wait_for_dataset_stats_json

    train_ds = instantiate(data_cfg.train)
    if data_cfg.get("val") is None:
        val_ds = train_ds
    else:
        train_stats_path = data_cfg.train.get("pretrained_norm_stats")
        default_stats_path = os.path.join(misc.get_work_dir(), "dataset_stats.json")
        val_stats_path = data_cfg.val.get("pretrained_norm_stats")
        pretrained_norm_stats = val_stats_path or train_stats_path or default_stats_path
        if int(os.environ.get("WORLD_SIZE", "1")) > 1:
            wait_for_dataset_stats_json(pretrained_norm_stats)
        logger.info("Building val dataset with pretrained_norm_stats: %s", pretrained_norm_stats)
        val_ds = instantiate(data_cfg.val, pretrained_norm_stats=pretrained_norm_stats)
    return train_ds, val_ds


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


def run_training(cfg: DictConfig):
    cfg = resolve_config(cfg)
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0 if torch.distributed.is_initialized() else True,
    )
    misc.register_work_dir(cfg.output_dir)
    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with open(Path(cfg.output_dir) / "config.yaml", "w") as f:
        OmegaConf.save(config_payload, f)

    model_device = _resolve_train_device()
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    _log_local_cuda_diagnostics(model_device, mixed_precision)
    _run_bf16_smoke_test_if_enabled(model_device, mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    train_ds, val_ds = build_datasets(cfg.data)

    trainer = FlexWAMTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()

def run_inference(cfg: DictConfig):
    cfg = resolve_config(cfg)
    setup_logging(log_level=logging.INFO)
    inference_cfg = cfg.inference
    mixed_precision = _normalize_mixed_precision(cfg.mixed_precision)
    model_dtype = _mixed_precision_to_model_dtype(mixed_precision)

    model = instantiate(cfg.model, model_dtype=model_dtype, device=str(inference_cfg.device))
    if "data" in cfg and "train" in cfg.data:
        configure_camera_metrics(model, cfg.data.train)
    checkpoint_path = inference_cfg.get("checkpoint_path")
    if checkpoint_path:
        ckpt = Path(checkpoint_path)
        if ckpt.exists():
            logger.info("Loading finetuned checkpoint: %s", checkpoint_path)
            model.load_checkpoint(checkpoint_path)
        else:
            logger.warning("Checkpoint not found, skipping load: %s", checkpoint_path)
    model.eval()
    
    def center_crop_resize(img: Image, width: int, height: int) -> Image.Image:
        src_w, src_h = img.size
        scale = max(width / src_w, height / src_h)
        resized = img.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        return resized.crop((left, top, left + width, top + height))

    input_image = Image.open(str(inference_cfg.input_image_path)).convert("RGB")
    input_image = center_crop_resize(input_image, width=inference_cfg.width, height=inference_cfg.height)
    arr = np.array(input_image, dtype=np.float32)
    x = torch.from_numpy(arr)
    x = x.to(device=model.device, dtype=model.torch_dtype)
    x = x * (2.0 / 255.0) - 1.0
    x = repeat(x, "H W C -> B C H W", B=1)
    output_mp4 = str(inference_cfg.output_mp4)

    infer_kwargs = {
        "prompt": str(inference_cfg.prompt),
        "negative_prompt": str(inference_cfg.negative_prompt),
        "text_cfg_scale": float(inference_cfg.text_cfg_scale),
        "action_cfg_scale": float(inference_cfg.action_cfg_scale),
        "input_image": x,
        "num_frames": int(inference_cfg.num_frames),
        "num_inference_steps": int(inference_cfg.num_inference_steps),
        "sigma_shift": None if inference_cfg.get("sigma_shift") is None else float(inference_cfg.sigma_shift),
        "seed": int(inference_cfg.seed),
        "rand_device": str(inference_cfg.rand_device),
        "tiled": bool(inference_cfg.tiled),
    }

    infer_out = model.infer(**infer_kwargs)
    video = infer_out["video"]
    save_mp4(video, output_mp4, fps=15)
    logger.info("Saved inference video to %s", output_mp4)
    return output_mp4

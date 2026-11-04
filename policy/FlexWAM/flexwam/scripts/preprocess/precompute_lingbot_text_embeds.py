from __future__ import annotations
from flexwam.config import resolve_config

import hashlib
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, ListConfig
from safetensors.torch import load_file
from tqdm import tqdm

from flexwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from flexwam.utils.config_resolvers import register_default_resolvers
from flexwam.utils.logging_config import get_logger, setup_logging

register_default_resolvers()
logger = get_logger(__name__)

DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 8
LINGBOT_ENCODER_ID = "lingbotqwen3vl"


def _iter_dataset_nodes(node: Any, path: str = "data"):
    if isinstance(node, DictConfig):
        if "dataset_dirs" in node and node.get("dataset_dirs") is not None:
            yield path, node
        for key, value in node.items():
            yield from _iter_dataset_nodes(value, f"{path}.{key}")
    elif isinstance(node, ListConfig):
        for idx, value in enumerate(node):
            yield from _iter_dataset_nodes(value, f"{path}[{idx}]")


def _collect_dataset_settings(data_cfg: DictConfig):
    dataset_dirs: list[str] = []
    cache_dirs: list[Path] = []
    context_lens = set()

    for node_path, node in _iter_dataset_nodes(data_cfg, path="data"):
        raw_dirs = node.get("dataset_dirs")
        if raw_dirs is None:
            continue
        cache_dir = node.get("text_embedding_cache_dir")
        if cache_dir is None or not str(cache_dir).strip():
            raise ValueError(f"Missing `text_embedding_cache_dir` for dataset node `{node_path}`.")
        for ds in raw_dirs:
            ds_str = str(ds)
            if ds_str not in dataset_dirs:
                dataset_dirs.append(ds_str)
        cache_dir_path = Path(str(cache_dir)).expanduser()
        if cache_dir_path not in cache_dirs:
            cache_dirs.append(cache_dir_path)
        context_len = node.get("context_len")
        if context_len is not None:
            context_lens.add(int(context_len))
        logger.info("Discovered dataset node `%s` with %d dataset_dirs.", node_path, len(raw_dirs))
    return dataset_dirs, cache_dirs, context_lens


def _resolve_context_len(context_lens: set[int]) -> int:
    if not context_lens:
        return DEFAULT_CONTEXT_LEN
    if len(context_lens) != 1:
        raise ValueError(f"Found multiple context_len values in data config: {sorted(context_lens)}.")
    return next(iter(context_lens))


def _read_unique_prompts(dataset_dirs: list[str]) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0
    for ds_dir in dataset_dirs:
        tasks_path = Path(ds_dir) / "meta" / "tasks.jsonl"
        if not tasks_path.exists():
            raise FileNotFoundError(f"Missing tasks file: {tasks_path}")
        with tasks_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "task" not in record:
                    raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
                prompt = DEFAULT_PROMPT.format(task=str(record["task"]))
                total_task_rows += 1
                if prompt not in seen:
                    seen.add(prompt)
                    prompts.append(prompt)
    logger.info(
        "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(dataset_dirs),
        len(prompts),
    )
    return prompts


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _load_lingbot_qwen(model_path: Path, device: str, dtype: torch.dtype):
    try:
        from transformers import Qwen3VLConfig, Qwen3VLProcessor, Qwen3VLTextModel
    except ImportError as exc:
        raise ImportError(
            "The installed transformers package does not provide the Qwen3VL text-encoder classes. "
            "Install a transformers release that includes Qwen3VL, for example `transformers==4.57.1`."
        ) from exc

    processor = Qwen3VLProcessor.from_pretrained(str(model_path / "tokenizer"), trust_remote_code=True)
    config = Qwen3VLConfig.from_pretrained(str(model_path / "tokenizer"), trust_remote_code=True).text_config
    text_encoder = Qwen3VLTextModel(config)
    state_dict = _load_lingbot_qwen_text_state_dict(model_path / "text_encoder")
    missing, unexpected = text_encoder.load_state_dict(state_dict, strict=False)
    unexpected = [key for key in unexpected if not key.startswith("visual.")]
    if missing or unexpected:
        raise ValueError(
            "LingBot/Qwen text encoder state_dict mismatch after fused-key conversion: "
            f"missing={list(missing)[:20]} unexpected={list(unexpected)[:20]}"
        )
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()
    return processor, text_encoder


def _load_lingbot_qwen_text_state_dict(text_encoder_dir: Path) -> dict[str, torch.Tensor]:
    index_path = text_encoder_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        raw_state: dict[str, torch.Tensor] = {}
        for shard_name in shard_names:
            raw_state.update(load_file(str(text_encoder_dir / shard_name), device="cpu"))
    else:
        raw_state = load_file(str(text_encoder_dir / "model.safetensors"), device="cpu")

    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if key.endswith(".self_attn.qkv_proj.weight"):
            prefix = key.removesuffix(".self_attn.qkv_proj.weight")
            q, k, v = value.split([4096, 1024, 1024], dim=0)
            state[f"{prefix}.self_attn.q_proj.weight"] = q.contiguous()
            state[f"{prefix}.self_attn.k_proj.weight"] = k.contiguous()
            state[f"{prefix}.self_attn.v_proj.weight"] = v.contiguous()
        elif key.endswith(".mlp.gate_up_proj.weight"):
            prefix = key.removesuffix(".mlp.gate_up_proj.weight")
            gate, up = value.chunk(2, dim=0)
            state[f"{prefix}.mlp.gate_proj.weight"] = gate.contiguous()
            state[f"{prefix}.mlp.up_proj.weight"] = up.contiguous()
        else:
            state[key] = value
    return state


def _extract_context(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("prompt_embeds", "last_hidden_state"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
        hidden_states = output.get("hidden_states")
        if hidden_states is not None:
            return hidden_states[-1]
    for attr in ("prompt_embeds", "last_hidden_state"):
        value = getattr(output, attr, None)
        if isinstance(value, torch.Tensor):
            return value
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        return hidden_states[-1]
    raise ValueError(f"Cannot extract text context from LingBot/Qwen output type: {type(output)}")


def _encode_batch(processor, text_encoder, prompts: list[str], context_len: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = processor(
        text=prompts,
        padding="max_length",
        max_length=context_len,
        truncation=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    mask = inputs.get("attention_mask")
    if mask is None:
        raise ValueError("LingBot/Qwen processor did not return `attention_mask`.")
    with torch.no_grad():
        try:
            output = text_encoder(**inputs, output_hidden_states=True)
        except TypeError:
            output = text_encoder(**inputs)
    context = _extract_context(output)
    if context.ndim != 3:
        raise ValueError(f"LingBot/Qwen context must be [B,L,D], got {tuple(context.shape)}")
    if context.shape[1] != context_len:
        raise ValueError(f"LingBot/Qwen context_len mismatch: expected {context_len}, got {context.shape[1]}")
    return context, mask.bool()


@hydra.main(config_path="../../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig):
    cfg = resolve_config(cfg)
    setup_logging(log_level=logging.INFO)
    if cfg.data is None:
        raise ValueError("`cfg.data` is required.")
    model_path = Path(str(cfg.model.get("model_path", "lingbot-video"))).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"LingBot model_path does not exist: {model_path}")

    overwrite = bool(cfg.get("overwrite", True))
    dataset_dirs, cache_dirs, context_lens = _collect_dataset_settings(cfg.data)
    if not cache_dirs:
        raise ValueError("No `text_embedding_cache_dir` found under `cfg.data`.")
    context_len = _resolve_context_len(context_lens)
    prompts = _read_unique_prompts(dataset_dirs)
    if not prompts:
        logger.warning("No prompts found; nothing to encode.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    logger.info(
        "Preparing LingBot/Qwen text encoder model_path=%s device=%s dtype=%s context_len=%d overwrite=%s",
        model_path,
        device,
        dtype,
        context_len,
        overwrite,
    )
    processor, text_encoder = _load_lingbot_qwen(model_path, device=device, dtype=dtype)

    stats = {str(cache_dir): {"new": 0, "overwrite": 0, "skip": 0} for cache_dir in cache_dirs}
    with tqdm(total=len(prompts), desc="Encoding LingBot/Qwen prompts", unit="prompt", dynamic_ncols=True) as pbar:
        for start in range(0, len(prompts), DEFAULT_BATCH_SIZE):
            batch_prompts = prompts[start : start + DEFAULT_BATCH_SIZE]
            context, mask = _encode_batch(processor, text_encoder, batch_prompts, context_len, device)
            for i, prompt in enumerate(batch_prompts):
                hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                payload = {
                    "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                }
                filename = f"{hashed}.text_len{context_len}.{LINGBOT_ENCODER_ID}.pt"
                for cache_dir in cache_dirs:
                    cache_path = cache_dir / filename
                    key = str(cache_dir)
                    if cache_path.exists() and not overwrite:
                        stats[key]["skip"] += 1
                        continue
                    if cache_path.exists():
                        stats[key]["overwrite"] += 1
                    else:
                        stats[key]["new"] += 1
                    _atomic_torch_save(payload, cache_path)
            pbar.update(len(batch_prompts))

    for cache_dir in cache_dirs:
        key = str(cache_dir)
        logger.info(
            "Cache dir: %s | new=%d overwrite=%d skip=%d",
            key,
            stats[key]["new"],
            stats[key]["overwrite"],
            stats[key]["skip"],
        )


if __name__ == "__main__":
    main()

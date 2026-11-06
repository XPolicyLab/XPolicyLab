#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import logging
import os
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from tqdm import tqdm

from lerobot.datasets.utils import DEFAULT_TASKS_PATH, LEGACY_TASKS_PATH, load_tasks
from lerobot.policies.fastwam.core.models.wan22.helpers.loader import _load_registered_model, _resolve_configs
from lerobot.policies.fastwam.core.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer
from lerobot.policies.fastwam.text_cache import build_fastwam_prompt, build_text_embedding_cache_path

DEFAULT_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
DEFAULT_TOKENIZER_MODEL_ID = "Wan-AI/Wan2.1-T2V-1.3B"
DEFAULT_CONTEXT_LEN = 128
DEFAULT_BATCH_SIZE = 16


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def _parse_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    if value == "float32":
        return torch.float32
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}. Expected one of: float32, float16, bfloat16.")


def _init_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 1, 0

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    if not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")

    return True, dist.get_rank(), dist.get_world_size(), local_rank


def _resolve_dataset_dirs(dataset_dirs: list[str], repo_id_file: str | None) -> list[str]:
    resolved = [str(Path(ds).expanduser()) for ds in dataset_dirs]
    if repo_id_file:
        repo_path = Path(repo_id_file).expanduser()
        if not repo_path.is_file():
            raise FileNotFoundError(f"repo_id_file does not exist: {repo_path}")
        with repo_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                ds_dir = line.strip()
                if ds_dir:
                    resolved.append(str(Path(ds_dir).expanduser()))
    deduped: list[str] = []
    seen = set()
    for ds_dir in resolved:
        if ds_dir not in seen:
            seen.add(ds_dir)
            deduped.append(ds_dir)
    if not deduped:
        raise ValueError("Provide at least one `--dataset-dir` or `--repo-id-file`.")
    return deduped


def _read_task_strings_from_parquet(dataset_dir: Path) -> list[str]:
    tasks = load_tasks(dataset_dir)
    if "task" in tasks.columns:
        task_values = tasks["task"].tolist()
    else:
        task_values = tasks.index.tolist()

    result = []
    for task in task_values:
        task = str(task).strip()
        if task:
            result.append(task)
    return result


def _read_task_strings_from_jsonl(tasks_path: Path) -> list[str]:
    result = []
    with tasks_path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if "task" not in record:
                raise KeyError(f"Missing `task` field at {tasks_path}:{line_idx}")
            task = str(record["task"]).strip()
            if task:
                result.append(task)
    return result


def _read_task_strings(dataset_dir: Path) -> list[str]:
    parquet_path = dataset_dir / DEFAULT_TASKS_PATH
    legacy_path = dataset_dir / LEGACY_TASKS_PATH
    if parquet_path.exists():
        return _read_task_strings_from_parquet(dataset_dir)
    if legacy_path.exists():
        return _read_task_strings_from_jsonl(legacy_path)
    raise FileNotFoundError(
        "Missing tasks file. Expected either "
        f"{parquet_path} (LeRobot v3.0) or {legacy_path} (legacy)."
    )


def _read_unique_prompts(dataset_dirs: list[str]) -> list[str]:
    prompts: list[str] = []
    seen = set()
    total_task_rows = 0

    for ds_dir in dataset_dirs:
        task_strings = _read_task_strings(Path(ds_dir))
        for task in task_strings:
            prompt = build_fastwam_prompt(task)
            total_task_rows += 1
            if prompt not in seen:
                seen.add(prompt)
                prompts.append(prompt)

    logging.info(
        "Loaded %d task rows from %d datasets, deduplicated to %d prompts.",
        total_task_rows,
        len(dataset_dirs),
        len(prompts),
    )
    return prompts


def _atomic_torch_save(payload: dict[str, torch.Tensor], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.parent / f".{output_path.name}.tmp.{uuid.uuid4().hex}"
    torch.save(payload, str(tmp_path))
    os.replace(tmp_path, output_path)


def _resolve_device(requested_device: str | None, local_rank: int) -> str:
    if requested_device:
        return requested_device
    if torch.cuda.is_available():
        return f"cuda:{local_rank}" if int(os.environ.get("WORLD_SIZE", "1")) > 1 else "cuda"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute FastWAM offline text embedding cache.")
    parser.add_argument(
        "--dataset-dir",
        action="append",
        default=[],
        help="Dataset directory containing `meta/tasks.parquet` or legacy `meta/tasks.jsonl`. Can be specified multiple times.",
    )
    parser.add_argument(
        "--repo-id-file",
        default=None,
        help="Optional file with one dataset directory per line.",
    )
    parser.add_argument(
        "--text-embedding-cache-dir",
        required=True,
        help="Output cache directory used later by FastWAM training.",
    )
    parser.add_argument("--context-len", type=int, default=DEFAULT_CONTEXT_LEN)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--tokenizer-model-id", default=DEFAULT_TOKENIZER_MODEL_ID)
    parser.add_argument("--redirect-common-files", default="true")
    parser.add_argument("--device", default=None, help="Optional explicit device such as cuda, cuda:0, or cpu.")
    parser.add_argument("--dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--override-instruction",
        default=None,
        help="If set, skip dataset scan and cache exactly one prompt built from this task string.",
    )
    parser.add_argument("--overwrite", default="true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    is_distributed, rank, world_size, local_rank = _init_distributed()
    if is_distributed and rank == 0:
        logging.info("Distributed enabled: world_size=%d", world_size)
    if (not is_distributed) and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        logging.info(
            "Multi-GPU available. To use it, run: torchrun --standalone --nproc_per_node=%d "
            "src/lerobot/scripts/fastwam_precompute_text_embeds.py ...",
            torch.cuda.device_count(),
        )

    dataset_dirs = _resolve_dataset_dirs(args.dataset_dir, args.repo_id_file)
    cache_dir = Path(args.text_embedding_cache_dir).expanduser()
    cache_dir.mkdir(parents=True, exist_ok=True)
    overwrite = _parse_bool(args.overwrite)
    context_len = int(args.context_len)

    if args.override_instruction is not None and str(args.override_instruction).strip():
        prompts = [build_fastwam_prompt(str(args.override_instruction).strip())]
        logging.info("Using override_instruction; skipping dataset scan and encoding exactly 1 prompt.")
    else:
        prompts = _read_unique_prompts(dataset_dirs)
    if not prompts:
        logging.warning("No prompts found; nothing to do.")
        return

    device = _resolve_device(args.device, local_rank)
    torch_dtype = _parse_dtype(args.dtype)
    redirect_common_files = _parse_bool(args.redirect_common_files)
    logging.info(
        "Preparing text encoder with model_id=%s tokenizer_model_id=%s device=%s dtype=%s context_len=%d overwrite=%s",
        args.model_id,
        args.tokenizer_model_id,
        device,
        torch_dtype,
        context_len,
        overwrite,
    )

    _, text_config, _, tokenizer_config = _resolve_configs(
        model_id=str(args.model_id),
        tokenizer_model_id=str(args.tokenizer_model_id),
        redirect_common_files=redirect_common_files,
    )
    text_config.download_if_necessary()
    tokenizer_config.download_if_necessary()

    text_encoder = _load_registered_model(
        text_config.path,
        "wan_video_text_encoder",
        torch_dtype=torch_dtype,
        device=device,
    ).eval()
    tokenizer = HuggingfaceTokenizer(
        name=tokenizer_config.path,
        seq_len=context_len,
        clean="whitespace",
    )

    prompts = prompts[rank::world_size] if is_distributed else prompts
    over_length_prompts = 0
    new_files = 0
    overwritten_files = 0
    skipped_files = 0

    if not overwrite:
        prompts_to_encode: list[str] = []
        fully_cached_local = 0
        for prompt in prompts:
            cache_path = build_text_embedding_cache_path(cache_dir, prompt, context_len)
            if cache_path.exists():
                fully_cached_local += 1
            else:
                prompts_to_encode.append(prompt)
        prompts = prompts_to_encode
        skipped_files = fully_cached_local

        fully_cached_global = fully_cached_local
        to_encode_global = len(prompts)
        if is_distributed:
            reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
            count_tensor = torch.tensor([fully_cached_local, len(prompts)], device=reduce_device, dtype=torch.long)
            dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
            fully_cached_global = int(count_tensor[0].item())
            to_encode_global = int(count_tensor[1].item())

        if (not is_distributed) or rank == 0:
            logging.info(
                "overwrite=false: fully cached prompts=%d, prompts to encode=%d",
                fully_cached_global,
                to_encode_global,
            )

    prompts_encoded_local = len(prompts)
    prompts_encoded_global = prompts_encoded_local
    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        count_tensor = torch.tensor([prompts_encoded_local], device=reduce_device, dtype=torch.long)
        dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM)
        prompts_encoded_global = int(count_tensor.item())
    with tqdm(
        total=len(prompts),
        desc=f"Encoding prompts (rank {rank}/{world_size})" if is_distributed else "Encoding prompts",
        unit="prompt",
        dynamic_ncols=True,
        disable=is_distributed and rank != 0,
    ) as pbar:
        with torch.no_grad():
            for start in range(0, len(prompts), int(args.batch_size)):
                batch_prompts = prompts[start : start + int(args.batch_size)]
                ids, mask = tokenizer(batch_prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device)
                mask = mask.to(device=device, dtype=torch.bool)
                over_length_prompts += int(mask.all(dim=1).sum().item())
                context = text_encoder(ids, mask)

                for i, prompt in enumerate(batch_prompts):
                    cache_path = build_text_embedding_cache_path(cache_dir, prompt, context_len)
                    if cache_path.exists() and not overwrite:
                        skipped_files += 1
                        continue

                    payload = {
                        "context": context[i].detach().to(device="cpu", dtype=torch.bfloat16).contiguous(),
                        "mask": mask[i].detach().to(device="cpu", dtype=torch.bool).contiguous(),
                    }
                    if cache_path.exists():
                        overwritten_files += 1
                    else:
                        new_files += 1
                    _atomic_torch_save(payload, cache_path)

                pbar.update(len(batch_prompts))

    if is_distributed:
        reduce_device = torch.device(device) if device.startswith("cuda") else torch.device("cpu")
        stat_tensor = torch.tensor(
            [over_length_prompts, new_files, overwritten_files, skipped_files],
            device=reduce_device,
            dtype=torch.long,
        )
        dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)
        over_length_prompts = int(stat_tensor[0].item())
        new_files = int(stat_tensor[1].item())
        overwritten_files = int(stat_tensor[2].item())
        skipped_files = int(stat_tensor[3].item())

    if (not is_distributed) or rank == 0:
        logging.info("Finished precomputing text embeddings.")
        logging.info(
            "Over-length prompts (mask all True, i.e. no padding after truncation/max_length=%d): %d/%d",
            context_len,
            over_length_prompts,
            prompts_encoded_global,
        )
        logging.info(
            "Cache dir: %s | new=%d overwrite=%d skip=%d",
            cache_dir,
            new_files,
            overwritten_files,
            skipped_files,
        )

    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

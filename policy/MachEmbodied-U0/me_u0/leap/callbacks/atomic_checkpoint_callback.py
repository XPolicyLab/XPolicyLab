"""Crash-consistent checkpoints for long-running Lance data streams."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import shutil
from typing import Optional

import torch
import torch.distributed as dist

from leap.core.callback import Callback
from leap.core.state import TrainingState


logger = logging.getLogger(__name__)

CHECKPOINT_COMPLETE_MARKER = "_SUCCESS"
LATEST_POINTER = "latest.json"
_STEP_PATTERN = re.compile(r"^step_(\d+)$")
_ZERO_OPTIMIZER_RANK_PATTERN = re.compile(r"(?:^|_)zero_pp_rank_(\d+)(?:_|$)")


def _is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def _barrier() -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    if dist.get_backend() == "nccl" and torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def _broadcast_main_bool(value: bool) -> bool:
    """Broadcast one rank-zero decision without relying on filesystem cache state."""

    if not (dist.is_available() and dist.is_initialized()):
        return bool(value)
    if dist.get_backend() == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    decision = torch.tensor(
        [1 if (_is_main_process() and value) else 0],
        dtype=torch.uint8,
        device=device,
    )
    dist.broadcast(decision, src=0)
    return bool(decision.item())


def _broadcast_main_error(value: Optional[str]) -> Optional[str]:
    """Make a rank-zero publication failure visible to every rank.

    Without this envelope, rank zero can raise on fsync/rename while its peers
    wait forever in the trailing barrier.  The collective save has already
    completed at this point, so broadcasting one small diagnostic is safe.
    """

    if not (dist.is_available() and dist.is_initialized()):
        return value
    objects = [value if _is_main_process() else None]
    device = None
    if dist.get_backend() == "nccl":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    dist.broadcast_object_list(objects, src=0, device=device)
    result = objects[0]
    return None if result is None else str(result)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def is_complete_checkpoint(path: str | os.PathLike) -> bool:
    """Validate the marker/trainer-state binding of a published checkpoint."""

    path = Path(path)
    marker_path = path / CHECKPOINT_COMPLETE_MARKER
    trainer_state_path = path / "trainer_state.json"
    if not path.is_dir() or not marker_path.is_file() or not trainer_state_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return False
    if not isinstance(marker, dict) or set(marker) != {"checkpoint", "global_step"}:
        return False
    if not isinstance(trainer_state, dict):
        return False
    marker_step = marker.get("global_step")
    trainer_step = trainer_state.get("global_step")
    if (
        marker.get("checkpoint") != path.name
        or isinstance(marker_step, bool)
        or not isinstance(marker_step, int)
        or isinstance(trainer_step, bool)
        or not isinstance(trainer_step, int)
        or trainer_step != marker_step
    ):
        return False
    match = _STEP_PATTERN.fullmatch(path.name)
    return match is None or int(match.group(1)) == marker_step


def list_complete_step_checkpoints(checkpoint_root: str | os.PathLike) -> list[Path]:
    """Return marker-complete step checkpoints in numeric order."""

    root = Path(checkpoint_root)
    if not root.is_dir():
        return []
    complete = []
    for child in root.iterdir():
        match = _STEP_PATTERN.match(child.name)
        if match and is_complete_checkpoint(child):
            complete.append((int(match.group(1)), child))
    complete.sort(key=lambda pair: pair[0])
    return [path for _step, path in complete]


def remove_optimizer_states(
    checkpoint_path: str | os.PathLike,
) -> tuple[int, int]:
    """Remove DeepSpeed optimizer shards from one published checkpoint.

    Older checkpoints remain usable for model loading and evaluation, while
    only the newest checkpoint retains the full optimizer state required for
    exact training resume.  Return the number of removed files and bytes.
    """

    model_dir = Path(checkpoint_path) / "pytorch_model"
    if not model_dir.is_dir():
        return 0, 0
    removed_files = 0
    removed_bytes = 0
    for path in sorted(model_dir.iterdir()):
        if not path.is_file() or "optim_states" not in path.name:
            continue
        removed_bytes += path.stat().st_size
        path.unlink()
        removed_files += 1
    if removed_files:
        _fsync_directory(model_dir)
    return removed_files, removed_bytes


def resolve_latest_checkpoint(work_dir: str | os.PathLike) -> Optional[str]:
    """Resolve ``latest`` by scanning only fully published checkpoints.

    The pointer is an acceleration hint, never the source of truth.  A stale or
    torn pointer cannot make an incomplete directory resumable.
    """

    root = Path(work_dir) / "checkpoints"
    candidates = list_complete_step_checkpoints(root)
    if not candidates:
        return None
    latest = candidates[-1]
    pointer = root / LATEST_POINTER
    if pointer.is_file():
        try:
            value = json.loads(pointer.read_text(encoding="utf-8"))
            pointed = root / str(value["checkpoint"])
            if pointed == latest and is_complete_checkpoint(pointed):
                return str(pointed)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass
    return str(latest)


def resolve_resume_checkpoint(
    resume_from: str | os.PathLike,
    work_dir: str | os.PathLike,
) -> Optional[str]:
    """Resolve a strict or automatic checkpoint request.

    ``latest`` requires at least one fully published checkpoint.
    ``auto`` selects the latest complete checkpoint when available and otherwise
    returns ``None`` so a failed pre-checkpoint attempt can restart from step 0.
    Explicit paths are returned unchanged and validated by the caller.
    """

    request = os.fspath(resume_from).strip()
    if not request:
        return None
    mode = request.lower()
    if mode not in {"auto", "latest"}:
        return request
    resolved = resolve_latest_checkpoint(work_dir)
    if resolved is not None:
        logger.info("Resolved --resume %s -> %s", mode, resolved)
        return resolved
    checkpoint_root = Path(work_dir) / "checkpoints"
    if mode == "auto":
        logger.warning(
            "--resume auto found no complete checkpoint below %s; "
            "retrying from step 0",
            checkpoint_root,
        )
        return None
    raise FileNotFoundError(
        f"No complete checkpoint exists below {checkpoint_root}"
    )


def validate_deepspeed_checkpoint_world_size(
    checkpoint_path: str | os.PathLike,
    expected_world_size: int,
) -> Optional[int]:
    """Validate the rank topology encoded by DeepSpeed optimizer shards.

    Return the saved world size when DeepSpeed optimizer shards are present.
    Non-DeepSpeed checkpoints contain no ``zero_pp_rank`` files and are left to
    their backend-specific loader.
    """

    checkpoint = Path(checkpoint_path)
    if not is_complete_checkpoint(checkpoint):
        raise ValueError(
            f"Checkpoint is not fully published or has an invalid marker: {checkpoint}"
        )
    if isinstance(expected_world_size, bool):
        raise ValueError("expected_world_size must be a positive integer")
    try:
        expected_world_size = int(expected_world_size)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_world_size must be a positive integer") from exc
    if expected_world_size <= 0:
        raise ValueError("expected_world_size must be a positive integer")

    model_dir = checkpoint / "pytorch_model"
    optimizer_files = (
        sorted(model_dir.rglob("*optim_states.pt"))
        if model_dir.is_dir()
        else []
    )
    if not optimizer_files:
        return None
    ranks = []
    for path in optimizer_files:
        match = _ZERO_OPTIMIZER_RANK_PATTERN.search(path.name)
        if match is None:
            raise ValueError(
                f"Cannot determine DeepSpeed rank from optimizer shard: {path}"
            )
        ranks.append(int(match.group(1)))
    unique_ranks = sorted(set(ranks))
    saved_world_size = len(unique_ranks)
    expected_ranks = list(range(saved_world_size))
    if len(ranks) != saved_world_size or unique_ranks != expected_ranks:
        raise ValueError(
            "DeepSpeed optimizer rank shards are duplicated or non-contiguous in "
            f"{checkpoint}: parsed ranks={unique_ranks[:16]}"
        )
    if saved_world_size != expected_world_size:
        raise ValueError(
            "Checkpoint world size does not match this launch: "
            f"checkpoint={saved_world_size}, launch={expected_world_size}, "
            f"path={checkpoint}"
        )
    return saved_world_size


class AtomicCheckpointCallback(Callback):
    """Publish a checkpoint only after every rank has finished writing it."""

    priority = 80

    def __init__(
        self,
        *,
        save_every_n_steps: int = 0,
        save_every_n_epochs: int = 0,
        max_keep: int = 5,
        save_at_fit_end: bool = True,
    ) -> None:
        self.save_every_n_steps = int(save_every_n_steps)
        self.save_every_n_epochs = int(save_every_n_epochs)
        self.max_keep = int(max_keep)
        self.save_at_fit_end = bool(save_at_fit_end)
        if self.save_every_n_steps < 0 or self.save_every_n_epochs < 0:
            raise ValueError("checkpoint intervals must be non-negative")
        if self.max_keep < 0:
            raise ValueError("max_keep must be non-negative")

    def batch_end(self, state: TrainingState) -> None:
        if self.save_every_n_steps and state.global_step % self.save_every_n_steps == 0:
            self._save(state, f"step_{state.global_step}")

    def epoch_end(self, state: TrainingState) -> None:
        if self.save_every_n_epochs and (state.epoch + 1) % self.save_every_n_epochs == 0:
            self._save(state, f"epoch_{state.epoch + 1}")

    def fit_end(self, state: TrainingState) -> None:
        """Always publish the terminal optimizer state.

        A sample-derived ``max_steps`` is not generally divisible by the periodic
        save interval after changing world size.  Without this hook a cleanly
        completed run can silently lose its last partial checkpoint interval.
        ``_save`` collectively detects and skips an already-published periodic
        checkpoint, so the divisible case does not write twice.
        """

        if self.save_at_fit_end and int(state.global_step) > 0:
            self._save(state, f"step_{int(state.global_step)}")

    def _save(self, state: TrainingState, name: str) -> None:
        save_fn = getattr(state, "_trainer_save_fn", None)
        if save_fn is None:
            raise RuntimeError("TrainingState has no trainer checkpoint save function")
        root = Path(state.work_dir) / "checkpoints"
        final = root / name
        temporary = root / f".{name}.incomplete"

        already_complete = False
        if _is_main_process():
            root.mkdir(parents=True, exist_ok=True)
            if is_complete_checkpoint(final):
                logger.info("Checkpoint already published, skipping: %s", final)
                already_complete = True
            else:
                if final.exists():
                    shutil.rmtree(final)
                if temporary.exists():
                    shutil.rmtree(temporary)
                temporary.mkdir(parents=False)
        _barrier()

        # Only rank zero consults filesystem state.  Independent NFS metadata
        # observations can disagree across nodes and would otherwise send some
        # ranks into Accelerate/DeepSpeed's collective save while others return.
        if _broadcast_main_bool(already_complete):
            return

        # Accelerate/DeepSpeed checkpoint state is collective: every rank must
        # enter the save call and use the same shared temporary directory.
        save_fn(str(temporary))
        _barrier()

        publish_error: Optional[str] = None
        if _is_main_process():
            try:
                trainer_state = temporary / "trainer_state.json"
                if not trainer_state.is_file():
                    raise RuntimeError(
                        "checkpoint save did not produce trainer_state.json: "
                        f"{temporary}"
                    )
                # BaseTrainer routes also use this callback. Force their small
                # progress file durable even if the trainer itself only closed it.
                with trainer_state.open("rb") as handle:
                    os.fsync(handle.fileno())
                _fsync_directory(temporary)
                marker = temporary / CHECKPOINT_COMPLETE_MARKER
                with marker.open("w", encoding="utf-8") as handle:
                    json.dump(
                        {"checkpoint": name, "global_step": state.global_step},
                        handle,
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                _fsync_directory(temporary)
                os.replace(temporary, final)
                _fsync_directory(root)
                _atomic_json(
                    root / LATEST_POINTER,
                    {"checkpoint": name, "global_step": int(state.global_step)},
                )
                self._prune(root)
                self._remove_older_optimizer_states(root, latest=final)
                logger.info("Atomically published checkpoint: %s", final)
            except Exception as exc:  # noqa: BLE001 - broadcast exact failure
                publish_error = f"{type(exc).__name__}: {exc}"
        publish_error = _broadcast_main_error(publish_error)
        if publish_error is not None:
            raise RuntimeError(
                "Checkpoint publication failed on rank zero: " + publish_error
            )
        _barrier()

    def _prune(self, root: Path) -> None:
        if self.max_keep <= 0:
            return
        complete = list_complete_step_checkpoints(root)
        for old in complete[:-self.max_keep]:
            shutil.rmtree(old)
            logger.info("Pruned complete checkpoint: %s", old)

    @staticmethod
    def _remove_older_optimizer_states(root: Path, *, latest: Path) -> None:
        """Keep full optimizer state only in the newly published checkpoint."""

        for checkpoint in list_complete_step_checkpoints(root):
            if checkpoint == latest:
                continue
            removed_files, removed_bytes = remove_optimizer_states(checkpoint)
            if removed_files:
                logger.info(
                    "Removed %d optimizer state files (%.2f GiB) from %s",
                    removed_files,
                    removed_bytes / (1024**3),
                    checkpoint,
                )


__all__ = [
    "AtomicCheckpointCallback",
    "CHECKPOINT_COMPLETE_MARKER",
    "LATEST_POINTER",
    "is_complete_checkpoint",
    "list_complete_step_checkpoints",
    "remove_optimizer_states",
    "resolve_latest_checkpoint",
    "resolve_resume_checkpoint",
    "validate_deepspeed_checkpoint_world_size",
]

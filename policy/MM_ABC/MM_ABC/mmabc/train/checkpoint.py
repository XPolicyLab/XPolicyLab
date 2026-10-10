"""Distributed resume checkpoints, inference milestones and weights-only loading."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys
from torch.distributed.tensor import DTensor, distribute_tensor

_OPTS = StateDictOptions(full_state_dict=False, cpu_offload=True)
_FULL = StateDictOptions(full_state_dict=True, cpu_offload=True)

# Written after everything else, so its presence means the save completed.
COMPLETE_MARKER = "COMPLETE"


@dataclass
class CheckpointConfig:
    root: str
    keep_resume: int = 2
    milestone_every: int = 10_000
    resume_every: int = 2_000
    min_free_gb: float = 150.0


def free_gb(path: str | Path) -> float:
    """Free space at `path`, or at its nearest existing ancestor."""
    p = Path(path).resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free / 2**30


class CheckpointManager:
    def __init__(self, cfg: CheckpointConfig, *, is_master: bool) -> None:
        self.cfg = cfg
        self.is_master = is_master
        self.root = Path(cfg.root)
        # Every rank writes its own shards, so every rank needs the directories.
        # mkdir is idempotent, which is cheaper than a barrier here.
        (self.root / "resume").mkdir(parents=True, exist_ok=True)
        (self.root / "milestones").mkdir(parents=True, exist_ok=True)

    def check_space(self) -> None:
        available = free_gb(self.root)
        if available < self.cfg.min_free_gb:
            raise RuntimeError(
                f"only {available:.0f} GiB free under {self.root}, need at least "
                f"{self.cfg.min_free_gb:.0f} GiB for checkpointing"
            )

    def save_resume(self, step: int, model, optimizer, scheduler, extra: dict | None = None) -> Path:
        self.check_space()
        # Release cached CUDA allocations before checkpoint collectives.
        torch.cuda.empty_cache()
        path = self.root / "resume" / f"step_{step:08d}"
        state = {
            "model": get_model_state_dict(model, options=_OPTS),
            "optimizer": get_optimizer_state_dict(model, optimizer, options=_OPTS),
        }
        dcp.save(state, checkpoint_id=str(path))
        if self.is_master:
            meta = {"step": step, "scheduler": scheduler.state_dict() if scheduler else None}
            meta.update(extra or {})
            (path / "meta.json").write_text(json.dumps(meta, indent=1))
            # Written last, and only on success. A save that dies partway leaves
            # a directory that looks resumable but is not, and the next launch
            # would fail on it instead of falling back to an older checkpoint.
            (path / COMPLETE_MARKER).write_text(str(step))
            self._prune()
        return path

    def save_milestone(self, step: int, model) -> Path:
        """Save weights-only bf16 milestones for evaluation and deployment."""
        self.check_space()
        torch.cuda.empty_cache()
        path = self.root / "milestones" / f"step_{step:08d}"
        state = get_model_state_dict(model, options=_OPTS)
        state = {
            k: (v.to(torch.bfloat16) if torch.is_floating_point(v) else v)
            for k, v in state.items()
        }
        dcp.save({"model": state}, checkpoint_id=str(path))
        return path

    def _prune(self) -> None:
        saves = sorted((self.root / "resume").glob("step_*"))
        for old in saves[: max(0, len(saves) - self.cfg.keep_resume)]:
            shutil.rmtree(old, ignore_errors=True)

    def latest_resume(self) -> Path | None:
        """Newest checkpoint that actually finished writing."""
        for path in sorted((self.root / "resume").glob("step_*"), reverse=True):
            if (path / COMPLETE_MARKER).exists() and (path / ".metadata").exists():
                return path
        return None

    def load_resume(self, path: Path, model, optimizer, scheduler) -> int:
        state = {
            "model": get_model_state_dict(model, options=_OPTS),
            "optimizer": get_optimizer_state_dict(model, optimizer, options=_OPTS),
        }
        dcp.load(state, checkpoint_id=str(path))
        set_model_state_dict(model, state["model"], options=_OPTS)
        set_optimizer_state_dict(model, optimizer, state["optimizer"], options=_OPTS)

        meta_path = path / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if scheduler is not None and meta.get("scheduler"):
                scheduler.load_state_dict(meta["scheduler"])
            return int(meta.get("step", 0))
        return 0

    def load_pretrained(self, path: str | Path, model, *, log=None) -> dict:
        """Warm-start weights from another run's checkpoint. No optimiser, no step."""
        path = Path(path)
        current = get_model_state_dict(model, options=_OPTS)

        reader = FileSystemReader(str(path))
        saved = {
            k[len("model.") :]: v
            for k, v in reader.read_metadata().state_dict_metadata.items()
            if k.startswith("model.")
        }

        matched, resized, missing = {}, {}, []
        for name, tensor in current.items():
            meta = saved.get(name)
            if meta is None:
                missing.append(name)
                continue
            if tuple(meta.size) == tuple(tensor.shape):
                matched[name] = tensor
            else:
                resized[name] = (tuple(meta.size), tuple(tensor.shape))

        # Read matching tensors sharded; reshape only explicitly supported position embeddings.
        dcp.load({"model": matched}, checkpoint_id=str(path))
        set_model_state_dict(
            model,
            matched,
            options=StateDictOptions(full_state_dict=False, cpu_offload=True, strict=False),
        )

        for name, (src_shape, dst_shape) in resized.items():
            if len(src_shape) != len(dst_shape) or any(
                s < d for s, d in zip(src_shape, dst_shape)
            ):
                raise ValueError(
                    f"cannot warm-start {name}: checkpoint shape {src_shape} "
                    f"does not contain the model shape {dst_shape}"
                )
            loaded = _load_state_dict_from_keys(f"model.{name}", checkpoint_id=str(path))
            # The helper nests only the top level of the saved state ("model"
            # here) and leaves the rest of the path flat, dots included.
            full = loaded["model"][name]
            sliced = full[tuple(slice(0, d) for d in dst_shape)].contiguous()
            param = model.get_parameter(name)
            local = param.data
            if isinstance(local, DTensor):
                # Re-shard the slice the same way the live parameter is shard-
                # ed; every rank passes an identical full tensor, so this is
                # deterministic without a collective.
                param.data.copy_(
                    distribute_tensor(
                        sliced.to(device=local.device, dtype=local.dtype),
                        local.device_mesh,
                        local.placements,
                    )
                )
            else:
                param.data.copy_(sliced.to(device=local.device, dtype=local.dtype))

        report = {
            "source": str(path),
            "loaded": len(matched),
            "resized": {k: f"{v[0]} -> {v[1]}" for k, v in resized.items()},
            "missing": missing,
        }
        if log is not None:
            log.info(
                "warm-started %d tensors from %s; resized %s; %d left at init",
                len(matched), path, report["resized"], len(missing),
            )
            if missing:
                log.warning("not in checkpoint, using fresh init: %s", missing[:10])
        return report

    def export_weights(self, model, out: str | Path) -> Path:
        """Consolidate to a single-file state dict for deployment and eval."""
        sd = get_model_state_dict(model, options=_FULL)
        out = Path(out)
        if self.is_master:
            out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": sd}, out)
        return out

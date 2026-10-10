"""Load a checkpoint and its saved contract for batched action-chunk inference."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mmabc.canonical.layout import CanonicalLayout
from mmabc.canonical.normalize import NormStats, Normalizer, identity_stats
from mmabc.canonical.transforms import EmbodimentSpec, TargetBuilder
from mmabc.data.dataset import CANONICAL_VIEWS
from mmabc.data.prompt import PromptSpec, build_prompt
from mmabc.data.video import resize_for_model
from mmabc.models.mmabc import MMABCPolicy, load_model_config


@dataclass
class Observation:
    """One control step's worth of input, in the robot's own units."""

    state: np.ndarray  # absolute state in the layout's state space
    images: dict[str, np.ndarray]  # view slot -> (H, W, 3) uint8 RGB
    instruction: str


def find_run_meta(checkpoint: str | Path) -> Path:
    """``<root>/milestones/step_N`` or ``<root>/resume/step_N`` -> ``<root>/run_meta``."""
    path = Path(checkpoint).resolve()
    for parent in (path, *path.parents):
        if (parent / "run_meta").is_dir():
            return parent / "run_meta"
    raise FileNotFoundError(f"no run_meta/ above {checkpoint}; was it produced by the MM-ABC trainer?")


class MMABCInferencePolicy:
    def __init__(
        self,
        checkpoint: str | Path | None,
        model_config: str | Path,
        *,
        embodiment_meta: str | Path,
        norm_stats: str | Path | None = None,
        allow_identity_stats: bool = False,
        action_type: str | None = None,
        reference_frame: str = "base",
        device: str = "cuda",
        repo_root: str | Path | None = None,
        num_inference_steps: int | None = None,
        execute_steps: int | None = None,
        seed: int | None = None,
        canonical: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
        self.device = torch.device(device)
        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        cfg = load_model_config(str(model_config), repo_root=str(self.repo_root))
        if canonical is not None:
            cfg.canonical = str(canonical)
        # The future stream exists only to shape training; nothing attends to it,
        # so building it at deployment would load a 1.3B teacher for nothing.
        cfg.future.enabled = False
        # Inference never runs backward; recomputation would only cost time.
        cfg.backbone.gradient_checkpointing = False
        if num_inference_steps:
            cfg.flow.num_inference_steps = int(num_inference_steps)
        self.model = MMABCPolicy(cfg, repo_root=str(self.repo_root))
        if checkpoint is not None:
            self._load_weights(checkpoint)
        else:
            warnings.warn("MMABCInferencePolicy: no checkpoint, running randomly initialised weights")
        self.model.to(self.device).eval()
        self.generator = None
        if seed is not None:
            self.generator = torch.Generator(device=self.device).manual_seed(int(seed))

        self.layout: CanonicalLayout = self.model.layout
        embodiment = json.loads(Path(embodiment_meta).read_text())
        self.spec = EmbodimentSpec.from_meta(embodiment)
        self.builder = TargetBuilder(self.layout, self.spec, reference_frame=reference_frame)
        self.action_type = action_type or self.builder.available_action_types[0]
        if self.action_type not in self.builder.available_action_types:
            raise ValueError(
                f"{self.action_type!r} unsupported by this embodiment; "
                f"available: {self.builder.available_action_types}"
            )

        # Fail loud on a missing stats path rather than silently using
        # mean=0/std=1, which gives plausible-looking but wrong commands.
        if norm_stats is not None:
            if not Path(norm_stats).exists():
                raise FileNotFoundError(f"norm stats not found: {norm_stats}")
            stats = NormStats.load(norm_stats)
        elif allow_identity_stats:
            warnings.warn(
                "MMABCInferencePolicy: using IDENTITY norm stats; only valid for smoke tests.",
                stacklevel=2,
            )
            stats = identity_stats(
                self.spec.embodiment_tag, self.layout.total_dim, self.layout.state_dim
            )
        else:
            raise ValueError("norm_stats is required for correct actions")
        self.normalizer = Normalizer(self.layout, stats)
        self.image_size = int(cfg.obs.image_size)
        self.prompt_header = bool(cfg.get("prompt", {}).get("header", False))
        self.chunk_size = int(cfg.flow.chunk_size)
        self.execute_steps = int(execute_steps or cfg.flow.execute_steps)

        # What the robot consumes is not the same as what was supervised: a
        # delta rotation is supervised as 3 dims but handed back as the full
        # 6-dim orientation, so take whole segments.
        supervised = self.builder.static_mask(self.action_type)
        output = np.zeros_like(supervised)
        for seg in self.layout.segments:
            if seg.kind == "reserved":
                continue
            if supervised[seg.target_slice].any():
                output[seg.slice] = self.spec.action_valid[seg.slice]
        self.action_dims = np.flatnonzero(output)
        self.supervised_dims = np.flatnonzero(supervised)

    @classmethod
    def from_run(
        cls,
        checkpoint: str | Path,
        *,
        embodiment_tag: str | None = None,
        **kwargs,
    ) -> "MMABCInferencePolicy":
        """Build from a trainer-produced checkpoint and the run_meta next to it."""
        meta = find_run_meta(checkpoint)
        if embodiment_tag is None:
            tags = sorted(p.name[len("embodiment_"):-len(".json")] for p in meta.glob("embodiment_*.json"))
            if len(tags) != 1:
                raise ValueError(f"run has embodiments {tags}; pass embodiment_tag")
            embodiment_tag = tags[0]
        policy = cls(
            checkpoint,
            meta / "model_config.yaml",
            embodiment_meta=meta / f"embodiment_{embodiment_tag}.json",
            norm_stats=meta / f"norm_stats_{embodiment_tag}.json",
            # The frozen copy next to the weights, so a moved run dir still loads.
            canonical=meta / "canonical.yaml",
            **kwargs,
        )
        policy.run_meta = meta
        policy.embodiment_tag = embodiment_tag
        contract = meta / f"contract_{embodiment_tag}.json"
        policy.contract = json.loads(contract.read_text()) if contract.exists() else None
        profiles = json.loads((meta / "profiles.json").read_text())
        policy.views = next(p["views"] for p in profiles.values() if p["embodiment_tag"] == embodiment_tag)
        return policy

    def _load_weights(self, checkpoint: str | Path) -> None:
        path = Path(checkpoint)
        if path.is_dir():
            import torch.distributed.checkpoint as dcp
            from torch.distributed.checkpoint.state_dict import set_model_state_dict

            state = {"model": self.model.state_dict()}
            dcp.load(state, checkpoint_id=str(path))
            set_model_state_dict(self.model, state["model"])
        else:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            # strict=False only because future.enabled=False drops the
            # training-only future stream; everything else must match.
            result = self.model.load_state_dict(blob.get("model", blob), strict=False)
            bad_missing = [k for k in result.missing_keys if "future" not in k]
            if bad_missing or result.unexpected_keys:
                raise RuntimeError(
                    f"checkpoint key mismatch loading {path}: missing {bad_missing[:10]}, "
                    f"unexpected {list(result.unexpected_keys)[:10]}"
                )

    def _prepare(self, observations: list[Observation]) -> dict:
        B = len(observations)
        # (B, V, T=1, H, W, 3): encode_context selects the current frame with
        # [:, :, 0], so the timestep axis must be present.
        images = np.zeros(
            (B, len(CANONICAL_VIEWS), 1, self.image_size, self.image_size, 3), dtype=np.uint8
        )
        view_mask = np.zeros((B, len(CANONICAL_VIEWS)), dtype=np.float32)
        prompts, states = [], []
        for b, obs in enumerate(observations):
            for i, slot in enumerate(CANONICAL_VIEWS):
                img = obs.images.get(slot)
                if img is None:
                    continue
                images[b, i, 0] = resize_for_model(img, self.image_size)
                view_mask[b, i] = 1.0
            prompts.append(
                build_prompt(
                    PromptSpec(
                        embodiment=self.spec.embodiment_tag,
                        fps=self.spec.fps,
                        action_type=self.action_type,
                        reference_frame=self.builder.reference_frame,
                        instruction=obs.instruction,
                    ),
                    dropout=0.0,
                    header=self.prompt_header,
                )
            )
            states.append(self.normalizer.normalize_state(np.asarray(obs.state, dtype=np.float32)))
        aux_active = float(self.builder.aux_is_active(self.action_type))
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device)
        return {
            "state": t(np.stack(states)).float(),
            "state_mask": t(np.repeat(self.spec.state_valid.astype(np.float32)[None], B, 0)),
            "aux_active": torch.full((B,), aux_active, device=self.device),
            "images": t(images),
            "view_mask": t(view_mask),
            "prompt": prompts,
        }

    @torch.no_grad()
    def act_batch(self, observations: list[Observation]) -> list[dict]:
        """One forward pass for several environments."""
        batch = self._prepare(observations)
        # Flash-Attention 2 only accepts fp16/bf16 and the modules are fp32, so
        # inference runs under bf16 autocast exactly like training.
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda"):
            normalised = self.model.predict_action(batch, generator=self.generator).float().cpu().numpy()

        results = []
        horizon = min(self.execute_steps, normalised.shape[1])
        for obs, pred in zip(observations, normalised):
            target = self.normalizer.denormalize_action(pred)
            absolute = self.builder.integrate(
                np.asarray(obs.state, dtype=np.float64), target, action_type=self.action_type
            )
            results.append(
                {
                    "action_canonical": absolute[:horizon],
                    "action_full_chunk": absolute,
                    "action_dims": self.action_dims,
                    "action": absolute[:horizon][:, self.action_dims],
                    "action_type": self.action_type,
                }
            )
        return results

    def act(self, obs: Observation) -> dict:
        return self.act_batch([obs])[0]

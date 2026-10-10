"""Vision-language backbone and rectified-flow action expert."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from mmabc.canonical.layout import CanonicalLayout, load_layout
from mmabc.data.dataset import CANONICAL_VIEWS
from mmabc.models.backbone import QwenVLBackbone, build_inputs
from mmabc.models.flow import FlowConfig, RectifiedFlow
from mmabc.models.heads import future_alignment_loss, masked_action_loss
from mmabc.models.joint_expert import MMABCJointExpert
from mmabc.models.teachers import TeacherConfig, build_teacher
from mmabc.paths import home_path


@dataclass
class StepOutput:
    loss: torch.Tensor
    metrics: dict[str, float] = field(default_factory=dict)


class MMABCPolicy(nn.Module):
    def __init__(self, cfg: DictConfig, *, repo_root: str | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.layout: CanonicalLayout = load_layout(
            str(_resolve(cfg.canonical, repo_root))
        )
        self.head_slices = {
            h.name: (h.start, h.end) for h in self.layout.heads
        }

        self.backbone = QwenVLBackbone(
            str(home_path(cfg.backbone.path)),
            memory_layers=tuple(int(x) for x in cfg.backbone.memory_layers),
            attn_implementation=str(cfg.backbone.get("attn_implementation", "flash_attention_2")),
            gradient_checkpointing=bool(cfg.backbone.get("gradient_checkpointing", True)),
        )

        self.flow = RectifiedFlow(
            FlowConfig(
                chunk_size=int(cfg.flow.chunk_size),
                execute_steps=int(cfg.flow.execute_steps),
                num_inference_steps=int(cfg.flow.num_inference_steps),
                repeated_noise_draws=int(cfg.flow.repeated_noise_draws),
                time_beta=tuple(float(x) for x in cfg.flow.time_beta),
                sigma_min=float(cfg.flow.sigma_min),
                loss_type=str(cfg.flow.get("loss_type", "v")),
                pred_type=str(cfg.flow.get("pred_type", "x")),
                normalize_weight=bool(cfg.flow.get("normalize_weight", True)),
            )
        )

        self.future_enabled = bool(cfg.future.get("enabled", False))
        self.future_weight = float(cfg.future.get("loss_weight", 0.05))
        # Scale the action objective relative to future alignment; report its unweighted metric.
        self.action_weight = float(cfg.get("action_loss_weight", 1.0))
        # Keep the frozen teacher outside the registered module tree, optimiser and checkpoints.
        self._teacher_holder: list[nn.Module] = []
        self._teacher_cfg: TeacherConfig | None = None
        teacher_dim = None
        if self.future_enabled:
            names = [str(t) for t in cfg.future.get("teachers", ["vggt"])]
            if len(names) != 1:
                raise ValueError("exactly one future teacher is supported")
            name = names[0]
            tcfg = cfg.future[name]
            self._teacher_cfg = TeacherConfig(
                name=name,
                repo_path=str(home_path(tcfg.repo_path)),
                ckpt_path=str(home_path(tcfg.get("ckpt_path", tcfg.get("weights_path", "")))),
                image_size=int(tcfg.get("image_size", 256)),
                feature_dim=int(tcfg.feature_dim),
                block_grid=int(cfg.future.get("block_grid", 8)),
            )
            teacher_dim = self._teacher_cfg.feature_dim

        self.expert = MMABCJointExpert(
            head_slices=self.head_slices,
            chunk_size=int(cfg.flow.chunk_size),
            near_steps=int(cfg.flow.near_steps),
            hidden=int(cfg.expert.hidden_size),
            num_blocks=int(cfg.expert.num_blocks),
            num_heads=int(cfg.expert.num_heads),
            ffn=int(cfg.expert.ffn_size),
            dropout=float(cfg.expert.get("dropout", 0.1)),
            context_dim=self.backbone.hidden_size,
            injection_blocks=tuple(int(x) for x in cfg.expert.injection_blocks),
            context_mode=str(cfg.expert.get("context_mode", "deepstack")),
            state_dim=int(cfg.get("state_dim", self.layout.state_dim)),
            future_tokens=int(cfg.future.get("num_tokens", 192)),
            future_dim=teacher_dim,
        )

    @property
    def future_offsets(self) -> tuple[int, ...]:
        """Frame offsets the two future segments predict.

        Defaults to the end of each action segment, so the near prediction
        anticipates where execution actually lands and the far one anticipates
        the horizon the speculative tail is planning toward.
        """
        configured = self.cfg.future.get("offsets")
        if configured:
            return tuple(int(o) for o in configured)
        return (int(self.cfg.flow.near_steps), int(self.cfg.flow.chunk_size))

    def teacher(self, device: torch.device) -> nn.Module:
        """Lazily built, and deliberately outside the module tree."""
        if not self._teacher_holder:
            assert self._teacher_cfg is not None
            # bf16 weights: the teacher is frozen and already runs its forward
            # under bf16 autocast, so fp32 copies would cost 2.7 GB of every
            # GPU for no change in the target.
            teacher = build_teacher(self._teacher_cfg.name, self._teacher_cfg).to(
                device=device, dtype=torch.bfloat16
            )
            teacher.eval()
            self._teacher_holder.append(teacher)
        return self._teacher_holder[0]

    def encode_context(self, batch: dict) -> tuple[list[torch.Tensor], torch.Tensor]:
        device = batch["state"].device
        inputs = batch.get("vlm")
        if inputs is None:
            # Tokenisation and image patching on the training process. The
            # dataloader normally does this in its workers (see data.collate.Collator).
            current = batch["images"][:, :, 0]  # (B, V, H, W, 3) uint8
            inputs = build_inputs(
                self.backbone.processor,
                batch["prompt"],
                current,
                batch["view_mask"],
                device=device,
            )
        out = self.backbone(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
        )
        return out.memory, out.attention_mask.float()

    @torch.no_grad()
    def _extract_future_targets(
        self, images: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """Independent teacher features for each future horizon.

        VGGT resolves reconstruction scale per sequence. Feeding the near and
        far frames as one sequence would couple their scales and give the far
        bank a target that is not actually "the scene at t+64" in isolation.
        Each horizon is therefore its own sequence of views. The two extracts
        are stacked as 2B so the aggregator still runs once.
        """
        futures = images[:, :, 1:]  # (B, V, T, H, W, 3)
        offsets = self.future_offsets
        # Banks that target the same frame get the same target; extract it once.
        unique = sorted(set(offsets), key=offsets.index)
        futures = futures[:, :, [offsets.index(o) for o in unique]]
        b, v, t = futures.shape[:3]
        stacked = futures.permute(2, 0, 1, 3, 4, 5).reshape(t * b, v, *futures.shape[3:])
        out = self.teacher(device)(stacked)  # (T*B, 1, N, D)
        out = out.reshape(t, b, *out.shape[2:])
        if out.shape[2] == 1:
            out = out.squeeze(2)
        out = out[[unique.index(o) for o in offsets]]
        return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)

    def forward(self, batch: dict) -> StepOutput:
        target = batch["target"]
        mask = batch["target_mask"]
        B = target.shape[0]
        device = target.device

        memory, context_valid = self.encode_context(batch)
        head_active = {
            "manip": (mask[..., self.head_slices["manip"][0] : self.head_slices["manip"][1]]
                      .sum((1, 2)) > 0).float(),
            "aux": batch["aux_active"],
        }

        future_target = None
        if self.future_enabled:
            with torch.no_grad():
                future_target = self._extract_future_targets(batch["images"], device)

        # The context does not depend on the noisy action input, so its keys and
        # values are identical for every draw. Building them once is the single
        # largest saving in the step.
        context_cache = self.expert.build_context_cache(memory)

        draws = self.flow.cfg.repeated_noise_draws
        total = torch.zeros((), device=device, dtype=torch.float32)
        action_total = torch.zeros((), device=device, dtype=torch.float32)
        future_loss_value = None
        per_head_acc: dict[str, float] = {}

        # Decoded-action error: the clean-action estimate implied by the raw
        # output vs ground truth, comparable across x- and v-prediction; in
        # normalised units and, with `action_scale`, in real command units.
        action_scale = batch.get("action_scale")
        recon_norm_acc = torch.zeros((), device=device, dtype=torch.float32)
        recon_real_acc = torch.zeros((), device=device, dtype=torch.float32)
        recon_head_acc = {n: torch.zeros((), device=device) for n in self.head_slices}

        for draw in range(draws):
            noise = torch.randn_like(target)
            t = self.flow.sample_time(B, device)
            x_t = self.flow.interpolate(target, noise, t)
            # Clean-sample ("x") or velocity ("v") regression target + weight.
            flow_target, flow_weight = self.flow.training_target(target, noise, t)
            want_future = self.future_enabled and draw == 0

            out = self.expert(
                noisy_actions=x_t,
                flow_time=t,
                state=batch["state"],
                state_mask=batch["state_mask"],
                memory=memory,
                context_valid=context_valid,
                aux_active=batch["aux_active"],
                want_future=want_future,
                context_cache=context_cache,
            )

            action_loss, per_head, _ = masked_action_loss(
                out.predictions,
                flow_target,
                mask,
                self.head_slices,
                weight=flow_weight,
                head_active=head_active,
            )
            action_total = action_total + action_loss / draws
            for k, v in per_head.items():
                per_head_acc[k] = per_head_acc.get(k, 0.0) + v.detach() / draws

            # Decode clean actions from either parameterisation before measuring command error.
            with torch.no_grad():
                a0_hat = torch.zeros_like(x_t)
                for name, (lo, hi) in self.head_slices.items():
                    a0_hat[..., lo:hi] = out.predictions[name].detach().to(a0_hat.dtype)
                if self.flow.cfg.pred_type == "v":
                    tt = t.view(-1, *([1] * (a0_hat.dim() - 1)))
                    a0_hat = x_t + (1.0 - tt) * a0_hat
                diff = (a0_hat.float() - target.float()).abs()
                m = mask.float()
                mdenom = m.sum().clamp(min=1.0)
                recon_norm_acc = recon_norm_acc + (diff * m).sum() / mdenom / draws
                diff_real = (
                    diff if action_scale is None
                    else diff * action_scale[:, None, :].float()
                )
                recon_real_acc = recon_real_acc + (diff_real * m).sum() / mdenom / draws
                for name, (lo, hi) in self.head_slices.items():
                    mh = m[..., lo:hi]
                    recon_head_acc[name] = recon_head_acc[name] + (
                        (diff_real[..., lo:hi] * mh).sum() / mh.sum().clamp(min=1.0) / draws
                    )

            if want_future and out.future_near is not None:
                # Each segment is scored against the frame at its own horizon.
                # `future_valid` drops frames that sit past the episode end so
                # a padded last-frame repeat cannot become a training target.
                valid = batch.get("future_valid")
                near_v = None if valid is None else valid[:, 0]
                far_v = None if valid is None else valid[:, 1]
                fut = 0.5 * (
                    future_alignment_loss(out.future_near, future_target[0], near_v)
                    + future_alignment_loss(out.future_far, future_target[1], far_v)
                )
                future_loss_value = fut.detach()
                total = total + self.future_weight * fut

        total = total + self.action_weight * action_total
        # Gathered as tensors and copied to the host in one transfer: a float()
        # per term would block the CPU on the GPU a dozen times per step.
        named = {
            "loss": total.detach(),
            "loss_action": action_total.detach(),
            "aux_active_frac": batch["aux_active"].float().mean(),
            **{f"loss_{k}": v for k, v in per_head_acc.items()},
            # Parametrisation-invariant decoded-action error (see loop comment).
            "recon_l1": recon_norm_acc.detach(),
            "recon_l1_real": recon_real_acc.detach(),
            **{f"recon_l1_real_{k}": v.detach() for k, v in recon_head_acc.items()},
        }
        if future_loss_value is not None:
            named["loss_future"] = future_loss_value
        values = torch.stack([v.float().reshape(()) for v in named.values()]).tolist()
        metrics = dict(zip(named, values))
        return StepOutput(loss=total, metrics=metrics)

    @torch.no_grad()
    def predict_action(self, batch: dict, generator: torch.Generator | None = None) -> torch.Tensor:
        """Normalised model-space action chunk, (B, chunk, 80).

        The future stream is not built here. Nothing attends to it, so its
        absence cannot change the result; skipping it saves the teacher and 192
        extra tokens of attention per step.
        """
        target_like = torch.zeros(
            (batch["state"].shape[0], self.flow.cfg.chunk_size, self.layout.total_dim),
            device=batch["state"].device,
        )
        memory, context_valid = self.encode_context(batch)
        context_cache = self.expert.build_context_cache(memory)

        def denoise(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            out = self.expert(
                noisy_actions=x,
                flow_time=t,
                state=batch["state"],
                state_mask=batch["state_mask"],
                memory=memory,
                context_valid=context_valid,
                aux_active=batch["aux_active"],
                want_future=False,
                context_cache=context_cache,
            )
            merged = torch.zeros_like(x)
            for name, (lo, hi) in self.head_slices.items():
                merged[..., lo:hi] = out.predictions[name].to(merged.dtype)
            return merged

        return self.flow.sample(
            denoise, target_like.shape, target_like.device, dtype=torch.float32,
            generator=generator,
        )

    def param_groups(self, cfg) -> list[dict]:
        """Separate learning rates for the pretrained trunk and the fresh head.

        The backbone already has useful representations and needs small steps;
        the expert is randomly initialised and needs large ones. Weight decay is
        skipped on norms, biases and learnt positional/query parameters.
        """
        def decay_ok(name: str, p: torch.Tensor) -> bool:
            return p.dim() >= 2 and not name.endswith("gate") and "pos_embed" not in name

        groups = []
        for tag, module, lr in (
            ("backbone", self.backbone, float(cfg.lr_backbone)),
            ("expert", self.expert, float(cfg.lr_expert)),
        ):
            decay, no_decay = [], []
            for name, p in module.named_parameters():
                if not p.requires_grad:
                    continue
                (decay if decay_ok(name, p) else no_decay).append(p)
            groups.append({"params": decay, "lr": lr, "weight_decay": float(cfg.weight_decay), "name": f"{tag}_decay"})
            groups.append({"params": no_decay, "lr": lr, "weight_decay": 0.0, "name": f"{tag}_nodecay"})
        return groups


def _resolve(rel, repo_root: str | None):
    from pathlib import Path

    p = Path(str(rel))
    if p.is_absolute():
        return p
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[2]
    return root / p


def load_model_config(path: str, repo_root: str | None = None) -> DictConfig:
    """Load a model config, following a single level of `defaults`."""
    cfg = OmegaConf.load(str(_resolve(path, repo_root)))
    base = cfg.pop("defaults", None)
    if base is not None:
        parent = OmegaConf.load(str(_resolve(base, repo_root)))
        parent.pop("defaults", None)
        cfg = OmegaConf.merge(parent, cfg)
    return cfg

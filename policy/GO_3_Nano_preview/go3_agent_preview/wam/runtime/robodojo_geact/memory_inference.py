from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from robodojo_geact.data.geact.transforms.geact_infer_transforms import (
    GeactCorobotPrepare,
)
from robodojo_geact.model.geact.task.geact_task_flow import GEActTaskFlow
from robodojo_geact.utils.geact import infer_utils


_single_frame_vgen = infer_utils.one_step_vgen_from_obs


class MemoryGeactCorobotPrepare(GeactCorobotPrepare):
    memory_state_tokens: int = 1

    def _from_legacy(self, obs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        images = np.asarray(obs["obs"])
        if images.ndim != 6:
            return super()._from_legacy(obs)
        states = np.asarray(obs["state"])
        if (
            states.ndim != 3
            or states.shape[0] != images.shape[0]
            or states.shape[1] != 1
        ):
            raise ValueError(
                "memory inference requires only the current state [B,1,D]; "
                f"got state={states.shape}, obs={images.shape}"
            )

        if self.memory_state_tokens == 1:
            selected_states = states
        else:
            memory_states = np.asarray(obs.get("memory_state"))
            if (
                memory_states.ndim != 3
                or memory_states.shape[0] != images.shape[0]
                or memory_states.shape[1] != self.memory_state_tokens
            ):
                raise ValueError(
                    "memory_state must be [B,T,D] with "
                    f"T={self.memory_state_tokens}; got {memory_states.shape}"
                )
            selected_states = memory_states

        latest = dict(obs)
        latest["obs"] = images[:, -1]
        latest["state"] = selected_states
        _, state = super()._from_legacy(latest)

        video = torch.from_numpy(images.astype(np.float32)).permute(0, 5, 2, 1, 3, 4)
        if video.max() > 1.0:
            video = (video / 127.5) - 1.0
        return (
            video.to(self.device, dtype=self.dtype),
            state,
        )


def _memory_vgen_from_obs(
    self,
    obs_frame,
    caption,
    view_ids=None,
    raw_t=None,
    t_sigma_val=None,
    return_at_t=0.0,
    precomputed_noisy_latents=None,
    vgen_model_override=None,
    noise_seed=None,
    future_noise=None,
    bboxes=None,
    bbox_reference_images=None,
):
    if bboxes is not None or bbox_reference_images is not None:
        raise ValueError("memory inference does not support bbox conditioning")
    memory = getattr(self, "_deploy_observation_memory", None)
    if memory is None:
        return _single_frame_vgen(
            self,
            obs_frame,
            caption,
            view_ids=view_ids,
            raw_t=raw_t,
            t_sigma_val=t_sigma_val,
            return_at_t=return_at_t,
            precomputed_noisy_latents=precomputed_noisy_latents,
            vgen_model_override=vgen_model_override,
            noise_seed=noise_seed,
            future_noise=future_noise,
        )

    if memory.dim() != 6:
        raise ValueError(
            "memory observation must be [B,C,V,T,H,W], "
            f"got {tuple(memory.shape)}"
        )
    expected = int(self.cfg.data.val.n_previous)
    if memory.shape[3] != expected:
        raise ValueError(
            f"expected {expected} memory frames, got {memory.shape[3]}"
        )

    if raw_t is None:
        raw_t = (
            expected
            + int(self.cfg.data.val.continious_vid_length)
            + int(self.cfg.data.val.sparse_length)
        )
    future_count = int(raw_t) - expected
    if future_count < 0:
        raise ValueError(f"raw_t={raw_t} is smaller than n_previous={expected}")

    current = memory[:, :, :, -1:]
    video_segment = torch.cat(
        [memory, current.repeat(1, 1, 1, future_count, 1, 1)], dim=3
    )
    batch_size, _channels, views, _time, _height, _width = video_segment.shape
    if isinstance(caption, str):
        caption = [caption] * batch_size
    if view_ids is None:
        view_ids = torch.arange(
            views, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
    batch_segment = {
        "caption": caption,
        "view_ids": view_ids,
        "video_slices": torch.arange(
            raw_t, device=self.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1),
    }
    return infer_utils._one_step_vgen_for_ae(
        self,
        video_segment,
        batch_segment,
        views,
        self.device,
        self.dtype,
        t_sigma_val=t_sigma_val,
        return_at_t=return_at_t,
        precomputed_noisy_latents=precomputed_noisy_latents,
        vgen_model_override=vgen_model_override,
        noise_seed=noise_seed,
        future_noise=future_noise,
    )


class MemoryGEActTaskFlow(GEActTaskFlow):
    def infer_deploy_step(
        self,
        video: torch.Tensor,
        state: torch.Tensor,
        caption: Any,
        num_vgen_steps: int = 25,
        num_ae_steps: int | None = None,
        replay: dict[str, Any] | None = None,
    ) -> np.ndarray:
        if video.dim() == 5:
            return super().infer_deploy_step(
                video,
                state,
                caption,
                num_vgen_steps=num_vgen_steps,
                num_ae_steps=num_ae_steps,
                replay=replay,
            )
        if video.dim() != 6:
            raise ValueError(
                "video must be [B,C,V,H,W] or [B,C,V,T,H,W], "
                f"got {tuple(video.shape)}"
            )

        expected = int(self.cfg.data.val.n_previous)
        if video.shape[3] != expected:
            raise ValueError(
                f"expected {expected} memory frames, got {video.shape[3]}"
            )
        object.__setattr__(self, "_deploy_observation_memory", video)
        try:
            return super().infer_deploy_step(
                video[:, :, :, -1],
                state,
                caption,
                num_vgen_steps=num_vgen_steps,
                num_ae_steps=num_ae_steps,
                replay=replay,
            )
        finally:
            object.__setattr__(self, "_deploy_observation_memory", None)

    def load_deploy_model(self, ckpt_path):
        from robodojo_geact.utils.geact.model_utils import stabilize_diffusers_config_for_compile
        import logging
        logger = logging.getLogger(__name__)
        super().load_deploy_model(ckpt_path)
        if (
            self.video_diffusion_model is not None
            and self.cfg.model.vgen_compile_forward
        ):
            if not hasattr(torch, "compile"):
                raise RuntimeError(
                    "vgen_compile_forward=true requires torch.compile support."
                )
            if self.cfg.model.video.vlm_fixed_sequence_length is None:
                raise ValueError(
                    "vgen_compile_forward=true requires vlm_fixed_sequence_length"
                )
            vgen_module = (
                self.video_diffusion_model.module
                if hasattr(self.video_diffusion_model, "module")
                else self.video_diffusion_model
            )
            if not getattr(vgen_module, "_vgen_compiled_forward_enabled", False):
                stabilize_diffusers_config_for_compile(vgen_module)
                compile_kwargs = {
                    "fullgraph": bool(self.cfg.model.vgen_compile_fullgraph),
                    "mode": self.cfg.model.vgen_compile_mode,
                }
                if self.cfg.model.vgen_compile_dynamic is not None:
                    compile_kwargs["dynamic"] = bool(
                        self.cfg.model.vgen_compile_dynamic
                    )
                vgen_module._vgen_eager_forward = vgen_module.forward
                vgen_module.forward = torch.compile(
                    vgen_module.forward, **compile_kwargs
                )
                vgen_module._vgen_compiled_forward_enabled = True
                logger.info(
                    "Enabled torch.compile for deploy VGen forward: "
                    "mode=%s, dynamic=%s, fullgraph=%s",
                    self.cfg.model.vgen_compile_mode,
                    self.cfg.model.vgen_compile_dynamic,
                    self.cfg.model.vgen_compile_fullgraph,
                )

        return self

    def _deploy_vgen_from_obs(self, *args, **kwargs):
        return _memory_vgen_from_obs(self, *args, **kwargs)

    def _deploy_ae_vision(self, dense_latents, sparse_latents, n_view, **kwargs):
        history_frames = int(self.cfg.data.val.n_previous) - 1
        if history_frames >= dense_latents.shape[2]:
            raise ValueError("Memory history would remove every dense AE frame")
        return super()._deploy_ae_vision(
            dense_latents[:, :, history_frames:], sparse_latents, n_view, **kwargs
        )


def build_memory_inference(cfg, settings):
    from robodojo_geact.inferrer.deploy_inferrer import DeployInferrer

    from robodojo_geact.data.geact.transforms.geact_infer_transforms import (
        GeactCorobotFormat,
    )
    from robodojo_geact.inferrer.geact_deploy_helpers import (
        load_geact_infer_stats,
        resolve_geact_infer_settings,
    )
    from robodojo_geact.model.geact.build_models import build_geact_task_flow
    from robodojo_geact.utils.common_utils import get_device

    if settings.geact_infer_return_head_waist:
        cfg.infer.return_head_waist = True
    resolved = resolve_geact_infer_settings(
        cfg,
        robot_name=settings.geact_infer_robot_name or None,
        stat_path=settings.geact_infer_stat_path or None,
    )
    cfg.infer.robot_name = resolved["robot_name"]
    cfg.infer.stat_path = resolved["stat_path"]
    cfg.infer.joint_dim = (
        6
        if cfg.infer.robot_name in {
            "sl_acone", "sl_arxvr", "sl_agilex", "robodojo", "robodojo_real"
        }
        else resolved["joint_dim"]
    )
    cfg.infer.action_mode = resolved["action_mode"]

    stats = load_geact_infer_stats(cfg.infer.stat_path, get_device())
    val = cfg.data.val
    memory_state_tokens = int(getattr(cfg.infer, "memory_state_tokens", 1))
    if memory_state_tokens not in {1, int(val.n_previous)}:
        raise ValueError(
            "cfg.infer.memory_state_tokens must be 1 or match "
            f"cfg.data.val.n_previous={val.n_previous}, got {memory_state_tokens}"
        )
    cameras = list(val.cam_use[0]) if val.cam_use else []
    if not cameras:
        raise ValueError("cfg.data.val.cam_use must define inference cameras")
    prepare = MemoryGeactCorobotPrepare(
        cam_order=cameras,
        target_h=val.sample_size[0],
        target_w=val.sample_size[1],
        s_mean=stats["s_mean"],
        s_std=stats["s_std"],
        device=get_device(),
        dtype=(
            torch.bfloat16
            if cfg.train.mixed_precision == "bf16"
            else torch.float32
        ),
        default_num_ae_steps=cfg.validation.num_inference_step,
        default_dense_action_steps=cfg.infer.dense_action_steps,
        legacy_state_layout=os.environ.get(
            "GEACT_INFER_LEGACY_STATE_LAYOUT", "identity"
        ),
    )
    prepare.memory_state_tokens = memory_state_tokens
    transforms = [prepare]
    output_transforms = [
        GeactCorobotFormat(
            robot_name=cfg.infer.robot_name,
            action_dim=stats["a_mean"].shape[0],
            default_dense_action_steps=cfg.infer.dense_action_steps,
            return_head_waist=cfg.infer.return_head_waist,
        )
    ]
    task_flow = build_geact_task_flow(
        cfg=cfg,
        cotrain_schedule=None,
        transforms=transforms,
        output_transforms=output_transforms,
    )
    task_flow.__class__ = MemoryGEActTaskFlow
    inferrer = None
    if settings.infer_checkpoint_dir:
        inferrer = DeployInferrer(
            run_name=settings.run_name,
            infer_ckpt_cfg={
                "pretrain_model": settings.infer_checkpoint_dir,
                "to_cuda": True,
            },
            infer_task_flow=task_flow,
            infer_type="deploy",
        )
    return task_flow, inferrer

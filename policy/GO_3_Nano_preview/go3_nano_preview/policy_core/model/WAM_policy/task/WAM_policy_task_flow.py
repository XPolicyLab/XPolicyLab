from __future__ import annotations

import logging
import os
from typing import Any, Optional, Sequence

import numpy as np
import torch
from einops import rearrange

from torchvision.transforms import Compose

from policy_core.model.WAM_policy.task.WAM_policy_task_model import WAM_policyTaskModel
from policy_core.utils.common_utils import get_device


logger = logging.getLogger(__name__)


class WAM_policyTaskFlow(torch.nn.Module):
    def __init__(
        self,
        *,
        task_model: WAM_policyTaskModel,
        cfg: Any,
        vae,
        ae_video_scaling_factor: float,
        stats_tracker=None,
        scheduler=None,
        scheduler_sigmas: Optional[torch.Tensor] = None,
        lpips_model=None,
        spatial_down_ratio: int = 1,
        temporal_down_ratio: int = 1,
        video_frame_indices: Optional[list[int]] = None,
        video_sparse_downsample_factor: int = 1,
        video_force_mixed_path: bool = False,
        video_sparse_spatial_rope_mode: str = "continuous",
        use_video_sparse_mode: bool = False,
        use_video_mixed_path: bool = False,
        expert_dual_train: bool = False,
        video_pipeline_class=None,
        action_pipeline_class=None,
        dsrl_enabled: bool = False,
        vlm_rm_vgen_mode: str = "moving",
        vlm_rm_head_type: str = "mlp",
        vlm_rm_lora_in_non_rm: bool = False,
        vlm_compute_with_grad: bool = False,
        vlm_rm_enabled: bool = False,
        transforms: Sequence[Any] = (),
        output_transforms: Sequence[Any] = (),
    ) -> None:
        super().__init__()
        self._task_model = task_model
        self._postprocess_layer = None
        self.cfg = cfg
        self.vae = vae
        self.stats_tracker = stats_tracker
        self.scheduler = scheduler
        self.scheduler_sigmas = scheduler_sigmas
        self.ae_video_scaling_factor = ae_video_scaling_factor
        self.spatial_down_ratio = spatial_down_ratio
        self.temporal_down_ratio = temporal_down_ratio
        self.video_frame_indices = video_frame_indices or []
        self.video_sparse_downsample_factor = video_sparse_downsample_factor
        self.video_force_mixed_path = video_force_mixed_path
        self.video_sparse_spatial_rope_mode = video_sparse_spatial_rope_mode
        self.use_video_sparse_mode = use_video_sparse_mode
        self.use_video_mixed_path = use_video_mixed_path
        object.__setattr__(self, "lpips_model", lpips_model)
        self.expert_dual_train = expert_dual_train
        self.device = get_device()
        self.dtype = (
            torch.bfloat16
            if cfg.train.mixed_precision == "bf16"
            else torch.float32
        )
        self.video_pipeline_class = video_pipeline_class
        self.action_pipeline_class = action_pipeline_class
        self.dsrl_enabled = dsrl_enabled
        self.vlm_rm_vgen_mode = vlm_rm_vgen_mode
        self.vlm_rm_head_type = vlm_rm_head_type
        self.vlm_rm_lora_in_non_rm = vlm_rm_lora_in_non_rm
        self.vlm_compute_with_grad = vlm_compute_with_grad
        self.vlm_rm_enabled = vlm_rm_enabled
        self._input_transform = Compose(list(transforms))
        self._output_transform = Compose(list(output_transforms))
        self._deploy_stats = None
        self._deploy_action_pipeline = None
        self._deploy_robot_name: Optional[str] = None
        self._deploy_joint_dim: Optional[int] = None
        self._deploy_action_mode: Optional[str] = None
        self._deploy_body_type: Optional[str] = None
        self._deploy_body_emb_id: Optional[int] = None
        self._deploy_camera_order: Optional[list[str]] = None
        self._deploy_camera_view_ids: Optional[list[int]] = None


    @property
    def video_diffusion_model(self):
        return getattr(self._task_model, "vgen", None)

    @property
    def video_diffusion_model_high(self):
        return getattr(self._task_model, "vgen_high", None)

    @property
    def action_diffusion_model(self):
        return self._task_model.action_expert


    @property
    def SPATIAL_DOWN_RATIO(self):
        return self.spatial_down_ratio

    @property
    def TEMPORAL_DOWN_RATIO(self):
        return self.temporal_down_ratio


    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self.training:
            raise RuntimeError("WAM_policy runtime is inference-only; call eval() first")
        return self._deploy_forward(inputs)

    def _deploy_forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        data = dict(inputs) if isinstance(inputs, dict) else inputs
        if not isinstance(data, dict):
            raise TypeError(
                f"deploy forward expects dict inputs, got {type(data).__name__}"
            )

        data = self._input_transform(data)

        if "video" in data and "state" in data:
            if "caption" in data:
                caption = data["caption"]
            elif "prompt" in data:
                caption = data["prompt"]
            else:
                raise KeyError("caption or prompt required after transforms")
            step_kwargs: dict[str, Any] = {}
            if "bbox" in data:
                step_kwargs["bboxes"] = data["bbox"]
            if "bbox_reference_images" in data:
                step_kwargs["bbox_reference_images"] = data["bbox_reference_images"]
            if "num_ae_steps" in data and data["num_ae_steps"] is not None:
                step_kwargs["num_ae_steps"] = data["num_ae_steps"]
            if "num_vgen_steps" in data and data["num_vgen_steps"] is not None:
                step_kwargs["num_vgen_steps"] = data["num_vgen_steps"]
            if data.get("replay") is not None:
                step_kwargs["replay"] = data["replay"]
            actions = self.infer_deploy_step(
                data["video"], data["state"], caption, **step_kwargs
            )
            out: dict[str, Any] = {
                "actions": actions,
                "dense_action_steps": data.get("dense_action_steps"),
            }
            return self._output_transform(out)

        outputs = self._task_model(data)
        if self._postprocess_layer:
            outputs = self._postprocess_layer(outputs)
        return outputs

    def load_deploy_model(self, ckpt_path: str) -> Any:
        import os

        from policy_core.utils.WAM_policy.model_utils import load_checkpoints

        ckpt_path = os.path.expanduser(str(ckpt_path))
        if not os.path.isdir(ckpt_path):
            raise FileNotFoundError(f"deploy ckpt dir not found: {ckpt_path}")
        action_subdir = os.path.join(ckpt_path, "action_model")
        if not os.path.isdir(action_subdir):
            raise FileNotFoundError(
                f"deploy ckpt missing action_model/: {action_subdir}"
            )
        logger.info(
            "load_deploy_model: vgen=%s action=%s", ckpt_path, action_subdir
        )
        if self.video_diffusion_model is not None:
            load_checkpoints(self.video_diffusion_model, ckpt_path)
        if self.action_diffusion_model is not None:
            load_checkpoints(self.action_diffusion_model, action_subdir)

        self.eval()
        self._task_model.eval()
        return self

    def _ensure_deploy_runtime(self) -> None:
        if self._deploy_action_pipeline is not None:
            return

        from policy_core.data.WAM_policy.utils.WAM_policy_camera import (
            infer_camera_view_ids,
        )
        from policy_core.inferrer.WAM_policy_deploy_helpers import (
            load_WAM_policy_infer_stats,
            resolve_WAM_policy_infer_settings,
        )
        from policy_core.utils.WAM_policy.body_embedding import resolve_body_emb_id

        if self.device is None or self.dtype is None:
            raise RuntimeError("task_flow.device/dtype must be set for deploy infer")
        if self.video_diffusion_model is None:
            raise RuntimeError("video_diffusion_model is required for deploy infer")
        if self.action_diffusion_model is None:
            raise RuntimeError("action_diffusion_model is required for deploy infer")
        if self.action_pipeline_class is None:
            raise RuntimeError("action_pipeline_class is required for deploy infer")

        settings = resolve_WAM_policy_infer_settings(self.cfg)
        self._deploy_robot_name = settings["robot_name"]
        self._deploy_joint_dim = settings["joint_dim"]
        self._deploy_action_mode = settings["action_mode"]

        val = self.cfg.data.val
        cam_use = list(val.cam_use[0]) if val.cam_use else []
        if not cam_use:
            raise ValueError("cfg.data.val.cam_use must define inference cameras")
        self._deploy_camera_order = cam_use
        self._deploy_camera_view_ids = infer_camera_view_ids(cam_use)

        body_type = self._deploy_robot_name
        self._deploy_body_type = body_type
        self._deploy_body_emb_id = resolve_body_emb_id(body_type)

        self._deploy_stats = load_WAM_policy_infer_stats(
            settings["stat_path"], self.device
        )
        self._deploy_action_pipeline = self.action_pipeline_class(
            vae=self.vae,
            transformer=self.action_diffusion_model,
        )
        for mod in (
            self.video_diffusion_model,
            self.action_diffusion_model,
            self.vae,
            self._task_model.text_encoder,
        ):
            if mod is not None and hasattr(mod, "eval"):
                mod.eval()
        logger.info(
            "deploy runtime ready robot=%s body_emb_id=%d cams=%s stats=%s",
            self._deploy_robot_name,
            self._deploy_body_emb_id,
            self._deploy_camera_order,
            settings["stat_path"],
        )

    def _condition_deploy_captions(self, caption: Any, batch_size: int) -> list[str]:
        from policy_core.data.WAM_policy.unified_dataset.embodiment_cue import condition_caption
        captions = [caption] * batch_size if isinstance(caption, str) else list(caption)
        if len(captions) != batch_size:
            raise ValueError(f'Expected {batch_size} captions, got {len(captions)}')
        return [condition_caption(c, self._deploy_body_type) for c in captions]

    def _apply_deploy_action_mode(
        self, pred: torch.Tensor, state_abs: torch.Tensor
    ) -> torch.Tensor:
        final = pred.clone()
        if self._deploy_action_mode == "rela":
            j = self._deploy_joint_dim
            final = pred + state_abs
            final[:, :, j] = pred[:, :, j]
            final[:, :, 2 * j + 1] = pred[:, :, 2 * j + 1]
        return final

    @torch.no_grad()
    def _deploy_vgen_from_obs(self, *args, **kwargs):
        from policy_core.utils.WAM_policy.infer_utils import one_step_vgen_from_obs
        return one_step_vgen_from_obs(self, *args, **kwargs)

    def _deploy_ae_vision(self, *args, **kwargs):
        from policy_core.utils.WAM_policy.infer_utils import build_ae_vision_feature_from_latents
        return build_ae_vision_feature_from_latents(self, *args, **kwargs)

    def infer_deploy_step(
        self,
        video: torch.Tensor,
        state: torch.Tensor,
        caption: Any,
        num_vgen_steps: int = 25,
        num_ae_steps: int | None = None,
        replay: dict[str, Any] | None = None,
        bboxes=None,
        bbox_reference_images=None,
    ) -> np.ndarray:
        del num_vgen_steps
        self._ensure_deploy_runtime()
        if num_ae_steps is None:
            num_ae_steps = self.cfg.validation.num_inference_step

        if video.dim() != 5:
            raise ValueError(
                f"video must be 5D [B, C, V, H, W], got shape {tuple(video.shape)}"
            )

        from policy_core.utils.WAM_policy.infer_utils import (
            build_ae_vision_feature_from_latents,
            one_step_vgen_from_obs,
        )
        from policy_core.utils.WAM_policy.profile_utils import (
            StageProfiler,
            stage_profile_enabled,
        )

        device, dtype = self.device, self.dtype
        profiler = StageProfiler("PROFILE", device)
        video = video.to(device, dtype=dtype)
        B, _C, V, _H, _W = video.shape
        caption_list = self._condition_deploy_captions(caption, B)
        body_emb_id = torch.full(
            (B,), self._deploy_body_emb_id, device=device, dtype=torch.long
        )

        if V != len(self._deploy_camera_view_ids):
            raise ValueError(
                f"Expected {len(self._deploy_camera_view_ids)} views "
                f"{self._deploy_camera_order}, got {V}"
            )
        view_ids = torch.tensor(
            self._deploy_camera_view_ids, device=device, dtype=torch.long
        ).unsqueeze(0).expand(B, -1)

        replay = replay or {}
        future_noise = replay.get("future_noise")
        if future_noise is not None and not torch.is_tensor(future_noise):
            future_noise = torch.as_tensor(future_noise)
        dense_topk_idx = replay.get("dense_topk_idx")
        if dense_topk_idx is not None and not torch.is_tensor(dense_topk_idx):
            dense_topk_idx = torch.as_tensor(dense_topk_idx)
        sparse_topk_idx = replay.get("sparse_topk_idx")
        if sparse_topk_idx is not None and not torch.is_tensor(sparse_topk_idx):
            sparse_topk_idx = torch.as_tensor(sparse_topk_idx)
        action_noise = replay.get("action_noise")
        if action_noise is not None and not torch.is_tensor(action_noise):
            action_noise = torch.as_tensor(action_noise)
        vgen_noise_seed = replay.get("vgen_noise_seed")
        if vgen_noise_seed is not None:
            vgen_noise_seed = int(vgen_noise_seed)

        profiler.mark("input_preparation")
        debug_video_dir = os.environ.get("WAM_policy_OPENLOOP_DEBUG_VIDEO_DIR")
        if debug_video_dir:
            from policy_core.utils.WAM_policy import save_video

            os.makedirs(debug_video_dir, exist_ok=True)
            video_index = getattr(self, "_openloop_debug_video_index", 0)
            input_path = os.path.join(
                debug_video_dir, f"input_video_{video_index:04d}.mp4"
            )
            fps = int(
                self.cfg.basic_fps
                / (self.cfg.data.train.action_chunk // self.cfg.data.train.chunk)
            )
            input_frames = rearrange(
                video[0].detach().float().cpu(), "c v h w -> c 1 h (v w)"
            )
            save_video(input_frames, input_path, fps=fps)
        vgen_latents = self._deploy_vgen_from_obs(
            video,
            caption_list,
            view_ids=view_ids,
            noise_seed=vgen_noise_seed,
            future_noise=future_noise,
            bboxes=bboxes,
            bbox_reference_images=bbox_reference_images,
        )
        profiler.mark("video_generation")
        if debug_video_dir:
            from policy_core.utils.WAM_policy.infer_utils import save_vgen_debug_video

            generated_path = os.path.join(
                debug_video_dir, f"generated_video_{video_index:04d}.mp4"
            )
            save_vgen_debug_video(
                self, vgen_latents, V, generated_path, fps=fps
            )
            if not os.path.isfile(generated_path):
                raise RuntimeError(f"failed to save generated video: {generated_path}")
            logger.info(
                "Saved open-loop input/generated videos at %s and %s (fps=%d)",
                input_path, generated_path, fps,
            )
            self._openloop_debug_video_index = video_index + 1
        if isinstance(vgen_latents, tuple):
            ae_latents_dense, ae_latents_sparse = vgen_latents
        else:
            sparse_length = self.cfg.data.val.sparse_length
            ae_latents_dense = vgen_latents[:, :, :-sparse_length]
            ae_latents_sparse = vgen_latents[:, :, -sparse_length:]
        profiler.mark("debug_io")
        v_cond = self._deploy_ae_vision(
            ae_latents_dense,
            ae_latents_sparse,
            V,
            dense_topk_idx=dense_topk_idx,
            sparse_topk_idx=sparse_topk_idx,
        )
        profiler.mark("vision_feature_build")

        state_cond = state.to(device, dtype=dtype)
        robot_name = self._deploy_robot_name
        stats = self._deploy_stats
        if state_cond.shape[2] < 32:
            pad_dim = 32 - state_cond.shape[2]
            pad = torch.zeros(
                state_cond.shape[0],
                state_cond.shape[1],
                pad_dim,
                device=device,
                dtype=dtype,
            )
            action_mask = torch.cat(
                [torch.ones_like(state_cond), torch.zeros_like(pad)], dim=2
            )
            state_cond = torch.cat([state_cond, pad], dim=2)
        elif state_cond.shape[2] == 32:
            action_mask = torch.ones_like(state_cond)
        else:
            raise ValueError(
                f"state_cond last dimension must be <= 32, got {state_cond.shape[2]}"
            )

        action_mask = action_mask[:, :1]

        j = self._deploy_joint_dim
        state_cond_drop = torch.cat(
            [
                state_cond[:, :, :j],
                state_cond[:, :, j + 1 : 2 * j + 1],
                state_cond[:, :, 2 * j + 2 :],
            ],
            dim=-1,
        )
        profiler.mark("state_preparation")

        action_dim = self.cfg.model.action.action_in_channels
        action_chunk = self.cfg.data.val.action_chunk
        ae_kwargs: dict[str, Any] = dict(
            vision_feature=v_cond,
            num_inference_steps=num_ae_steps,
            action_chunk=action_chunk,
            state_cond=state_cond_drop,
            action_dim=action_dim,
            action_mask=action_mask,
            view_ids=view_ids,
            body_emb_id=body_emb_id,
        )
        if action_noise is not None:
            ae_kwargs["noise"] = action_noise.to(device=device, dtype=dtype)
        ae_res = self._deploy_action_pipeline.infer(**ae_kwargs)[0]
        profiler.mark("action_inference")

        stats_dim = stats["a_std"].shape[-1]
        pred_action = ae_res["action"].clone()
        curr_state = state_cond.clone()
        pred_action[:, :, :stats_dim] = (
            pred_action[:, :, :stats_dim]
            * stats["a_std"][:stats_dim].to(pred_action.dtype)
            + stats["a_mean"][:stats_dim].to(pred_action.dtype)
        )
        curr_state[:, :, :stats_dim] = (
            curr_state[:, :, :stats_dim]
            * stats["s_std"][:stats_dim].to(curr_state.dtype)
            + stats["s_mean"][:stats_dim].to(curr_state.dtype)
        )

        final_action = self._apply_deploy_action_mode(pred_action, curr_state)
        profiler.mark("action_postprocess")

        output = final_action.detach().cpu().float().numpy()
        profiler.mark("to_cpu_numpy")
        total_ms = profiler.report(
            order=(
                "input_preparation",
                "video_generation",
                "debug_io",
                "vision_feature_build",
                "state_preparation",
                "action_inference",
                "action_postprocess",
                "to_cpu_numpy",
            )
        )
        if stage_profile_enabled():
            print(
                "-----------------------------------------------------------------------------------",
                flush=True,
            )
            print(f"运行耗时：{total_ms / 1000.0:.3f} 秒", flush=True)
        return output
    def predict_policy_step_actions(
        self,
        batch: dict[str, Any],
        *,
        num_ae_steps: int | None = None,
    ) -> dict[str, Any]:
        if "video" not in batch or "actions" not in batch or "state" not in batch:
            raise KeyError(
                "policy-step eval batch requires keys video / actions / state; "
                f"got {sorted(batch)}"
            )
        video = batch["video"]
        if video.dim() == 6:
            obs = video[:, :, :, 0]
        elif video.dim() == 5:
            obs = video
        else:
            raise ValueError(
                f"policy-step eval video must be 5D or 6D, got shape {tuple(video.shape)}"
            )

        state = batch["state"]
        if state.dim() == 2:
            state = state.unsqueeze(1)
        captions = batch.get("caption", batch.get("prompt"))
        if captions is None:
            captions = [""] * obs.shape[0]
        step_kwargs: dict[str, Any] = {}
        if num_ae_steps is not None:
            step_kwargs["num_ae_steps"] = num_ae_steps
        pred = self.infer_deploy_step(obs, state, captions, **step_kwargs)
        gt = batch["actions"]
        if torch.is_tensor(gt):
            gt_np = gt.detach().cpu().float().numpy()
        else:
            gt_np = np.asarray(gt, dtype=np.float32)

        inferred_list = [pred[i] for i in range(pred.shape[0])]
        gt_list = [gt_np[i] for i in range(gt_np.shape[0])]
        sparse_length = self.cfg.data.val.sparse_length
        mode_name = str(batch.get("_val_metric_mode_name", "policy_step"))
        return {
            "model_outputs": {"inferred_actions": inferred_list},
            "model_inputs": {
                "ground_truth_actions": gt_list,
                "sparse_length": sparse_length,
                "metric_track": (mode_name, ""),
            },
        }

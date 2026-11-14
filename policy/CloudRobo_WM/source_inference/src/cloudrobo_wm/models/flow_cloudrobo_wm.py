from __future__ import annotations
import copy
from typing import Dict, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F
from cloudrobo_wm.utils.logging_config import get_logger
from cloudrobo_wm.utils.pytorch_utils import _staged_randn
from .cloudrobo_wm import CloudRoboWM
from .wan_video_dit import sinusoidal_embedding_1d
logger = get_logger(__name__)
FLOW_LAYOUT_VERSION = 1
FLOW_ANCHOR_MODE = 'omitted'
FLOW_CACHE_SCHEMA = 'cloudrobo_wm.flow.fp16_x264rgb.final_tile.stride4.v1'

class FlowCloudRoboWM(CloudRoboWM):
    _FLOW_CKPT_KEYS: tuple[str, ...] = ('flow_patch_embedding', 'flow_head')

    def _init_flow_branch(self, *, flow_loss_lambda: float, p_joint_flow: float, cross_modal_predict_flow: bool, flow_layout_version: int, flow_anchor_mode: str, flow_cache_schema: str) -> None:
        if hasattr(self, 'flow_patch_embedding'):
            raise RuntimeError('FlowCloudRoboWM flow branch was initialized twice.')
        if not 0.0 <= float(p_joint_flow) <= 1.0:
            raise ValueError(f'p_joint_flow must be in [0,1], got {p_joint_flow}')
        if not bool(cross_modal_predict_flow):
            raise ValueError('S0 has no observed O_f range, so U_f must remain an active cross-modal prediction target (cross_modal_predict_flow=true).')
        if int(flow_layout_version) != FLOW_LAYOUT_VERSION:
            raise ValueError(f'unsupported flow_layout_version={flow_layout_version}; expected {FLOW_LAYOUT_VERSION}')
        if str(flow_anchor_mode) != FLOW_ANCHOR_MODE:
            raise ValueError("S0 supports only flow_anchor_mode='omitted'; present/history flow tokens are intentionally not part of this layout.")
        if str(flow_cache_schema) != FLOW_CACHE_SCHEMA:
            raise ValueError(f'unsupported flow_cache_schema={flow_cache_schema!r}; expected {FLOW_CACHE_SCHEMA!r}')
        self.flow_patch_embedding = copy.deepcopy(self.video_expert.patch_embedding)
        self.flow_head = copy.deepcopy(self.video_expert.head)
        self.loss_lambda_flow = float(flow_loss_lambda)
        self.p_joint_flow = float(p_joint_flow)
        self.cross_modal_predict_flow = True
        self.flow_layout_version = FLOW_LAYOUT_VERSION
        self.flow_anchor_mode = FLOW_ANCHOR_MODE
        self.flow_cache_schema = str(flow_cache_schema)
        self.train_flow_scheduler = copy.deepcopy(self.train_video_scheduler)
        self.infer_flow_scheduler = copy.deepcopy(self.infer_video_scheduler)
        self._batch_joint_flow: Optional[torch.Tensor] = None
        self._infer_joint_flow = False
        self.to(device=self.device, dtype=self.torch_dtype)

    def _mode_ckpt_keys(self) -> tuple:
        return super()._mode_ckpt_keys() + self._FLOW_CKPT_KEYS

    def prepare_for_inference(self, *args, **kwargs) -> None:
        super().prepare_for_inference(*args, **kwargs)
        self.infer_flow_scheduler.set_solver(self.infer_video_scheduler.solver)

    @torch.no_grad()
    def infer_action(self, prompt=None, input_image=None, action_horizon=None, num_video_frames: Optional[int]=None, proprio=None, context=None, context_mask=None, negative_prompt=None, text_cfg_scale: float=1.0, num_inference_steps: int=20, sigma_shift=None, seed=None, rand_device: str='cpu', tiled: bool=False, camera_intrinsics: Optional[torch.Tensor]=None, per_cam: Optional[Dict[str, torch.Tensor]]=None, per_cam_depth: Optional[Dict[str, torch.Tensor]]=None, dynamic_step_skip: bool=False, step_skip_thresholds: Sequence[Tuple[float, int]]=((0.95, 4), (0.93, 2)), step_skip_sim_sources: Optional[Sequence[str]]=None, step_skip_sim_aggregation: str='mean', return_stream_latents: bool=False, joint_video: Optional[bool]=None, joint_dino: Optional[bool]=None, joint_pointmap: Optional[bool]=None, present_video: Optional[bool]=None, present_dino: Optional[bool]=None, present_pointmap: Optional[bool]=None, joint_flow: bool=False, return_flow_latents: bool=False, present_flow: Optional[bool]=None):
        if present_flow:
            raise ValueError("present_flow=True is unsupported by S0: flow_anchor_mode is 'omitted' and no O_f module/range exists.")
        if not joint_flow:
            return super().infer_action(prompt=prompt, input_image=input_image, action_horizon=action_horizon, num_video_frames=num_video_frames, proprio=proprio, context=context, context_mask=context_mask, negative_prompt=negative_prompt, text_cfg_scale=text_cfg_scale, num_inference_steps=num_inference_steps, sigma_shift=sigma_shift, seed=seed, rand_device=rand_device, tiled=tiled, camera_intrinsics=camera_intrinsics, per_cam=per_cam, per_cam_depth=per_cam_depth, dynamic_step_skip=dynamic_step_skip, step_skip_thresholds=step_skip_thresholds, step_skip_sim_sources=step_skip_sim_sources, step_skip_sim_aggregation=step_skip_sim_aggregation, return_stream_latents=return_stream_latents, joint_video=joint_video, joint_dino=joint_dino, joint_pointmap=joint_pointmap, present_video=present_video, present_dino=present_dino, present_pointmap=present_pointmap)
        return self._unsupported_joint_flow(prompt=prompt, input_image=input_image, action_horizon=action_horizon, num_video_frames=num_video_frames, proprio=proprio, context=context, context_mask=context_mask, negative_prompt=negative_prompt, text_cfg_scale=text_cfg_scale, num_inference_steps=num_inference_steps, sigma_shift=sigma_shift, seed=seed, rand_device=rand_device, tiled=tiled, camera_intrinsics=camera_intrinsics, per_cam=per_cam, per_cam_depth=per_cam_depth, dynamic_step_skip=dynamic_step_skip, step_skip_thresholds=step_skip_thresholds, step_skip_sim_sources=step_skip_sim_sources, step_skip_sim_aggregation=step_skip_sim_aggregation, return_stream_latents=return_stream_latents, joint_video=joint_video, joint_dino=joint_dino, joint_pointmap=joint_pointmap, present_video=present_video, present_dino=present_dino, present_pointmap=present_pointmap, return_flow_latents=return_flow_latents)

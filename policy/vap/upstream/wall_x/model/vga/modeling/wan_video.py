from torch._tensor import Tensor
import gc
import logging
import math
import torch
import types
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from einops import repeat
from collections import defaultdict
from typing import Callable, List, Optional, Union

from wall_x.utils.io import load_state_dict as _load_state_dict_from_path
from einops import rearrange
from tqdm import tqdm
from typing_extensions import Literal

from wall_x.model.vga.pipeline.config import ModelConfig

from wall_x.model.vga.pipeline.units import (
    PipelineUnitRunner,
    PipelineUnit,
    WanVideoUnit_ShapeChecker,
    WanVideoUnit_NoiseInitializer,
    WanVideoUnit_PromptEmbedder,
    WanVideoUnit_InputVideoEmbedder,
    WanVideoUnit_ImageEmbedderCLIP,
    WanVideoUnit_ImageEmbedderVAE,
    WanVideoUnit_ImageEmbedderFused,
    WanVideoUnit_ActionContext,
    WanVideoUnit_SightconeViewMask,
)
from wall_x.model.vga.modeling.base import BasePipeline

from wall_x.model.vga.pipeline.model_manager import ModelManager

from wall_x.model.vga.modeling.wan_dit import (
    WanModel,
    RMSNorm,
    sinusoidal_embedding_1d,
    SelfAttention,
)
from wall_x.model.vga.modeling.wan_action import ActionTransformer, ActionConditioner
from wall_x.model.vga.modeling.wan_text_encoder import (
    WanTextEncoder,
    T5RelativeEmbedding,
    T5LayerNorm,
)
from wall_x.model.vga.modeling.wan_vae import (
    WanVideoVAE,
    RMS_norm,
    CausalConv3d,
    Upsample,
)
from wall_x.model.vga.modeling.wan_image_encoder import WanImageEncoder

from wall_x.model.vga.pipeline.scheduler import FlowMatchScheduler

from wall_x.model.vga.pipeline.prompter import WanPrompter
from wall_x.distributed import sp_shard, get_sp_world_size, sp_gather

logger = logging.getLogger(__name__)


def _rank0_print(msg: str) -> None:
    """Log only on rank 0 when distributed; log everywhere otherwise.

    Default ``log_fn`` for ``WanVideoPipeline.load_partial_state_dict``.
    Callers (FSDP trainer, inference wrapper) usually pass their own
    structured logger and bypass this fallback.
    """
    try:
        import torch.distributed as dist

        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except (ImportError, RuntimeError):
        pass
    logger.info("%s", msg)


# ---------------------------------------------------------------------------
# Helpers used by ActionTransformer init-from-DiT routines.
# Resize a Linear / 1D tensor across dim mismatch using linear interpolation,
# applying sqrt(old_in / new_in) variance-preserving scaling on the input axis.
# ---------------------------------------------------------------------------
def _at_resize_2d(W: torch.Tensor, out_dim: int, in_dim: int) -> torch.Tensor:
    old_out, old_in = W.shape
    w = W.detach().to(torch.float32).cpu()
    if old_in != in_dim:
        w = F.interpolate(
            w.unsqueeze(0), size=in_dim, mode="linear", align_corners=False
        ).squeeze(0)
    if old_out != out_dim:
        w = (
            F.interpolate(
                w.t().unsqueeze(0), size=out_dim, mode="linear", align_corners=False
            )
            .squeeze(0)
            .t()
            .contiguous()
        )
    if old_in != in_dim:
        w = w * (old_in / in_dim) ** 0.5
    return w


def _at_resize_1d(b: torch.Tensor, dim: int) -> torch.Tensor:
    x = b.detach().to(torch.float32).cpu()
    if x.shape[-1] == dim:
        return x.clone()
    return F.interpolate(
        x.reshape(1, 1, -1), size=dim, mode="linear", align_corners=False
    ).reshape(-1)


def _at_copy_linear(dst: nn.Linear, src: nn.Linear):
    out_dim, in_dim = dst.weight.shape
    new_w = _at_resize_2d(src.weight.data, out_dim, in_dim)
    dst.weight.data.copy_(new_w.to(dtype=dst.weight.dtype, device=dst.weight.device))
    if dst.bias is not None and src.bias is not None:
        new_b = _at_resize_1d(src.bias.data, out_dim)
        dst.bias.data.copy_(new_b.to(dtype=dst.bias.dtype, device=dst.bias.device))


def _at_copy_norm(dst, src):
    w_dst = getattr(dst, "weight", None)
    w_src = getattr(src, "weight", None)
    if w_dst is not None and w_src is not None:
        w_dst.data.copy_(
            _at_resize_1d(w_src.data, w_dst.shape[0]).to(
                dtype=w_dst.dtype, device=w_dst.device
            )
        )
    b_dst = getattr(dst, "bias", None)
    b_src = getattr(src, "bias", None)
    if b_dst is not None and b_src is not None:
        b_dst.data.copy_(
            _at_resize_1d(b_src.data, b_dst.shape[0]).to(
                dtype=b_dst.dtype, device=b_dst.device
            )
        )


def _at_copy_modulation(dst_p: torch.nn.Parameter, src_p: torch.nn.Parameter):
    target_dim = dst_p.shape[-1]
    s = src_p.detach().to(torch.float32).cpu()
    if s.shape[-1] != target_dim:
        s_flat = s.reshape(-1, s.shape[-1])
        s_resized = F.interpolate(
            s_flat.unsqueeze(0), size=target_dim, mode="linear", align_corners=False
        ).squeeze(0)
        s = s_resized.reshape(*src_p.shape[:-1], target_dim)
    dst_p.data.copy_(s.to(dtype=dst_p.dtype, device=dst_p.device))


try:
    from .vram_management import (
        enable_vram_management,
        AutoWrappedModule,
        AutoWrappedLinear,
        WanAutoCastLayerNorm,
    )
except ImportError:
    enable_vram_management = None
    AutoWrappedModule = None
    AutoWrappedLinear = None
    WanAutoCastLayerNorm = None


class WanVideoPipeline(BasePipeline):

    def __init__(
        self,
        device="cuda",
        torch_dtype=torch.bfloat16,
        tokenizer_path=None,
        ###
        enable_action=True,
        action_dim=14,
        action_chunk_t=40,
        action_batch_size: int = 1,
        action_loss_weight=1.0,
        video_loss_weight=1.0,
        video_loss_weight_start_step=15000,
        action_video_block_mapping=None,
        ignore_video_loss=False,
        constraint_loss_weight=0.0,
        use_qwen25: bool = False,
        qwen25_path: str = None,
        use_qwen35: bool = False,
        qwen35_path: str = None,
        qwen_proj_out_dim: int = 4096,
        qwen25_cascade_t5: bool = False,
        qwen35_cascade_t5: bool = False,
        qwen35_system_prompt: Optional[str] = None,
        print_qwen_output: bool = False,
        separate_flow_matching: bool = False,
        num_views: int = 3,
        use_sightcone_matching: bool = False,
        cross_view_masking_prob: float = 0.0,
        cross_view_masking_condition_prob: float = 0.0,
        cross_view_masking_max_len: int = 5,
        mask_region_loss_weight: float = 1.0,
        share_state_action_encoder: bool = True,
        local_alignment_pe: bool = False,
        state_attn: bool = False,
        view_as_tokens: bool = False,
        action_timestep_mapping: Optional[dict] = None,
        video_early_stop_for_action: bool = True,
        video_train_anchors: Optional[Union[str, List[int]]] = None,
        video_train_anchor_inference_steps: int = 50,
        video_train_window: int = 20,
        # In inference, run action denoising inside the video loop while video features stay on GPU.
        sync_action_denoising: bool = False,
        # In action-only inference, skip VAE decoding and return only predicted actions.
        skip_vae_decode: bool = False,
        action_pred_type: str = "v_pred",
        second_head_loss_weight: float = 1.0,
        dct_loss_weight: float = 0.0,
        action_as_condition: bool = False,
        action_tokenizer_ckpt_path: Optional[str] = None,
        skip_text_encoder: bool = False,
        action_transformer_dim: Optional[int] = None,
        action_per_video_frame: int = 8,
        scheduler_shift: float = 5.0,
        timestep_sample_mode: str = "uniform",
        action_beta_alpha: float = 1.0,
        action_beta_beta: float = 3.0,
        enable_action_training_weight: bool = False,
        init_action_from_dit: bool = False,
        init_cross_attn_q_from_dit: bool = False,
        zero_init_cross_attn_o: bool = False,
        zero_init_action_output: bool = False,
        vga_post_training: bool = False,
        vga_post_action_updates_per_video: int = 1,
        detach_video_features: bool = True,
        **kwargs,
    ):
        super().__init__(
            device=device,
            torch_dtype=torch_dtype,
            height_division_factor=16,
            width_division_factor=16,
            time_division_factor=4,
            time_division_remainder=1,
        )
        self.skip_text_encoder = skip_text_encoder
        # scheduler_shift controls the sigma(index) curvature. Default 5 is
        # the Wan default (concave, compresses clean region to the tail).
        # shift=1 gives a linear sigma curve (no compression). If you change
        # this, remember that the optimal action Beta(alpha,beta) sampling
        # also changes — see tools/plot_beta_shift_analysis.py.
        self.scheduler_shift = scheduler_shift
        self.timestep_sample_mode = timestep_sample_mode
        self.action_beta_alpha = action_beta_alpha
        self.action_beta_beta = action_beta_beta
        self.enable_action_training_weight = enable_action_training_weight
        self.scheduler = FlowMatchScheduler(
            shift=scheduler_shift, sigma_min=0.0, extra_one_step=True
        )
        self.action_batch_size = action_batch_size
        # Dual flow matching support
        self.separate_flow_matching = separate_flow_matching
        logger.info(
            "separate_flow_matching: %s, scheduler_shift: %s",
            self.separate_flow_matching,
            scheduler_shift,
        )
        if separate_flow_matching:
            # Create separate scheduler for action
            self.action_scheduler = FlowMatchScheduler(
                shift=scheduler_shift, sigma_min=0.0, extra_one_step=True
            )
        else:
            # Use the same scheduler
            self.action_scheduler = self.scheduler

        # Action prediction type: "v_pred" (velocity) or "x_pred" (data, JiT-style)
        # "v_pred_uni" / "x_pred_uni": dual-head, train both v_pred and x_pred losses,
        # inference uses the prefix before "_uni" (v_pred or x_pred).
        self.action_pred_type = action_pred_type
        self.second_head_loss_weight = second_head_loss_weight
        # For inference: strip "_uni" to get actual pred type
        self._action_infer_pred_type = action_pred_type.replace("_uni", "")
        logger.info(
            "action_pred_type: %s, dual_head=%s, infer_pred_type=%s",
            self.action_pred_type,
            action_pred_type.endswith("_uni"),
            self._action_infer_pred_type,
        )

        # Inference: map action_progress_id -> video_progress_id for conditioning (e.g. share_ts)
        self.action_timestep_mapping = action_timestep_mapping
        # When True and action_timestep_mapping is set, only run video steps 0..max(mapping.values()) to save compute
        self.video_early_stop_for_action = video_early_stop_for_action
        if isinstance(video_train_anchors, str):
            video_train_anchors = [
                int(x.strip()) for x in video_train_anchors.split(",") if x.strip()
            ]
        elif video_train_anchors is not None:
            video_train_anchors = [int(x) for x in video_train_anchors]
        self.video_train_anchors = video_train_anchors
        self.video_train_anchor_inference_steps = int(
            video_train_anchor_inference_steps
        )
        self.video_train_window = int(video_train_window)
        self._video_train_anchor_cache_key = None
        self._video_train_anchor_ids = None
        self.vga_post_training = vga_post_training
        self.vga_post_action_updates_per_video = max(
            1, int(vga_post_action_updates_per_video)
        )
        self.detach_video_features = detach_video_features
        logger.info(
            "vga_post_training: %s (alternate 1:%s video:action), "
            "detach_video_features: %s",
            self.vga_post_training,
            self.vga_post_action_updates_per_video,
            self.detach_video_features,
        )
        self.sync_action_denoising = sync_action_denoising
        self.skip_vae_decode = skip_vae_decode

        # Qwen2.5/3.5 configuration
        self.use_qwen25 = use_qwen25
        self.qwen25_path = qwen25_path
        self.use_qwen35 = use_qwen35
        self.qwen35_path = qwen35_path
        self.qwen_proj_out_dim = qwen_proj_out_dim
        self.qwen25_cascade_t5 = qwen25_cascade_t5
        self.qwen35_cascade_t5 = qwen35_cascade_t5
        self.qwen35_system_prompt = qwen35_system_prompt
        self.print_qwen_output = print_qwen_output
        self.prompter = WanPrompter(
            tokenizer_path=tokenizer_path,
            use_qwen25=self.use_qwen25,
            qwen25_path=self.qwen25_path,
            use_qwen35=self.use_qwen35,
            qwen35_path=self.qwen35_path,
            qwen35_system_prompt=self.qwen35_system_prompt,
            qwen_proj_out_dim=self.qwen_proj_out_dim,
            qwen25_cascade_t5=self.qwen25_cascade_t5,
            qwen35_cascade_t5=self.qwen35_cascade_t5,
            skip_text_encoder=self.skip_text_encoder,
        )

        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        ###
        self.action_loss_weight = action_loss_weight  # Action loss weight
        self.video_loss_weight = video_loss_weight  # Video loss weight
        self.video_loss_weight_start_step = (
            video_loss_weight_start_step  # Step at which video_loss_weight takes effect
        )
        self.action_video_block_mapping = (
            action_video_block_mapping  # Mapping from action to video blocks
        )
        self.ignore_video_loss = ignore_video_loss
        self.constraint_loss_weight = constraint_loss_weight
        self.dct_loss_weight = dct_loss_weight
        self.use_sightcone_matching: bool = bool(use_sightcone_matching)
        self.cross_view_masking_prob: float = float(cross_view_masking_prob)
        self.cross_view_masking_condition_prob: float = float(
            cross_view_masking_condition_prob
        )
        self.cross_view_masking_max_len: int = int(cross_view_masking_max_len)
        self.mask_region_loss_weight: float = float(mask_region_loss_weight)

        self.dit: WanModel = None
        self.dit2: WanModel = None
        self.vae: WanVideoVAE = None
        self.in_iteration_models = (
            "dit",
            "motion_controller",
            "vace",
            "animate_adapter",
            "action_transformer",
            "action_conditioner",
        )
        self.in_iteration_models_2 = (
            "dit2",
            "motion_controller",
            "vace2",
            "animate_adapter",
            "action_transformer",
            "action_conditioner",
        )
        self.unit_runner = PipelineUnitRunner()
        self.units = [
            WanVideoUnit_ShapeChecker(),
            WanVideoUnit_NoiseInitializer(),
            WanVideoUnit_PromptEmbedder(),
            # WanVideoUnit_S2V(),
            WanVideoUnit_InputVideoEmbedder(),
            WanVideoUnit_ImageEmbedderVAE(),
            WanVideoUnit_ImageEmbedderCLIP(),
            WanVideoUnit_ImageEmbedderFused(),
            WanVideoUnit_ActionContext(),
            # WanVideoUnit_FunControl(),
            # WanVideoUnit_FunReference(),
            # WanVideoUnit_FunCameraControl(),
            # WanVideoUnit_SpeedControl(),
            # WanVideoUnit_VACE(),
            WanVideoPostUnit_AnimateVideoSplit(),
            # WanVideoPostUnit_AnimatePoseLatents(),
            # WanVideoPostUnit_AnimateFacePixelValues(),
            # WanVideoPostUnit_AnimateInpaint(),
            # WanVideoUnit_UnifiedSequenceParallel(),
            # WanVideoUnit_TeaCache(),
            # WanVideoUnit_CfgMerger(),
        ]
        self.post_units = [
            # WanVideoPostUnit_S2V(),
        ]
        self.model_fn = model_fn_wan_video
        self.camera_buffer = {}
        self.sight_cone_buffer = {}
        if self.use_sightcone_matching or self.cross_view_masking_prob > 0:
            self.units += [WanVideoUnit_SightconeViewMask()]

        ### Action prediction control
        self.enable_action: bool = enable_action
        self.action_as_condition: bool = action_as_condition

        # Action config (ActionTransformer is created lazily in _build_action_transformer
        # after DiT is loaded, so that dim/num_heads/num_layers can match the DiT)
        self.action_transformer = None
        self.action_conditioner = None
        # When True, copy DiT block weights into ActionTransformer blocks after build
        # (interpolating across dim mismatch and applying variance-preserving scaling).
        self.init_action_from_dit: bool = bool(init_action_from_dit)
        self.init_cross_attn_q_from_dit: bool = bool(init_cross_attn_q_from_dit)
        self.zero_init_cross_attn_o: bool = bool(zero_init_cross_attn_o)
        self.zero_init_action_output: bool = bool(zero_init_action_output)
        if self.enable_action:
            self.action_embed_dim: int = 768
            self.action_mlp_ratio: float = 4.0
            self.action_dim: int = action_dim
            self.action_transformer_dim: Optional[int] = action_transformer_dim
            self.num_views: int = num_views
            self.share_state_action_encoder: bool = share_state_action_encoder
            self.local_alignment_pe: bool = local_alignment_pe
            self.state_attn: bool = state_attn
            self.view_as_tokens: bool = view_as_tokens
            self.action_per_video_frame: int = action_per_video_frame

    def _get_camera(
        self,
        robot_id,
        file_path,
        dataset_name,
        view_map=None,
        read_whs=None,
        target_wh=(224, 176),
    ):
        if robot_id in self.camera_buffer:
            return self.camera_buffer[robot_id]
        try:
            from wall_x.model.vga.utils_camera import MAPPING_DATASET_TO_CAMERA

            if not file_path:
                logger.warning(
                    "[camera_loading] camera_information_path is not specified, no files to read."
                )
                return None
            construct_whs = {view_map[key]: read_whs[key] for key in read_whs}
            camera_result = MAPPING_DATASET_TO_CAMERA[dataset_name](
                file_path, target_wh_or_whs=construct_whs, resize_mode="crop"
            )
            for key in view_map:
                view_name = view_map[key]
                camera_result[f"camera_{view_name}"].resize_intrinsics(
                    dst_wh=target_wh, resize_mode="resize"
                )
            self.camera_buffer[robot_id] = camera_result
            return camera_result
        except Exception:
            logger.warning(
                "[camera_loading] Fail to load camera object for robot %s.",
                robot_id,
                exc_info=True,
            )
        return None

    def _get_sight_cone(
        self,
        robot_id,
        file_path,
        dataset_name,
        view_name=None,
        view_map=None,
        read_whs=None,
        target_wh=(224, 176),
    ):
        cache_key = (
            robot_id,
            view_map[view_name] if view_map and view_name else view_name,
        )
        if cache_key in self.sight_cone_buffer:
            return self.sight_cone_buffer[cache_key]
        try:
            from wall_x.model.vga.utils_camera import MAPPING_DATASET_TO_SIGHTCONE

            camera_result = self._get_camera(
                robot_id=robot_id,
                file_path=file_path,
                dataset_name=dataset_name,
                view_map=view_map,
                read_whs=read_whs,
                target_wh=target_wh,
            )
            sight_cone = MAPPING_DATASET_TO_SIGHTCONE[dataset_name](
                camera_result[f"camera_{view_map[view_name]}"]
            )
            sight_cone.to(self.device)
            self.sight_cone_buffer[cache_key] = sight_cone
            return sight_cone
        except Exception:
            logger.warning(
                "[camera_loading] Fail to load sightcones object for robot %s.",
                robot_id,
                exc_info=True,
            )
        return None

    def _build_action_transformer(
        self, dit_dim: int, dit_num_heads: int, dit_num_layers: int
    ):
        """Build ActionTransformer with dimensions matching the loaded DiT.

        Called from from_pretrained() after DiT weights are loaded.
        """
        if not self.enable_action:
            return

        # Compute effective action transformer dim and num_heads
        act_dim = self.action_transformer_dim or dit_dim
        assert dit_num_heads > 0 and dit_dim >= dit_num_heads, (
            f"Invalid DiT layout: need dit_num_heads>0 and dit_dim>=dit_num_heads, "
            f"got dit_dim={dit_dim}, dit_num_heads={dit_num_heads}"
        )
        head_dim = dit_dim // dit_num_heads
        assert head_dim > 0 and act_dim % head_dim == 0, (
            f"action_transformer_dim ({act_dim}) must be divisible by "
            f"head_dim ({head_dim} = dit_dim//dit_num_heads = {dit_dim}//{dit_num_heads})"
        )
        act_num_heads = act_dim // head_dim

        # Scale action ffn_dim proportionally to act_dim (base: 4096 for dim=1536)
        action_ffn_dim = max(4096, int(4096 * act_dim / 1536))

        logger.info(
            "[ActionTransformer] Building with act_dim=%s, dit_dim=%s, "
            "act_num_heads=%s, dit_num_layers=%s, action_ffn_dim=%s",
            act_dim,
            dit_dim,
            act_num_heads,
            dit_num_layers,
            action_ffn_dim,
        )

        self.action_transformer = ActionTransformer(
            dim=act_dim,
            num_heads=act_num_heads,
            num_layers=dit_num_layers,
            ffn_dim=action_ffn_dim,
            dit_dim=dit_dim if act_dim != dit_dim else None,
            action_dim=self.action_dim,
            action_embed_dim=self.action_embed_dim,
            action_mlp_ratio=self.action_mlp_ratio,
            video_block_mapping=self.action_video_block_mapping,
            num_views=self.num_views,
            view_as_tokens=self.view_as_tokens,
            share_state_action_encoder=self.share_state_action_encoder,
            local_alignment_pe=self.local_alignment_pe,
            state_attn=self.state_attn,
            action_pred_type=self.action_pred_type,
            action_per_video_frame=self.action_per_video_frame,
        ).to(device=self.device, dtype=self.torch_dtype)

    @torch.no_grad()
    def _init_action_transformer_from_dit(self) -> None:
        """Initialize ActionTransformer block weights from the corresponding DiT blocks.

        For each action block i, copy weights from DiT block ``video_block_mapping[i]``.
        When ``act_dim != dit_dim`` (e.g. Wan 5B/14B with smaller action_transformer_dim),
        weights are linearly interpolated along the relevant dim(s) and rescaled by
        ``alpha = sqrt(old_in / new_in)`` so that forward output variance is preserved
        (paper trick: action tokens start with a distribution comparable to video tokens).

        Modules copied per block:
          - self_attn.{q, k, v, o, norm_q, norm_k}
          - norm3 (LayerNorm with affine, used for video cross-attn norm)
          - ffn[0], ffn[2]
          - modulation parameter (1, 6, dim)

        Skipped (action-only or shape-incompatible):
          - cross_attn (multi-view, different K/V context dim)
          - cross_attn_state, norm4 (action-only)
          - norm1, norm2 (no affine params)
        """
        if not self.enable_action or self.action_transformer is None:
            logger.info(
                "[ActionTransformer init-from-DiT] skipped: action disabled or transformer not built"
            )
            return
        if self.dit is None or not hasattr(self.dit, "blocks"):
            logger.info("[ActionTransformer init-from-DiT] skipped: pipe.dit missing")
            return

        act = self.action_transformer
        dit_blocks = self.dit.blocks
        n_dit = len(dit_blocks)
        mapping = act.video_block_mapping
        act_dim = act.dim
        dit_dim = dit_blocks[0].self_attn.q.weight.shape[0]

        n_copied = 0
        for act_idx, dit_idx in enumerate(mapping):
            # Match forward's clamp behavior so trailing past-the-end mapping values
            # (e.g. legacy "1..30" with n_dit==30) still copy from the last DiT block.
            dit_idx = max(0, min(int(dit_idx), n_dit - 1))
            ab = act.blocks[act_idx]
            db = dit_blocks[dit_idx]
            _at_copy_linear(ab.self_attn.q, db.self_attn.q)
            _at_copy_linear(ab.self_attn.k, db.self_attn.k)
            _at_copy_linear(ab.self_attn.v, db.self_attn.v)
            _at_copy_linear(ab.self_attn.o, db.self_attn.o)
            _at_copy_norm(ab.self_attn.norm_q, db.self_attn.norm_q)
            _at_copy_norm(ab.self_attn.norm_k, db.self_attn.norm_k)
            _at_copy_norm(ab.norm3, db.norm3)
            _at_copy_linear(ab.ffn[0], db.ffn[0])
            _at_copy_linear(ab.ffn[2], db.ffn[2])
            _at_copy_modulation(ab.modulation, db.modulation)
            n_copied += 1

        logger.info(
            "[ActionTransformer init-from-DiT] copied %s/%s blocks "
            "(act_dim=%s, dit_dim=%s, alpha=sqrt(%s/%s)=%.4f)",
            n_copied,
            len(act.blocks),
            act_dim,
            dit_dim,
            dit_dim,
            act_dim,
            (dit_dim / max(act_dim, 1)) ** 0.5,
        )

    @torch.no_grad()
    def _init_cross_attn_q_from_dit(self) -> None:
        """Init ActionBlock.cross_attn.q (and norm_q) from DiT's self_attn.q at the
        same mapped block. Aligns the query-side projection with how DiT projects
        same-distribution tokens, so multi-view video features (acting as K/V here)
        get queried with a geometry consistent with the just-copied self_attn output.

        Uses the same interp + alpha=sqrt(dit_dim/act_dim) variance scaling as
        _init_action_transformer_from_dit. K/V are intentionally NOT copied (DiT's K/V
        learned to project text/self-features, not multi-view video).
        """
        if not self.enable_action or self.action_transformer is None:
            logger.info(
                "[cross_attn Q init] skipped: action disabled or transformer not built"
            )
            return
        if self.dit is None or not hasattr(self.dit, "blocks"):
            logger.info("[cross_attn Q init] skipped: pipe.dit missing")
            return
        act = self.action_transformer
        dit_blocks = self.dit.blocks
        n_dit = len(dit_blocks)
        mapping = act.video_block_mapping

        n = 0
        for act_idx, dit_idx in enumerate(mapping):
            dit_idx = max(0, min(int(dit_idx), n_dit - 1))
            ab = act.blocks[act_idx]
            db = dit_blocks[dit_idx]
            _at_copy_linear(ab.cross_attn.q, db.self_attn.q)
            _at_copy_norm(ab.cross_attn.norm_q, db.self_attn.norm_q)
            n += 1
        logger.info(
            "[cross_attn Q init] copied DiT self_attn.q -> ActionBlock.cross_attn.q for %s/%s blocks",
            n,
            len(act.blocks),
        )

    @torch.no_grad()
    def _zero_init_cross_attn_o(self) -> None:
        """Zero-init ActionBlock.cross_attn.o (and cross_attn_state.o if present).

        ControlNet-style residual zero-init: the entire cross-attn contribution
        starts at 0, so step-0 forward = action self-attn path only (which was
        DiT-init). cross-attn gradients still flow through `o`, so K/V/Q learn
        organically once `o` moves off zero.
        """
        if not self.enable_action or self.action_transformer is None:
            logger.info(
                "[cross_attn O zero-init] skipped: action disabled or transformer not built"
            )
            return
        act = self.action_transformer
        n = 0
        for ab in act.blocks:
            nn.init.zeros_(ab.cross_attn.o.weight)
            if ab.cross_attn.o.bias is not None:
                nn.init.zeros_(ab.cross_attn.o.bias)
            cas = getattr(ab, "cross_attn_state", None)
            if cas is not None:
                nn.init.zeros_(cas.o.weight)
                if cas.o.bias is not None:
                    nn.init.zeros_(cas.o.bias)
            n += 1
        logger.info(
            "[cross_attn O zero-init] zeroed cross_attn.o for %s/%s blocks",
            n,
            len(act.blocks),
        )

    @torch.no_grad()
    def _zero_init_action_output(self) -> None:
        """Zero-init the last Linear of action_decoder so step-0 prediction = 0.

        For v_pred this gives initial loss = E[||v_target||^2] (~2D in normalized
        space), still lower than typical random-init loss because random output
        also adds its own variance to the residual. Cleaner gradient: only the
        last layer learns the output scale initially; upper layers (DiT-init) are
        not perturbed by mis-scaled output gradients.

        For dual_head mode, both action_decoder and action_decoder_second are zeroed.
        """
        if not self.enable_action or self.action_transformer is None:
            logger.info(
                "[action output zero-init] skipped: action disabled or transformer not built"
            )
            return
        act = self.action_transformer

        def _zero_last_linear(seq: nn.Sequential, tag: str):
            last = seq[-1]
            nn.init.zeros_(last.weight)
            if last.bias is not None:
                nn.init.zeros_(last.bias)
            logger.info(
                "[action output zero-init] zeroed %s[-1] (Linear %s)",
                tag,
                tuple(last.weight.shape),
            )

        _zero_last_linear(act.action_decoder, "action_decoder")
        if getattr(act, "dual_head", False):
            _zero_last_linear(act.action_decoder_second, "action_decoder_second")

    def _build_action_conditioner(self, dit_dim: int, action_tokenizer_ckpt_path: str):
        """Build ActionConditioner with dimensions matching the loaded DiT.

        Called from from_pretrained() after DiT weights are loaded.
        """
        if not self.action_as_condition:
            return

        logger.info(
            "[ActionConditioner] Building with dit_dim=%s, action_tokenizer_ckpt_path=%s",
            dit_dim,
            action_tokenizer_ckpt_path,
        )

        self.action_conditioner = ActionConditioner(
            dim=dit_dim,
            checkpoint_path=action_tokenizer_ckpt_path,
            device=self.device,
        ).to(device=self.device, dtype=self.torch_dtype)

    def _ensure_video_train_anchor_ids(self) -> None:
        video_anchors = getattr(self, "video_train_anchors", None)
        if not video_anchors:
            return
        n_inf = int(getattr(self, "video_train_anchor_inference_steps", 50))
        cache_key = (
            tuple(video_anchors),
            n_inf,
            float(self.scheduler_shift),
            len(self.scheduler.timesteps),
        )
        if self._video_train_anchor_cache_key == cache_key:
            return
        ref_scheduler = FlowMatchScheduler(
            shift=self.scheduler_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        ref_scheduler.set_timesteps(n_inf)
        train_timesteps = self.scheduler.timesteps.detach().to("cpu")
        ref_timesteps = ref_scheduler.timesteps.detach().to("cpu")
        anchor_ids = []
        for anchor in video_anchors:
            if anchor < 0 or anchor >= n_inf:
                raise ValueError(
                    f"video_train_anchor {anchor} out of range [0,{n_inf})"
                )
            anchor_ids.append(
                int(torch.argmin((train_timesteps - ref_timesteps[anchor]).abs()))
            )
        self._video_train_anchor_ids = anchor_ids
        self._video_train_anchor_cache_key = cache_key

    def _sample_anchor_timestep_id(
        self,
        *,
        min_timestep_boundary: int,
        max_timestep_boundary: int,
    ) -> torch.Tensor:
        self._ensure_video_train_anchor_ids()
        window = int(getattr(self, "video_train_window", 20))
        if window < 0:
            raise ValueError(f"video_train_window must be non-negative, got {window}")
        anchor_id = self._video_train_anchor_ids[
            int(torch.randint(0, len(self._video_train_anchor_ids), (1,)).item())
        ]
        lo = max(min_timestep_boundary, anchor_id - window)
        hi = min(max_timestep_boundary, anchor_id + window + 1)
        if hi <= lo:
            return torch.tensor(
                [
                    max(
                        min(anchor_id, max_timestep_boundary - 1),
                        min_timestep_boundary,
                    )
                ]
            )
        return torch.randint(lo, hi, (1,))

    def _post_training_phase(self, batch_idx: int) -> str:
        """Alternating post-train: one video step per ``1 + N`` steps, then N action steps."""
        action_updates = max(
            1, int(getattr(self, "vga_post_action_updates_per_video", 1))
        )
        cycle = 1 + action_updates
        return "video" if batch_idx % cycle == 0 else "action"

    def _inject_action_context(self, inputs: dict) -> None:
        """Populate ``inputs["action_context"]`` when action_as_condition is enabled."""
        if not self.action_as_condition:
            return
        actions_cond = inputs.get("actions", None)
        agent_pos_cond = inputs.get("agent_pos", None)
        dof_mask = inputs.get("dof_mask", None)
        if actions_cond is not None:
            actions_cond = actions_cond.to(device=self.device, dtype=self.torch_dtype)
            if agent_pos_cond is not None:
                agent_pos_cond = agent_pos_cond.to(
                    device=self.device, dtype=self.torch_dtype
                )
            inputs["action_context"] = self.action_conditioner(
                actions_cond, agent_pos_cond, dof_mask
            )

    def training_loss(self, batch_idx=0, **inputs):
        max_timestep_boundary = int(
            inputs.get("max_timestep_boundary", 1) * self.scheduler.num_train_timesteps
        )
        min_timestep_boundary = int(
            inputs.get("min_timestep_boundary", 0) * self.scheduler.num_train_timesteps
        )
        # Video timestep sampling.
        # Default: uniform over [min_timestep_boundary, max_timestep_boundary).
        # When configured, video_train_anchors restricts sampling to a window
        # around the matching training index for an inference schedule anchor.
        post_training = getattr(self, "vga_post_training", False)
        post_phase = self._post_training_phase(batch_idx) if post_training else None
        video_anchors = getattr(self, "video_train_anchors", None)

        apply_video_loss = post_phase is None or post_phase == "video"
        apply_action_loss = post_phase is None or post_phase == "action"
        run_action_branch = (
            self.enable_action
            and not self.action_as_condition
            and inputs.get("actions", None) is not None
            and (post_phase is None or post_phase in ("video", "action"))
        )

        loss_dict = {"total_loss": 0.0, "video_loss": None, "action_loss": None}
        if post_phase is not None:
            loss_dict["post_train_phase"] = post_phase

        block_time_features = None
        training_target = None
        noise_pred = None
        timestep = None
        video_loss = None
        actual_video_loss_weight = 1.0
        latent_mask = inputs.pop("latent_mask", None)
        view_masking_condition = inputs.pop("view_masking_condition", True)

        def apply_latent_mask() -> None:
            nonlocal latent_mask
            if latent_mask is None:
                return
            latent_mask = latent_mask.to(
                inputs["latents"].device, inputs["latents"].dtype
            )
            latent_mask = latent_mask.repeat(1, 1, inputs["latents"].shape[2], 1, 1)
            if not view_masking_condition:
                latent_mask[:, :, 0] = 0.0
            inputs["latents"] = (1.0 - latent_mask) * inputs[
                "latents"
            ] + latent_mask * inputs["noise"]

        if post_phase == "action":
            if not video_anchors:
                raise ValueError(
                    "vga_post_training=True requires video_train_anchors to be set"
                )
            # Match action_pretrain video feature path: anchor-window t sampling,
            # pipeline inputs["noise"] (fp32 randn -> bf16), then detach below.
            timestep_id = self._sample_anchor_timestep_id(
                min_timestep_boundary=min_timestep_boundary,
                max_timestep_boundary=max_timestep_boundary,
            )
            timestep = (
                self.scheduler.timesteps[timestep_id]
                .to(dtype=torch.float32, device=self.device)
                .reshape(-1)[0]
            )
            inputs["latents"] = self.scheduler.add_noise(
                inputs["input_latents"], inputs["noise"], timestep
            )
            apply_latent_mask()
            self._inject_action_context(inputs)
            noise_pred, _, block_time_features = self.model_fn(
                **inputs, timestep=timestep, return_video_features=True
            )
            if self.detach_video_features:
                block_time_features = [bt.detach() for bt in block_time_features]
        else:
            if post_training:
                timestep_id = torch.randint(
                    min_timestep_boundary, max_timestep_boundary, (1,)
                )
            elif video_anchors:
                timestep_id = self._sample_anchor_timestep_id(
                    min_timestep_boundary=min_timestep_boundary,
                    max_timestep_boundary=max_timestep_boundary,
                )
            else:
                timestep_id = torch.randint(
                    min_timestep_boundary, max_timestep_boundary, (1,)
                )
            timestep = self.scheduler.timesteps[timestep_id].to(
                dtype=torch.float32, device=self.device
            )

            inputs["latents"] = self.scheduler.add_noise(
                inputs["input_latents"], inputs["noise"], timestep
            )
            apply_latent_mask()
            training_target = self.scheduler.training_target(
                inputs["input_latents"], inputs["noise"], timestep
            )

            self._inject_action_context(inputs)

            if run_action_branch:
                noise_pred, video_time_features, block_time_features = self.model_fn(
                    **inputs, timestep=timestep, return_video_features=True
                )
            else:
                noise_pred = self.model_fn(
                    **inputs, timestep=timestep, return_video_features=False
                )

        if post_phase == "action":
            # Feature-only DiT forward; skip video MSE (not trained on action steps).
            if noise_pred is not None:
                loss_dict["total_loss"] = (
                    loss_dict["total_loss"] + noise_pred.sum() * 0.0
                )
        elif not self.ignore_video_loss and noise_pred is not None:
            if latent_mask is not None:
                video_loss = torch.nn.functional.mse_loss(
                    noise_pred.float(), training_target.float(), reduction="none"
                )
                video_loss = (
                    video_loss
                    * (1.0 + (self.mask_region_loss_weight - 1.0) * latent_mask)
                ).mean()
            else:
                video_loss = torch.nn.functional.mse_loss(
                    noise_pred.float(), training_target.float()
                )
            video_loss = video_loss * self.scheduler.training_weight(timestep)

            if self.video_loss_weight != 1.0:
                if batch_idx > self.video_loss_weight_start_step:
                    actual_video_loss_weight = self.video_loss_weight
                else:
                    actual_video_loss_weight = 1.0
            else:
                actual_video_loss_weight = 1.0

            loss_dict["video_loss"] = (
                (actual_video_loss_weight * video_loss).detach().float()
            )
            if apply_video_loss:
                loss_dict["total_loss"] += actual_video_loss_weight * video_loss
        else:
            loss_dict["video_loss"] = torch.tensor(0.0, device=self.device)

        # Action prediction loss (if enabled and actions provided)
        actions = inputs.get("actions", None)  # e.g., [B, 21, action_dim]
        agent_pos = inputs.get("agent_pos", None)
        action_history = inputs.get("action_history", None)

        if run_action_branch:
            actions = actions.to(device=self.device)
            B, T_action, action_dim = actions.shape
            action_batch_size = getattr(self, "action_batch_size", 1)

            # Compute action timestep(s) (may differ from video)
            # ----- ACTION timestep sampling -----
            # self.timestep_sample_mode (wired from args.timestep_sample_mode):
            #   "uniform"         — original behavior (uniform over all t)
            #   "high_noise_beta" — Beta(alpha, beta) biased toward LOW index
            #                       (= HIGH sigma = HIGH noise). Default
            #                       (alpha, beta) = (1, 3) is tuned for
            #                       scheduler_shift=5. At shift=1 use (1,12)
            #                       to roughly match the same sigma
            #                       distribution (mean sigma ≈ 0.92, % clean
            #                       ~ 0%). Tune via config keys:
            #                         action_beta_alpha / action_beta_beta
            #
            # NOTE on the (index ↔ sigma) mapping in FlowMatchScheduler:
            #   scheduler.sigmas is DECREASING — sigmas[0]=1.0 (pure noise),
            #   sigmas[-1]≈0 (clean). At shift=5 the sigma curve is concave:
            #   index 0..500 all have sigma>0.83 (noisy), only index 876..999
            #   are in the clean regime. At shift=1 the curve is linear, so
            #   clean region is index > 800 (20% of indices). In both cases
            #   LOW index = HIGH sigma, so Beta(α<β) is correct.
            _action_sample_mode = getattr(self, "timestep_sample_mode", "uniform")
            _action_beta_alpha = float(getattr(self, "action_beta_alpha", 1.0))
            _action_beta_beta = float(getattr(self, "action_beta_beta", 3.0))

            def _sample_action_ts_ids(k: int) -> torch.Tensor:
                """Sample k action timestep indices in [0, num_ts).

                high_noise_beta mode: draws u ~ Beta(alpha, beta) and maps
                to index = floor(u * num_ts). With alpha < beta the pdf is
                skewed toward u=0, so samples concentrate near index 0 =
                high sigma = high noise.
                """
                if _action_sample_mode == "high_noise_beta":
                    u = torch.distributions.Beta(
                        _action_beta_alpha, _action_beta_beta
                    ).sample((k,))
                    ids = (u * num_ts).long().clamp(max=num_ts - 1)
                    return ids.to(device=self.device)
                else:
                    return torch.randperm(num_ts, device=self.device)[:k]

            if self.separate_flow_matching:
                # When action_batch_size > 1, ensure we always include the video timestep.
                # The remaining (K-1) timesteps are sampled without replacement.
                num_ts = self.action_scheduler.num_train_timesteps
                all_action_ts = self.action_scheduler.timesteps.to(
                    dtype=torch.float32, device=self.device
                )

                if action_batch_size > 1:
                    video_ts = timestep
                    if video_ts.dim() != 0:
                        # Best-effort: take first element if not scalar
                        video_ts = video_ts.reshape(-1)[0]
                    video_ts = video_ts.to(device=self.device)

                    if _action_sample_mode == "high_noise_beta":
                        # Beta-biased high-noise sampling (optional mode).
                        action_timestep_ids = _sample_action_ts_ids(action_batch_size)
                        action_timesteps = all_action_ts[action_timestep_ids]
                    else:
                        # Original uniform-mode behavior: include video_ts
                        # as the first slot, randperm the rest.
                        all_action_ts_f = all_action_ts.float()
                        video_ts_f = video_ts.float()
                        eq_mask = all_action_ts_f == video_ts_f
                        if not torch.any(eq_mask):
                            eq_mask = torch.isclose(all_action_ts_f, video_ts_f)
                        video_idx = torch.nonzero(eq_mask, as_tuple=False).reshape(-1)

                        remaining_perm = torch.randperm(num_ts, device=self.device)
                        if video_idx.numel() > 0:
                            remaining_perm = remaining_perm[
                                remaining_perm != video_idx[0]
                            ]
                            action_timestep_ids = torch.cat(
                                [
                                    video_idx[:1],
                                    remaining_perm[: action_batch_size - 1],
                                ],
                                dim=0,
                            )
                            action_timesteps = all_action_ts[action_timestep_ids]
                        else:
                            action_timestep_ids = remaining_perm[
                                : action_batch_size - 1
                            ]
                            action_timesteps = torch.cat(
                                [
                                    video_ts.unsqueeze(0),
                                    all_action_ts[action_timestep_ids],
                                ],
                                dim=0,
                            )
                else:
                    action_timestep_ids = _sample_action_ts_ids(1)
                    action_timesteps = all_action_ts[action_timestep_ids]

                action_timesteps = action_timesteps.to(
                    dtype=torch.float32, device=self.device
                )
                action_scheduler = self.action_scheduler
            else:
                if action_batch_size > 1:
                    raise ValueError(
                        "action_batch_size > 1 requires separate_flow_matching=True"
                    )
                action_timesteps = (
                    timestep.unsqueeze(0) if timestep.dim() == 0 else timestep
                )
                action_timesteps = action_timesteps.to(
                    dtype=torch.float32, device=self.device
                )
                if action_timesteps.dim() == 0:
                    action_timesteps = action_timesteps.unsqueeze(0)
                action_scheduler = self.scheduler

            # Unified path: expand and process (action_batch_size may be 1 or >1)
            actions_expanded = actions.repeat(action_batch_size, 1, 1)
            if agent_pos is not None:
                # agent_pos shape is [1, 1, 14] -> repeat on batch dim
                agent_pos_expanded = agent_pos.repeat(action_batch_size, 1, 1)
            else:
                agent_pos_expanded = None
            agent_pos_expanded = (
                agent_pos_expanded.to(dtype=self.torch_dtype)
                if agent_pos_expanded is not None
                else None
            )
            if action_history is not None:
                action_history_expanded = action_history.repeat(action_batch_size, 1, 1)
            else:
                action_history_expanded = None
            action_history_expanded = (
                action_history_expanded.to(dtype=self.torch_dtype)
                if action_history_expanded is not None
                else None
            )

            action_latents = actions_expanded.to(dtype=self.torch_dtype).clone()
            if T_action > 1:
                action_noise = self.generate_noise(
                    (action_batch_size * B, T_action, action_dim),
                    rand_device=self.device,
                    rand_torch_dtype=self.torch_dtype,
                    device=self.device,
                    torch_dtype=self.torch_dtype,
                )
                for i in range(action_batch_size):
                    s = i * B
                    e = (i + 1) * B
                    action_latents[s:e, :] = action_scheduler.add_noise(
                        actions_expanded[s:e, :].to(dtype=self.torch_dtype),
                        action_noise[s:e, :],
                        action_timesteps[i],
                    )

            action_t_embeds = self.dit.time_embedding(
                sinusoidal_embedding_1d(
                    self.dit.freq_dim,
                    action_timesteps,
                    output_dtype=self.torch_dtype,
                )
            )
            action_t_mod = self.dit.time_projection(action_t_embeds).unflatten(
                1, (6, self.dit.dim)
            )
            action_t_mod = action_t_mod.repeat_interleave(
                B, dim=0
            )  # [action_batch_size * B, 6, dim]
            if post_training:
                action_t_mod = action_t_mod.detach()

            _dual = self.action_transformer.dual_head
            expected_views = self.action_transformer.blocks[0].cross_attn.num_views
            if block_time_features:
                block0_shape = tuple(block_time_features[0].shape)
                b_video = int(block_time_features[0].shape[0])
                if expected_views <= 0 or b_video % expected_views != 0:
                    msg = (
                        "[WALLX_VIEW_MISMATCH] "
                        f"video feature batch dim={b_video} is not divisible by "
                        f"action transformer num_views={expected_views}; "
                        f"block0_shape={block0_shape}; "
                        f"actions_shape={tuple(actions.shape)}; "
                        f"agent_pos_shape={tuple(agent_pos.shape) if agent_pos is not None else None}; "
                        f"input_num_views={inputs.get('num_views')}"
                    )
                    logger.error("%s", msg)
                    raise ValueError(msg)
            action_transformer_out = self.action_transformer(
                action_latents,
                agent_pos_expanded,
                block_time_features,
                action_t_mod,
                action_history_expanded,
            )

            # Unpack dual-head output; pick the infer head for downstream (constraint/dct)
            if _dual:
                # primary_out corresponds to _action_infer_pred_type, second_out to the other
                action_noise_pred_primary, action_noise_pred_second = (
                    action_transformer_out
                )
                action_noise_pred = (
                    action_noise_pred_primary  # infer head for constraint/dct
                )
                # The other pred type
                _second_pred_type = (
                    "x_pred" if self._action_infer_pred_type == "v_pred" else "v_pred"
                )
            else:
                action_noise_pred = action_transformer_out

            if T_action > 1:
                primary_losses = []
                second_losses = []
                _train_pred_types = (
                    ["v_pred", "x_pred"] if _dual else [self.action_pred_type]
                )
                # Map pred_type -> head_output
                if _dual:
                    _pred_by_type = {
                        self._action_infer_pred_type: action_noise_pred_primary,
                        _second_pred_type: action_noise_pred_second,
                    }
                else:
                    _pred_by_type = {self.action_pred_type: action_noise_pred}

                for i in range(action_batch_size):
                    s = i * B
                    e = (i + 1) * B
                    for _pt in _train_pred_types:
                        _pred_out = _pred_by_type[_pt]
                        action_training_target = action_scheduler.training_target(
                            actions_expanded[s:e, :].to(dtype=self.torch_dtype),
                            action_noise[s:e, :],
                            action_timesteps[i],
                            pred_type=_pt,
                        )
                        loss_i = torch.nn.functional.mse_loss(
                            _pred_out[s:e, :].float(),
                            action_training_target.float(),
                            reduction="none",
                        )
                        dof_mask = inputs.get("dof_mask", None)
                        if dof_mask is not None:
                            loss_i = loss_i * dof_mask
                        loss_i = loss_i.mean()
                        # Optional per-timestep Gaussian training weight
                        # (matches DreamZero's behavior — DZ applies
                        # training_weight to BOTH video and action loss,
                        # while the wall-x default applied it to video
                        # only, leaving action loss floor-limited at low
                        # sigma). Toggle via
                        # args.enable_action_training_weight.
                        if getattr(self, "enable_action_training_weight", False):
                            loss_i = loss_i * action_scheduler.training_weight(
                                action_timesteps[i]
                            )
                        # Route to primary or second loss list
                        if _dual and _pt == _second_pred_type:
                            second_losses.append(loss_i)
                        else:
                            primary_losses.append(loss_i)

                action_loss = (
                    torch.stack(primary_losses).mean()
                    if primary_losses
                    else torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)
                )
                second_head_loss = (
                    torch.stack(second_losses).mean()
                    if second_losses
                    else torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)
                )
            else:
                action_loss = torch.tensor(
                    0.0, device=self.device, dtype=self.torch_dtype
                )
                second_head_loss = torch.tensor(
                    0.0, device=self.device, dtype=self.torch_dtype
                )

            if self.constraint_loss_weight > 0:
                total_constraint_losses = []
                constraint_details = {}
                valid_constraint_count = 0
                for i in range(action_batch_size):
                    s = i * B
                    e = (i + 1) * B
                    if action_timesteps[i].item() < 1001:
                        x0_hat = action_scheduler.invert_to_data_domain(
                            action_noise_pred[s:e, :],
                            action_latents[s:e, :],
                            action_timesteps[i],
                            pred_type=self._action_infer_pred_type,
                        )
                        constraint_losses = self.compute_action_constraint_losses(
                            x0_hat,
                            actions_expanded[s:e, :].to(dtype=self.torch_dtype),
                        )
                        t_norm = action_timesteps[i].item() / 1000.0
                        time_weight = (1.0 - t_norm) ** 3
                        total_constraint_losses.append(
                            constraint_losses["total"] * time_weight
                        )
                        valid_constraint_count += 1
                        for loss_name, loss_value in constraint_losses.items():
                            if loss_name != "total":
                                key = f"constraint_{loss_name}"
                                if key not in constraint_details:
                                    constraint_details[key] = []
                                constraint_details[key].append(
                                    loss_value.detach().float()
                                )
                if len(total_constraint_losses) > 0:
                    total_constraint_loss = torch.stack(total_constraint_losses).mean()
                    for key, values in constraint_details.items():
                        loss_dict[key] = torch.stack(values).mean()
                    loss_dict["constraint_total_raw"] = (
                        total_constraint_loss.detach().float()
                    )
                    loss_dict["constraint_valid_ratio"] = (
                        float(valid_constraint_count) / action_batch_size
                    )
                    if action_batch_size == 1:
                        loss_dict["constraint_timestep"] = action_timesteps[0].item()
                        loss_dict["constraint_time_weight"] = (
                            1.0 - action_timesteps[0].item() / 1000.0
                        ) ** 3
                else:
                    total_constraint_loss = torch.tensor(
                        0.0, device=self.device, dtype=self.torch_dtype
                    )
                    loss_dict["constraint_skipped"] = torch.tensor(1.0)
            else:
                total_constraint_loss = torch.tensor(
                    0.0, device=self.device, dtype=self.torch_dtype
                )
        else:
            action_loss = torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)
            second_head_loss = torch.tensor(
                0.0, device=self.device, dtype=self.torch_dtype
            )
            total_constraint_loss = torch.tensor(
                0.0, device=self.device, dtype=self.torch_dtype
            )

        # --- DCT frequency-domain alignment auxiliary loss ---
        # With frequency weighting so that low-freq (smooth trajectory) components
        # are emphasised more than high-freq (jitter/noise) components.
        # Without such weighting, orthonormal-DCT + uniform MSE ≡ plain MSE (Parseval).
        dct_loss = torch.tensor(0.0, device=self.device, dtype=self.torch_dtype)
        if self.dct_loss_weight > 0 and run_action_branch and T_action > 1:
            dct_losses = []
            # Build frequency weights once: exponential decay over DCT index k
            # Low-freq (k=0,1,...) get high weight; high-freq get low weight.
            _T_dct = actions_expanded.shape[-2]  # T
            freq_weights = torch.exp(
                -torch.arange(_T_dct, device=self.device, dtype=torch.float32)
                / max(_T_dct / 4.0, 1.0)
            )  # [T]
            freq_weights = freq_weights.unsqueeze(0).unsqueeze(-1)  # [1, T, 1]

            for i in range(action_batch_size):
                s = i * B
                e = (i + 1) * B
                # Recover x0_hat from model prediction
                if self._action_infer_pred_type == "x_pred":
                    x0_hat = action_noise_pred[s:e, :]
                else:
                    x0_hat = action_scheduler.invert_to_data_domain(
                        action_noise_pred[s:e, :],
                        action_latents[s:e, :],
                        action_timesteps[i],
                        pred_type=self._action_infer_pred_type,
                    )
                x0_gt = actions_expanded[s:e, :].to(dtype=self.torch_dtype)

                # Compute DCT-II along time axis for both pred and gt
                # x0_hat, x0_gt: [1, T, action_dim]
                dct_pred = self._dct(x0_hat.float())
                dct_gt = self._dct(x0_gt.float())

                # Frequency-weighted MSE: emphasise low-frequency components
                dct_loss_i = (dct_pred - dct_gt) ** 2 * freq_weights
                dof_mask_dct = inputs.get("dof_mask", None)
                if dof_mask_dct is not None:
                    dct_loss_i = dct_loss_i * dof_mask_dct
                dct_loss_i = dct_loss_i.mean()

                # Time-dependent weighting: DCT loss is more meaningful at lower noise
                t_norm = action_timesteps[i].item() / 1000.0
                time_weight = (1.0 - t_norm) ** 2
                dct_losses.append(dct_loss_i * time_weight)
            dct_loss = torch.stack(dct_losses).mean()

        loss_dict["dct_loss"] = dct_loss.detach().float()

        # action_loss: primary head FM loss; second_head_loss: secondary head FM loss (0 if single-head)
        loss_dict["action_loss"] = action_loss.detach().float()
        loss_dict["second_head_loss"] = second_head_loss.detach().float()
        loss_dict["constraint_total"] = total_constraint_loss.detach().float()

        # Add to total loss:
        # - primary action loss with action_loss_weight
        # - secondary head loss with action_loss_weight * second_head_loss_weight
        # - constraint loss with constraint_loss_weight
        if run_action_branch:
            if apply_action_loss:
                loss_dict["total_loss"] += self.action_loss_weight * action_loss
                loss_dict["total_loss"] += (
                    self.action_loss_weight
                    * self.second_head_loss_weight
                    * second_head_loss
                )
                loss_dict["total_loss"] += (
                    self.constraint_loss_weight * total_constraint_loss
                )
                loss_dict["total_loss"] += self.dct_loss_weight * dct_loss
            elif post_phase == "video":
                # FSDP2 requires every sharded submodule to participate in the
                # autograd graph even on video-only alternating steps. We add
                # action-side losses with ``* 0.0`` (not ``detach()``) so their
                # tensors still flow through backward and touch action params,
                # while contributing zero gradient to the optimizer step.
                loss_dict["total_loss"] = (
                    loss_dict["total_loss"]
                    + action_loss * 0.0
                    + second_head_loss * 0.0
                    + total_constraint_loss * 0.0
                    + dct_loss * 0.0
                )

        return loss_dict

    @staticmethod
    def _dct(x: torch.Tensor) -> torch.Tensor:
        """Compute Type-II Discrete Cosine Transform along the time axis (dim=-2).

        Uses the orthonormal definition so that energy is preserved.
        Input:  x of shape [..., T, D]
        Output: DCT coefficients of the same shape.
        """
        T = x.shape[-2]
        # Build DCT-II basis matrix: W[k, n] = cos(pi * k * (2n+1) / (2T))
        n = torch.arange(T, device=x.device, dtype=x.dtype)
        k = torch.arange(T, device=x.device, dtype=x.dtype)
        # [T, T] basis matrix
        basis = torch.cos(
            math.pi * k.unsqueeze(1) * (2.0 * n.unsqueeze(0) + 1.0) / (2.0 * T)
        )
        # Orthonormal scaling
        basis[0, :] *= math.sqrt(1.0 / T)
        basis[1:, :] *= math.sqrt(2.0 / T)
        # x: [..., T, D]  ->  basis @ x  along time dim
        return torch.matmul(basis, x)

    def compute_action_constraint_losses(
        self, action_pred: torch.Tensor, action_gt: torch.Tensor
    ) -> dict:
        """
        action_pred/gt: [B, T, 14]
        0-6 : Left arm [x,y,z, rx,ry,rz, g]
        7-13: Right arm [x,y,z, rx,ry,rz, g]
        Returns: dict(individual losses, total)
        """
        B, T, D = action_pred.shape
        device = action_pred.device
        eps = 1e-8

        # ===== Hyperparameters (start moderate, increase later) =====
        window_radius = 3  # 3 frames before and after
        dgrip_thresh = 0.1  # Threshold for gripper change detection
        w_event_boost = 3.0  # Supervision amplification in event window
        pos_small = 0.02  # Expected position displacement upper bound (data units)
        # Individual weights
        λ_pos_cont = 0.5  # Position continuity (2nd-order diff)
        λ_rot_cont = 0.4  # Rotation continuity (2nd-order diff in sincos)
        λ_evt_xyz = 5.0  # Event window XYZ alignment (to GT)
        λ_evt_rot = 3.0  # Event window rotation alignment
        λ_evt_grip = 10.0  # Event window gripper alignment
        λ_g_tv_out = 0.4  # Non-event segment gripper TV (keep constant)
        λ_g_lin_in = 0.6  # Event segment gripper linear (small 2nd-order diff)
        λ_g_range = 0.05  # Gripper value range constraint (optional)
        λ_mutex = 1.0  # Mutual exclusion (small event displacement + small non-event grip + product)
        λ_prod = 0.3  # Velocity product suppression coefficient

        # ===== Split =====
        def split14(x):
            L_xyz = x[:, :, 0:3]
            L_rpy = x[:, :, 3:6]
            L_g = x[:, :, 6:7]
            R_xyz = x[:, :, 7:10]
            R_rpy = x[:, :, 10:13]
            R_g = x[:, :, 13:14]
            return L_xyz, L_rpy, L_g, R_xyz, R_rpy, R_g

        # ===== Basic differentiation tools =====
        def diff1(x):  # 1st-order diff, preserve time dimension
            return torch.diff(x, dim=1, prepend=x[:, :1])

        def diff2(x):  # 2nd-order diff, preserve time dimension
            # First compute 1st-order diff
            d1 = torch.diff(x, dim=1, prepend=x[:, :1])
            # Then compute 2nd-order diff, set first frame to 0
            d2 = torch.diff(d1, dim=1, prepend=torch.zeros_like(d1[:, :1]))
            return d2

        pL_xyz, pL_rpy, pL_g, pR_xyz, pR_rpy, pR_g = split14(action_pred)
        gL_xyz, gL_rpy, gL_g, gR_xyz, gR_rpy, gR_g = split14(action_gt)

        # Merge left/right for unified computation
        pXYZ = torch.stack([pL_xyz, pR_xyz], dim=2)  # [B,T,2,3]
        gXYZ = torch.stack([gL_xyz, gR_xyz], dim=2)
        pRPY = torch.stack([pL_rpy, pR_rpy], dim=2)  # [B,T,2,3]
        gRPY = torch.stack([gL_rpy, gR_rpy], dim=2)
        pG = torch.cat([pL_g, pR_g], dim=-1)  # [B,T,2]
        gG = torch.cat([gL_g, gR_g], dim=-1)

        # ===== Event window (constructed from GT gripper, based on total diff sum of 5 frames before/after) =====
        # Compute gripper change per frame
        dG = diff1(gG).abs().max(dim=-1).values  # [B,T] max diff of both arms per frame

        total_diff_thresh = window_radius * dgrip_thresh  # Total diff sum threshold

        mask_event = torch.zeros(B, T, device=device)
        for b in range(B):
            for t in range(T):
                # Get range of 5 frames before/after this frame
                lo = max(0, t - window_radius)
                hi = min(T, t + window_radius + 1)
                # Compute sum of absolute diffs in this window
                total_diff = dG[b, lo:hi].sum().item()

                # If total diff sum exceeds threshold, mark as event window
                if total_diff > total_diff_thresh:
                    mask_event[b, t] = 1.0

        mask_out = 1.0 - mask_event
        w_evt = w_event_boost * mask_event  # Event amplification weight [B,T]

        # ===== 1) Predicted 3D position continuity (small acceleration) =====
        # Compute 2nd-order diff for left/right arms
        pos_acc = diff2(pXYZ).abs().mean(dim=(-1, -2))  # [B,T] -> mean over arm,xyz
        L_pos_cont = pos_acc.mean()

        # ===== 2) Predicted rotation continuity (in sincos space) =====
        pSC = torch.cat([torch.sin(pRPY), torch.cos(pRPY)], dim=-1)  # [B,T,2,6]
        rot_acc = diff2(pSC).abs().mean(dim=(-1, -2))  # [B,T]
        L_rot_cont = rot_acc.mean()

        # ===== 3) Event window: XYZ+rotation+gripper alignment (to GT, enhanced supervision) =====
        # XYZ direct L1; rotation in sincos space L1; gripper direct L1
        gSC = torch.cat([torch.sin(gRPY), torch.cos(gRPY)], dim=-1)
        evt_xyz_err = (pXYZ - gXYZ).abs().mean(dim=(-1, -2))  # [B,T]
        evt_rot_err = (pSC - gSC).abs().mean(dim=(-1, -2))  # [B,T]
        evt_grip_err = (pG - gG).abs().mean(dim=-1)  # [B,T]
        L_evt_xyz = (evt_xyz_err * w_evt).sum() / (w_evt.sum() + eps)
        L_evt_rot = (evt_rot_err * w_evt).sum() / (w_evt.sum() + eps)
        L_evt_grip = (evt_grip_err * w_evt).sum() / (w_evt.sum() + eps)

        # ===== 4) Predicted gripper shape prior =====
        # Non-event segment: small TV (keep constant)
        pG_sig = pG  # If outputting logits, change to torch.sigmoid(pG)
        d_pred_g = diff1(pG_sig).abs().mean(dim=-1)  # [B,T], mean of both arms
        L_g_tv_out = (d_pred_g * mask_out).sum() / (mask_out.sum() + eps)
        # Event segment: linear change (small 2nd-order diff)
        d2_pred_g = diff2(pG_sig).abs().mean(dim=-1)  # [B,T]
        L_g_lin_in = (d2_pred_g * mask_event).sum() / (mask_event.sum() + eps)
        # (Optional) Value range (0~1) constraint: ReLU penalty for out-of-bounds
        L_g_range = (F.relu(-pG_sig) + F.relu(pG_sig - 1.0)).pow(2).mean()

        # ===== 5) Mutual exclusion: gripper change <-> position change =====
        # Event window: position velocity should be small; Non-event: gripper velocity should be small; plus velocity product suppression
        v_pos = diff1(pXYZ).norm(dim=-1).mean(dim=-1)  # [B,T] mean of both arms
        v_grip = diff1(pG_sig).abs().mean(dim=-1)  # [B,T] mean of both arms
        # Event window: penalize excessive displacement (soft threshold)
        L_mutex_evt = F.relu(v_pos - pos_small).pow(2) * mask_event
        L_mutex_evt = L_mutex_evt.sum() / (mask_event.sum() + eps)
        # Non-event window: gripper velocity penalty (already in L_g_tv_out, can add here too)
        L_mutex_out = (v_grip * mask_out).sum() / (mask_out.sum() + eps)
        # Simultaneous movement and grasping (at any time)
        L_prod = (v_pos * v_grip).mean()
        L_mutex = L_mutex_evt + L_mutex_out + λ_prod * L_prod

        # ===== Summary =====
        losses = {
            # Continuity
            "pos_continuity": λ_pos_cont * L_pos_cont,
            "rot_continuity": λ_rot_cont * L_rot_cont,
            # Event window enhanced supervision
            "event_xyz_supervise": λ_evt_xyz * L_evt_xyz,
            "event_rot_supervise": λ_evt_rot * L_evt_rot,
            "event_grip_supervise": λ_evt_grip * L_evt_grip,
            # Gripper shape prior
            "grip_tv_out": λ_g_tv_out * L_g_tv_out,
            "grip_linear_in": λ_g_lin_in * L_g_lin_in,
            "grip_range": λ_g_range * L_g_range,
            # Mutual exclusion
            "mutual_exclusion": λ_mutex * L_mutex,
        }
        losses["total"] = sum(losses.values())
        return losses

    def enable_vram_management(
        self, num_persistent_param_in_dit=None, vram_limit=None, vram_buffer=0.5
    ):
        self.vram_management_enabled = True
        if num_persistent_param_in_dit is not None:
            vram_limit = None
        else:
            if vram_limit is None:
                vram_limit = self.get_vram()
            vram_limit = vram_limit - vram_buffer
        if self.text_encoder is not None:
            dtype = next(iter(self.text_encoder.parameters())).dtype
            enable_vram_management(
                self.text_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Embedding: AutoWrappedModule,
                    T5RelativeEmbedding: AutoWrappedModule,
                    T5LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit is not None:
            dtype = next(iter(self.dit.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                    torch.nn.Embedding: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.dit2 is not None:
            dtype = next(iter(self.dit2.parameters())).dtype
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.dit2,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: WanAutoCastLayerNorm,
                    RMSNorm: AutoWrappedModule,
                    torch.nn.Conv2d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                max_num_param=num_persistent_param_in_dit,
                overflow_module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.vae is not None:
            dtype = next(iter(self.vae.parameters())).dtype
            enable_vram_management(
                self.vae,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    RMS_norm: AutoWrappedModule,
                    CausalConv3d: AutoWrappedModule,
                    Upsample: AutoWrappedModule,
                    torch.nn.SiLU: AutoWrappedModule,
                    torch.nn.Dropout: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=self.device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.motion_controller is not None:
            dtype = next(iter(self.motion_controller.parameters())).dtype
            enable_vram_management(
                self.motion_controller,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        if self.vace is not None:
            device = "cpu" if vram_limit is not None else self.device
            enable_vram_management(
                self.vace,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv3d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    RMSNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device=device,
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
                vram_limit=vram_limit,
            )
        if self.audio_encoder is not None:
            # TODO: need check
            dtype = next(iter(self.audio_encoder.parameters())).dtype
            enable_vram_management(
                self.audio_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.LayerNorm: AutoWrappedModule,
                    torch.nn.Conv1d: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )

    def initialize_usp(self):
        import torch.distributed as dist
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        dist.init_process_group(backend="nccl", init_method="env://")
        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )
        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=1,
            ulysses_degree=dist.get_world_size(),
        )
        torch.cuda.set_device(dist.get_rank())

    def enable_usp(self):
        from xfuser.core.distributed import get_sequence_parallel_world_size
        from wall_x.model.distributed.xdit_context_parallel import (
            usp_attn_forward,
            usp_dit_forward,
        )

        for block in self.dit.blocks:
            block.self_attn.forward = types.MethodType(
                usp_attn_forward, block.self_attn
            )
        self.dit.forward = types.MethodType(usp_dit_forward, self.dit)
        if self.dit2 is not None:
            for block in self.dit2.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.dit2.forward = types.MethodType(usp_dit_forward, self.dit2)
        self.sp_size = get_sequence_parallel_world_size()
        self.use_unified_sequence_parallel = True

    def _init_action_transformer_from_loaded_dit(
        self, log_fn: Optional[Callable] = None
    ) -> None:
        """Re-init ActionTransformer from the **already loaded** DiT.

        Called after ckpt load when ``enable_action`` is on but the ckpt(s)
        did not contain ``action_transformer.*`` weights (video-only warmstart).
        Respects pipe flags: ``init_action_from_dit``, etc.
        """
        if self.action_transformer is None:
            return
        if log_fn is None:
            log_fn = _rank0_print

        if getattr(self, "init_action_from_dit", False):
            log_fn(
                "[ActionTransformer post-load init] Re-running _init_action_transformer_from_dit from LOADED DiT (ckpt had no action_transformer keys)."
            )
            self._init_action_transformer_from_dit()
        if getattr(self, "init_cross_attn_q_from_dit", False):
            log_fn(
                "[ActionTransformer post-load init] Re-running _init_cross_attn_q_from_dit from LOADED DiT."
            )
            self._init_cross_attn_q_from_dit()
        if getattr(self, "zero_init_cross_attn_o", False):
            log_fn(
                "[ActionTransformer post-load init] Re-running _zero_init_cross_attn_o."
            )
            self._zero_init_cross_attn_o()
        if getattr(self, "zero_init_action_output", False):
            log_fn(
                "[ActionTransformer post-load init] Re-running _zero_init_action_output."
            )
            self._zero_init_action_output()

    def load_partial_state_dict(
        self,
        paths: Union[str, List[str]],
        *,
        enable_action: bool = False,
        init_action_from_dit_if_missing: bool = False,
        log_fn: Optional[Callable] = None,
    ) -> None:
        """Load ckpt(s) from ``model_paths`` into pipe submodules.

        Primary job: warmstart dit / action_transformer / action_conditioner
        from one or two checkpoint files (see key formats below).

        Optional follow-up (``init_action_from_dit_if_missing=True``):
        when ``enable_action`` is on but the ckpt(s) had no
        ``action_transformer.*`` keys — i.e. only video weights were loaded —
        re-initialize the action branch from the **already loaded** DiT via
        ``_init_action_transformer_from_loaded_dit``.

        Supported key formats:
          - current FSDP-style ``pipe.dit.*`` /
            ``pipe.action_transformer.*`` keys.
          - bare-key (Wan2.1 / legacy VGA) format:
          - dit weights live at top level: ``blocks.0.weight``,
            ``text_embedding.*``, ``head.*``, ``patch_embedding.*``, ...
          - action_transformer weights (optional): keys prefixed
            ``pipe.action_transformer.<x>``.
          - action_conditioner weights (optional): keys prefixed
            ``pipe.action_conditioner.<x>``.

        Args:
            paths: One ckpt path, or list/comma-string of two.
                ``len(paths) == 1`` warmstarts dit (and, when
                ``enable_action``, also action_transformer /
                action_conditioner if those keys are present in the same
                file). ``len(paths) == 2`` treats ``paths[0]`` as the
                video ckpt (dit + optional action_conditioner) and
                ``paths[1]`` as the action ckpt (action_transformer +
                optional action_conditioner).
            enable_action: when False, skip loading into action_transformer
                even if matching keys are present.
            init_action_from_dit_if_missing: after load, if action was
                enabled but ``model_paths`` contained no
                ``action_transformer.*`` weights (video-only ckpt),
                init action from loaded DiT (training warmstart path).
            log_fn: optional callable for status lines (defaults to
                ``print`` on rank 0).

        Raises:
            ValueError: when ``paths`` does not resolve to 1 or 2 entries.
        """
        if log_fn is None:
            log_fn = _rank0_print

        if isinstance(paths, str):
            paths = [p.strip() for p in paths.split(",") if p.strip()]
        else:
            paths = list(paths)

        action_weights_in_model_paths = False
        if len(paths) == 1:
            sd = _load_state_dict_from_path(paths[0], device="cpu")
            log_fn(f"[warmstart] {paths[0]}: {len(sd)} keys")
            action_weights_in_model_paths = self._route_warmstart_into_pipe(
                sd,
                load_dit=True,
                load_action_transformer=enable_action,
                load_action_conditioner=True,
                log_fn=log_fn,
            )
            del sd
        elif len(paths) == 2:
            vid_sd = _load_state_dict_from_path(paths[0], device="cpu")
            log_fn(f"[warmstart] video={paths[0]}: {len(vid_sd)} keys")
            self._route_warmstart_into_pipe(
                vid_sd,
                load_dit=True,
                load_action_transformer=False,
                load_action_conditioner=True,
                log_fn=log_fn,
            )
            del vid_sd
            gc.collect()
            torch.cuda.empty_cache()

            act_sd = _load_state_dict_from_path(paths[1], device="cpu")
            log_fn(f"[warmstart] action={paths[1]}: {len(act_sd)} keys")
            action_weights_in_model_paths = self._route_warmstart_into_pipe(
                act_sd,
                load_dit=False,
                load_action_transformer=enable_action,
                load_action_conditioner=True,
                log_fn=log_fn,
            )
            del act_sd
        else:
            raise ValueError(
                f"load_partial_state_dict expects 1 or 2 paths, got {len(paths)}"
            )

        gc.collect()
        torch.cuda.empty_cache()

        if (
            init_action_from_dit_if_missing
            and enable_action
            and not action_weights_in_model_paths
        ):
            self._init_action_transformer_from_loaded_dit(log_fn)

    @staticmethod
    def _checked_load(module, sd, name, log_fn, *, subset=False):
        """load_state_dict(strict=False) but SURFACE what didn't load and REFUSE
        to run a submodule that loaded nothing.

        The old code discarded the (missing_keys, unexpected_keys) return, so an
        arch mismatch between the checkpoint and the model built from the train
        config (e.g. wrong action_transformer_dim / action_video_block_mapping)
        silently left the submodule at random init -> inf actions -> 0% eval with
        no error. ``subset=True`` is for split loads into a shared module (dit
        blocks vs others), where "loaded" means ckpt keys consumed, not full-module
        coverage.
        """
        res = module.load_state_dict(sd, strict=False)
        n_unexp = len(res.unexpected_keys)
        if subset:
            n_ok = len(sd) - n_unexp
            log_fn(
                f"  -> {name}: {len(sd)} ckpt keys -> {n_ok} matched ({n_unexp} unexpected)"
            )
            detail = f"none of the {len(sd)} checkpoint keys matched the module"
        else:
            n_model = len(module.state_dict())
            n_ok = n_model - len(res.missing_keys)
            log_fn(
                f"  -> {name}: {len(sd)} ckpt keys -> loaded {n_ok}/{n_model} params "
                f"(missing {len(res.missing_keys)}, unexpected {n_unexp})"
            )
            detail = f"loaded 0/{n_model} params (all {n_unexp} checkpoint keys unused)"
        if n_ok == 0:
            raise RuntimeError(
                f"[warmstart] {name}: {detail} -> checkpoint/model ARCH MISMATCH. The "
                f"model is built from the train config; ensure it matches the checkpoint "
                f"(action_transformer_dim, action_video_block_mapping, action_dim, layer "
                f"counts). Refusing to run a half-initialized model."
            )
        return res

    def _route_warmstart_into_pipe(
        self,
        sd: dict,
        *,
        load_dit: bool,
        load_action_transformer: bool,
        load_action_conditioner: bool,
        log_fn: Callable,
    ) -> bool:
        """Filter+route a single state dict into pipe submodules.

        Each submodule load is followed by ``del`` + ``gc.collect()`` to
        release the materialized batch promptly — matters for warmstart
        from a 14B-class video ckpt where holding the full sd plus
        intermediates can OOM.

        Returns:
            bool: whether ``sd`` contained ``action_transformer.*`` keys
            (and they were loaded when ``load_action_transformer`` is True).
        """
        action_transformer_keys_in_sd = False
        p_blocks = "blocks"
        p_dit = "pipe.dit"
        p_action_xfm = "pipe.action_transformer"
        p_action_xfm_bare = "action_transformer"
        p_action_cond = "pipe.action_conditioner"
        p_action_cond_bare = "action_conditioner"

        if load_action_transformer and self.action_transformer is not None:
            action_sd = {}
            for k, v in sd.items():
                if k.startswith(p_action_xfm + "."):
                    action_sd[k[len(p_action_xfm) + 1 :]] = v
                elif k.startswith(p_action_xfm_bare + "."):
                    action_sd[k[len(p_action_xfm_bare) + 1 :]] = v
            if action_sd:
                self._checked_load(
                    self.action_transformer, action_sd, "action_transformer", log_fn
                )
                action_transformer_keys_in_sd = True
            del action_sd
            gc.collect()

        if (
            load_action_conditioner
            and getattr(self, "action_as_condition", False)
            and self.action_conditioner is not None
        ):
            cond_sd = {}
            for k, v in sd.items():
                if k.startswith(p_action_cond + "."):
                    cond_sd[k[len(p_action_cond) + 1 :]] = v
                elif k.startswith(p_action_cond_bare + "."):
                    cond_sd[k[len(p_action_cond_bare) + 1 :]] = v
            if cond_sd:
                self._checked_load(
                    self.action_conditioner, cond_sd, "action_conditioner", log_fn
                )
            del cond_sd
            gc.collect()

        if load_dit and self.dit is not None:
            pipe_dit_key_count = 0
            bare_block_key_count = 0
            bare_other_key_count = 0
            blocks_sd = {}
            others_sd = {}
            for k, v in sd.items():
                if k.startswith(p_dit + "."):
                    pipe_dit_key_count += 1
                    dit_key = k[len(p_dit) + 1 :]
                    if dit_key.startswith(p_blocks + ".") or dit_key == p_blocks:
                        blocks_sd[dit_key] = v
                    else:
                        others_sd[dit_key] = v
                elif k.startswith(p_blocks + ".") or k == p_blocks:
                    bare_block_key_count += 1
                    blocks_sd[k] = v
                elif not (
                    k.startswith(p_action_xfm + ".")
                    or k.startswith(p_action_xfm_bare + ".")
                    or k.startswith(p_action_cond + ".")
                    or k.startswith(p_action_cond_bare + ".")
                    or k.startswith("pipe.")
                ):
                    bare_other_key_count += 1
                    others_sd[k] = v

            if pipe_dit_key_count and (bare_block_key_count or bare_other_key_count):
                dit_format = "mixed pipe.dit.* + legacy bare"
            elif pipe_dit_key_count:
                dit_format = "new pipe.dit.*"
            elif bare_block_key_count or bare_other_key_count:
                dit_format = "legacy bare"
            else:
                dit_format = "none"
            log_fn(
                "  -> dit checkpoint format: "
                f"{dit_format} "
                f"(pipe.dit={pipe_dit_key_count}, "
                f"bare_blocks={bare_block_key_count}, "
                f"bare_others={bare_other_key_count})"
            )

            if blocks_sd:
                self._checked_load(
                    self.dit, blocks_sd, "dit blocks", log_fn, subset=True
                )
            del blocks_sd
            gc.collect()

            if others_sd:
                self._checked_load(
                    self.dit, others_sd, "dit others", log_fn, subset=True
                )
            del others_sd
            gc.collect()

        return action_transformer_keys_in_sd

    @staticmethod
    def from_pretrained(
        torch_dtype: torch.dtype = torch.bfloat16,
        device: Union[str, torch.device] = "cuda",
        model_configs: list[ModelConfig] = [],
        tokenizer_config: ModelConfig = ModelConfig(
            model_id="Wan-AI/Wan2.1-T2V-1.3B", origin_file_pattern="google/*"
        ),
        audio_processor_config: ModelConfig = None,
        redirect_common_files: bool = True,
        use_usp=False,
        # action
        enable_action: bool = True,
        action_dim: int = 14,
        action_batch_size: int = 1,
        action_loss_weight: float = 1.0,
        video_loss_weight: float = 1.0,
        video_loss_weight_start_step: int = 15000,
        action_video_block_mapping: list = None,
        ignore_video_loss: bool = False,
        constraint_loss_weight: float = 0.0,
        # Qwen
        use_qwen25: bool = False,
        qwen25_path: str = None,
        use_qwen35: bool = False,
        qwen35_path: str = None,
        qwen_proj_out_dim: int = 4096,
        qwen25_cascade_t5: bool = False,
        qwen35_cascade_t5: bool = False,
        qwen35_system_prompt: Optional[str] = None,
        print_qwen_output: bool = False,
        # flow matching
        separate_flow_matching: bool = False,  # flag-separate
        action_timestep_mapping: Optional[dict] = None,
        video_early_stop_for_action: bool = True,
        video_train_anchors: Optional[Union[str, List[int]]] = None,
        video_train_anchor_inference_steps: int = 50,
        video_train_window: int = 20,
        sync_action_denoising: bool = False,
        skip_vae_decode: bool = False,
        num_views: int = 3,
        use_sightcone_matching: bool = False,
        cross_view_masking_prob: float = 0.0,
        cross_view_masking_condition_prob: float = 0.0,
        cross_view_masking_max_len: int = 5,
        mask_region_loss_weight: float = 1.0,
        share_state_action_encoder: bool = True,
        local_alignment_pe: bool = False,
        state_attn: bool = False,
        action_pred_type: str = "v_pred",
        second_head_loss_weight: float = 1.0,
        dct_loss_weight: float = 0.0,
        action_as_condition: bool = False,
        action_tokenizer_ckpt_path: Optional[str] = None,
        view_attn_block_interval: int = 1,
        skip_text_encoder: bool = False,
        action_transformer_dim: Optional[int] = None,
        action_per_video_frame: int = 8,
        init_action_from_dit: bool = False,
        init_cross_attn_q_from_dit: bool = False,
        zero_init_cross_attn_o: bool = False,
        zero_init_action_output: bool = False,
        vga_post_training: bool = False,
        vga_post_action_updates_per_video: int = 1,
        detach_video_features: bool = True,
        **kwargs,
    ):
        import warnings

        # Deprecated kwargs (single source of truth: qwen_proj_out_dim on WanVideoPipeline)
        if "qwen25_proj_out_dim" in kwargs:
            qwen_proj_out_dim = kwargs.pop("qwen25_proj_out_dim")
            warnings.warn(
                "qwen25_proj_out_dim in from_pretrained is deprecated; use qwen_proj_out_dim.",
                DeprecationWarning,
                stacklevel=2,
            )
        if "qwen35_proj_out_dim" in kwargs:
            qwen_proj_out_dim = kwargs.pop("qwen35_proj_out_dim")
            warnings.warn(
                "qwen35_proj_out_dim in from_pretrained is deprecated; use qwen_proj_out_dim.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Redirect model path
        if redirect_common_files:
            redirect_dict = {
                "models_t5_umt5-xxl-enc-bf16.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "Wan2.1_VAE.pth": "Wan-AI/Wan2.1-T2V-1.3B",
                "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": "Wan-AI/Wan2.1-I2V-14B-480P",
                # Wan2.2
                # "models_t5_umt5-xxl-enc-bf16.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
                # "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.safetensors"),
                # "Wan2.1_VAE.pth": ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.1_VAE.safetensors"),
                "Wan2.2_VAE.pth": "Wan-AI/Wan2.2-TI2V-5B",
            }
            for model_config in model_configs:
                if (
                    model_config.origin_file_pattern is None
                    or model_config.model_id is None
                ):
                    continue
                if (
                    model_config.origin_file_pattern in redirect_dict
                    and model_config.model_id
                    != redirect_dict[model_config.origin_file_pattern]
                ):
                    logger.info(
                        "To avoid repeatedly downloading model files, (%s, %s) is redirected to (%s, %s). You can use `redirect_common_files=False` to disable file redirection.",
                        model_config.model_id,
                        model_config.origin_file_pattern,
                        redirect_dict[model_config.origin_file_pattern],
                        model_config.origin_file_pattern,
                    )
                    model_config.model_id = redirect_dict[
                        model_config.origin_file_pattern
                    ]

        # Initialize pipeline
        pipe = WanVideoPipeline(
            device=device,
            torch_dtype=torch_dtype,
            enable_action=enable_action,
            action_dim=action_dim,
            action_batch_size=action_batch_size,
            action_loss_weight=action_loss_weight,
            video_loss_weight=video_loss_weight,
            video_loss_weight_start_step=video_loss_weight_start_step,
            action_video_block_mapping=action_video_block_mapping,
            ignore_video_loss=ignore_video_loss,
            constraint_loss_weight=constraint_loss_weight,
            use_qwen25=use_qwen25,
            qwen25_path=qwen25_path,
            use_qwen35=use_qwen35,
            qwen35_path=qwen35_path,
            qwen_proj_out_dim=qwen_proj_out_dim,
            qwen25_cascade_t5=qwen25_cascade_t5,
            qwen35_cascade_t5=qwen35_cascade_t5,
            qwen35_system_prompt=qwen35_system_prompt,
            print_qwen_output=print_qwen_output,
            separate_flow_matching=separate_flow_matching,
            action_timestep_mapping=action_timestep_mapping,
            video_early_stop_for_action=video_early_stop_for_action,
            video_train_anchors=video_train_anchors,
            video_train_anchor_inference_steps=video_train_anchor_inference_steps,
            video_train_window=video_train_window,
            sync_action_denoising=sync_action_denoising,
            skip_vae_decode=skip_vae_decode,
            num_views=num_views,
            use_sightcone_matching=use_sightcone_matching,
            cross_view_masking_prob=cross_view_masking_prob,
            cross_view_masking_condition_prob=cross_view_masking_condition_prob,
            cross_view_masking_max_len=cross_view_masking_max_len,
            mask_region_loss_weight=mask_region_loss_weight,
            share_state_action_encoder=share_state_action_encoder,
            local_alignment_pe=local_alignment_pe,
            state_attn=state_attn,
            action_pred_type=action_pred_type,
            second_head_loss_weight=second_head_loss_weight,
            dct_loss_weight=dct_loss_weight,
            action_as_condition=action_as_condition,
            action_tokenizer_ckpt_path=action_tokenizer_ckpt_path,
            skip_text_encoder=skip_text_encoder,
            action_transformer_dim=action_transformer_dim,
            action_per_video_frame=action_per_video_frame,
            init_action_from_dit=init_action_from_dit,
            init_cross_attn_q_from_dit=init_cross_attn_q_from_dit,
            zero_init_cross_attn_o=zero_init_cross_attn_o,
            zero_init_action_output=zero_init_action_output,
            vga_post_training=vga_post_training,
            vga_post_action_updates_per_video=vga_post_action_updates_per_video,
            detach_video_features=detach_video_features,
            **kwargs,
        )
        if use_usp:
            pipe.initialize_usp()

        # Download and load models
        model_manager = ModelManager()
        model_init_kwargs = {}

        for model_config in model_configs:
            model_config.download_if_necessary(use_usp=use_usp)
            model_manager.load_model(
                model_config.path,
                device=model_config.offload_device or device,
                torch_dtype=model_config.offload_dtype or torch_dtype,
                model_init_kwargs=model_init_kwargs,
            )

        # Load models
        # In cascade mode, T5 encoder is also needed; in non-cascade mode, load T5 when not using Qwen.
        # When skip_text_encoder (e.g. text feature cache), do not load T5/Qwen — encode_prompt uses cache only.
        if not skip_text_encoder:
            need_t5 = (
                not pipe.prompter.use_qwen25 or pipe.prompter.qwen25_cascade_t5
            ) and (not pipe.prompter.use_qwen35 or pipe.prompter.qwen35_cascade_t5)
            if need_t5:
                pipe.text_encoder = model_manager.fetch_model("wan_video_text_encoder")

        dit = model_manager.fetch_model("wan_video_dit", index=2)
        if isinstance(dit, list):
            pipe.dit, pipe.dit2 = dit
        else:
            pipe.dit = dit

        # Get DiT dimensions for dynamic ActionTransformer construction
        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        dit_num_heads = pipe.dit.blocks[0].self_attn.num_heads
        dit_num_layers = len(pipe.dit.blocks)

        # Build ActionTransformer with DiT-matching dimensions (deferred from __init__)
        pipe._build_action_transformer(dim, dit_num_heads, dit_num_layers)
        # Build ActionConditioner with DiT-matching dimensions (deferred from __init__)
        pipe._build_action_conditioner(dim, action_tokenizer_ckpt_path)

        # Optionally initialize ActionTransformer from DiT block weights.
        # Paper trick: interpolate by dim and scale by sqrt(d_v / d_a) so action tokens
        # start with output distribution comparable to video tokens, stabilizing joint
        # attention early in training. Done before view_attn cloning so the copy reflects
        # the original loaded DiT state.
        if init_action_from_dit:
            pipe._init_action_transformer_from_dit()
        # Cross-attn Q-side alignment: copy DiT self_attn Q into ActionBlock cross_attn Q.
        # Useful when self_attn was already DiT-init (output distribution matches DiT).
        if init_cross_attn_q_from_dit:
            pipe._init_cross_attn_q_from_dit()
        # ControlNet-style zero-init of cross_attn output projection: cross-attn
        # contribution starts at 0, opens up gradually as o learns.
        if zero_init_cross_attn_o:
            pipe._zero_init_cross_attn_o()
        # Zero-init the last linear of action_decoder so step-0 prediction = 0.
        if zero_init_action_output:
            pipe._zero_init_action_output()

        if num_views > 1:
            for block_idx, block in enumerate(pipe.dit.blocks):
                if block_idx % view_attn_block_interval != 0:
                    continue
                block.view_projector = nn.Linear(dim, dim)
                block.view_projector.weight = nn.Parameter(torch.zeros(dim, dim))
                block.view_projector.bias = nn.Parameter(torch.zeros(dim))
                block.norm_view = nn.LayerNorm(
                    dim, eps=block.norm1.eps, elementwise_affine=False
                )
                block.modulation_view = nn.Parameter(torch.randn(1, 3, dim) / dim**0.5)
                block.modulation_view.data = block.modulation.data[:, :3, :].clone()
                block.view_attn = SelfAttention(
                    dim, block.self_attn.num_heads, block.self_attn.norm_q.eps
                )
                block.view_attn.load_state_dict(
                    block.self_attn.state_dict(), strict=True
                )
        pipe.dit.to(dtype=torch_dtype, device=device)

        pipe.vae = model_manager.fetch_model("wan_video_vae")
        pipe.image_encoder = model_manager.fetch_model("wan_video_image_encoder")
        pipe.motion_controller = model_manager.fetch_model(
            "wan_video_motion_controller"
        )
        vace = model_manager.fetch_model("wan_video_vace", index=2)
        if isinstance(vace, list):
            pipe.vace, pipe.vace2 = vace
        else:
            pipe.vace = vace
        pipe.audio_encoder = model_manager.fetch_model("wans2v_audio_encoder")
        pipe.animate_adapter = model_manager.fetch_model("wan_video_animate_adapter")

        # Size division factor
        if pipe.vae is not None:
            pipe.height_division_factor = pipe.vae.upsampling_factor * 2
            pipe.width_division_factor = pipe.vae.upsampling_factor * 2

        # Initialize tokenizer and text encoder models (skipped when using precomputed text cache)
        if not skip_text_encoder:
            tokenizer_config.download_if_necessary(use_usp=use_usp)
            pipe.prompter.fetch_models(pipe.text_encoder)
            # If Qwen2.5/3.5 is enabled and not in cascade mode, register Qwen adapter as pipeline's text_encoder
            # In cascade mode, pipe.text_encoder remains T5, prompter internally manages Qwen and T5
            if (pipe.prompter.use_qwen25 and not pipe.prompter.qwen25_cascade_t5) or (
                pipe.prompter.use_qwen35 and not pipe.prompter.qwen35_cascade_t5
            ):
                pipe.text_encoder = pipe.prompter.text_encoder
            pipe.prompter.fetch_tokenizer(tokenizer_config.path)

        if audio_processor_config is not None:
            audio_processor_config.download_if_necessary(use_usp=use_usp)
            from transformers import Wav2Vec2Processor

            pipe.audio_processor = Wav2Vec2Processor.from_pretrained(
                audio_processor_config.path
            )
        # Unified Sequence Parallel
        if use_usp:
            pipe.enable_usp()
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        # Prompt
        prompt: str,
        negative_prompt: Optional[str] = "",
        # Image-to-video (single PIL, or list for multi-view / per-view frames)
        input_image: Optional[Union[Image.Image, List[Image.Image]]] = None,
        # First-last-frame-to-video
        end_image: Optional[Image.Image] = None,
        # Video-to-video
        input_video: Optional[list[Image.Image]] = None,
        denoising_strength: Optional[float] = 1.0,
        # Speech-to-video
        input_audio: Optional[np.array] = None,
        audio_embeds: Optional[torch.Tensor] = None,
        audio_sample_rate: Optional[int] = 16000,
        s2v_pose_video: Optional[list[Image.Image]] = None,
        s2v_pose_latents: Optional[torch.Tensor] = None,
        motion_video: Optional[list[Image.Image]] = None,
        # ControlNet
        control_video: Optional[list[Image.Image]] = None,
        reference_image: Optional[Image.Image] = None,
        # Camera control
        camera_control_direction: Optional[
            Literal[
                "Left",
                "Right",
                "Up",
                "Down",
                "LeftUp",
                "LeftDown",
                "RightUp",
                "RightDown",
            ]
        ] = None,
        camera_control_speed: Optional[float] = 1 / 54,
        camera_control_origin: Optional[tuple] = (
            0,
            0.532139961,
            0.946026558,
            0.5,
            0.5,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            1,
            0,
        ),
        # VACE
        vace_video: Optional[list[Image.Image]] = None,
        vace_video_mask: Optional[Image.Image] = None,
        vace_reference_image: Optional[Image.Image] = None,
        vace_scale: Optional[float] = 1.0,
        # Animate
        animate_pose_video: Optional[list[Image.Image]] = None,
        animate_face_video: Optional[list[Image.Image]] = None,
        animate_inpaint_video: Optional[list[Image.Image]] = None,
        animate_mask_video: Optional[list[Image.Image]] = None,
        # Randomness
        seed: Optional[int] = None,
        rand_device: Optional[str] = "cpu",
        # Shape
        height: Optional[int] = 480,
        width: Optional[int] = 832,
        num_frames=81,
        # Classifier-free guidance
        cfg_scale: Optional[float] = 5.0,  # todomike  default 5.0 --> 1.0
        cfg_merge: Optional[bool] = False,
        # Boundary
        switch_DiT_boundary: Optional[float] = 0.875,
        # Scheduler
        num_inference_steps: Optional[int] = 50,
        sigma_shift: Optional[float] = None,
        # Speed control
        motion_bucket_id: Optional[int] = None,
        # VAE tiling
        tiled: Optional[bool] = True,
        tile_size: Optional[tuple[int, int]] = (30, 52),
        tile_stride: Optional[tuple[int, int]] = (15, 26),
        # Sliding window
        sliding_window_size: Optional[int] = None,
        sliding_window_stride: Optional[int] = None,
        # Teacache
        tea_cache_l1_thresh: Optional[float] = None,
        tea_cache_model_id: Optional[str] = "",
        # progress_bar
        progress_bar_cmd=tqdm,
        ###
        # Action prediction
        action_num_frames: Optional[int] = None,
        actions: Optional[torch.Tensor] = None,
        agent_pos: Optional[torch.Tensor] = None,
        action_history=None,
        ###
        num_views: int = 3,
        view_ids: Optional[torch.Tensor] = None,
        # Batch inference support
        inference_batch_size: int = 1,
        skip_video_decode: bool = False,
    ):
        # Scheduler
        if sigma_shift is None:
            sigma_shift = self.scheduler_shift
        self.scheduler.set_timesteps(
            num_inference_steps,
            denoising_strength=denoising_strength,
            shift=sigma_shift,
        )

        # If separate flow matching is enabled, set timesteps for action scheduler
        if self.separate_flow_matching:
            self.action_scheduler.set_timesteps(
                num_inference_steps,
                denoising_strength=denoising_strength,
                shift=sigma_shift,
            )

        # Prepare action prediction
        if self.enable_action and not self.action_as_condition:
            # ActionTransformer inference: initialize noisy actions for denoising
            if not isinstance(actions, torch.Tensor):
                actions = torch.as_tensor(actions)
            if actions.ndim == 2:
                actions = actions.unsqueeze(0)
            actions = actions.to(device=self.device, dtype=torch.float32)
            agent_pos = agent_pos.to(device=self.device, dtype=self.torch_dtype)
            if action_history is not None:
                action_history = action_history.to(
                    device=self.device, dtype=self.torch_dtype
                )
            B, T_action, _ = actions.shape
            noisy_actions = self.generate_noise(
                (B, T_action, self.action_dim),
                seed=seed,
                rand_device=rand_device,
            )
            noisy_actions = noisy_actions.to(device=self.device, dtype=self.torch_dtype)

        inputs_posi = {
            "prompt": prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_nega = {
            "negative_prompt": negative_prompt,
            "tea_cache_l1_thresh": tea_cache_l1_thresh,
            "tea_cache_model_id": tea_cache_model_id,
            "num_inference_steps": num_inference_steps,
        }
        inputs_shared = {
            "input_image": input_image,
            "end_image": end_image,
            "input_video": input_video,
            "denoising_strength": denoising_strength,
            "control_video": control_video,
            "reference_image": reference_image,
            "camera_control_direction": camera_control_direction,
            "camera_control_speed": camera_control_speed,
            "camera_control_origin": camera_control_origin,
            "vace_video": vace_video,
            "vace_video_mask": vace_video_mask,
            "vace_reference_image": vace_reference_image,
            "vace_scale": vace_scale,
            "seed": seed,
            "rand_device": rand_device,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "num_views": num_views,
            "batch_size": inference_batch_size,
            "cfg_scale": cfg_scale,
            "cfg_merge": cfg_merge,
            "sigma_shift": sigma_shift,
            "motion_bucket_id": motion_bucket_id,
            "tiled": tiled,
            "tile_size": tile_size,
            "tile_stride": tile_stride,
            "sliding_window_size": sliding_window_size,
            "sliding_window_stride": sliding_window_stride,
            # model_fn_wan_video strictly requires torch.LongTensor of shape [B,V];
            # inference path accepts a plain list (e.g. from vga_policy), so normalize here.
            "view_ids": (
                view_ids
                if view_ids is None or isinstance(view_ids, torch.Tensor)
                else torch.as_tensor(view_ids, dtype=torch.long).reshape(1, -1)
            ),
            "input_audio": input_audio,
            "audio_sample_rate": audio_sample_rate,
            "s2v_pose_video": s2v_pose_video,
            "audio_embeds": audio_embeds,
            "s2v_pose_latents": s2v_pose_latents,
            "motion_video": motion_video,
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
            # Action conditioning inputs (consumed by WanVideoUnit_ActionContext)
            "actions": actions,
            "agent_pos": agent_pos,
            "dof_mask": None,  # dof_mask is only used in training
        }
        for unit in self.units:
            inputs_shared, inputs_posi, inputs_nega = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        # Denoise
        self.load_models_to_device(self.in_iteration_models)
        if self.enable_action and not self.action_as_condition:
            models = {name: getattr(self, name) for name in self.in_iteration_models}
        else:
            models = {
                name: getattr(self, name)
                for name in self.in_iteration_models
                if name != "action_transformer"
            }

        video_block_features_list = []
        video_block_features_nega_list = []

        _sync_active = (
            self.sync_action_denoising
            and self.enable_action
            and not self.action_as_condition
        )
        if _sync_active:
            sync_action_scheduler = (
                self.action_scheduler if self.separate_flow_matching else self.scheduler
            )
            video_to_action_steps = defaultdict(list)
            if self.action_timestep_mapping is not None:
                for action_step, video_step in self.action_timestep_mapping.items():
                    video_to_action_steps[video_step].append(action_step)
            else:
                for step_idx in range(len(sync_action_scheduler.timesteps)):
                    video_to_action_steps[step_idx].append(step_idx)
            for video_step in video_to_action_steps:
                video_to_action_steps[video_step].sort()

        # Early stop: when action_timestep_mapping is set and video_early_stop_for_action, only run video steps 0..max_video_step (e.g. 0..3) to collect features for action conditioning
        if (
            self.enable_action
            and not self.action_as_condition
            and self.action_timestep_mapping is not None
            and self.video_early_stop_for_action
        ):
            max_video_step = max(self.action_timestep_mapping.values())
            video_steps = min(max_video_step + 1, len(self.scheduler.timesteps))
            timesteps_to_run = list[tuple[int, Tensor]](
                enumerate(self.scheduler.timesteps[:video_steps])
            )
        else:
            timesteps_to_run = list(enumerate(self.scheduler.timesteps))

        for progress_id, timestep in timesteps_to_run:
            # Switch DiT if necessary
            if (
                timestep.item()
                < switch_DiT_boundary * self.scheduler.num_train_timesteps
                and self.dit2 is not None
                and models["dit"] is not self.dit2
            ):
                self.load_models_to_device(self.in_iteration_models_2)
                models["dit"] = self.dit2
                models["vace"] = self.vace2

            # Timestep is always fp32. (The bf16-timestep legacy path was retired:
            # bf16-vs-fp32 is within RoboTwin eval noise and fp32 is the correct value.)
            timestep = timestep.unsqueeze(0).to(dtype=torch.float32, device=self.device)

            # Inference
            if self.enable_action and not self.action_as_condition:
                # If action prediction is needed, first get video features and collect for action loop
                noise_pred_posi, video_time_features, block_time_features = (
                    self.model_fn(
                        **models,
                        **inputs_shared,
                        **inputs_posi,
                        timestep=timestep,
                        return_video_features=True,
                    )
                )
                if not _sync_active:
                    video_block_features_list.append(
                        [bt.to("cpu") for bt in block_time_features]
                    )
            else:
                noise_pred_posi = self.model_fn(
                    **models, **inputs_shared, **inputs_posi, timestep=timestep
                )

            if cfg_scale != 1.0:
                if cfg_merge:
                    noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
                    # In cfg_merge case, need to separately get negative features for action
                    if self.enable_action and not self.action_as_condition:
                        _, _, block_time_features_nega = self.model_fn(
                            **models,
                            **inputs_shared,
                            **inputs_nega,
                            timestep=timestep,
                            return_video_features=True,
                        )
                        if not _sync_active:
                            video_block_features_nega_list.append(
                                [bt.to("cpu") for bt in block_time_features_nega]
                            )
                else:
                    # If action exists, nega also needs to get features (if cfg_merge=False)
                    if self.enable_action and not self.action_as_condition:
                        noise_pred_nega, _, block_time_features_nega = self.model_fn(
                            **models,
                            **inputs_shared,
                            **inputs_nega,
                            timestep=timestep,
                            return_video_features=True,
                        )
                        if not _sync_active:
                            video_block_features_nega_list.append(
                                [bt.to("cpu") for bt in block_time_features_nega]
                            )
                    else:
                        noise_pred_nega = self.model_fn(
                            **models, **inputs_shared, **inputs_nega, timestep=timestep
                        )
                noise_pred = noise_pred_nega + cfg_scale * (
                    noise_pred_posi - noise_pred_nega
                )
            else:
                noise_pred = noise_pred_posi

            if _sync_active:
                sync_nega_features = (
                    block_time_features_nega if cfg_scale != 1.0 else None
                )
                for action_step_id in video_to_action_steps.get(progress_id, []):
                    action_timestep = (
                        sync_action_scheduler.timesteps[action_step_id]
                        .unsqueeze(0)
                        .to(dtype=self.torch_dtype, device=self.device)
                    )
                    action_t_embed = self.dit.time_embedding(
                        sinusoidal_embedding_1d(self.dit.freq_dim, action_timestep)
                    )
                    action_t_mod = self.dit.time_projection(action_t_embed).unflatten(
                        1, (6, self.dit.dim)
                    )

                    action_noise_pred_posi = self.action_transformer(
                        noisy_actions,
                        agent_pos,
                        block_time_features,
                        action_t_mod,
                        action_history,
                        active_head=self._action_infer_pred_type,
                    )
                    if cfg_scale != 1.0 and sync_nega_features is not None:
                        action_noise_pred_nega = self.action_transformer(
                            noisy_actions,
                            agent_pos,
                            sync_nega_features,
                            action_t_mod,
                            action_history,
                            active_head=self._action_infer_pred_type,
                        )
                        action_noise_pred = action_noise_pred_nega + cfg_scale * (
                            action_noise_pred_posi - action_noise_pred_nega
                        )
                    else:
                        action_noise_pred = action_noise_pred_posi

                    noisy_actions = sync_action_scheduler.step(
                        action_noise_pred,
                        sync_action_scheduler.timesteps[action_step_id],
                        noisy_actions,
                        pred_type=self._action_infer_pred_type,
                    )

            inputs_shared["latents"] = self.scheduler.step(
                noise_pred,
                self.scheduler.timesteps[progress_id],
                inputs_shared["latents"],
            )
            if "first_frame_latents" in inputs_shared:
                inputs_shared["latents"][:, :, 0:1] = inputs_shared[
                    "first_frame_latents"
                ]

        if (
            self.enable_action
            and not self.action_as_condition
            and len(video_block_features_list) > 0
        ):
            if self.separate_flow_matching:
                action_scheduler = self.action_scheduler
            else:
                action_scheduler = self.scheduler

            for action_progress_id, action_timestep in enumerate(
                action_scheduler.timesteps
            ):
                # Map action step -> video step for conditioning (e.g. 0..9 -> 0 for share_ts)
                if self.action_timestep_mapping is not None:
                    video_progress_id = self.action_timestep_mapping.get(
                        action_progress_id, action_progress_id
                    )
                else:
                    video_progress_id = action_progress_id

                video_progress_id = min(
                    video_progress_id, len(video_block_features_list) - 1
                )
                block_time_features = [
                    bt.to(self.device)
                    for bt in video_block_features_list[video_progress_id]
                ]
                if cfg_scale != 1.0 and len(video_block_features_nega_list) > 0:
                    nega_idx = min(
                        video_progress_id,
                        len(video_block_features_nega_list) - 1,
                    )
                    block_time_features_nega = [
                        bt.to(self.device)
                        for bt in video_block_features_nega_list[nega_idx]
                    ]
                else:
                    block_time_features_nega = None

                action_timestep = action_timestep.unsqueeze(0).to(
                    dtype=torch.float32, device=self.device
                )
                action_t_embed = self.dit.time_embedding(
                    sinusoidal_embedding_1d(
                        self.dit.freq_dim,
                        action_timestep,
                        output_dtype=self.torch_dtype,
                    )
                )
                action_t_mod = self.dit.time_projection(action_t_embed).unflatten(
                    1, (6, self.dit.dim)
                )

                # active_head=infer_pred_type: dual_head skips unused head; single head ignores it
                action_noise_pred_posi = self.action_transformer(
                    noisy_actions,
                    agent_pos,
                    block_time_features,
                    action_t_mod,
                    action_history,
                    active_head=self._action_infer_pred_type,
                )

                if cfg_scale != 1.0 and block_time_features_nega is not None:
                    action_noise_pred_nega = self.action_transformer(
                        noisy_actions,
                        agent_pos,
                        block_time_features_nega,
                        action_t_mod,
                        action_history,
                        active_head=self._action_infer_pred_type,
                    )
                    action_noise_pred = action_noise_pred_nega + cfg_scale * (
                        action_noise_pred_posi - action_noise_pred_nega
                    )
                else:
                    action_noise_pred = action_noise_pred_posi

                noisy_actions = action_scheduler.step(
                    action_noise_pred,
                    action_scheduler.timesteps[action_progress_id],
                    noisy_actions,
                    pred_type=self._action_infer_pred_type,
                )

        # post-denoising, pre-decoding processing logic
        for unit in self.post_units:
            inputs_shared, _, _ = self.unit_runner(
                unit, self, inputs_shared, inputs_posi, inputs_nega
            )

        if self.skip_vae_decode and self.enable_action and not self.action_as_condition:
            return None, noisy_actions

        # Decode
        all_videos = []
        if not skip_video_decode:
            for view in range(num_views):
                self.load_models_to_device(["vae"])
                view_indices = list(
                    range(view, inference_batch_size * num_views, num_views)
                )
                video = self.vae.decode(
                    inputs_shared["latents"][view_indices],
                    device=self.device,
                    tiled=tiled,
                    tile_size=tile_size,
                    tile_stride=tile_stride,
                )
                video = self.vae_output_to_video(video)
                self.load_models_to_device([])
                all_videos.append(video)

        self.load_models_to_device([])

        # Return video and action predictions if requested
        if self.enable_action and not self.action_as_condition:
            return all_videos, noisy_actions
        return all_videos


# ============ WanVideo-specific PipelineUnits (new style) ============
# Base units (ShapeChecker, NoiseInitializer, PromptEmbedder, InputVideoEmbedder,
# ImageEmbedderCLIP, ImageEmbedderVAE, ImageEmbedderFused, CfgMerger) imported from units.py


class WanVideoUnit_FunControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "control_video",
                "num_frames",
                "height",
                "width",
                "tiled",
                "tile_size",
                "tile_stride",
                "clip_feature",
                "y",
                "latents",
            ),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, control_video, **kwargs) -> bool:
        return control_video is not None

    def run(
        self,
        pipe: WanVideoPipeline,
        control_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        clip_feature,
        y,
        latents,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        control_video = pipe.preprocess_video(control_video)
        control_latents = pipe.vae.encode(
            control_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        control_latents = control_latents.to(dtype=pipe.torch_dtype, device=pipe.device)
        y_dim = pipe.dit.in_dim - control_latents.shape[1] - latents.shape[1]
        if clip_feature is None or y is None:
            clip_feature = torch.zeros(
                (1, 257, 1280), dtype=pipe.torch_dtype, device=pipe.device
            )
            y = torch.zeros(
                (1, y_dim, (num_frames - 1) // 4 + 1, height // 8, width // 8),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        else:
            y = y[:, -y_dim:]
        y = torch.concat([control_latents, y], dim=1)
        return {"clip_feature": clip_feature, "y": y}


class WanVideoUnit_FunReference(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("reference_image", "height", "width"),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, reference_image, **kwargs) -> bool:
        return reference_image is not None

    def run(
        self, pipe: WanVideoPipeline, reference_image, height, width, **kwargs
    ) -> dict:
        pipe.load_models_to_device(["vae"])
        reference_image = reference_image.resize((width, height))
        reference_latents = pipe.preprocess_video([reference_image])
        reference_latents = pipe.vae.encode(reference_latents, device=pipe.device)
        if pipe.image_encoder is None:
            return {"reference_latents": reference_latents}
        clip_feature = pipe.preprocess_image(reference_image)
        clip_feature = pipe.image_encoder.encode_image([clip_feature])
        return {"reference_latents": reference_latents, "clip_feature": clip_feature}


class WanVideoUnit_FunCameraControl(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "height",
                "width",
                "num_frames",
                "camera_control_direction",
                "camera_control_speed",
                "camera_control_origin",
                "latents",
                "input_image",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, camera_control_direction, **kwargs) -> bool:
        return camera_control_direction is not None

    def run(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        num_frames,
        camera_control_direction,
        camera_control_speed,
        camera_control_origin,
        latents,
        input_image,
        tiled,
        tile_size,
        tile_stride,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        camera_control_plucker_embedding = (
            pipe.dit.control_adapter.process_camera_coordinates(
                camera_control_direction,
                num_frames,
                height,
                width,
                camera_control_speed,
                camera_control_origin,
            )
        )

        control_camera_video = (
            camera_control_plucker_embedding[:num_frames]
            .permute([3, 0, 1, 2])
            .unsqueeze(0)
        )
        control_camera_latents = torch.concat(
            [
                torch.repeat_interleave(
                    control_camera_video[:, :, 0:1], repeats=4, dim=2
                ),
                control_camera_video[:, :, 1:],
            ],
            dim=2,
        ).transpose(1, 2)
        b, f, c, h, w = control_camera_latents.shape
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, 4, c, h, w)
            .transpose(2, 3)
        )
        control_camera_latents = (
            control_camera_latents.contiguous()
            .view(b, f // 4, c * 4, h, w)
            .transpose(1, 2)
        )
        control_camera_latents_input = control_camera_latents.to(
            device=pipe.device, dtype=pipe.torch_dtype
        )

        input_image = input_image.resize((width, height))
        input_latents = pipe.preprocess_video([input_image])
        input_latents = pipe.vae.encode(input_latents, device=pipe.device)
        y = torch.zeros_like(latents).to(pipe.device)
        y[:, :, :1] = input_latents
        y = y.to(dtype=pipe.torch_dtype, device=pipe.device)

        if y.shape[1] != pipe.dit.in_dim - latents.shape[1]:
            image = pipe.preprocess_image(input_image.resize((width, height))).to(
                pipe.device
            )
            vae_input = torch.concat(
                [
                    image.transpose(0, 1),
                    torch.zeros(3, num_frames - 1, height, width).to(image.device),
                ],
                dim=1,
            )
            y = pipe.vae.encode(
                [vae_input.to(dtype=pipe.torch_dtype, device=pipe.device)],
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )[0]
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=pipe.device)
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)
            msk = msk.transpose(1, 2)[0]
            y = torch.cat([msk, y])
            y = y.unsqueeze(0)
            y = y.to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"control_camera_latents_input": control_camera_latents_input, "y": y}


class WanVideoUnit_SpeedControl(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=("motion_bucket_id",))

    def check_condition(self, pipe, motion_bucket_id, **kwargs) -> bool:
        return motion_bucket_id is not None

    def run(self, pipe: WanVideoPipeline, motion_bucket_id, **kwargs) -> dict:
        motion_bucket_id = torch.Tensor((motion_bucket_id,)).to(
            dtype=pipe.torch_dtype, device=pipe.device
        )
        return {"motion_bucket_id": motion_bucket_id}


class WanVideoUnit_VACE(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "vace_video",
                "vace_video_mask",
                "vace_reference_image",
                "vace_scale",
                "height",
                "width",
                "num_frames",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def check_condition(
        self, pipe, vace_video, vace_video_mask, vace_reference_image, **kwargs
    ) -> bool:
        return (
            vace_video is not None
            or vace_video_mask is not None
            or vace_reference_image is not None
        )

    def get_fallback_output(self, pipe, vace_scale, **kwargs) -> dict:
        return {"vace_context": None, "vace_scale": vace_scale}

    def run(
        self,
        pipe: WanVideoPipeline,
        vace_video,
        vace_video_mask,
        vace_reference_image,
        vace_scale,
        height,
        width,
        num_frames,
        tiled,
        tile_size,
        tile_stride,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(["vae"])
        if vace_video is None:
            vace_video = torch.zeros(
                (1, 3, num_frames, height, width),
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
        else:
            vace_video = pipe.preprocess_video(vace_video)

        if vace_video_mask is None:
            vace_video_mask = torch.ones_like(vace_video)
        else:
            vace_video_mask = pipe.preprocess_video(
                vace_video_mask, min_value=0, max_value=1
            )

        inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
        reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
        inactive = pipe.vae.encode(
            inactive,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        reactive = pipe.vae.encode(
            reactive,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        vace_video_latents = torch.concat((inactive, reactive), dim=1)

        vace_mask_latents = rearrange(
            vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=8, Q=8
        )
        vace_mask_latents = torch.nn.functional.interpolate(
            vace_mask_latents,
            size=(
                (vace_mask_latents.shape[2] + 3) // 4,
                vace_mask_latents.shape[3],
                vace_mask_latents.shape[4],
            ),
            mode="nearest-exact",
        )

        if vace_reference_image is not None:
            if not isinstance(vace_reference_image, list):
                vace_reference_image = [vace_reference_image]

            vace_reference_image = pipe.preprocess_video(vace_reference_image)

            bs, c, f, h, w = vace_reference_image.shape
            new_vace_ref_images = []
            for j in range(f):
                new_vace_ref_images.append(vace_reference_image[0, :, j : j + 1])
            vace_reference_image = new_vace_ref_images

            vace_reference_latents = pipe.vae.encode(
                vace_reference_image,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_reference_latents = torch.concat(
                (vace_reference_latents, torch.zeros_like(vace_reference_latents)),
                dim=1,
            )
            vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

            vace_video_latents = torch.concat(
                (*vace_reference_latents, vace_video_latents), dim=2
            )
            vace_mask_latents = torch.concat(
                (torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents),
                dim=2,
            )

        vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
        return {"vace_context": vace_context, "vace_scale": vace_scale}


class WanVideoUnit_UnifiedSequenceParallel(PipelineUnit):
    def __init__(self):
        super().__init__(input_params=())

    def check_condition(self, pipe, **kwargs) -> bool:
        return (
            hasattr(pipe, "use_unified_sequence_parallel")
            and pipe.use_unified_sequence_parallel
        )

    def run(self, pipe: WanVideoPipeline, **kwargs) -> dict:
        return {"use_unified_sequence_parallel": True}


class WanVideoUnit_TeaCache(PipelineUnit):
    def __init__(self):
        super().__init__(
            seperate_cfg=True,
            input_params_posi={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
            input_params_nega={
                "num_inference_steps": "num_inference_steps",
                "tea_cache_l1_thresh": "tea_cache_l1_thresh",
                "tea_cache_model_id": "tea_cache_model_id",
            },
        )

    def check_condition(self, pipe, tea_cache_l1_thresh, **kwargs) -> bool:
        return tea_cache_l1_thresh is not None

    def run(
        self,
        pipe: WanVideoPipeline,
        num_inference_steps,
        tea_cache_l1_thresh,
        tea_cache_model_id,
        **kwargs,
    ) -> dict:
        return {
            "tea_cache": TeaCache(
                num_inference_steps,
                rel_l1_thresh=tea_cache_l1_thresh,
                model_id=tea_cache_model_id,
            )
        }


# WanVideoUnit_CfgMerger imported from units.py


class WanVideoUnit_S2V(PipelineUnit):
    """S2V (Sound to Video) unit - processes audio input (take_over mode)"""

    def __init__(self):
        super().__init__(
            take_over=True,
            onload_model_names=(
                "audio_encoder",
                "vae",
            ),
        )

    def check_condition(
        self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega, **kwargs
    ):
        if (
            (
                inputs_shared.get("input_audio") is None
                and inputs_shared.get("audio_embeds") is None
            )
            or pipe.audio_encoder is None
            or pipe.audio_processor is None
        ):
            return False
        return True

    def get_fallback_output(
        self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega
    ):
        return inputs_shared, inputs_posi, inputs_nega

    def process_audio(
        self,
        pipe: WanVideoPipeline,
        input_audio,
        audio_sample_rate,
        num_frames,
        fps=16,
        audio_embeds=None,
        return_all=False,
    ):
        if audio_embeds is not None:
            return {"audio_embeds": audio_embeds}
        pipe.load_models_to_device(["audio_encoder"])
        audio_embeds = pipe.audio_encoder.get_audio_feats_per_inference(
            input_audio,
            audio_sample_rate,
            pipe.audio_processor,
            fps=fps,
            batch_frames=num_frames - 1,
            dtype=pipe.torch_dtype,
            device=pipe.device,
        )
        if return_all:
            return audio_embeds
        else:
            return {"audio_embeds": audio_embeds[0]}

    def process_motion_latents(
        self,
        pipe: WanVideoPipeline,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        motion_video=None,
    ):
        pipe.load_models_to_device(["vae"])
        motion_frames = 73
        kwargs = {}
        if motion_video is not None and len(motion_video) > 0:
            assert (
                len(motion_video) == motion_frames
            ), f"motion video must have {motion_frames} frames, but got {len(motion_video)}"
            motion_latents = pipe.preprocess_video(motion_video)
            kwargs["drop_motion_frames"] = False
        else:
            motion_latents = torch.zeros(
                [1, 3, motion_frames, height, width],
                dtype=pipe.torch_dtype,
                device=pipe.device,
            )
            kwargs["drop_motion_frames"] = True
        motion_latents = pipe.vae.encode(
            motion_latents,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        kwargs.update({"motion_latents": motion_latents})
        return kwargs

    def process_pose_cond(
        self,
        pipe: WanVideoPipeline,
        s2v_pose_video,
        num_frames,
        height,
        width,
        tiled,
        tile_size,
        tile_stride,
        s2v_pose_latents=None,
        num_repeats=1,
        return_all=False,
    ):
        if s2v_pose_latents is not None:
            return {"s2v_pose_latents": s2v_pose_latents}
        if s2v_pose_video is None:
            return {"s2v_pose_latents": None}
        pipe.load_models_to_device(["vae"])
        infer_frames = num_frames - 1
        input_video = pipe.preprocess_video(s2v_pose_video)[
            :, :, : infer_frames * num_repeats
        ]
        # pad if not enough frames
        padding_frames = infer_frames * num_repeats - input_video.shape[2]
        input_video = torch.cat(
            [
                input_video,
                -torch.ones(
                    1,
                    3,
                    padding_frames,
                    height,
                    width,
                    device=input_video.device,
                    dtype=input_video.dtype,
                ),
            ],
            dim=2,
        )
        input_videos = input_video.chunk(num_repeats, dim=2)
        pose_conds = []
        for r in range(num_repeats):
            cond = input_videos[r]
            cond = torch.cat([cond[:, :, 0:1].repeat(1, 1, 1, 1, 1), cond], dim=2)
            cond_latents = pipe.vae.encode(
                cond,
                device=pipe.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            ).to(dtype=pipe.torch_dtype, device=pipe.device)
            pose_conds.append(cond_latents[:, :, 1:])
        if return_all:
            return pose_conds
        else:
            return {"s2v_pose_latents": pose_conds[0]}

    def run(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        num_frames, height, width, tiled, tile_size, tile_stride = (
            inputs_shared.get("num_frames"),
            inputs_shared.get("height"),
            inputs_shared.get("width"),
            inputs_shared.get("tiled"),
            inputs_shared.get("tile_size"),
            inputs_shared.get("tile_stride"),
        )
        input_audio, audio_embeds, audio_sample_rate = (
            inputs_shared.pop("input_audio"),
            inputs_shared.pop("audio_embeds"),
            inputs_shared.get("audio_sample_rate"),
        )
        s2v_pose_video, s2v_pose_latents, motion_video = (
            inputs_shared.pop("s2v_pose_video"),
            inputs_shared.pop("s2v_pose_latents"),
            inputs_shared.pop("motion_video"),
        )

        audio_input_positive = self.process_audio(
            pipe, input_audio, audio_sample_rate, num_frames, audio_embeds=audio_embeds
        )
        inputs_posi.update(audio_input_positive)
        inputs_nega.update({"audio_embeds": 0.0 * audio_input_positive["audio_embeds"]})

        inputs_shared.update(
            self.process_motion_latents(
                pipe, height, width, tiled, tile_size, tile_stride, motion_video
            )
        )
        inputs_shared.update(
            self.process_pose_cond(
                pipe,
                s2v_pose_video,
                num_frames,
                height,
                width,
                tiled,
                tile_size,
                tile_stride,
                s2v_pose_latents=s2v_pose_latents,
            )
        )
        return inputs_shared, inputs_posi, inputs_nega

    @staticmethod
    def pre_calculate_audio_pose(
        pipe: WanVideoPipeline,
        input_audio=None,
        audio_sample_rate=16000,
        s2v_pose_video=None,
        num_frames=81,
        height=448,
        width=832,
        fps=16,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ):
        assert (
            pipe.audio_encoder is not None and pipe.audio_processor is not None
        ), "Please load audio encoder and audio processor first."
        shapes = (
            WanVideoUnit_ShapeChecker().process(pipe, height, width, num_frames).outputs
        )
        height, width, num_frames = (
            shapes["height"],
            shapes["width"],
            shapes["num_frames"],
        )
        unit = WanVideoUnit_S2V()
        audio_embeds = unit.process_audio(
            pipe, input_audio, audio_sample_rate, num_frames, fps, return_all=True
        )
        pose_latents = unit.process_pose_cond(
            pipe,
            s2v_pose_video,
            num_frames,
            height,
            width,
            num_repeats=len(audio_embeds),
            return_all=True,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )
        pose_latents = None if s2v_pose_video is None else pose_latents
        return audio_embeds, pose_latents, len(audio_embeds)


class WanVideoPostUnit_S2V(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("latents", "motion_latents", "drop_motion_frames")
        )

    def check_condition(
        self, pipe, motion_latents, drop_motion_frames, **kwargs
    ) -> bool:
        if pipe.audio_encoder is None:
            return False
        if motion_latents is None:
            return False
        if drop_motion_frames:
            return False
        return True

    def run(
        self,
        pipe: WanVideoPipeline,
        latents,
        motion_latents,
        drop_motion_frames,
        **kwargs,
    ) -> dict:
        latents = torch.cat([motion_latents, latents[:, :, 1:]], dim=2)
        return {"latents": latents}


class WanVideoPostUnit_AnimateVideoSplit(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "input_video",
                "animate_pose_video",
                "animate_face_video",
                "animate_inpaint_video",
                "animate_mask_video",
            )
        )

    def check_condition(self, pipe, input_video, **kwargs) -> bool:
        return input_video is not None

    def run(
        self,
        pipe: WanVideoPipeline,
        input_video,
        animate_pose_video,
        animate_face_video,
        animate_inpaint_video,
        animate_mask_video,
        **kwargs,
    ) -> dict:
        if animate_pose_video is not None:
            animate_pose_video = animate_pose_video[: len(input_video) - 4]
        if animate_face_video is not None:
            animate_face_video = animate_face_video[: len(input_video) - 4]
        if animate_inpaint_video is not None:
            animate_inpaint_video = animate_inpaint_video[: len(input_video) - 4]
        if animate_mask_video is not None:
            animate_mask_video = animate_mask_video[: len(input_video) - 4]
        return {
            "animate_pose_video": animate_pose_video,
            "animate_face_video": animate_face_video,
            "animate_inpaint_video": animate_inpaint_video,
            "animate_mask_video": animate_mask_video,
        }


class WanVideoPostUnit_AnimatePoseLatents(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=("animate_pose_video", "tiled", "tile_size", "tile_stride"),
            onload_model_names=("vae",),
        )

    def check_condition(self, pipe, animate_pose_video, **kwargs) -> bool:
        return animate_pose_video is not None

    def run(
        self,
        pipe: WanVideoPipeline,
        animate_pose_video,
        tiled,
        tile_size,
        tile_stride,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)
        animate_pose_video = pipe.preprocess_video(animate_pose_video)
        pose_latents = pipe.vae.encode(
            animate_pose_video,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        return {"pose_latents": pose_latents}


class WanVideoPostUnit_AnimateFacePixelValues(PipelineUnit):
    """Animate Face Pixel Values unit - take_over mode"""

    def __init__(self):
        super().__init__(take_over=True)

    def check_condition(
        self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega, **kwargs
    ):
        return inputs_shared.get("animate_face_video", None) is not None

    def get_fallback_output(
        self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega
    ):
        return inputs_shared, inputs_posi, inputs_nega

    def run(self, pipe: WanVideoPipeline, inputs_shared, inputs_posi, inputs_nega):
        inputs_posi["face_pixel_values"] = pipe.preprocess_video(
            inputs_shared["animate_face_video"]
        )
        inputs_nega["face_pixel_values"] = (
            torch.zeros_like(inputs_posi["face_pixel_values"]) - 1
        )
        return inputs_shared, inputs_posi, inputs_nega


class WanVideoPostUnit_AnimateInpaint(PipelineUnit):
    def __init__(self):
        super().__init__(
            input_params=(
                "animate_inpaint_video",
                "animate_mask_video",
                "input_image",
                "tiled",
                "tile_size",
                "tile_stride",
            ),
            onload_model_names=("vae",),
        )

    def get_i2v_mask(
        self, lat_t, lat_h, lat_w, mask_len=1, mask_pixel_values=None, device="cuda"
    ):
        if mask_pixel_values is None:
            msk = torch.zeros(1, (lat_t - 1) * 4 + 1, lat_h, lat_w, device=device)
        else:
            msk = mask_pixel_values.clone()
        msk[:, :mask_len] = 1
        msk = torch.concat(
            [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1
        )
        msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
        msk = msk.transpose(1, 2)[0]
        return msk

    def check_condition(
        self, pipe, animate_inpaint_video, animate_mask_video, **kwargs
    ) -> bool:
        return animate_inpaint_video is not None and animate_mask_video is not None

    def run(
        self,
        pipe: WanVideoPipeline,
        animate_inpaint_video,
        animate_mask_video,
        input_image,
        tiled,
        tile_size,
        tile_stride,
        **kwargs,
    ) -> dict:
        pipe.load_models_to_device(self.onload_model_names)

        bg_pixel_values = pipe.preprocess_video(animate_inpaint_video)
        y_reft = pipe.vae.encode(
            bg_pixel_values,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        )[0].to(dtype=pipe.torch_dtype, device=pipe.device)
        _, lat_t, lat_h, lat_w = y_reft.shape

        ref_pixel_values = pipe.preprocess_video([input_image])
        ref_latents = pipe.vae.encode(
            ref_pixel_values,
            device=pipe.device,
            tiled=tiled,
            tile_size=tile_size,
            tile_stride=tile_stride,
        ).to(dtype=pipe.torch_dtype, device=pipe.device)
        mask_ref = self.get_i2v_mask(1, lat_h, lat_w, 1, device=pipe.device)
        y_ref = torch.concat([mask_ref, ref_latents[0]]).to(
            dtype=torch.bfloat16, device=pipe.device
        )

        mask_pixel_values = 1 - pipe.preprocess_video(
            animate_mask_video, max_value=1, min_value=0
        )
        mask_pixel_values = rearrange(mask_pixel_values, "b c t h w -> (b t) c h w")
        mask_pixel_values = torch.nn.functional.interpolate(
            mask_pixel_values, size=(lat_h, lat_w), mode="nearest"
        )
        mask_pixel_values = rearrange(
            mask_pixel_values, "(b t) c h w -> b t c h w", b=1
        )[:, :, 0]
        msk_reft = self.get_i2v_mask(
            lat_t,
            lat_h,
            lat_w,
            0,
            mask_pixel_values=mask_pixel_values,
            device=pipe.device,
        )

        y_reft = torch.concat([msk_reft, y_reft]).to(
            dtype=torch.bfloat16, device=pipe.device
        )
        y = torch.concat([y_ref, y_reft], dim=1).unsqueeze(0)
        return {"y": y}


class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None

        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [
                -5.21862437e04,
                9.23041404e03,
                -5.28275948e02,
                1.36987616e01,
                -4.99875664e-02,
            ],
            "Wan2.1-T2V-14B": [
                -3.03318725e05,
                4.90537029e04,
                -2.65530556e03,
                5.87365115e01,
                -3.15583525e-01,
            ],
            "Wan2.1-I2V-14B-480P": [
                2.57151496e05,
                -3.54229917e04,
                1.40286849e03,
                -1.35890334e01,
                1.32517977e-01,
            ],
            "Wan2.1-I2V-14B-720P": [
                8.10705460e03,
                2.13393892e03,
                -3.72934672e02,
                1.66203073e01,
                -4.17769401e-02,
            ],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(
                f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids})."
            )
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(
                (
                    (modulated_inp - self.previous_modulated_input).abs().mean()
                    / self.previous_modulated_input.abs().mean()
                )
                .cpu()
                .item()
            )
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states


class TemporalTiler_BCTHW:
    def __init__(self):
        pass

    def build_1d_mask(self, length, left_bound, right_bound, border_width):
        x = torch.ones((length,))
        if border_width == 0:
            return x

        shift = 0.5
        if not left_bound:
            x[:border_width] = (torch.arange(border_width) + shift) / border_width
        if not right_bound:
            x[-border_width:] = torch.flip(
                (torch.arange(border_width) + shift) / border_width, dims=(0,)
            )
        return x

    def build_mask(self, data, is_bound, border_width):
        _, _, T, _, _ = data.shape
        t = self.build_1d_mask(T, is_bound[0], is_bound[1], border_width[0])
        mask = repeat(t, "T -> 1 1 T 1 1")
        return mask

    def run(
        self,
        model_fn,
        sliding_window_size,
        sliding_window_stride,
        computation_device,
        computation_dtype,
        model_kwargs,
        tensor_names,
        batch_size=None,
    ):
        tensor_names = [
            tensor_name
            for tensor_name in tensor_names
            if model_kwargs.get(tensor_name) is not None
        ]
        tensor_dict = {
            tensor_name: model_kwargs[tensor_name] for tensor_name in tensor_names
        }
        B, C, T, H, W = tensor_dict[tensor_names[0]].shape
        if batch_size is not None:
            B *= batch_size
        data_device, data_dtype = (
            tensor_dict[tensor_names[0]].device,
            tensor_dict[tensor_names[0]].dtype,
        )
        value = torch.zeros((B, C, T, H, W), device=data_device, dtype=data_dtype)
        weight = torch.zeros((1, 1, T, 1, 1), device=data_device, dtype=data_dtype)
        for t in range(0, T, sliding_window_stride):
            if (
                t - sliding_window_stride >= 0
                and t - sliding_window_stride + sliding_window_size >= T
            ):
                continue
            t_ = min(t + sliding_window_size, T)
            model_kwargs.update(
                {
                    tensor_name: tensor_dict[tensor_name][:, :, t:t_:, :].to(
                        device=computation_device, dtype=computation_dtype
                    )
                    for tensor_name in tensor_names
                }
            )
            model_output = model_fn(**model_kwargs).to(
                device=data_device, dtype=data_dtype
            )
            mask = self.build_mask(
                model_output,
                is_bound=(t == 0, t_ == T),
                border_width=(sliding_window_size - sliding_window_stride,),
            ).to(device=data_device, dtype=data_dtype)
            value[:, :, t:t_, :, :] += model_output * mask
            weight[:, :, t:t_, :, :] += mask
        value /= weight
        model_kwargs.update(tensor_dict)
        return value


def _check_dit_context_text_dim(dit: WanModel, context: torch.Tensor) -> None:
    """Raise if context last dim does not match DiT text_embedding input (e.g. T5 or Qwen adapter after project_out)."""
    if context.shape[-1] != dit.text_embedding[0].in_features:
        raise ValueError(
            f"Context dim {context.shape[-1]} != dit.text_embedding in_features "
            f"{dit.text_embedding[0].in_features}. Align text encoder output with DiT "
            f"(e.g. qwen_proj_out_dim / Qwen35TextEncoderAdapter.project_out)."
        )


def model_fn_wan_video(
    dit: WanModel,
    # motion_controller: WanMotionControllerModel = None,
    # vace: VaceWanModel = None,
    # animate_adapter: WanAnimateAdapter = None,
    motion_controller=None,
    vace=None,
    animate_adapter=None,
    latents: torch.Tensor = None,
    timestep: torch.Tensor = None,
    context: torch.Tensor = None,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    num_views: int = None,
    reference_latents=None,
    vace_context=None,
    vace_scale=1.0,
    audio_embeds: Optional[torch.Tensor] = None,
    motion_latents: Optional[torch.Tensor] = None,
    s2v_pose_latents: Optional[torch.Tensor] = None,
    drop_motion_frames: bool = True,
    tea_cache: TeaCache = None,
    use_unified_sequence_parallel: bool = False,
    motion_bucket_id: Optional[torch.Tensor] = None,
    pose_latents=None,
    face_pixel_values=None,
    sliding_window_size: Optional[int] = None,
    sliding_window_stride: Optional[int] = None,
    cfg_merge: bool = False,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
    control_camera_latents_input=None,
    fuse_vae_embedding_in_latents: bool = False,
    ###
    return_video_features: bool = False,
    action_context: Optional[torch.Tensor] = None,
    view_mask: Optional[torch.Tensor] = None,
    **kwargs,
):
    if sliding_window_size is not None and sliding_window_stride is not None:
        model_kwargs = dict(
            dit=dit,
            motion_controller=motion_controller,
            vace=vace,
            latents=latents,
            timestep=timestep,
            context=context,
            clip_feature=clip_feature,
            y=y,
            reference_latents=reference_latents,
            vace_context=vace_context,
            vace_scale=vace_scale,
            tea_cache=tea_cache,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
            motion_bucket_id=motion_bucket_id,
            view_mask=view_mask,
        )
        return TemporalTiler_BCTHW().run(
            model_fn_wan_video,
            sliding_window_size,
            sliding_window_stride,
            latents.device,
            latents.dtype,
            model_kwargs=model_kwargs,
            tensor_names=["latents", "y"],
            batch_size=2 if cfg_merge else 1,
        )
    # wan2.2 s2v
    if audio_embeds is not None:
        return model_fn_wans2v(
            dit=dit,
            latents=latents,
            timestep=timestep,
            context=context,
            audio_embeds=audio_embeds,
            motion_latents=motion_latents,
            s2v_pose_latents=s2v_pose_latents,
            drop_motion_frames=drop_motion_frames,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_unified_sequence_parallel=use_unified_sequence_parallel,
        )

    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
        )

    # Timestep
    if dit.seperated_timestep and fuse_vae_embedding_in_latents:
        timestep = torch.concat(
            [
                torch.zeros(
                    (1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=timestep.dtype,
                    device=latents.device,
                ),
                torch.ones(
                    (latents.shape[2] - 1, latents.shape[3] * latents.shape[4] // 4),
                    dtype=timestep.dtype,
                    device=latents.device,
                )
                * timestep,
            ]
        ).flatten()
        t = dit.time_embedding(
            sinusoidal_embedding_1d(
                dit.freq_dim, timestep, output_dtype=latents.dtype
            ).unsqueeze(0)
        )
        if (
            use_unified_sequence_parallel
            and dist.is_initialized()
            and dist.get_world_size() > 1
        ):
            t_chunks = torch.chunk(t, get_sequence_parallel_world_size(), dim=1)
            t_chunks = [
                torch.nn.functional.pad(
                    chunk, (0, 0, 0, t_chunks[0].shape[1] - chunk.shape[1]), value=0
                )
                for chunk in t_chunks
            ]
            t = t_chunks[get_sequence_parallel_rank()]
        t_mod = dit.time_projection(t).unflatten(2, (6, dit.dim))
    else:
        t = dit.time_embedding(
            sinusoidal_embedding_1d(dit.freq_dim, timestep, output_dtype=latents.dtype)
        )
        t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))

    # Motion Controller
    if motion_bucket_id is not None and motion_controller is not None:
        t_mod = t_mod + motion_controller(motion_bucket_id).unflatten(1, (6, dit.dim))

    x = latents
    # Merged cfg
    if x.shape[0] != context.shape[0]:
        x = torch.concat([x] * context.shape[0], dim=0)
    if timestep.shape[0] != context.shape[0]:
        timestep = torch.concat([timestep] * context.shape[0], dim=0)

    # Image Embedding
    img_seq_len = 257
    if y is not None and dit.require_vae_embedding:
        x = torch.cat([x, y], dim=1)

    # context shape: [B, L, text_dim] -> [B, L, dit.dim]; text_dim must match first Linear in_features
    _check_dit_context_text_dim(dit, context)
    context = dit.text_embedding(context)

    if clip_feature is not None and dit.require_clip_embedding:
        clip_embdding = dit.img_emb(clip_feature)
        img_seq_len = clip_embdding.shape[1]
        context = torch.cat([clip_embdding, context], dim=1)

    # Action conditioning: append action tokens to context for cross-attention
    if action_context is not None:
        nv = num_views if num_views is not None else 1
        batch_size = context.shape[0] // nv
        # Expand action_context from (batch, seq, dim) to (batch * num_views, seq, dim)
        if len(action_context.shape) == 3 and action_context.shape[0] == batch_size:
            action_context = action_context.repeat_interleave(nv, dim=0)
        elif len(action_context.shape) == 3 and action_context.shape[0] == 1:
            action_context = action_context.expand(context.shape[0], -1, -1)
        else:
            raise ValueError(
                f"action_context batch dim {action_context.shape[0]} must be "
                f"{batch_size} or 1, got shape {action_context.shape}"
            )

        context = torch.cat([action_context, context], dim=1)

    # Camera control
    x = dit.patchify(x, control_camera_latents_input)

    # Animate
    if pose_latents is not None and face_pixel_values is not None:
        x, motion_vec = animate_adapter.after_patch_embedding(
            x, pose_latents, face_pixel_values
        )

    # Patchify
    f, h, w = x.shape[2:]
    x = rearrange(x, "b c f h w -> b (f h w) c").contiguous()

    # Reference image
    if reference_latents is not None:
        if len(reference_latents.shape) == 5:
            reference_latents = reference_latents[:, :, 0]
        reference_latents = dit.ref_conv(reference_latents).flatten(2).transpose(1, 2)
        x = torch.concat([reference_latents, x], dim=1)
        f += 1

    # Build cos/sin freqs for xcompute rope kernel
    (f_cos, f_sin), (h_cos, h_sin), (w_cos, w_sin) = dit.freqs
    cos_freqs = (
        torch.cat(
            [
                f_cos[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                h_cos[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                w_cos[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(1, f * h * w, -1)
        .to(x.device)
    )
    sin_freqs = (
        torch.cat(
            [
                f_sin[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                h_sin[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                w_sin[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(1, f * h * w, -1)
        .to(x.device)
    )
    freqs = (cos_freqs, sin_freqs)

    v = num_views
    view_ids = kwargs.get("view_ids", None)
    if view_ids is None:
        raise ValueError("Missing required 'view_ids' in model_fn_wan_video")

    # Strict format: view_ids must be torch.LongTensor of shape [b, v],
    # where b is logical batch size (views are stacked into batch dim).
    if not isinstance(view_ids, torch.Tensor):
        raise ValueError(
            f"view_ids must be torch.Tensor with shape [B,V], got {type(view_ids)}"
        )
    view_positions = view_ids.to(f_cos.device, dtype=torch.long)
    if view_positions.dim() != 2:
        raise ValueError(
            f"view_ids must be 2D tensor [B,V], got shape {tuple(view_positions.shape)}"
        )

    b_v = x.shape[0]
    if b_v % v != 0:
        raise ValueError(
            f"Batch dim not divisible by num_views: Bv={b_v}, num_views={v}"
        )
    b = b_v // v
    if tuple(view_positions.shape) != (b, v):
        raise ValueError(
            f"view_ids shape mismatch: expected ({b},{v}), got {tuple(view_positions.shape)}"
        )

    # view_positions: [b, v]
    f_cos_view = f_cos[view_positions]  # [b, v, d]
    f_sin_view = f_sin[view_positions]  # [b, v, d]

    # Build cos/sin view_freqs for xcompute rope kernel
    # Build per-sample view_freqs then repeat along frame-batch: [b, ...] -> [b*f, ...]
    view_cos = torch.cat(
        [
            f_cos_view.view(b, v, 1, 1, -1).expand(b, v, h, w, -1),
            h_cos[:h].view(1, 1, h, 1, -1).expand(b, v, h, w, -1),
            w_cos[:w].view(1, 1, 1, w, -1).expand(b, v, h, w, -1),
        ],
        dim=-1,
    ).reshape(b, v * h * w, -1)
    view_sin = torch.cat(
        [
            f_sin_view.view(b, v, 1, 1, -1).expand(b, v, h, w, -1),
            h_sin[:h].view(1, 1, h, 1, -1).expand(b, v, h, w, -1),
            w_sin[:w].view(1, 1, 1, w, -1).expand(b, v, h, w, -1),
        ],
        dim=-1,
    ).reshape(b, v * h * w, -1)

    # q/k in view-attn uses batch dim (b*f), so we must match it exactly (no broadcasting).
    view_cos = view_cos.repeat_interleave(f, dim=0).to(x.device)
    view_sin = view_sin.repeat_interleave(f, dim=0).to(x.device)
    view_freqs = (view_cos, view_sin)

    # TeaCache
    if tea_cache is not None:
        tea_cache_update = tea_cache.check(dit, x, t_mod)
    else:
        tea_cache_update = False

    if vace_context is not None:
        vace_hints = vace(
            x,
            vace_context,
            context,
            t_mod,
            freqs,
            use_gradient_checkpointing=use_gradient_checkpointing,
            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,
        )

    block_time_features = []
    # blocks
    import torch.distributed as dist

    if get_sp_world_size() > 1:
        x, original_seq_len = sp_shard(x)
    if tea_cache_update:
        x = tea_cache.update(x)
    else:

        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs, view_mask=view_mask)

            return custom_forward

        for block_id, block in enumerate(dit.blocks):
            # Block
            if use_gradient_checkpointing_offload:
                with torch.autograd.graph.save_on_cpu(pin_memory=True):
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x,
                        context,
                        t_mod,
                        freqs,
                        view_freqs,
                        v,
                        f,
                        h,
                        w,
                        img_seq_len,
                        use_reentrant=False,
                    )
            elif use_gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    freqs,
                    view_freqs,
                    v,
                    f,
                    h,
                    w,
                    img_seq_len,
                    use_reentrant=False,
                )
            else:
                x = block(
                    x,
                    context,
                    t_mod,
                    freqs,
                    view_freqs,
                    v,
                    f,
                    h,
                    w,
                    img_seq_len=img_seq_len,
                    view_mask=view_mask,
                )
            ###
            # collect spatio-temporal features for this block: [B, f, h, w, dim]
            # Keep full spatial-temporal information, allowing ActionTransformer to interact with all video information
            # TODO: Use shard tensor, process it in action dit
            if get_sp_world_size() > 1:
                gathered_x = sp_gather(x, original_seq_len)
                bt = rearrange(gathered_x, "b (f h w) c -> b f h w c", f=f, h=h, w=w)
            else:
                bt = rearrange(x, "b (f h w) c -> b f h w c", f=f, h=h, w=w)
            block_time_features.append(bt)
            # VACE
            if vace_context is not None and block_id in vace.vace_layers_mapping:
                current_vace_hint = vace_hints[vace.vace_layers_mapping[block_id]]
                if (
                    use_unified_sequence_parallel
                    and dist.is_initialized()
                    and dist.get_world_size() > 1
                ):
                    vace_chunks = torch.chunk(
                        current_vace_hint, get_sequence_parallel_world_size(), dim=1
                    )
                    current_vace_hint = vace_chunks[get_sequence_parallel_rank()]
                    current_vace_hint = torch.nn.functional.pad(
                        current_vace_hint,
                        (0, 0, 0, vace_chunks[0].shape[1] - current_vace_hint.shape[1]),
                        value=0,
                    )
                x = x + current_vace_hint * vace_scale

            # Animate
            if pose_latents is not None and face_pixel_values is not None:
                x = animate_adapter.after_transformer_block(block_id, x, motion_vec)
        if tea_cache is not None:
            tea_cache.store(x)

    x_tokens = x  # tokens before head: [B, S, dim]
    if get_sp_world_size() > 1:
        x_tokens = sp_gather(x_tokens, original_seq_len)

    # RGB output
    rgb_output = dit.head(x_tokens, t)
    # Remove reference latents
    if reference_latents is not None:
        rgb_output = rgb_output[:, reference_latents.shape[1] :]
        f -= 1
    rgb_output = dit.unpatchify(rgb_output, (f, h, w))

    # Return results
    if return_video_features:
        # reshape tokens to [B, f, h, w, dim] then spatial mean -> [B, f, dim]
        video_time_features = rearrange(
            x_tokens, "b (f h w) c -> b f h w c", f=f, h=h, w=w
        ).mean(dim=(2, 3))
        return rgb_output, video_time_features, block_time_features
    return rgb_output


def model_fn_wans2v(
    dit,
    latents,
    timestep,
    context,
    audio_embeds,
    motion_latents,
    s2v_pose_latents,
    drop_motion_frames=True,
    use_gradient_checkpointing_offload=False,
    use_gradient_checkpointing=False,
    use_unified_sequence_parallel=False,
):
    if use_unified_sequence_parallel:
        import torch.distributed as dist
        from xfuser.core.distributed import (
            get_sequence_parallel_rank,
            get_sequence_parallel_world_size,
            get_sp_group,
        )
    origin_ref_latents = latents[:, :, 0:1]
    x = latents[:, :, 1:]

    # context embedding
    _check_dit_context_text_dim(dit, context)
    context = dit.text_embedding(context)

    # audio encode
    audio_emb_global, merged_audio_emb = dit.cal_audio_emb(audio_embeds)

    # x and s2v_pose_latents
    s2v_pose_latents = (
        torch.zeros_like(x) if s2v_pose_latents is None else s2v_pose_latents
    )
    x, (f, h, w) = dit.patchify(
        dit.patch_embedding(x) + dit.cond_encoder(s2v_pose_latents)
    )
    seq_len_x = seq_len_x_global = x.shape[
        1
    ]  # global used for unified sequence parallel

    # reference image
    ref_latents, (rf, rh, rw) = dit.patchify(dit.patch_embedding(origin_ref_latents))
    grid_sizes = dit.get_grid_sizes((f, h, w), (rf, rh, rw))  # noqa: F841
    x = torch.cat([x, ref_latents], dim=1)
    # mask
    mask = (
        torch.cat(
            [torch.zeros([1, seq_len_x]), torch.ones([1, ref_latents.shape[1]])], dim=1
        )
        .to(torch.long)
        .to(x.device)
    )
    # freqs
    # rope_precompute is not yet ported from upstream; this code path
    # (model_fn_wans2v) is inference-only and currently unreachable.
    raise NotImplementedError(
        "rope_precompute is not available; model_fn_wans2v is not yet ported"
    )
    pre_compute_freqs = None  # noqa: F841 — unreachable, keeps downstream parseable
    # motion
    x, pre_compute_freqs, mask = dit.inject_motion(
        x,
        pre_compute_freqs,
        mask,
        motion_latents,
        drop_motion_frames=drop_motion_frames,
        add_last_motion=2,
    )

    x = x + dit.trainable_cond_mask(mask).to(x.dtype)

    # tmod
    timestep = torch.cat(
        [timestep, torch.zeros([1], dtype=timestep.dtype, device=timestep.device)]
    )
    t = dit.time_embedding(
        sinusoidal_embedding_1d(dit.freq_dim, timestep, output_dtype=x.dtype)
    )
    t_mod = (
        dit.time_projection(t).unflatten(1, (6, dit.dim)).unsqueeze(2).transpose(0, 2)
    )

    if (
        use_unified_sequence_parallel
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        world_size, sp_rank = (
            get_sequence_parallel_world_size(),
            get_sequence_parallel_rank(),
        )
        assert (
            x.shape[1] % world_size == 0
        ), f"the dimension after chunk must be divisible by world size, but got {x.shape[1]} and {get_sequence_parallel_world_size()}"
        x = torch.chunk(x, world_size, dim=1)[sp_rank]
        seg_idxs = [0] + list(
            torch.cumsum(torch.tensor([x.shape[1]] * world_size), dim=0).cpu().numpy()
        )
        seq_len_x_list = [
            min(max(0, seq_len_x - seg_idxs[i]), x.shape[1])
            for i in range(len(seg_idxs) - 1)
        ]
        seq_len_x = seq_len_x_list[sp_rank]

    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    for block_id, block in enumerate(dit.blocks):
        if use_gradient_checkpointing_offload:
            with torch.autograd.graph.save_on_cpu(pin_memory=True):
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x,
                    context,
                    t_mod,
                    seq_len_x,
                    pre_compute_freqs[0],
                    use_reentrant=False,
                )
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(
                        lambda x: dit.after_transformer_block(
                            block_id, x, audio_emb_global, merged_audio_emb, seq_len_x
                        )
                    ),
                    x,
                    use_reentrant=False,
                )
        elif use_gradient_checkpointing:
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(block),
                x,
                context,
                t_mod,
                seq_len_x,
                pre_compute_freqs[0],
                use_reentrant=False,
            )
            x = torch.utils.checkpoint.checkpoint(
                create_custom_forward(
                    lambda x: dit.after_transformer_block(
                        block_id, x, audio_emb_global, merged_audio_emb, seq_len_x
                    )
                ),
                x,
                use_reentrant=False,
            )
        else:
            x = block(x, context, t_mod, seq_len_x, pre_compute_freqs[0])
            x = dit.after_transformer_block(
                block_id,
                x,
                audio_emb_global,
                merged_audio_emb,
                seq_len_x_global,
                use_unified_sequence_parallel,
            )

    if (
        use_unified_sequence_parallel
        and dist.is_initialized()
        and dist.get_world_size() > 1
    ):
        x = get_sp_group().all_gather(x, dim=1)

    x = x[:, :seq_len_x_global]
    x = dit.head(x, t[:-1])
    x = dit.unpatchify(x, (f, h, w))
    # make compatible with wan video
    x = torch.cat([origin_ref_latents, x], dim=2)
    return x

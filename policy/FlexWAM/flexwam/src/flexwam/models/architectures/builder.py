from typing import Any, Optional
import torch
from flexwam.utils.logging_config import get_logger
logger = get_logger(__name__)

from flexwam.models.backbone.wan22.action_dit import ActionDiT
from flexwam.models.backbone.wan22.loader import load_wan22_ti2v_5b_components
from flexwam.models.architectures.mot import MoT
from omegaconf import DictConfig, OmegaConf


def _plain_dict(value):
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    return value

class PretrainedBuilderMixin:
    @classmethod
    def from_wan22_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B",
        tokenizer_model_id: str = "Wan-AI/Wan2.1-T2V-1.3B",
        tokenizer_max_len: int = 512,
        load_text_encoder: bool = True,
        proprio_dim: Optional[int] = None,
        redirect_common_files: bool = True,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 5.0,
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_router_config: Optional[dict[str, Any]] = None,
        action_critic_config: Optional[dict[str, Any]] = None,
        action_context_as_prefix: bool = False,
        router_cf_config: Optional[dict[str, Any]] = None,
        context_input_dim: Optional[int] = None,
        state_conditioning: Optional[dict[str, Any]] = None,
        action_attention: Optional[dict[str, Any]] = None,
    ):
        video_dit_config = dict(video_dit_config)
        video_state_config = video_dit_config.pop("state_conditioning", None)
        video_dit_config.setdefault("cross_view_attention_interval", 4)
        components = load_wan22_ti2v_5b_components(
            device=device,
            torch_dtype=torch_dtype,
            model_id=model_id,
            tokenizer_model_id=tokenizer_model_id,
            tokenizer_max_len=tokenizer_max_len,
            redirect_common_files=redirect_common_files,
            dit_config=video_dit_config,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            load_text_encoder=load_text_encoder,
        )

        video_expert = components.dit
        logger.info(
            "WAN cross-view attention layers: %s.",
            list(video_expert.cross_view_layer_indices),
        )
        action_expert = ActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )
        if int(action_expert.text_dim) != int(video_dit_config["text_dim"]):
            raise ValueError("WAN video and action experts must use the same text_dim.")

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )

        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=components.text_encoder,
            tokenizer=components.tokenizer,
            text_dim=int(video_dit_config["text_dim"]),
            context_input_dim=context_input_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            action_router_config=action_router_config,
            action_critic_config=action_critic_config,
            action_context_as_prefix=action_context_as_prefix,
            router_cf_config=router_cf_config,
            video_state_config=video_state_config,
            state_conditioning=state_conditioning,
            action_attention=action_attention,
        )
        model.model_paths = {
            "video_dit": components.dit_path,
            "vae": components.vae_path,
            "text_encoder": components.text_encoder_path,
            "tokenizer": components.tokenizer_path,
            "action_dit_backbone": (
                "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path
            ),
        }
        return model

    @classmethod
    def from_lingbot_pretrained(
        cls,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        model_path: str = "lingbot-video",
        load_text_encoder: bool = False,
        proprio_dim: Optional[int] = None,
        video_dit_config: dict[str, Any] | None = None,
        action_dit_config: dict[str, Any] | None = None,
        action_dit_pretrained_path: str | None = None,
        skip_dit_load_from_pretrain: bool = False,
        mot_checkpoint_mixed_attn: bool = True,
        video_train_shift: float = 1.0,
        video_infer_shift: float = 1.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 1.0,
        action_infer_shift: float = 1.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,
        loss_lambda_action: float = 1.0,
        action_router_config: Optional[dict[str, Any]] = None,
        action_critic_config: Optional[dict[str, Any]] = None,
        context_input_dim: Optional[int] = None,
        router_cf_config: Optional[dict[str, Any]] = None,
        state_conditioning: Optional[dict[str, Any]] = None,
        action_attention: Optional[dict[str, Any]] = None,
    ):
        del load_text_encoder
        from flexwam.models.backbone.lingbot_video.action_dit import LingBotActionDiT
        from flexwam.models.backbone.lingbot_video import load_lingbot_components

        components = load_lingbot_components(
            model_path=model_path,
            torch_dtype=torch_dtype,
            device=device,
        )
        video_expert = components.transformer
        video_expert.num_views = int(video_dit_config.get("num_views", getattr(video_expert, "num_views", 1)))
        use_camera_embedding = video_dit_config.get("use_camera_embedding", None)
        video_expert.configure_camera_embedding(
            num_views=int(video_expert.num_views),
            enabled=None if use_camera_embedding is None else bool(use_camera_embedding),
        )
        if video_expert.use_camera_embedding:
            logger.info(
                "LingBot per-view camera embedding enabled: num_views=%d.",
                int(video_expert.num_views),
            )
        video_expert.video_attention_mask_mode = str(
            video_dit_config.get("video_attention_mask_mode", "first_frame_causal")
        )
        prefix_layout = str(video_dit_config.get("prefix_layout", "global"))
        if prefix_layout != "global":
            raise ValueError(
                "LingBot video expert only supports `video_dit_config.prefix_layout: global`, "
                f"got {prefix_layout!r}."
            )
        video_expert.prefix_layout = prefix_layout
        video_expert.configure_cross_view_attention(
            interval=int(video_dit_config.get("cross_view_attention_interval", 4)),
            layers=video_dit_config.get("cross_view_attention_layers"),
        )
        logger.info(
            "LingBot cross-view attention layers: %s.",
            list(video_expert.cross_view_layer_indices),
        )

        action_expert = LingBotActionDiT.from_pretrained(
            action_dit_config=action_dit_config,
            action_dit_pretrained_path=action_dit_pretrained_path,
            skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
            device=device,
            torch_dtype=torch_dtype,
        )

        mot = MoT(
            mixtures={"video": video_expert, "action": action_expert},
            mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
        )
        model = cls(
            video_expert=video_expert,
            action_expert=action_expert,
            mot=mot,
            vae=components.vae,
            text_encoder=None,
            tokenizer=None,
            text_dim=int(video_dit_config.get("text_dim", getattr(video_expert, "text_dim", 2560))),
            context_input_dim=context_input_dim,
            proprio_dim=proprio_dim,
            device=device,
            torch_dtype=torch_dtype,
            video_train_shift=video_train_shift,
            video_infer_shift=video_infer_shift,
            video_num_train_timesteps=video_num_train_timesteps,
            action_train_shift=action_train_shift,
            action_infer_shift=action_infer_shift,
            action_num_train_timesteps=action_num_train_timesteps,
            loss_lambda_video=loss_lambda_video,
            loss_lambda_action=loss_lambda_action,
            action_router_config=action_router_config,
            action_critic_config=action_critic_config,
            router_cf_config=router_cf_config,
            video_state_config=video_dit_config.get("state_conditioning"),
            state_conditioning=state_conditioning,
            action_attention=action_attention,
        )
        model.model_paths = {
            "video_dit": components.transformer_path,
            "vae": components.vae_path,
            "text_encoder": None,
            "tokenizer": None,
            "action_dit_backbone": "SKIPPED_PRETRAIN" if skip_dit_load_from_pretrain else action_dit_pretrained_path,
        }
        return model


def create_flexwam(
    model_id: str,
    tokenizer_model_id: str,
    video_dit_config,
    tokenizer_max_len: int = 512,
    load_text_encoder: bool = True,
    proprio_dim: int | None = None,
    action_dit_config=None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    action_router=None,
    action_critic=None,
    router_cf=None,
    mot_checkpoint_mixed_attn: bool = True,
    redirect_common_files: bool = True,
    action_context_as_prefix: bool = False,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    context_input_dim: int | None = None,
    state_conditioning=None,
    action_attention=None,
    use_cuda_graph: bool = False,
):
    from flexwam.models.architectures.flexwam import FlexWAM

    video_dit_config = _plain_dict(video_dit_config)
    action_dit_config = _plain_dict(action_dit_config) or {}
    video_scheduler = _plain_dict(video_scheduler) or {}
    action_scheduler = _plain_dict(action_scheduler)
    loss = _plain_dict(loss) or {}
    action_router = _plain_dict(action_router)
    action_critic = _plain_dict(action_critic)
    router_cf = _plain_dict(router_cf)

    model = FlexWAM.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=model_id,
        tokenizer_model_id=tokenizer_model_id,
        tokenizer_max_len=int(tokenizer_max_len),
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=proprio_dim,
        redirect_common_files=bool(redirect_common_files),
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_router_config=action_router,
        action_critic_config=action_critic,
        action_context_as_prefix=bool(action_context_as_prefix),
        router_cf_config=router_cf,
        context_input_dim=context_input_dim,
        state_conditioning=_plain_dict(state_conditioning),
        action_attention=_plain_dict(action_attention),
    )
    model.use_cuda_graph = bool(use_cuda_graph)
    return model


def create_lingbot_flexwam(
    model_path: str,
    video_dit_config,
    action_dit_config,
    load_text_encoder: bool = False,
    context_input_dim: int | None = None,
    proprio_dim: int | None = None,
    action_dit_pretrained_path: str | None = None,
    skip_dit_load_from_pretrain: bool = False,
    video_scheduler=None,
    action_scheduler=None,
    loss=None,
    action_router=None,
    action_critic=None,
    router_cf=None,
    state_conditioning=None,
    action_attention=None,
    mot_checkpoint_mixed_attn: bool = True,
    model_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    use_cuda_graph: bool = False,
):
    from flexwam.models.architectures.flexwam import FlexWAM

    video_dit_config = _plain_dict(video_dit_config)
    action_dit_config = _plain_dict(action_dit_config)
    video_scheduler = _plain_dict(video_scheduler) or {}
    action_scheduler = _plain_dict(action_scheduler)
    loss = _plain_dict(loss) or {}
    action_router = _plain_dict(action_router)
    action_critic = _plain_dict(action_critic)
    router_cf = _plain_dict(router_cf)

    model = FlexWAM.from_lingbot_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_path=model_path,
        load_text_encoder=bool(load_text_encoder),
        proprio_dim=proprio_dim,
        video_dit_config=video_dit_config,
        action_dit_config=action_dit_config,
        action_dit_pretrained_path=action_dit_pretrained_path,
        skip_dit_load_from_pretrain=bool(skip_dit_load_from_pretrain),
        mot_checkpoint_mixed_attn=bool(mot_checkpoint_mixed_attn),
        video_train_shift=float(video_scheduler.get("train_shift", 1.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 1.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler["train_shift"]),
        action_infer_shift=float(action_scheduler["infer_shift"]),
        action_num_train_timesteps=int(action_scheduler["num_train_timesteps"]),
        loss_lambda_video=float(loss.get("lambda_video", 1.0)),
        loss_lambda_action=float(loss.get("lambda_action", 1.0)),
        action_router_config=action_router,
        action_critic_config=action_critic,
        context_input_dim=context_input_dim,
        router_cf_config=router_cf,
        state_conditioning=_plain_dict(state_conditioning),
        action_attention=_plain_dict(action_attention),
    )
    model.use_cuda_graph = bool(use_cuda_graph)
    return model

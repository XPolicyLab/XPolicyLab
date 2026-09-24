import math

import torch
import torch.nn.functional as F  # noqa: N812
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from giga_models.utils.action_horizon import (
    downsample_flow_action_tensors,
    flow_action_horizon_indices,
    resolve_flow_action_steps,
)

from .paligemma2_with_expert import PaliGemma2WithExpertModel
from .paligemma_with_expert import PaliGemmaWithExpertModel


class _GradientWeight(torch.autograd.Function):
    """Identity in forward, scalar gradient weight in backward."""

    @staticmethod
    def forward(ctx, tensor: Tensor, weight: float) -> Tensor:
        ctx.weight = weight
        return tensor

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:
        return grad_output * ctx.weight, None


def _apply_gradient_weight(tensor: Tensor, weight: float) -> Tensor:
    if weight == 0.0:
        return tensor.detach()
    if weight == 1.0:
        return tensor
    return _GradientWeight.apply(tensor, weight)


def _weight_kv_cache_gradients(past_key_values: dict, weight: float) -> None:
    """Weight gradients crossing from an action suffix into the VLM prefix."""
    for layer_cache in past_key_values.values():
        layer_cache['key_states'] = _apply_gradient_weight(layer_cache['key_states'], weight)
        layer_cache['value_states'] = _apply_gradient_weight(layer_cache['value_states'], weight)


def _compact_language_logits(
    model: nn.Module,
    hidden_states: Tensor,
    lang_loss_masks: Tensor | None,
) -> dict[str, Tensor]:
    if lang_loss_masks is None:
        return {'lang_logits': model.language_out_proj(hidden_states)}

    if hidden_states.shape[:2] != lang_loss_masks.shape:
        raise ValueError(
            f'hidden_states prefix shape {tuple(hidden_states.shape[:2])} does not match '
            f'lang_loss_masks shape {tuple(lang_loss_masks.shape)}'
        )

    loss_positions = lang_loss_masks[:, 1:].to(device=hidden_states.device, dtype=torch.bool)
    hidden_for_loss = hidden_states[:, :-1, :][loss_positions]
    return {
        'lang_logits': model.language_out_proj(hidden_for_loss),
        'lang_logits_mask': loss_positions,
    }


def _chunked_language_token_loss(
    model: nn.Module,
    hidden_states: Tensor,
    lang_tokens: Tensor,
    lang_loss_masks: Tensor,
    chunk_size: int,
    use_checkpoint: bool,
) -> dict[str, Tensor]:
    if hidden_states.shape[:2] != lang_loss_masks.shape:
        raise ValueError(
            f'hidden_states prefix shape {tuple(hidden_states.shape[:2])} does not match '
            f'lang_loss_masks shape {tuple(lang_loss_masks.shape)}'
        )
    if hidden_states.shape[:2] != lang_tokens.shape:
        raise ValueError(
            f'hidden_states prefix shape {tuple(hidden_states.shape[:2])} does not match '
            f'lang_tokens shape {tuple(lang_tokens.shape)}'
        )

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f'chunk_size must be positive, got {chunk_size}')

    loss_positions = lang_loss_masks[:, 1:].to(device=hidden_states.device, dtype=torch.bool)
    hidden_for_loss = hidden_states[:, :-1, :][loss_positions]
    targets = lang_tokens[:, 1:].to(device=hidden_states.device)[loss_positions].long()

    if hidden_for_loss.shape[0] == 0:
        token_loss = hidden_states[:, :-1, 0].float() * 0.0
        return {'lang_token_loss': token_loss}

    def _ce_for_hidden(hidden_chunk: Tensor, target_chunk: Tensor) -> Tensor:
        logits = model.language_out_proj(hidden_chunk)
        return F.cross_entropy(logits, target_chunk, reduction='none')

    losses = []
    for start in range(0, hidden_for_loss.shape[0], chunk_size):
        end = min(start + chunk_size, hidden_for_loss.shape[0])
        hidden_chunk = hidden_for_loss[start:end]
        target_chunk = targets[start:end]
        if use_checkpoint and hidden_chunk.requires_grad:
            losses.append(checkpoint(_ce_for_hidden, hidden_chunk, target_chunk, use_reentrant=False))
        else:
            losses.append(_ce_for_hidden(hidden_chunk, target_chunk))

    compact_loss = torch.cat(losses, dim=0)
    token_loss = compact_loss.new_zeros(loss_positions.shape)
    token_loss[loss_positions] = compact_loss
    return {'lang_token_loss': token_loss}


def _mask_action_suffix_target_attention(
    full_att_2d: Tensor,
    suffix_len: int,
    fast_action_indicator: Tensor | None,
    subtask_indicator: Tensor | None,
) -> Tensor:
    """Keep flow-action suffix independent from autoregressive target tokens."""
    if suffix_len <= 0:
        return full_att_2d

    batch_size = full_att_2d.shape[0]
    start = full_att_2d.shape[1] - suffix_len
    full_fast_indicator = None

    if fast_action_indicator is not None:
        suffix_fast_indicator = fast_action_indicator.new_zeros(batch_size, suffix_len)
        full_fast_indicator = torch.cat([fast_action_indicator, suffix_fast_indicator], dim=1)
        # Continuous flow action tokens must not condition on FAST action targets.
        full_att_2d[:, start:, :] &= ~full_fast_indicator[:, None, :]

    if subtask_indicator is not None:
        suffix_subtask_indicator = subtask_indicator.new_zeros(batch_size, suffix_len)
        full_subtask_indicator = torch.cat([subtask_indicator, suffix_subtask_indicator], dim=1)
        # Continuous flow action tokens must not condition on predicted subtask targets.
        full_att_2d[:, start:, :] &= ~full_subtask_indicator[:, None, :]

        if full_fast_indicator is not None:
            # FAST action target tokens also stay independent from predicted subtask targets.
            full_att_2d &= ~(full_fast_indicator[:, :, None] & full_subtask_indicator[:, None, :])

    return full_att_2d


class GigaBrain0Policy(ModelMixin, ConfigMixin):
    @register_to_config
    def __init__(
        self,
        max_state_dim: int = 32,
        max_action_dim: int = 32,
        proj_width: int = 1024,
        vlm_type: str = 'paligemma2',
        vlm_hidden_size: int = 2304,
        n_action_steps: int = 50,
        flow_action_horizon: int | None = None,
        num_steps: int = 10,
        use_cache: bool = True,
        observation_memory_size: int = 1,
        state_input_mode: str = 'prompt',
        agent_pos_config: dict | None = None,
        state_hidden_size: int = 2048,
        proj_with_mask: bool = True,
        propri_token_id: int | None = None,
        vision_in_channels: int = 3,
        temporal_attention_interval: int = 4,
        detach_history_frames: bool = True,
        enable_knowledge_insulation: bool = False,
        enable_next_token_prediction: bool = True,
        compact_language_logits: bool = True,
        language_loss_chunk_size: int = 0,
        language_loss_checkpoint: bool = True,
        enable_learnable_traj_token: bool = False,
        num_traj_tokens: int = 10,
        max_traj_dim: int = 4,
        traj_hidden_dim: int = 256,
        num_embodiments: int = 1,
        expert_intermediate_size: int = None,
        has_action_expert: bool = False,
        # Action-expert variant. 'dual_stream' = state-bridged dual-stream expert
        # (qwen3_5_with_expert.py). 'cross_dit' = decoupled layer-wise cross-attention
        # DiT expert (qwen3_5_crossdit_expert.py); currently qwen3_5-only. For cross_dit,
        # proj_width is reused as the DiT hidden size and the DiT depth equals the VLM's
        # num_hidden_layers (block i cross-attends to VLM layer i).
        expert_type: str = 'dual_stream',
        crossdit_attention_head_dim: int = 64,
        crossdit_num_target_vision_tokens: int = 32,
        # 'gemma3' expert: warm-started decoupled Gemma3-1B cross-attention expert
        # (gemma3_action_expert.py); qwen3_5-only for now. proj_width MUST equal the
        # Gemma3-1B hidden (1152). gate_self_mlp: True=DiT-gated (identity at init),
        # False=Flamingo-active (warm self-attn/mlp live from step 0).
        gemma3_gate_self_mlp: bool = True,
        # 'robomanip' expert (Qwen-RobotManip-style DiT): N blocks, modality-alternating
        # cross-attn to the VLM's last layer, self-attn per block, AdaZeroRMSNorm.
        # proj_width is reused as the DiT hidden Dact (paper: 768).
        robomanip_num_layers: int = 10,
        robomanip_attention_head_dim: int = 64,
        robomanip_ff_dim: int = None,
        # Learnable query/register tokens (Darcet et al. 2024) condensing the VLM V/L tokens
        # for the action expert. 0 disables them. The paper's adopted variant-3 uses them with
        # joint (non-alternating) cross-attention to the VLM last-layer hidden.
        robomanip_num_query_tokens: int = 0,
        robomanip_cross_attn_modality_alternating: bool = True,
        qwen3_5_config_dict: dict | None = None,
        qwen3_vl_config_dict: dict | None = None,
        qwen2_5_vl_config_dict: dict | None = None,
        gemma3_config_dict: dict | None = None,
        gemma4_config_dict: dict | None = None,
        gemma3_expert_full_attends_prefix: bool = False,
        gemma4_expert_full_attends_prefix: bool = False,
        # Gemma4-only: at the KV-shared layers (24-41), have the Expert cross-attend
        # to a fresh per-layer projection of the VLM hidden state instead of the
        # frozen cross-layer-shared VLM K/V. Default False = baseline dual-stream.
        # See Gemma4ExpertConfig.expert_cross_attend_shared_layers.
        gemma4_expert_cross_attend_shared_layers: bool = False,
        fast_token_vocab_mode: str | None = None,
        fast_token_tail_skip_tokens: int = 128,
        fast_token_tail_vocab_size: int | None = None,
        fast_vocab_size: int = 2048,
        # TEMP(ada-norm-compat): only exists to load pre-f0e9f6f gemma4 ckpts (trained with a
        # plain RMSNorm final norm, before AdaRMSNorm). Remove this flag + all non-Ada paths
        # once every gemma4 ckpt uses AdaRMSNorm.
        gemma4_expert_final_norm_ada: bool = True,
        # F8: opt-in timestep-conditioned AdaRMSNorm for the gemma3 expert final norm.
        # Default False — every pre-F8 gemma3 flow ckpt has a plain RMSNorm final norm
        # (single .weight); @register_to_config fills this default when a config.json
        # predates the flag, so old ckpts keep loading. Set True in new gemma3 flow configs.
        gemma3_expert_final_norm_ada: bool = False,
        # Qwen3-VL MoT (Cosmos3-style dual tower): give the expert AdaRMSNorms a loadable
        # base weight so VLM layernorm weights can be copied in at warm-init. Default
        # False keeps the pre-MoT expert state_dict, so every existing qwen3_vl flow
        # ckpt (whose config.json predates the flag) keeps loading unchanged.
        qwen3_vl_expert_adarms_base: bool = False,
        # PaliGemma2 only: gradient weight at the action-expert -> VLM prefix
        # boundary. None preserves the legacy boolean behavior (0.0 with KI,
        # 1.0 without KI). An explicit value takes precedence over the boolean.
        action_loss_vlm_gradient_weight: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.has_action_expert = has_action_expert
        self.expert_type = expert_type
        self.crossdit_attention_head_dim = crossdit_attention_head_dim
        self.crossdit_num_target_vision_tokens = crossdit_num_target_vision_tokens
        self.gemma3_gate_self_mlp = gemma3_gate_self_mlp
        self.robomanip_num_layers = robomanip_num_layers
        self.robomanip_attention_head_dim = robomanip_attention_head_dim
        self.robomanip_ff_dim = robomanip_ff_dim
        self.robomanip_num_query_tokens = robomanip_num_query_tokens
        self.robomanip_cross_attn_modality_alternating = robomanip_cross_attn_modality_alternating
        if self.expert_type not in ('dual_stream', 'cross_dit', 'robomanip', 'gemma3'):
            raise ValueError(f"expert_type must be 'dual_stream', 'cross_dit', 'robomanip' or 'gemma3', got {self.expert_type!r}")
        if self.expert_type in ('cross_dit', 'robomanip', 'gemma3') and vlm_type != 'qwen3_5':
            raise ValueError(f"expert_type={self.expert_type!r} is currently only supported for vlm_type='qwen3_5', got {vlm_type!r}")

        # Store the parameters
        self.max_state_dim = max_state_dim
        if int(max_action_dim) != 32:
            raise ValueError(f'GigaBrain action heads require max_action_dim=32, got {max_action_dim}')
        self.max_action_dim = 32
        self._action_denoise_mask: Tensor | None = None
        self.proj_width = proj_width
        self.raw_action_steps, self.n_action_steps, self.flow_action_horizon = resolve_flow_action_steps(
            n_action_steps,
            flow_action_horizon,
        )
        self.num_steps = num_steps
        self.use_cache = use_cache
        if observation_memory_size < 1:
            raise ValueError(f'observation_memory_size must be positive, got {observation_memory_size}')
        self.observation_memory_size = observation_memory_size
        if state_input_mode not in ('prompt', 'proprio_memory', 'proprio_anchor'):
            raise ValueError(
                "state_input_mode must be 'prompt', 'proprio_memory', or "
                f"'proprio_anchor', got {state_input_mode!r}"
            )
        self.state_input_mode = state_input_mode
        self.agent_pos_config = dict(agent_pos_config or {})
        self.propri_dim = sum(int(dim) for dim in self.agent_pos_config.values())
        self.state_hidden_size = int(state_hidden_size)
        self.proj_with_mask = bool(proj_with_mask)
        self.propri_token_id = None if propri_token_id is None else int(propri_token_id)
        if self.state_input_mode == 'proprio_anchor':
            if vlm_type != 'paligemma2':
                raise ValueError(
                    "state_input_mode='proprio_anchor' is currently supported only for "
                    f"vlm_type='paligemma2', got {vlm_type!r}"
                )
            # Observation memory controls visual history; the anchor remains one token.
            if self.propri_dim <= 0:
                raise ValueError(
                    "state_input_mode='proprio_anchor' requires a non-empty agent_pos_config"
                )
            if self.state_hidden_size <= 0 or self.state_hidden_size > int(vlm_hidden_size):
                raise ValueError(
                    'state_hidden_size must be positive and no larger than vlm_hidden_size, '
                    f'got state_hidden_size={self.state_hidden_size}, vlm_hidden_size={vlm_hidden_size}'
                )
            if self.propri_token_id is None:
                raise ValueError(
                    "state_input_mode='proprio_anchor' requires propri_token_id from the tokenizer"
                )
        self.vision_in_channels = vision_in_channels
        self.temporal_attention_interval = temporal_attention_interval

        self.enable_knowledge_insulation = enable_knowledge_insulation
        if action_loss_vlm_gradient_weight is not None and vlm_type != 'paligemma2':
            raise ValueError(
                "action_loss_vlm_gradient_weight is currently supported only for "
                f"vlm_type='paligemma2', got {vlm_type!r}"
            )
        if action_loss_vlm_gradient_weight is None:
            action_loss_vlm_gradient_weight = 0.0 if enable_knowledge_insulation else 1.0
        action_loss_vlm_gradient_weight = float(action_loss_vlm_gradient_weight)
        if not math.isfinite(action_loss_vlm_gradient_weight) or action_loss_vlm_gradient_weight < 0.0:
            raise ValueError(
                'action_loss_vlm_gradient_weight must be finite and greater than or equal to 0, '
                f'got {action_loss_vlm_gradient_weight}'
            )
        self.action_loss_vlm_gradient_weight = action_loss_vlm_gradient_weight
        # @register_to_config already stored the caller-provided value. Keep None in
        # the config so a later KI override can resolve a new legacy default.
        self.vlm_type = vlm_type
        self.vlm_hidden_size = vlm_hidden_size
        tail_default_vlm_types = ('paligemma2', 'gemma4')
        tail_supported_vlm_types = ('paligemma', 'paligemma2', 'gemma4')
        if fast_token_vocab_mode is None:
            fast_token_vocab_mode = 'tail' if self.vlm_type in tail_default_vlm_types else 'expanded'
        self.fast_token_vocab_mode = fast_token_vocab_mode
        if self.fast_token_vocab_mode not in ('expanded', 'tail'):
            raise ValueError(
                f"fast_token_vocab_mode must be 'expanded' or 'tail', got {self.fast_token_vocab_mode!r}"
            )
        if self.fast_token_vocab_mode == 'tail' and self.vlm_type not in tail_supported_vlm_types:
            raise ValueError(
                "fast_token_vocab_mode='tail' is currently supported only for "
                f"vlm_type in {tail_supported_vlm_types}, got {self.vlm_type!r}"
            )
        self.fast_token_tail_skip_tokens = int(fast_token_tail_skip_tokens)
        self.fast_token_tail_vocab_size = (
            None if fast_token_tail_vocab_size is None else int(fast_token_tail_vocab_size)
        )
        self.fast_vocab_size = int(fast_vocab_size)
        self.register_to_config(fast_token_vocab_mode=self.fast_token_vocab_mode)
        self.gemma4_fast_token_min_id = None
        self.gemma4_fast_token_max_id = None

        if self.vlm_type == 'paligemma2':
            self.paligemma_with_expert = PaliGemma2WithExpertModel(
                vision_in_channels=vision_in_channels,
                enable_next_token_prediction=enable_next_token_prediction,
                temporal_attention_interval=temporal_attention_interval,
                detach_history_frames=detach_history_frames,
            )
        elif self.vlm_type == 'paligemma':
            self.paligemma_with_expert = PaliGemmaWithExpertModel(
                vision_in_channels=vision_in_channels,
                enable_next_token_prediction=enable_next_token_prediction,
                pi05_enabled=True,
            )
        elif self.vlm_type == 'qwen3_5':
            from .qwen3_5_config import Qwen3_5Config

            # Priority: explicit Qwen3_5Config object (kwargs) > serialized dict (in config.json) > defaults.
            # The kwargs path is used by the trainer at construction time (it has the live Qwen3_5Config
            # object in memory). The dict path is what `from_pretrained` will hit, since the dict is what
            # gets serialized into config.json by ConfigMixin.
            qwen_config_obj = kwargs.pop('qwen3_5_config', None)
            if qwen_config_obj is not None:
                qwen_config = qwen_config_obj
            elif qwen3_5_config_dict is not None:
                qwen_config = Qwen3_5Config(**qwen3_5_config_dict)
            else:
                qwen_config = Qwen3_5Config()

            if self.has_action_expert and self.expert_type == 'cross_dit':
                # Decoupled layer-wise cross-attention DiT expert: a pure VLM (exposing
                # per-layer hidden states) + an external DiT that cross-attends to each
                # VLM layer. This sidesteps the hybrid-attention state bridging entirely.
                from .qwen3_5_crossdit_expert import Qwen3_5CrossDiTExpert
                from .qwen3_5_vlm import Qwen3_5VLMModel

                self.qwen3_5_vlm = Qwen3_5VLMModel(
                    config=qwen_config,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
                self.qwen3_5_crossdit_expert = Qwen3_5CrossDiTExpert(
                    vlm_hidden_size=qwen_config.text_config.hidden_size,
                    dit_hidden=proj_width,
                    num_dit_layers=qwen_config.text_config.num_hidden_layers,
                    n_action_steps=self.n_action_steps,
                    attention_head_dim=crossdit_attention_head_dim,
                    num_target_vision_tokens=crossdit_num_target_vision_tokens,
                )
            elif self.has_action_expert and self.expert_type == 'robomanip':
                # Qwen-RobotManip-style DiT: pure VLM (last-layer hidden) + N-block DiT
                # with self-attn + modality-alternating cross-attn (visual/language).
                from .qwen3_5_robomanip_expert import Qwen3_5RoboManipExpert
                from .qwen3_5_vlm import Qwen3_5VLMModel

                self.qwen3_5_vlm = Qwen3_5VLMModel(
                    config=qwen_config,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
                self.qwen3_5_robomanip_expert = Qwen3_5RoboManipExpert(
                    vlm_hidden_size=qwen_config.text_config.hidden_size,
                    dim=proj_width,
                    num_layers=robomanip_num_layers,
                    num_heads=proj_width // robomanip_attention_head_dim,
                    head_dim=robomanip_attention_head_dim,
                    ff_dim=robomanip_ff_dim if robomanip_ff_dim is not None else 2048,
                    n_action_steps=self.n_action_steps,
                    # Paper-faithful continuous proprioceptive state: 2-layer MLP prepended
                    # to the action tokens. The robomanip config must set
                    # discrete_state_input=False (state arrives via forward(state=...)).
                    state_dim=self.max_state_dim,
                    num_query_tokens=robomanip_num_query_tokens,
                    cross_attn_modality_alternating=robomanip_cross_attn_modality_alternating,
                )
            elif self.has_action_expert and self.expert_type == 'gemma3':
                # Warm-started decoupled Gemma3-1B cross-attention expert: pure VLM
                # (per-layer hidden states) + an independent Gemma3-1B stack that
                # cross-attends to those hiddens. Warm-start is applied by the trainer
                # (load_gemma3_pretrained_into_expert), NOT here, so rollout
                # reconstruction loads the trained expert without touching gemma-3-1b-pt.
                from .gemma3_action_expert import Gemma3ActionExpert, Gemma3ActionExpertConfig
                from .qwen3_5_vlm import Qwen3_5VLMModel

                self.qwen3_5_vlm = Qwen3_5VLMModel(
                    config=qwen_config,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
                g3_cfg = Gemma3ActionExpertConfig(
                    vlm_hidden_size=qwen_config.text_config.hidden_size,
                    gate_self_mlp=gemma3_gate_self_mlp,
                )
                if proj_width != g3_cfg.hidden_size:
                    raise ValueError(
                        f"expert_type='gemma3' requires proj_width == {g3_cfg.hidden_size} "
                        f"(Gemma3-1B hidden), got proj_width={proj_width}"
                    )
                self.gemma3_expert = Gemma3ActionExpert(g3_cfg)
                self._gemma3_num_vlm_layers = qwen_config.text_config.num_hidden_layers
            elif self.has_action_expert:
                # Dual-Stream with Expert
                from .qwen3_5_config import Qwen3_5ExpertConfig
                from .qwen3_5_with_expert import Qwen3_5WithExpertModel

                expert_config = Qwen3_5ExpertConfig.from_vlm_config(
                    qwen_config.text_config, expert_hidden_size=proj_width,
                    expert_intermediate_size=expert_intermediate_size)
                self.qwen3_5_with_expert = Qwen3_5WithExpertModel(
                    vlm_config=qwen_config, expert_config=expert_config)
            else:
                # Pure VLM (no action expert)
                from .qwen3_5_vlm import Qwen3_5VLMModel

                self.qwen3_5_vlm = Qwen3_5VLMModel(
                    config=qwen_config,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
        elif self.vlm_type == 'qwen3_vl':
            from .qwen3_vl_config import Qwen3VLConfig

            # Priority: explicit Qwen3VLConfig object (kwargs) > serialized dict (in config.json) > defaults.
            # The kwargs path is used by the trainer at construction time (it has the live Qwen3VLConfig
            # object in memory). The dict path is what `from_pretrained` will hit, since the dict is what
            # gets serialized into config.json by ConfigMixin.
            qwen_vl_config_obj = kwargs.pop('qwen3_vl_config', None)
            if qwen_vl_config_obj is not None:
                qwen_config = qwen_vl_config_obj
            elif qwen3_vl_config_dict is not None:
                qwen_config = Qwen3VLConfig(**qwen3_vl_config_dict)
            else:
                qwen_config = Qwen3VLConfig()

            if self.has_action_expert:
                from .qwen3_vl_config import Qwen3VLExpertConfig
                from .qwen3_vl_with_expert import Qwen3VLWithExpertModel

                expert_config = Qwen3VLExpertConfig.from_vlm_config(
                    qwen_config.text_config,
                    expert_hidden_size=proj_width,
                    expert_intermediate_size=expert_intermediate_size,
                    adarms_base_weight=qwen3_vl_expert_adarms_base,
                )
                self.qwen3_vl_with_expert = Qwen3VLWithExpertModel(
                    vlm_config=qwen_config, expert_config=expert_config)
            else:
                from .qwen3_vl_vlm import Qwen3VLVLMModel

                self.qwen3_vl_vlm = Qwen3VLVLMModel(
                    config=qwen_config,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
        elif self.vlm_type == 'qwen2_5_vl':
            from .qwen2_5_vl_config import Qwen2_5VLConfig

            # Priority: explicit Qwen2_5VLConfig (kwargs) > serialized dict (config.json) > defaults.
            qwen_vl_config_obj = kwargs.pop('qwen2_5_vl_config', None)
            if qwen_vl_config_obj is not None:
                qwen_config = qwen_vl_config_obj
            elif qwen2_5_vl_config_dict is not None:
                qwen_config = Qwen2_5VLConfig(**qwen2_5_vl_config_dict)
            else:
                qwen_config = Qwen2_5VLConfig()

            if self.has_action_expert:
                from .qwen2_5_vl_config import Qwen2_5VLExpertConfig
                from .qwen2_5_vl_with_expert import Qwen2_5VLWithExpertModel

                expert_config = Qwen2_5VLExpertConfig.from_vlm_config(
                    qwen_config.text_config, expert_hidden_size=proj_width)
                self.qwen2_5_vl_with_expert = Qwen2_5VLWithExpertModel(
                    vlm_config=qwen_config, expert_config=expert_config)
            else:
                from .qwen2_5_vl_vlm import Qwen2_5VLVLMModel

                self.qwen2_5_vl_vlm = Qwen2_5VLVLMModel(
                    config=qwen_config,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
        elif self.vlm_type == 'gemma3':
            from .gemma3_config import Gemma3Config

            # Priority: explicit Gemma3Config object (kwargs) > serialized dict (config.json) > defaults.
            gemma3_config_obj = kwargs.pop('gemma3_config', None)
            if gemma3_config_obj is not None:
                gemma3_config = gemma3_config_obj
            elif gemma3_config_dict is not None:
                gemma3_config = Gemma3Config(**gemma3_config_dict)
            else:
                gemma3_config = Gemma3Config()

            if self.has_action_expert:
                # Phase 3 dual-stream + flow matching.
                from .gemma3_config import Gemma3ExpertConfig
                from .gemma3_with_expert import Gemma3WithExpertModel

                expert_config = Gemma3ExpertConfig.from_vlm_config(
                    gemma3_config.text_config, expert_hidden_size=proj_width,
                    expert_full_attends_prefix=gemma3_expert_full_attends_prefix,
                )
                self.gemma3_with_expert = Gemma3WithExpertModel(
                    vlm_config=gemma3_config, expert_config=expert_config,
                    expert_final_norm_ada=gemma3_expert_final_norm_ada,
                )
            else:
                from .gemma3_vlm import Gemma3VLMModel

                self.gemma3_vlm = Gemma3VLMModel(
                    config=gemma3_config,
                    vision_in_channels=vision_in_channels,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
        elif self.vlm_type == 'gemma4':
            from .gemma4_config import Gemma4Config

            # Priority: explicit Gemma4Config object (kwargs) > serialized dict (config.json) > defaults.
            gemma4_config_obj = kwargs.pop('gemma4_config', None)
            if gemma4_config_obj is not None:
                gemma4_config = gemma4_config_obj
            elif gemma4_config_dict is not None:
                gemma4_config = Gemma4Config(**gemma4_config_dict)
            else:
                gemma4_config = Gemma4Config()
            # VLA never uses the audio tower.
            gemma4_config.audio_config = None
            # gemma-4-E4B-it is causal-pretrained (config use_bidirectional_attention=null).
            # Leave it as-is so the PURE-VLM path stays faithful to HF (causal image
            # attention -> correct spatial grounding; benchmarked: bidirectional costs
            # grounding/counting/spatial accuracy). Opt into image-bidirectional by setting
            # use_bidirectional_attention='vision' explicitly in the gemma4 config. The VLA
            # dual-stream path is unaffected — its image attention is controlled by the
            # transform-side `image_attn` flag, not this config field.
            if self.fast_token_vocab_mode == 'expanded':
                model_vocab_size = int(gemma4_config.text_config.vocab_size)
                base_vocab_size = int(getattr(gemma4_config.text_config, 'vocab_size_per_layer_input', 0) or 262144)
                if model_vocab_size == base_vocab_size:
                    gemma4_config.text_config.vocab_size = model_vocab_size + self.fast_vocab_size
            if self.fast_token_vocab_mode == 'tail':
                tail_vocab_size = self.fast_token_tail_vocab_size
                model_vocab_size = int(gemma4_config.text_config.vocab_size)
                if tail_vocab_size is None:
                    tail_vocab_size = model_vocab_size
                if tail_vocab_size > model_vocab_size:
                    raise ValueError(
                        f'fast_token_tail_vocab_size={tail_vocab_size} exceeds Gemma4 model vocab_size={model_vocab_size}'
                    )
                base_token_id = (
                    tail_vocab_size
                    - 1
                    - self.fast_token_tail_skip_tokens
                )
                min_token_id = base_token_id - self.fast_vocab_size + 1
                if min_token_id < 0:
                    raise ValueError(
                        f'Cannot place {self.fast_vocab_size} FAST tokens in Gemma4 vocab tail: '
                        f'vocab_size={tail_vocab_size}, '
                        f'fast_token_tail_skip_tokens={self.fast_token_tail_skip_tokens}'
                    )
                self.gemma4_fast_token_min_id = min_token_id
                self.gemma4_fast_token_max_id = base_token_id

            if self.has_action_expert:
                # Phase 3 dual-stream + flow matching.
                from .gemma4_config import Gemma4ExpertConfig
                from .gemma4_with_expert import Gemma4WithExpertModel

                expert_config = Gemma4ExpertConfig.from_vlm_config(
                    gemma4_config.text_config, expert_hidden_size=proj_width,
                    expert_full_attends_prefix=gemma4_expert_full_attends_prefix,
                    expert_cross_attend_shared_layers=gemma4_expert_cross_attend_shared_layers,
                )
                self.gemma4_with_expert = Gemma4WithExpertModel(
                    vlm_config=gemma4_config, expert_config=expert_config,
                    observation_memory_size=self.observation_memory_size,
                    detach_history_frames=detach_history_frames,
                    expert_final_norm_ada=gemma4_expert_final_norm_ada,  # TEMP(ada-norm-compat)
                )
            else:
                from .gemma4_vlm import Gemma4VLMModel

                self.gemma4_vlm = Gemma4VLMModel(
                    config=gemma4_config,
                    vision_in_channels=vision_in_channels,
                    enable_next_token_prediction=enable_next_token_prediction,
                )
        else:
            raise NotImplementedError(
                f"Unknown vlm_type: {self.vlm_type!r}. "
                "Supported: 'paligemma', 'paligemma2', 'qwen3_5', 'qwen3_vl', 'qwen2_5_vl', 'gemma3', 'gemma4'."
            )

        # Projections are float32
        if self.state_input_mode == 'proprio_memory':
            self.proprio_state_proj = EmbodimentSpecificLinear(
                self.max_state_dim,
                self.vlm_hidden_size,
                num_categories=num_embodiments,
                dtype=torch.float32,
            )
        else:
            self.proprio_state_proj = None
        if self.state_input_mode == 'proprio_anchor':
            propri_input_dim = self.propri_dim * (2 if self.proj_with_mask else 1)
            self.propri_proj = nn.Linear(
                propri_input_dim,
                self.state_hidden_size,
                bias=False,
                dtype=torch.float32,
            )
            nn.init.normal_(self.propri_proj.weight, mean=0.0, std=0.02)
            num_embeddings = self.paligemma_with_expert.embed_tokens.num_embeddings
            if not 0 <= self.propri_token_id < num_embeddings:
                raise ValueError(
                    f'propri_token_id={self.propri_token_id} is outside the PaliGemma2 '
                    f'embedding table with {num_embeddings} rows'
                )
        else:
            self.propri_proj = None
        self.action_in_proj = EmbodimentSpecificLinear(self.max_action_dim, self.proj_width, num_categories=num_embodiments, dtype=torch.float32)
        self.action_out_proj = EmbodimentSpecificLinear(self.proj_width, self.max_action_dim, num_categories=num_embodiments, dtype=torch.float32)
        self.time_mlp_in = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)
        self.time_mlp_out = nn.Linear(self.proj_width, self.proj_width, dtype=torch.float32)

        self.enable_next_token_prediction = enable_next_token_prediction
        self.compact_language_logits = compact_language_logits
        self.language_loss_chunk_size = int(language_loss_chunk_size)
        self.language_loss_checkpoint = bool(language_loss_checkpoint)
        self.enable_learnable_traj_token = enable_learnable_traj_token
        if self.enable_learnable_traj_token:
            self.num_traj_tokens = num_traj_tokens
            self.max_traj_dim = max_traj_dim
            self.traj_hidden_dim = traj_hidden_dim
            self.traj_token = nn.Parameter(torch.randn(num_traj_tokens, self.vlm_hidden_size), requires_grad=True)
            self.traj_decoder = nn.GRU(input_size=self.vlm_hidden_size, hidden_size=self.traj_hidden_dim, batch_first=True)
            self.traj_out_proj = nn.Linear(self.traj_hidden_dim, self.max_traj_dim, dtype=torch.float32)

    def _gemma4_tail_fast_token_mask(self, token_ids: Tensor) -> Tensor:
        if self.vlm_type != 'gemma4' or self.fast_token_vocab_mode != 'tail':
            return torch.zeros_like(token_ids, dtype=torch.bool)
        if self.gemma4_fast_token_min_id is None or self.gemma4_fast_token_max_id is None:
            return torch.zeros_like(token_ids, dtype=torch.bool)
        return (token_ids >= self.gemma4_fast_token_min_id) & (token_ids <= self.gemma4_fast_token_max_id)

    def _gemma4_prefix_ple_remap_mask(
        self,
        token_ids: Tensor,
        image_token_id: int,
        fast_action_indicator: Tensor | None = None,
    ) -> Tensor:
        ple_remap_mask = token_ids == image_token_id
        if fast_action_indicator is None:
            return ple_remap_mask

        fast_mask = fast_action_indicator.to(device=token_ids.device, dtype=torch.bool)
        if self.fast_token_vocab_mode == 'tail':
            fast_mask = fast_mask & self._gemma4_tail_fast_token_mask(token_ids)
        return ple_remap_mask | fast_mask

    def sync_registered_vlm_config(self) -> None:
        """Refresh the serialized ``<vlm_type>_config_dict`` from the live VLM config.

        FAST-token vocab expansion (``resize_token_embeddings``) grows the live
        ``vlm_config``'s embed_tokens / lm_head to base+2048, but NOT the
        ``@register_to_config`` snapshot captured at ``__init__`` — so without this
        the saved ``config.json`` keeps the pre-resize ``vocab_size`` and a later
        ``from_pretrained`` rebuilds those layers at the stale size and dies on load.
        Re-deriving the dict from the live config restores the invariant "saved
        config == live model". Idempotent (a no-op when nothing drifted); skips
        backbones (paligemma/paligemma2) that register no ``<vlm_type>_config_dict``.
        """
        key = f'{self.vlm_type}_config_dict'
        if key not in self.config:
            return
        sub = getattr(self, f'{self.vlm_type}_with_expert', None) or getattr(self, f'{self.vlm_type}_vlm', None)
        if sub is None:
            return
        live = getattr(sub, 'vlm_config', None) or getattr(sub, 'config', None)
        if live is None or not hasattr(live, 'to_dict'):
            return
        self.register_to_config(**{key: live.to_dict()})

    def _embed_proprio_state_tokens(
        self,
        state_memory: Tensor | None,
        state_memory_masks: Tensor | None,
        emb_ids: Tensor | None,
        dtype: torch.dtype,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
        if self.state_input_mode != 'proprio_memory':
            return None, None, None
        if self.proprio_state_proj is None:
            raise RuntimeError('proprio_state_proj is not initialized for proprio_memory mode')
        if state_memory is None:
            raise ValueError("state_input_mode='proprio_memory' requires state_memory")
        if emb_ids is None:
            raise ValueError("state_input_mode='proprio_memory' requires emb_ids for embodiment-specific projection")

        if state_memory.ndim == 2:
            state_memory = state_memory[:, None, :]
        if state_memory.ndim != 3:
            raise ValueError(f'state_memory must have shape [B,K,D], got {tuple(state_memory.shape)}')
        batch_size, memory_size, state_dim = state_memory.shape
        if memory_size != self.observation_memory_size:
            raise ValueError(
                f'state_memory has K={memory_size}, but observation_memory_size={self.observation_memory_size}'
            )

        if state_dim < self.max_state_dim:
            pad_width = self.max_state_dim - state_dim
            state_memory = F.pad(state_memory, (0, pad_width), value=0.0)
        elif state_dim > self.max_state_dim:
            state_memory = state_memory[..., : self.max_state_dim]

        if state_memory_masks is None:
            state_memory_masks = torch.ones(
                batch_size,
                memory_size,
                dtype=torch.bool,
                device=state_memory.device,
            )
        else:
            state_memory_masks = state_memory_masks.to(device=state_memory.device, dtype=torch.bool)
            if state_memory_masks.shape != (batch_size, memory_size):
                raise ValueError(
                    f'state_memory_masks shape {tuple(state_memory_masks.shape)} does not match '
                    f'state_memory shape {(batch_size, memory_size)}'
                )

        state_memory = state_memory.to(dtype=self.proprio_state_proj.weight.dtype)
        state_embs = self.proprio_state_proj(state_memory, emb_ids=emb_ids).to(dtype=dtype)
        state_att_masks = torch.zeros(
            batch_size,
            memory_size,
            dtype=torch.bool,
            device=state_memory.device,
        )
        return state_embs, state_memory_masks, state_att_masks

    def _project_proprioception_anchor(
        self,
        proprioception: Tensor | None,
        agent_pos_mask: Tensor | None,
        *,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if self.state_input_mode != 'proprio_anchor':
            return None
        if self.propri_proj is None:
            raise RuntimeError('propri_proj is not initialized for proprio_anchor mode')
        if proprioception is None:
            raise ValueError("state_input_mode='proprio_anchor' requires proprioception")

        if proprioception.ndim == 2:
            proprioception = proprioception[:, None, :]
        expected_shape = (proprioception.shape[0], 1, self.propri_dim)
        if tuple(proprioception.shape) != expected_shape:
            raise ValueError(
                f'proprioception must have shape [B,1,{self.propri_dim}], '
                f'got {tuple(proprioception.shape)}'
            )

        projector_input = proprioception.to(
            device=self.propri_proj.weight.device,
            dtype=self.propri_proj.weight.dtype,
        )
        if self.proj_with_mask:
            if agent_pos_mask is None:
                raise ValueError('proj_with_mask=True requires agent_pos_mask')
            if agent_pos_mask.ndim == 2:
                agent_pos_mask = agent_pos_mask[:, None, :]
            if tuple(agent_pos_mask.shape) != expected_shape:
                raise ValueError(
                    f'agent_pos_mask must have shape [B,1,{self.propri_dim}], '
                    f'got {tuple(agent_pos_mask.shape)}'
                )
            projector_input = torch.cat(
                [
                    projector_input,
                    agent_pos_mask.to(
                        device=projector_input.device,
                        dtype=projector_input.dtype,
                    ),
                ],
                dim=-1,
            )

        state_hidden = self.propri_proj(projector_input)
        if self.state_hidden_size < self.vlm_hidden_size:
            state_hidden = F.pad(
                state_hidden,
                (0, self.vlm_hidden_size - self.state_hidden_size),
                value=0.0,
            )
        return state_hidden.to(dtype=dtype)

    def _scatter_proprioception_anchor(
        self,
        lang_tokens: Tensor,
        lang_embs: Tensor,
        proprioception: Tensor | None,
        agent_pos_mask: Tensor | None,
        proprioception_present: Tensor | None = None,
    ) -> Tensor:
        if self.state_input_mode != 'proprio_anchor':
            return lang_embs

        if lang_tokens.ndim != 2:
            raise ValueError(
                f'lang_tokens must have shape [B,L], got {tuple(lang_tokens.shape)}'
            )
        if lang_embs.ndim != 3 or tuple(lang_embs.shape[:2]) != tuple(lang_tokens.shape):
            raise ValueError(
                'lang_embs must have shape [B,L,H] matching lang_tokens; '
                f'got lang_tokens={tuple(lang_tokens.shape)}, lang_embs={tuple(lang_embs.shape)}'
            )

        positions = lang_tokens == self.propri_token_id
        counts = positions.sum(dim=1)
        if proprioception_present is None:
            expected_counts = torch.ones_like(counts)
        else:
            expected_counts = proprioception_present.to(
                device=counts.device,
                dtype=torch.bool,
            )
            if expected_counts.ndim == 0 and counts.shape[0] == 1:
                expected_counts = expected_counts[None]
            if tuple(expected_counts.shape) != tuple(counts.shape):
                raise ValueError(
                    'proprioception_present must have shape [B], '
                    f'got {tuple(expected_counts.shape)} for batch size {counts.shape[0]}'
                )
            expected_counts = expected_counts.to(dtype=counts.dtype)
        if not torch.equal(counts, expected_counts):
            raise ValueError(
                'Unexpected <|propri|> count per sample: '
                f'expected {expected_counts.detach().cpu().tolist()}, '
                f'got {counts.detach().cpu().tolist()}'
            )
        if not torch.any(expected_counts):
            return lang_embs
        state_embed = self._project_proprioception_anchor(
            proprioception,
            agent_pos_mask,
            dtype=lang_embs.dtype,
        )
        if state_embed is None:
            raise RuntimeError('proprio_anchor projection unexpectedly returned None')
        expected_state_shape = (
            lang_tokens.shape[0],
            1,
            lang_embs.shape[-1],
        )
        if tuple(state_embed.shape) != expected_state_shape:
            raise ValueError(
                'Projected proprioception must match the language batch and hidden size; '
                f'expected {expected_state_shape}, got {tuple(state_embed.shape)}'
            )
        return torch.where(positions.unsqueeze(-1), state_embed, lang_embs)

    def _prepend_proprio_prefix_tokens(
        self,
        prefix_embs: Tensor,
        prefix_pad_masks: Tensor,
        prefix_att_masks: Tensor,
        state_memory: Tensor | None,
        state_memory_masks: Tensor | None,
        emb_ids: Tensor | None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        state_embs, state_pad_masks, state_att_masks = self._embed_proprio_state_tokens(
            state_memory,
            state_memory_masks,
            emb_ids,
            dtype=prefix_embs.dtype,
        )
        if state_embs is None:
            return prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, None

        state_att_masks = state_att_masks.to(dtype=prefix_att_masks.dtype)
        prefix_embs = torch.cat([state_embs, prefix_embs], dim=1)
        prefix_pad_masks = torch.cat([state_pad_masks, prefix_pad_masks], dim=1)
        prefix_att_masks = torch.cat([state_att_masks, prefix_att_masks], dim=1)
        state_len = state_embs.shape[1]
        if fast_action_indicator is not None:
            fast_action_indicator = F.pad(fast_action_indicator, (state_len, 0), value=False)
        if subtask_indicator is not None:
            subtask_indicator = F.pad(subtask_indicator, (state_len, 0), value=False)
        return prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks

    @staticmethod
    def _prepend_proprio_position_ids(position_ids: Tensor, state_pad_masks: Tensor | None) -> Tensor:
        if state_pad_masks is None:
            return position_ids

        state_pos = (torch.cumsum(state_pad_masks.long(), dim=1) - 1).clamp_min(0)
        state_offsets = state_pad_masks.long().sum(dim=1)
        if position_ids.ndim == 3:
            state_pos = state_pos.unsqueeze(0).expand(position_ids.shape[0], -1, -1)
            offset = state_offsets.view(1, -1, 1)
            return torch.cat([state_pos, position_ids + offset], dim=-1)

        offset = state_offsets.view(-1, 1)
        return torch.cat([state_pos, position_ids + offset], dim=1)

    @staticmethod
    def _last_valid_indices(pad_masks: Tensor) -> Tensor:
        positions = torch.arange(pad_masks.shape[1], device=pad_masks.device)
        positions = positions.view(1, -1).expand_as(pad_masks)
        return positions.masked_fill(~pad_masks.to(dtype=torch.bool), 0).max(dim=1).values

    @staticmethod
    def _gemma4_continuous_per_layer_inputs(model, inputs_embeds: Tensor) -> Tensor | None:
        if getattr(model, 'embed_tokens_per_layer', None) is None:
            return None
        projection = model.per_layer_model_projection(inputs_embeds) * model.per_layer_model_projection_scale
        projection = projection.reshape(
            *inputs_embeds.shape[:-1],
            model.text_cfg.num_hidden_layers,
            model.text_cfg.hidden_size_per_layer_input,
        )
        projection = model.per_layer_projection_norm(projection)
        return projection * model.per_layer_input_scale

    def _attach_full_prefill_generation_state(
        self,
        state: dict,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_att_masks: Tensor | None,
        image_grid_thw: Tensor | None,
        emb_ids: Tensor | None,
        state_memory: Tensor | None,
        state_memory_masks: Tensor | None,
    ) -> dict:
        """Store enough context to decode by recomputing the whole prefix.

        Some expert-backed backbones expose VLM-prefix prefill for flow inference
        but no token-by-token VLM decode API. For autoregressive FAST tokens with
        continuous proprio prefix tokens, recomputing the compact prefix is slower
        but preserves the exact observation prefix layout.
        """
        if lang_tokens.shape[0] != 1:
            raise NotImplementedError(
                "full-prefix autoregressive fallback currently supports batch_size=1"
            )

        valid_len = int(lang_masks[0].sum().item())
        state['full_prefill_decode'] = True
        state['images'] = images
        state['img_masks'] = img_masks
        state['lang_tokens'] = lang_tokens[:, :valid_len]
        state['lang_masks'] = lang_masks[:, :valid_len]
        state['lang_att_masks'] = None if lang_att_masks is None else lang_att_masks[:, :valid_len]
        state['image_grid_thw'] = image_grid_thw
        state['emb_ids'] = emb_ids
        state['state_memory'] = state_memory
        state['state_memory_masks'] = state_memory_masks
        return state

    @torch.no_grad()
    def _next_lang_logits_full_prefill(
        self,
        state: dict,
        input_token: Tensor,
    ) -> tuple[Tensor, dict]:
        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        lang_tokens = torch.cat([state['lang_tokens'], input_token], dim=1)
        new_mask = torch.ones(
            lang_tokens.shape[0],
            1,
            dtype=state['lang_masks'].dtype,
            device=state['lang_masks'].device,
        )
        lang_masks = torch.cat([state['lang_masks'], new_mask], dim=1)
        lang_att_masks = state['lang_att_masks']
        if lang_att_masks is not None:
            new_att = torch.ones(
                lang_att_masks.shape[0],
                1,
                dtype=lang_att_masks.dtype,
                device=lang_att_masks.device,
            )
            lang_att_masks = torch.cat([lang_att_masks, new_att], dim=1)

        return self.init_lang_generation(
            state['images'],
            state['img_masks'],
            lang_tokens,
            lang_masks,
            image_grid_thw=state.get('image_grid_thw'),
            lang_att_masks=lang_att_masks,
            emb_ids=state.get('emb_ids'),
            state_memory=state.get('state_memory'),
            state_memory_masks=state.get('state_memory_masks'),
        )

    def forward(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        image_grid_thw: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
        state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Runs the full model forward pass.

        This method processes image and language inputs, generates key-value caches,
        and then uses them to denoise an action sequence `x_t` at a given `timestep`.
        It also computes language logits and, if enabled, trajectory predictions.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            x_t: The noisy action tensor at the current timestep.
            timestep: The current timestep value.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.
            lang_att_masks: Optional attention masks for language tokens.
            fast_action_indicator: Optional tensor indicating fast action token positions.
            subtask_indicator: Optional tensor indicating subtask token positions.
              When both are provided, FAST token rows are blocked from attending to
              subtask token columns so the two groups are independent of each other.
            image_grid_thw: Optional grid dimensions for Qwen3.5 vision model.
                Shape (num_images, 3) with [T, H_patches, W_patches] per image.
                Only used when vlm_type='qwen3_5'. If None, computed from images.

        Returns:
            A dictionary containing:
            - 'v_t': The predicted velocity (denoised action).
            - 'lang_logits': Logits for the language model output.
            - 'traj_pred' (optional): The predicted trajectory if enabled.
        """
        if self.vlm_type == 'qwen3_vl' and getattr(self, 'has_action_expert', False):
            return self._forward_qwen3_vl_with_expert(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                image_grid_thw=image_grid_thw,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
            )
        elif self.vlm_type == 'qwen3_vl':
            if self.state_input_mode == 'proprio_memory':
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._forward_qwen3_vl(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
        elif self.vlm_type == 'qwen2_5_vl' and getattr(self, 'has_action_expert', False):
            return self._forward_qwen2_5_vl_with_expert(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                image_grid_thw=image_grid_thw,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
            )
        elif self.vlm_type == 'qwen2_5_vl':
            if self.state_input_mode == 'proprio_memory':
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._forward_qwen2_5_vl(images, img_masks, lang_tokens, lang_masks, image_grid_thw)
        elif (
            self.vlm_type == 'qwen3_5'
            and getattr(self, 'has_action_expert', False)
            and getattr(self, 'expert_type', 'dual_stream') == 'cross_dit'
        ):
            return self._forward_qwen3_5_crossdit(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                image_grid_thw=image_grid_thw,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                lang_loss_masks=lang_loss_masks,
            )
        elif (
            self.vlm_type == 'qwen3_5'
            and getattr(self, 'has_action_expert', False)
            and getattr(self, 'expert_type', 'dual_stream') == 'robomanip'
        ):
            return self._forward_qwen3_5_robomanip(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                image_grid_thw=image_grid_thw,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                lang_loss_masks=lang_loss_masks,
                state=state,
            )
        elif (
            self.vlm_type == 'qwen3_5'
            and getattr(self, 'has_action_expert', False)
            and getattr(self, 'expert_type', 'dual_stream') == 'gemma3'
        ):
            return self._forward_qwen3_5_gemma3(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                image_grid_thw=image_grid_thw,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                lang_loss_masks=lang_loss_masks,
            )
        elif self.vlm_type == 'qwen3_5' and getattr(self, 'has_action_expert', False):
            return self._forward_qwen3_5_with_expert(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                image_grid_thw=image_grid_thw,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
                lang_loss_masks=lang_loss_masks,
            )
        elif self.vlm_type == 'qwen3_5':
            if self.state_input_mode == 'proprio_memory':
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._forward_qwen3_5(
                images, img_masks, lang_tokens, lang_masks, image_grid_thw,
                lang_loss_masks=lang_loss_masks,
            )
        elif self.vlm_type == 'gemma3' and getattr(self, 'has_action_expert', False):
            return self._forward_gemma3_with_expert(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
                lang_loss_masks=lang_loss_masks,
            )
        elif self.vlm_type == 'gemma3':
            if self.state_input_mode == 'proprio_memory':
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            # Phase 1/2 VLM-only.
            return self._forward_gemma3(
                images, img_masks, lang_tokens, lang_masks,
                lang_loss_masks=lang_loss_masks,
            )
        elif self.vlm_type == 'gemma4' and getattr(self, 'has_action_expert', False):
            return self._forward_gemma4_with_expert(
                images, img_masks, lang_tokens, lang_masks,
                x_t, timestep, emb_ids,
                lang_att_masks=lang_att_masks,
                fast_action_indicator=fast_action_indicator,
                subtask_indicator=subtask_indicator,
                lang_loss_masks=lang_loss_masks,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
            )
        elif self.vlm_type == 'gemma4':
            if self.state_input_mode == 'proprio_memory':
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            # Phase 1/2 VLM-only.
            return self._forward_gemma4(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                fast_action_indicator=fast_action_indicator,
                lang_loss_masks=lang_loss_masks,
            )

        lang_length = lang_masks.shape[-1]

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, traj_token_start_idx = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks,
            fast_action_indicator,
            subtask_indicator,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks = (
            self._prepend_proprio_prefix_tokens(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state_memory,
                state_memory_masks,
                emb_ids,
                fast_action_indicator,
                subtask_indicator,
            )
        )
        if state_pad_masks is not None and traj_token_start_idx is not None:
            traj_token_start_idx += state_pad_masks.shape[1]
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        # Construct 2d attention mask and position ids
        batch_size, prefix_len = prefix_pad_masks.shape
        suffix_len = suffix_pad_masks.shape[1]

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_len,
            fast_action_indicator,
            subtask_indicator,
        )

        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1, keepdim=True)
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        if fast_action_indicator is not None:
            suffix_position_ids = suffix_position_ids - fast_action_indicator.sum(dim=-1, keepdim=True)

        # VLM forward pass
        if self.action_loss_vlm_gradient_weight != 1.0:
            # Two-stage forward: prefix first, then suffix with gradient-weighted KV.
            prefix_att_2d_masks = full_att_2d[:, :prefix_len, :prefix_len]

            (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
                fill_kv_cache=True,
                adarms_cond=[None, None],
            )

            _weight_kv_cache_gradients(
                past_key_values,
                self.action_loss_vlm_gradient_weight,
            )

            suffix_att_2d_masks = full_att_2d[:, prefix_len:, :]

            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=suffix_att_2d_masks,
                position_ids=suffix_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=True,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        else:
            full_position_ids = torch.cat([prefix_position_ids, suffix_position_ids], dim=1)

            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d,
                position_ids=full_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        output = {}

        # Compute denoised output
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out, emb_ids=emb_ids)
        output['v_t'] = v_t

        # Compute language logits
        if self.enable_next_token_prediction:
            lang_out = prefix_out[:, -lang_length:, :]
            lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            output.update(_compact_language_logits(self.paligemma_with_expert, lang_out, compact_masks))

        # Compute trajectory predictions
        if self.enable_learnable_traj_token:
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            traj_pred = self.traj_out_proj(traj_out)
            output['traj_pred'] = traj_pred

        return output

    def _forward_qwen3_5(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        image_grid_thw: Tensor | None,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Pure VLM forward for Qwen3.5. Returns lang_logits + dummy v_t.

        Unlike PaliGemma2 which uses embed_prefix/embed_suffix + dual-stream,
        Qwen3.5 handles vision-text fusion internally (方案 A). The input_ids
        (lang_tokens) must contain image_token placeholders (248056) which the
        model replaces with vision embeddings.
        """
        from .qwen3_5_vlm import Qwen3_5VLMModel

        # 1. Convert images list to Qwen3.5 pixel_values format
        #    images: List[(B, C, H, W)] × num_cameras → flattened patches
        #    Must be batch-major: [b0_cam0, b0_cam1, b1_cam0, b1_cam1, ...]
        #    because masked_scatter fills batch 0 first, then batch 1.
        images_stacked = torch.stack(images, dim=1)  # (B, num_cameras, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])  # (B*N, C, H, W)
        pixel_values, computed_grid_thw = Qwen3_5VLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        else:
            # image_grid_thw from collator: (B, num_cameras, 3) → (B*num_cameras, 3)
            if image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.reshape(-1, 3)

        # 2. Attention mask (1D) from lang_masks
        attention_mask = lang_masks.long()

        # 3. Forward through Qwen3.5 (vision-text fusion handled internally)
        hidden_states, _ = self.qwen3_5_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        # 4. Compute language logits
        result = {}
        if self.enable_next_token_prediction:
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            result.update(_compact_language_logits(self.qwen3_5_vlm, hidden_states, compact_masks))

        # 5. Dummy v_t for loss framework compatibility (no diffusion in VLM-only mode)
        batch_size = lang_tokens.shape[0]
        result['v_t'] = torch.zeros(
            batch_size,
            self.n_action_steps,
            self.max_action_dim,
            device=lang_tokens.device,
            dtype=hidden_states.dtype,
        )

        return result

    def _forward_qwen3_vl(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        image_grid_thw: Tensor | None,
    ) -> dict[str, Tensor]:
        """Pure VLM forward for Qwen3-VL. Returns lang_logits + dummy v_t."""
        from .qwen3_vl_vlm import Qwen3VLVLMModel

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3VLVLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        else:
            if image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.reshape(-1, 3)

        attention_mask = lang_masks.long()
        hidden_states, _ = self.qwen3_vl_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        result = {}
        if self.enable_next_token_prediction:
            result['lang_logits'] = self.qwen3_vl_vlm.language_out_proj(hidden_states)

        batch_size = lang_tokens.shape[0]
        result['v_t'] = torch.zeros(
            batch_size, self.n_action_steps, self.max_action_dim,
            device=lang_tokens.device, dtype=hidden_states.dtype,
        )
        return result

    def _forward_qwen2_5_vl(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        image_grid_thw: Tensor | None,
    ) -> dict[str, Tensor]:
        """Pure VLM forward for Qwen2.5-VL. Returns lang_logits + dummy v_t."""
        from .qwen2_5_vl_vlm import Qwen2_5VLVLMModel

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen2_5VLVLMModel.images_to_pixel_values(
            images_flat, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        )
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        else:
            if image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.reshape(-1, 3)

        attention_mask = lang_masks.long()
        hidden_states, _ = self.qwen2_5_vl_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=attention_mask,
        )

        result = {}
        if self.enable_next_token_prediction:
            result['lang_logits'] = self.qwen2_5_vl_vlm.language_out_proj(hidden_states)

        batch_size = lang_tokens.shape[0]
        result['v_t'] = torch.zeros(
            batch_size, self.n_action_steps, self.max_action_dim,
            device=lang_tokens.device, dtype=hidden_states.dtype,
        )
        return result

    def _forward_gemma3(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Pure VLM forward for Gemma3 (Phase 1). Returns lang_logits + dummy v_t.

        Gemma3 SigLIP accepts standard (B*N_cam, C, H, W) — no flattened-patch
        format like Qwen. mm_tokens_per_image is fixed at 256, so no
        image_grid_thw is needed; the inner Gemma3Model handles vision encoding,
        AvgPool projection, masked_scatter, sliding/full mask construction, and
        double-theta RoPE internally.

        token_type_ids is derived inside Gemma3VLMModel.forward from input_ids
        (image_token_id positions = 1, others = 0) and overlays a bidirectional
        mask on image-token blocks (Phase 1 simplification: all image tokens
        share one image_group_id, equivalent to PaliGemma prefix-LM behavior).
        """
        # Stack images batch-major: [b0_cam0, b0_cam1, ..., b1_cam0, ...] so that
        # masked_scatter inside Gemma3Model fills batch 0 first.
        images_stacked = torch.stack(images, dim=1)  # (B, num_cameras, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])  # (B*N, C, H, W)

        attention_mask = lang_masks.long()
        hidden_states, _ = self.gemma3_vlm(
            input_ids=lang_tokens,
            pixel_values=images_flat,
            attention_mask=attention_mask,
        )

        result = {}
        if self.enable_next_token_prediction:
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            result.update(_compact_language_logits(self.gemma3_vlm, hidden_states, compact_masks))

        batch_size = lang_tokens.shape[0]
        result['v_t'] = torch.zeros(
            batch_size, self.n_action_steps, self.max_action_dim,
            device=lang_tokens.device, dtype=hidden_states.dtype,
        )
        return result

    def _forward_qwen3_vl_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Dual-stream forward with flow matching for Qwen3-VL.

        Standard Dual-Stream (all 36 layers Full Attention QKV concat).
        Simpler than Qwen3.5 (no GatedDeltaNet, no State-Bridged).
        """
        from .qwen3_vl_vlm import Qwen3VLVLMModel

        model = self.qwen3_vl_with_expert
        batch_size = lang_tokens.shape[0]

        # 1. Build prefix embeddings (vision + language)
        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3VLVLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs, deepstack_features = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)

        image_token_id = model.vlm_config.image_token_id
        visual_pos_masks = (lang_tokens == image_token_id)  # (B, L_prefix) bool
        special_image_mask = visual_pos_masks.unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))
        lang_length = lang_tokens.shape[1]

        # 2. Build suffix embeddings (action + timestep)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        # 3. Build attention mask
        prefix_pad_masks = lang_masks
        if lang_att_masks is None:
            prefix_att_masks = torch.ones(batch_size, lang_masks.shape[1], dtype=suffix_att_masks.dtype, device=lang_masks.device)
        else:
            prefix_att_masks = lang_att_masks.to(dtype=suffix_att_masks.dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks = (
            self._prepend_proprio_prefix_tokens(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state_memory,
                state_memory_masks,
                emb_ids,
                fast_action_indicator,
                subtask_indicator,
            )
        )

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_pad_masks.shape[1],
            fast_action_indicator,
            subtask_indicator,
        )

        full_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        # 4. Compute M-RoPE position IDs for prefix
        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1
        spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
        position_ids, mrope_deltas = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
            pad_masks=lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)
        if state_pad_masks is not None:
            state_visual_pos_masks = torch.zeros(
                batch_size,
                state_pad_masks.shape[1],
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([state_visual_pos_masks, visual_pos_masks], dim=1)

        # 5. Dual-stream forward
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        if self.enable_knowledge_insulation:
            prefix_att_mask = full_att_mask_4d[:, :, :prefix_len, :prefix_len]

            prefix_out, vlm_cache = model.forward_prefix_only(
                prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_features)

            vlm_cache = self._detach_dynamic_cache(vlm_cache)

            suffix_att_mask = full_att_mask_4d[:, :, prefix_len:, :]
            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1], fast_action_indicator=fast_action_indicator)
            suffix_out = model.forward_expert_only(
                suffix_embs, vlm_cache, adarms_cond=adarms_cond, attention_mask=suffix_att_mask,
                position_ids=expert_pos_ids)
        else:
            [prefix_out, suffix_out], _ = model.forward(
                attention_mask=full_att_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_features,
            )

        # 5. Extract outputs
        result = {}
        suffix_out = suffix_out[:, -self.n_action_steps:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        result['v_t'] = self.action_out_proj(suffix_out, emb_ids=emb_ids)

        if self.enable_next_token_prediction:
            lang_out = prefix_out[:, -lang_length:, :].to(dtype=model.lm_head.weight.dtype)
            result['lang_logits'] = model.language_out_proj(lang_out)

        return result

    def _forward_qwen2_5_vl_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Dual-stream forward with flow matching for Qwen2.5-VL.

        Standard Dual-Stream (all 36 layers Full Attention QKV concat). No DeepStack
        and no Q/K per-head norm (both Qwen3-VL specific) — simpler than the Qwen3-VL
        path. M-RoPE uses split section [16,24,24].
        """
        from .qwen2_5_vl_vlm import Qwen2_5VLVLMModel

        model = self.qwen2_5_vl_with_expert
        batch_size = lang_tokens.shape[0]

        # 1. Build prefix embeddings (vision + language)
        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen2_5VLVLMModel.images_to_pixel_values(
            images_flat, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        )
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)

        image_token_id = model.vlm_config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))
        lang_length = lang_tokens.shape[1]

        # 2. Build suffix embeddings (action + timestep)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        # 3. Build attention mask
        prefix_pad_masks = lang_masks
        if lang_att_masks is None:
            prefix_att_masks = torch.ones(batch_size, lang_masks.shape[1], dtype=suffix_att_masks.dtype, device=lang_masks.device)
        else:
            prefix_att_masks = lang_att_masks.to(dtype=suffix_att_masks.dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks = (
            self._prepend_proprio_prefix_tokens(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state_memory,
                state_memory_masks,
                emb_ids,
                fast_action_indicator,
                subtask_indicator,
            )
        )

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_pad_masks.shape[1],
            fast_action_indicator,
            subtask_indicator,
        )

        full_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        # 4. Compute prefix position IDs.
        # 1D RoPE (lingbot-vla / π0 style): cumsum(lang_masks)-1 → (B, L_prefix).
        # M-RoPE 3D (Qwen2.5-VL native): groupby vision/text, returns (3, B, L_prefix).
        if model.use_1d_rope:
            position_ids = self._compute_1d_position_ids(lang_masks)
        else:
            mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
            mm_token_type_ids[lang_tokens == image_token_id] = 1
            spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
            position_ids, _ = self._compute_mrope_position_ids(
                lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
                pad_masks=lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        # 5. Dual-stream forward
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        if self.enable_knowledge_insulation:
            prefix_att_mask = full_att_mask_4d[:, :, :prefix_len, :prefix_len]

            prefix_out, vlm_cache = model.forward_prefix_only(
                prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

            vlm_cache = self._detach_dynamic_cache(vlm_cache)

            suffix_att_mask = full_att_mask_4d[:, :, prefix_len:, :]
            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1], fast_action_indicator=fast_action_indicator)
            suffix_out = model.forward_expert_only(
                suffix_embs, vlm_cache, adarms_cond=adarms_cond, attention_mask=suffix_att_mask,
                position_ids=expert_pos_ids)
        else:
            [prefix_out, suffix_out], _ = model.forward(
                attention_mask=full_att_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        # 6. Extract outputs
        result = {}
        suffix_out = suffix_out[:, -self.n_action_steps:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        result['v_t'] = self.action_out_proj(suffix_out, emb_ids=emb_ids)

        if self.enable_next_token_prediction:
            lang_out = prefix_out[:, -lang_length:, :].to(dtype=model.lm_head.weight.dtype)
            result['lang_logits'] = model.language_out_proj(lang_out)

        return result

    def _forward_gemma3_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Dual-stream forward + flow matching for Gemma3 (Phase 3).

        Compared to the Qwen2.5-VL path this is *simpler*: no M-RoPE
        (1D positions only), no rope_deltas, no DeepStack injection. The only
        bookkeeping unique to Gemma3 is the per-layer-type position embedding
        dispatch, which lives inside Gemma3WithExpertModel itself — at this
        level we just pass standard 1D position_ids and let the model split.

        Image-token bidirectional attention is handled here the same way
        PaliGemma2 does it: lang_att_masks=False on vision-token positions
        means make_att_2d_masks emits a single causal block covering them, so
        every image token attends to every other image token within the prefix
        (Phase 1 simplification: all cameras share one image_group; Phase 4
        can split by camera).
        """
        model = self.gemma3_with_expert
        batch_size = lang_tokens.shape[0]
        image_token_id = model.vlm_config.image_token_id

        # 1. Build prefix embeddings (vision + language).
        # Gemma3 SigLIP takes standard (B*N_cam, C, H, W) — no flatten/Conv3d trick.
        # Stack batch-major: [b0_cam0, b0_cam1, ..., b1_cam0, ...] so masked_scatter
        # fills batch 0 first, matching how lang_tokens lays out per-camera image_token blocks.
        images_stacked = torch.stack(images, dim=1)  # (B, N_cam, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])

        img_embs = model.embed_image(images_flat)  # (B*N_cam, mm_tokens=256, text_hidden)
        img_embs = img_embs.reshape(-1, img_embs.shape[-1])  # flatten across cams + tokens

        lang_embs = model.embed_language_tokens(lang_tokens)
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))
        lang_length = lang_tokens.shape[1]

        # 2. Build suffix embeddings (action + timestep).
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep, emb_ids=emb_ids,
        )

        # 3. Build the unified attention mask.
        prefix_pad_masks = lang_masks
        if lang_att_masks is None:
            prefix_att_masks = torch.ones(
                batch_size, lang_masks.shape[1],
                dtype=suffix_att_masks.dtype, device=lang_masks.device,
            )
        else:
            prefix_att_masks = lang_att_masks.to(dtype=suffix_att_masks.dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks = (
            self._prepend_proprio_prefix_tokens(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state_memory,
                state_memory_masks,
                emb_ids,
                fast_action_indicator,
                subtask_indicator,
            )
        )

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_pad_masks.shape[1],
            fast_action_indicator,
            subtask_indicator,
        )

        full_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min,
        ).to(dtype=prefix_embs.dtype)

        # 4. 1D position IDs (cumsum from lang_masks, 0-indexed).
        position_ids = self._compute_1d_position_ids(lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        # 5. Dual-stream forward (with optional KI prefill+detach split).
        prefix_len = prefix_pad_masks.shape[1]

        if self.enable_knowledge_insulation:
            prefix_att_mask = full_att_mask_4d[:, :, :prefix_len, :prefix_len]
            prefix_out, vlm_cache = model.forward_prefix_only(
                prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
            )
            vlm_cache = self._detach_dynamic_cache(vlm_cache)

            suffix_att_mask = full_att_mask_4d[:, :, prefix_len:, :]
            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1], fast_action_indicator=fast_action_indicator)
            suffix_out = model.forward_expert_only(
                suffix_embs, vlm_cache, adarms_cond=adarms_cond,
                attention_mask=suffix_att_mask, position_ids=expert_pos_ids,
                prefix_position_ids=position_ids,
            )
        else:
            [prefix_out, suffix_out], _ = model.forward(
                attention_mask=full_att_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        # 6. Project outputs.
        result = {}
        suffix_out = suffix_out[:, -self.n_action_steps:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        result['v_t'] = self.action_out_proj(suffix_out, emb_ids=emb_ids)

        if self.enable_next_token_prediction:
            prefix_out = prefix_out.to(dtype=model.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            if compact_masks is not None and self.language_loss_chunk_size > 0:
                result.update(
                    _chunked_language_token_loss(
                        model,
                        prefix_out,
                        lang_tokens,
                        compact_masks,
                        self.language_loss_chunk_size,
                        self.language_loss_checkpoint,
                    )
                )
            else:
                result.update(_compact_language_logits(model, prefix_out, compact_masks))

        return result

    def _forward_gemma4(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        fast_action_indicator: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Pure VLM forward for Gemma4 (Phase 1/2). Returns lang_logits + dummy v_t.

        Gemma4 vision tower takes flattened patches plus 2D position ids
        (NOT raw image tensors like SigLIP). We patchify with
        Gemma4VLMModel.images_to_pixel_values and let the inner Gemma4Model
        handle vision encoding, multimodal embedder, masked_scatter, sliding/
        full mask construction, double-theta + proportional RoPE, PLE, and
        KV cross-layer sharing internally.
        """
        from .gemma4_vlm import Gemma4VLMModel

        # Stack images batch-major: [b0_cam0, b0_cam1, ..., b1_cam0, ...] so
        # the inner masked_scatter fills batch 0's image-token slots first.
        images_stacked = torch.stack(images, dim=1)  # (B, N_cam, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, image_position_ids = Gemma4VLMModel.images_to_pixel_values(images_flat)

        attention_mask = lang_masks.long()
        ple_remap_mask = None
        if self.fast_token_vocab_mode == 'tail' and fast_action_indicator is not None:
            ple_remap_mask = fast_action_indicator.to(device=lang_tokens.device, dtype=torch.bool)
            ple_remap_mask = ple_remap_mask & self._gemma4_tail_fast_token_mask(lang_tokens)
        hidden_states, _ = self.gemma4_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=attention_mask,
            ple_remap_mask=ple_remap_mask,
        )

        result = {}
        if self.enable_next_token_prediction:
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            result.update(_compact_language_logits(self.gemma4_vlm, hidden_states, compact_masks))

        batch_size = lang_tokens.shape[0]
        result['v_t'] = torch.zeros(
            batch_size, self.n_action_steps, self.max_action_dim,
            device=lang_tokens.device, dtype=hidden_states.dtype,
        )
        return result

    def _gemma4_embed_images_temporal(
        self, model, images: list[Tensor], img_masks: list[Tensor]
    ) -> Tensor:
        """Encode multi-frame (observation_memory_size > 1) camera images for gemma4.

        Each images[i] is (B, T, C, H, W). Every frame is encoded independently by the
        unchanged Gemma4 vision tower, then `model.temporal_mixer` mixes across the T
        frames per soft-token position and returns only the current frame's soft tokens,
        so the downstream masked_scatter sees the same flat (sum_cam B*S, hidden) tensor
        as the single-frame path. History frames are encoded under no_grad when training
        and detach_history_frames is set. See
        docs/design/gemma4_observation_memory_design_doc.md (strategy C).
        """
        mixer = model.temporal_mixer
        if mixer is None:
            raise RuntimeError(
                'gemma4 received multi-frame images but temporal_mixer is None; '
                'construct GigaBrain0Policy with observation_memory_size > 1.'
            )
        cam_shapes = {tuple(img.shape[-2:]) for img in images}
        if len(cam_shapes) != 1:
            raise NotImplementedError(
                'gemma4 observation_memory_size > 1 with per-camera resolution '
                '(high_res_cam) is not supported yet; use a single shared resolution.'
            )

        images_stacked = torch.stack(images, dim=1)  # (B, N_cam, T, C, H, W)
        bsz, n_cam, num_frames = images_stacked.shape[:3]
        spatial = images_stacked.shape[3:]
        imgs_bt = images_stacked.reshape(bsz * n_cam, num_frames, *spatial)  # (B*N_cam, T, C, H, W)

        def _encode(frames_4d: Tensor) -> Tensor:  # (M, C, H, W) -> (M, S, hidden)
            pixel_values, image_position_ids = model.images_to_pixel_values(frames_4d)
            embs = model.embed_image(pixel_values, image_position_ids)  # FLAT (M*S, hidden)
            return embs.reshape(frames_4d.shape[0], -1, embs.shape[-1])

        detach = mixer.detach_history_frames and self.training and num_frames > 1
        if detach:
            with torch.no_grad():
                history = imgs_bt[:, :-1].reshape(bsz * n_cam * (num_frames - 1), *spatial)
                soft_history = _encode(history)  # (B*N_cam*(T-1), S, hidden)
            soft_history = soft_history.reshape(bsz * n_cam, num_frames - 1, *soft_history.shape[1:])
            soft_current = _encode(imgs_bt[:, -1])  # (B*N_cam, S, hidden)
            soft = torch.cat([soft_history, soft_current[:, None]], dim=1)  # (B*N_cam, T, S, hidden)
        else:
            soft_all = _encode(imgs_bt.reshape(bsz * n_cam * num_frames, *spatial))
            soft = soft_all.reshape(bsz * n_cam, num_frames, *soft_all.shape[1:])

        frame_mask = torch.stack(img_masks, dim=1).reshape(bsz * n_cam, num_frames)  # (B*N_cam, T)
        soft_current = mixer(soft, frame_mask)  # (B*N_cam, S, hidden)
        return soft_current.reshape(-1, soft_current.shape[-1])  # FLAT (B*N_cam*S, hidden)

    @staticmethod
    def _present_camera_mask(img_masks: list[Tensor]) -> Tensor:
        """方案 B: per-(sample, camera) current-frame validity, present_img_keys order.

        Each img_masks[c] is (B,) single-frame or (B, T) temporal; take the current
        (last) frame for the temporal case. Returns (B, N_cam) bool. This must match
        the camera set ImageTransform emitted prompt blocks for (it uses the same
        current-frame validity), so the gather below stays aligned with masked_scatter.
        """
        return torch.stack(
            [m if m.dim() == 1 else m[..., -1] for m in img_masks], dim=1
        ).bool()

    @staticmethod
    def _zero_grad_graph_anchor(*tensors: Tensor) -> Tensor | None:
        """Attach otherwise-unused tensors to the graph with exactly zero gradient."""
        anchor = None
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor) or not tensor.requires_grad:
                continue
            term = tensor.sum(dtype=torch.float32) * tensor.new_zeros((), dtype=torch.float32)
            anchor = term if anchor is None else anchor + term
        return anchor

    @staticmethod
    def _gather_present_image_embs(img_embs: Tensor, present_mask: Tensor, num_image_token_slots: int) -> Tensor:
        """方案 B: keep only present cameras' image embeddings (uniform soft-token count).

        Args:
            img_embs: (B, N_cam, S, hidden) per-camera soft tokens, present_img_keys order.
            present_mask: (B, N_cam) bool from _present_camera_mask.
            num_image_token_slots: number of image_token_id positions in the batch.

        Returns a flat (num_image_token_slots, hidden) tensor in batch-major,
        present-camera-major order — exactly masked_scatter's fill order. When nothing
        was omitted (num_image_token_slots == B*N_cam*S) this equals a plain flatten, so
        a full-camera sample is bit-for-bit unchanged.
        """
        B, N_cam, S, hidden = img_embs.shape
        flat = img_embs.reshape(B * N_cam, S, hidden)
        anchor = GigaBrain0Policy._zero_grad_graph_anchor(img_embs)
        if num_image_token_slots == B * N_cam * S:
            gathered = flat.reshape(-1, hidden)
            return gathered if anchor is None else gathered + anchor.to(dtype=gathered.dtype)
        gathered = flat[present_mask.reshape(-1)].reshape(-1, hidden)
        assert gathered.shape[0] == num_image_token_slots, (
            f"方案 B image-token mismatch: {gathered.shape[0]} present-camera embeddings vs "
            f"{num_image_token_slots} image-token slots in the prompt; present_mask and the "
            f"prompt's emitted camera blocks are out of sync."
        )
        return gathered if anchor is None else gathered + anchor.to(dtype=gathered.dtype)

    @staticmethod
    def _reshape_gemma4_image_embs(img_embs: Tensor, batch_size: int, num_cameras: int) -> Tensor:
        """Return Gemma4 image embeddings as (B, N_cam, S, hidden).

        Gemma4VisionModel strips padded soft tokens with boolean indexing, so the
        current vision path returns a flat (B*N_cam*S, hidden) tensor. Keep a 3D
        fallback because older wrappers/documentation describe (B*N_cam, S, hidden).
        """
        if img_embs.dim() == 3:
            return img_embs.reshape(batch_size, num_cameras, img_embs.shape[1], img_embs.shape[2])

        if img_embs.dim() != 2:
            raise ValueError(f"Gemma4 image embeddings must be 2D or 3D, got shape {tuple(img_embs.shape)}")

        denom = batch_size * num_cameras
        if denom <= 0 or img_embs.shape[0] % denom != 0:
            raise ValueError(
                f"Gemma4 flat image embeddings have {img_embs.shape[0]} tokens, not divisible by "
                f"batch_size*num_cameras={denom}; vision tower output unexpected."
            )
        soft = img_embs.shape[0] // denom
        return img_embs.reshape(batch_size, num_cameras, soft, img_embs.shape[-1])

    def _forward_gemma4_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Dual-stream forward + flow matching for Gemma4 (Phase 3).

        Compared to the Gemma3 path this is structurally identical at the
        wiring level (1D RoPE, no rope_deltas, no DeepStack), but the
        Gemma4 dual-stream model needs two extra inputs and produces an
        extra cache:

          - `per_layer_inputs` (PLE per-token-identity residual signal,
            shape (B, L_prefix, num_layers, 256)) computed here via
            `model.compute_per_layer_inputs(...)`. Image and FAST tokens
            are remapped to PAD before the PLE lookup so the PLE signal
            is only defined for real text tokens.

          - `shared_kv_states` dict (returned from forward_prefix_only
            alongside the standard DynamicCache). KI requires BOTH the
            cache and this dict to be detached — otherwise layers 24-41,
            which read from shared_kv_states, leak gradients back to the
            VLM prefix.

        Image-token bidirectional attention is handled the same way as
        gemma3 / paligemma2: `lang_att_masks=False` on vision-token
        positions causes `make_att_2d_masks` to emit a single causal block
        covering them, so every image token attends to every other image
        token within the prefix.
        """
        from .gemma4_with_expert import detach_cross_kv, detach_kv_cache, detach_shared_kv_states

        model = self.gemma4_with_expert
        batch_size = lang_tokens.shape[0]
        image_token_id = model.vlm_config.image_token_id

        # 1. Vision encode. Fast path when all cameras share resolution: stack into
        #    one batch and run the vision tower once. With per-camera resolution
        #    (high_res_cam), cameras have different H/W and can't be stacked, so we
        #    embed each camera separately and re-interleave the soft tokens in
        #    (sample, camera) order. Either way the flattened img_embs ordering is
        #    [b0c0, b0c1, ..., b1c0, ...], matching the prompt's vision-token blocks
        #    so the masked_scatter below lands each camera's tokens correctly even
        #    when per-camera token counts differ.
        # 方案 B: a missing camera (current frame invalid) emits no image-token block in
        # the prompt, so we scatter only present cameras' embeddings. present_cam is in
        # present_img_keys order — the same order the prompt emitted blocks and that
        # masked_scatter fills in. n_img_slots is how many image-token positions the
        # prompt actually emitted; when nothing was omitted the gather is a no-op.
        n_img_slots = int((lang_tokens == image_token_id).sum().item())
        present_cam = self._present_camera_mask(img_masks)  # (B, N_cam)

        cam_shapes = {tuple(img.shape[-2:]) for img in images}
        if images[0].dim() == 5:
            # Multi-frame observation memory (observation_memory_size > 1): encode each
            # frame, mix across time, keep only the current frame's soft tokens.
            img_embs = self._gemma4_embed_images_temporal(model, images, img_masks)  # (B*N_cam*S, hidden)
            soft = img_embs.shape[0] // (batch_size * len(images))
            img_embs = self._gather_present_image_embs(
                img_embs.reshape(batch_size, len(images), soft, img_embs.shape[-1]),
                present_cam, n_img_slots,
            )
        elif len(cam_shapes) == 1:
            images_stacked = torch.stack(images, dim=1)  # (B, N_cam, C, H, W)
            images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
            pixel_values, image_position_ids = model.images_to_pixel_values(images_flat)
            img_embs = model.embed_image(pixel_values, image_position_ids)  # (B*N_cam*S, hidden)
            img_embs = self._gather_present_image_embs(
                self._reshape_gemma4_image_embs(img_embs, batch_size, len(images)),
                present_cam, n_img_slots,
            )
        else:
            # Per-camera embed (high_res_cam): variable soft-token count per camera, so
            # we can't reshape to (B, N_cam, S, hidden). embed_image returns a FLAT tensor
            # (B*soft_c, hidden) because Gemma4VisionModel strips padding via
            # `hidden_states[pooler_mask]` (gemma4_modeling.py:2058) — NOT (B, soft_c,
            # hidden). Reshape to add the batch dim back so we can index per sample, then
            # interleave in (sample, camera) order to match the prompt's block layout.
            per_cam_embs = []  # camera-major: each (B, soft_c, hidden)
            for cam_img in images:
                pv, pos = model.images_to_pixel_values(cam_img)
                embs_flat = model.embed_image(pv, pos)  # (B*soft_c, hidden)
                hidden = embs_flat.shape[-1]
                assert embs_flat.shape[0] % batch_size == 0, (
                    f"embed_image returned {embs_flat.shape[0]} tokens, not divisible by "
                    f"batch_size={batch_size}; vision tower output unexpected (padding?)"
                )
                soft_c = embs_flat.shape[0] // batch_size
                per_cam_embs.append(embs_flat.reshape(batch_size, soft_c, hidden))
            # 方案 B: drop absent cameras only when the prompt actually omitted blocks
            # (n_img_slots < full count); otherwise keep all (bit-identical legacy path).
            omit_active = n_img_slots != sum(e.shape[0] * e.shape[1] for e in per_cam_embs)
            img_embs = torch.cat(
                [
                    per_cam_embs[c][b]
                    for b in range(batch_size)
                    for c in range(len(per_cam_embs))
                    if not omit_active or bool(present_cam[b, c])
                ],
                dim=0,
            )  # (present soft tokens, hidden), in (sample, camera) order
            anchor = self._zero_grad_graph_anchor(*per_cam_embs)
            if anchor is not None:
                img_embs = img_embs + anchor.to(dtype=img_embs.dtype)
            assert img_embs.shape[0] == n_img_slots, (
                f"方案 B image-token mismatch (high_res): {img_embs.shape[0]} present vs "
                f"{n_img_slots} prompt slots"
            )

        # 2. Language embed + masked_scatter image features into image-token slots.
        lang_embs = model.embed_language_tokens(lang_tokens)
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))
        lang_length = lang_tokens.shape[1]

        # 3. PLE compute. Remap image/FAST tokens to PAD before the per-layer
        # embedding lookup. In tail-vocab mode the FAST ids are inside the base
        # vocab, so combine the action mask with the reserved tail-id range.
        ple_remap_mask = self._gemma4_prefix_ple_remap_mask(
            lang_tokens,
            image_token_id,
            fast_action_indicator=fast_action_indicator,
        )
        per_layer_inputs = model.compute_per_layer_inputs(
            lang_tokens, prefix_embs, ple_remap_mask=ple_remap_mask,
        )

        # 4. Suffix embeddings (action + timestep).
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            x_t, timestep, emb_ids=emb_ids,
        )

        # 5. Build the unified attention mask.
        prefix_pad_masks = lang_masks
        if lang_att_masks is None:
            prefix_att_masks = torch.ones(
                batch_size, lang_masks.shape[1],
                dtype=suffix_att_masks.dtype, device=lang_masks.device,
            )
        else:
            prefix_att_masks = lang_att_masks.to(dtype=suffix_att_masks.dtype)

        original_prefix_embs = prefix_embs
        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks = (
            self._prepend_proprio_prefix_tokens(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state_memory,
                state_memory_masks,
                emb_ids,
                fast_action_indicator,
                subtask_indicator,
            )
        )
        if state_pad_masks is not None:
            state_len = state_pad_masks.shape[1]
            state_per_layer_inputs = self._gemma4_continuous_per_layer_inputs(
                model,
                prefix_embs[:, :state_len, :],
            )
            if state_per_layer_inputs is not None:
                per_layer_inputs = torch.cat([state_per_layer_inputs, per_layer_inputs], dim=1)
        elif prefix_embs is not original_prefix_embs:
            raise RuntimeError('unexpected prefix embedding replacement without state_pad_masks')

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_pad_masks.shape[1],
            fast_action_indicator,
            subtask_indicator,
        )

        full_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min,
        ).to(dtype=prefix_embs.dtype)

        # 6. 1D position IDs.
        position_ids = self._compute_1d_position_ids(lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        # 7. Dual-stream forward (with optional KI prefill+detach split).
        prefix_len = prefix_pad_masks.shape[1]

        if self.enable_knowledge_insulation:
            prefix_att_mask = full_att_mask_4d[:, :, :prefix_len, :prefix_len]
            prefix_out, vlm_cache, shared_kv_states, cached_cross_kv = model.forward_prefix_only(
                prefix_embs, position_ids=position_ids,
                attention_mask=prefix_att_mask, per_layer_inputs=per_layer_inputs,
                return_cross_kv=True,
            )
            # Gemma4-specific KI: detach BOTH cache and shared_kv_states.
            # Without detaching shared_kv_states, layers 24-41 leak gradients
            # back to the VLM prefix via the shared-KV path. With
            # expert_cross_attend_shared_layers on, the fresh cross K/V also
            # carries VLM-prefix gradients and must be detached (empty no-op
            # otherwise).
            vlm_cache = detach_kv_cache(vlm_cache)
            shared_kv_states = detach_shared_kv_states(shared_kv_states)
            cached_cross_kv = detach_cross_kv(cached_cross_kv)

            suffix_att_mask = full_att_mask_4d[:, :, prefix_len:, :]
            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1], fast_action_indicator=fast_action_indicator)
            suffix_out = model.forward_expert_only(
                suffix_embs, vlm_cache, shared_kv_states,
                adarms_cond=adarms_cond,
                attention_mask=suffix_att_mask, position_ids=expert_pos_ids,
                prefix_position_ids=position_ids,
                cached_cross_kv=cached_cross_kv,
            )
        else:
            [prefix_out, suffix_out], _ = model.forward(
                attention_mask=full_att_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
                per_layer_inputs=per_layer_inputs,
            )

        # 8. Project outputs.
        result = {}
        suffix_out = suffix_out[:, -self.n_action_steps:]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        result['v_t'] = self.action_out_proj(suffix_out, emb_ids=emb_ids)

        if self.enable_next_token_prediction:
            prefix_out = prefix_out.to(dtype=model.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            if compact_masks is not None and self.language_loss_chunk_size > 0:
                result.update(
                    _chunked_language_token_loss(
                        model,
                        prefix_out[:, -lang_length:, :],
                        lang_tokens,
                        compact_masks,
                        self.language_loss_chunk_size,
                        self.language_loss_checkpoint,
                    )
                )
            else:
                result.update(_compact_language_logits(model, prefix_out[:, -lang_length:, :], compact_masks))

        return result

    def _forward_qwen3_5_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Dual-stream forward with flow matching.

        Manually constructs prefix embeddings (vision + language) and suffix
        embeddings (noisy action + timestep), then runs through the dual-stream
        Qwen3_5WithExpertModel.
        """
        from .qwen3_5_vlm import Qwen3_5VLMModel

        model = self.qwen3_5_with_expert
        batch_size = lang_tokens.shape[0]
        lang_length = lang_masks.shape[-1]

        # ── 1. Build prefix embeddings (vision + language) ──
        # Convert images to pixel_values (batch-major for masked_scatter)
        images_stacked = torch.stack(images, dim=1)  # (B, num_cameras, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3_5VLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        else:
            if image_grid_thw.dim() == 3:
                image_grid_thw = image_grid_thw.reshape(-1, 3)

        # Vision encoding
        img_embs = model.embed_image(pixel_values, image_grid_thw)
        # img_embs: (total_merged_patches, vlm_hidden_size)

        # Language embedding
        lang_embs = model.embed_language_tokens(lang_tokens)
        # lang_embs: (B, L_text, vlm_hidden_size)

        # Replace image_token placeholders with vision embeddings
        image_token_id = self.qwen3_5_with_expert.vlm_config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))
        lang_length = lang_tokens.shape[1]

        # ── 2. Build suffix embeddings (action + timestep) ──
        # Cast x_t to model dtype (float32 from loss → bf16 when model is bf16)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        # ── 3. Build attention mask ──
        # Prefix: lang_masks (padding), lang_att_masks (causal blocks)
        prefix_pad_masks = lang_masks
        if lang_att_masks is None:
            # Default: fully causal (all 1s)
            prefix_att_masks = torch.ones(batch_size, lang_masks.shape[1], dtype=suffix_att_masks.dtype, device=lang_masks.device)
        else:
            prefix_att_masks = lang_att_masks.to(dtype=suffix_att_masks.dtype)

        prefix_embs, prefix_pad_masks, prefix_att_masks, fast_action_indicator, subtask_indicator, state_pad_masks = (
            self._prepend_proprio_prefix_tokens(
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                state_memory,
                state_memory_masks,
                emb_ids,
                fast_action_indicator,
                subtask_indicator,
            )
        )

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)  # (B, L_total, L_total) bool

        full_att_2d = _mask_action_suffix_target_attention(
            full_att_2d,
            suffix_pad_masks.shape[1],
            fast_action_indicator,
            subtask_indicator,
        )

        # Convert bool mask to float mask for SDPA: True→0.0, False→-inf
        # Shape: (B, 1, L_total, L_total) for head broadcasting
        full_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        # ── 4. Compute M-RoPE position IDs for prefix ──
        # Use Qwen3.5's internal mm_token_type_ids to distinguish vision vs text
        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1  # 1 = image

        spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
        position_ids, mrope_deltas = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
            pad_masks=lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        # ── 5. Dual-stream forward ──
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = suffix_pad_masks.shape[1]

        if self.enable_knowledge_insulation:
            # Two-stage forward: VLM prefix → detach cache → Expert suffix
            prefix_att_mask = full_att_mask_4d[:, :, :prefix_len, :prefix_len]

            # Stage 1: VLM prefix only, builds KV + recurrent state cache
            prefix_out, vlm_cache = model.forward_prefix_only(
                prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

            # Detach cache: gradient isolation between VLM and Expert
            vlm_cache = self._detach_qwen3_5_cache(vlm_cache)

            # Stage 2: Expert suffix from detached cache
            suffix_att_mask = full_att_mask_4d[:, :, prefix_len:, :]
            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1], fast_action_indicator=fast_action_indicator)
            suffix_out = model.forward_expert_only(
                suffix_embs, vlm_cache, adarms_cond=adarms_cond, attention_mask=suffix_att_mask,
                position_ids=expert_pos_ids)
        else:
            # Single forward pass with both streams
            [prefix_out, suffix_out], _ = model.forward(
                attention_mask=full_att_mask_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                fill_kv_cache=False,
                adarms_cond=[None, adarms_cond],
            )

        # ── 5. Extract outputs ──
        result = {}

        # v_t: velocity field from Expert suffix output
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        result['v_t'] = self.action_out_proj(suffix_out, emb_ids=emb_ids)

        # lang_logits: NTP from VLM prefix output
        if self.enable_next_token_prediction:
            prefix_out = prefix_out.to(dtype=model.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            if compact_masks is not None and self.language_loss_chunk_size > 0:
                result.update(
                    _chunked_language_token_loss(
                        model,
                        prefix_out,
                        lang_tokens,
                        compact_masks,
                        self.language_loss_chunk_size,
                        self.language_loss_checkpoint,
                    )
                )
            else:
                result.update(_compact_language_logits(model, prefix_out, compact_masks))

        return result

    @torch.no_grad()
    def _sample_actions_qwen3_5(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> Tensor:
        """Dual-stream inference: VLM prefill once → Expert denoising loop."""
        from .qwen3_5_vlm import Qwen3_5VLMModel

        model = self.qwen3_5_with_expert
        B = lang_tokens.shape[0]
        device = lang_tokens.device

        # ── 1. Build prefix embeddings (batch-major for masked_scatter) ──
        images_stacked = torch.stack(images, dim=1)  # (B, num_cameras, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3_5VLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)

        image_token_id = model.vlm_config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        # Compute M-RoPE position IDs
        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1
        spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
        position_ids, _ = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
            pad_masks=lang_masks)

        if lang_att_masks is None:
            prefix_att_masks = torch.ones(B, lang_masks.shape[1], dtype=torch.long, device=device)
        else:
            prefix_att_masks = lang_att_masks.to(device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        # Build attention mask matching training (_forward_qwen3_5_with_expert).
        # `lang_att_masks` from the tokenizer encodes prefix-LM (vision bidirectional,
        # text/action causal); when None we fall back to fully causal prefix.
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = self.n_action_steps
        prefix_att_mask, suffix_att_mask = self._build_dual_stream_inference_masks(
            prefix_pad_masks, prefix_len, suffix_len, dtype=prefix_embs.dtype, device=device,
            lang_att_masks=prefix_att_masks)

        # ── 2. VLM Prefill (only once) ──
        prefix_out, cached_vlm_states = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

        # ── 3. Expert Denoising Loop ──
        x_t = self._prepare_action_noise(noise, B, device)
        # Keep x_t in model dtype for consistency
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)

        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        # Expert position IDs: start from prefix_max_pos (matches training)
        if position_ids.ndim == 3:
            # Reduce across T/H/W so expert tokens start from a single scalar; per-dim max would let
            # vision H/W grid push H/W beyond T, breaking 3D-equal RoPE and 1D degeneracy for expert.
            prefix_max_pos = position_ids.max(dim=-1).values.max(dim=0).values.unsqueeze(0).expand(3, -1)  # (3, B)
        else:
            prefix_max_pos = position_ids.max(dim=-1).values.unsqueeze(0).expand(3, -1)

        for t in timesteps:
            t_batch = t.expand(B)

            # Embed action + time (cast to model dtype)
            x_t_cast = x_t.to(dtype=self.action_in_proj.weight.dtype)
            suffix_embs, _, _, adarms_cond = self.embed_suffix(x_t_cast, t_batch, emb_ids=emb_ids)

            # Snapshot cache (each step starts fresh from VLM state)
            step_cache = cached_vlm_states.snapshot()

            if self.enable_knowledge_insulation:
                step_cache = self._detach_qwen3_5_cache(step_cache)

            # Expert forward
            L_suffix = suffix_embs.shape[1]
            suffix_offsets = torch.arange(1, L_suffix + 1, device=device).view(1, 1, -1)
            expert_pos_ids = prefix_max_pos.unsqueeze(-1) + suffix_offsets
            expert_out = model.forward_expert_only(
                suffix_embs, step_cache, adarms_cond=adarms_cond, position_ids=expert_pos_ids,
                attention_mask=suffix_att_mask)

            # Euler step
            v_t = self.action_out_proj(expert_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t)

        return x_t

    def _build_dual_stream_inference_masks(
        self,
        lang_masks: Tensor,
        prefix_len: int,
        suffix_len: int,
        dtype: torch.dtype,
        device: torch.device,
        lang_att_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Build prefix/suffix attention masks matching the training KI path.

        Mirrors the construction in `_forward_qwen3_5_with_expert` /
        `_forward_qwen3_vl_with_expert`.

        `lang_att_masks` follows `make_att_2d_masks` semantics: a `1` starts a
        new causal block, `0` continues the current block. The tokenizer
        produces `False` for vision tokens (bidirectional) and `True` for
        text/subtask/action/eos (causal) — pass that through unchanged. If
        `None`, defaults to all-ones (fully causal prefix).

        Returns:
            prefix_att_mask: (B, 1, prefix_len, prefix_len) float, additive (-inf for masked).
            suffix_att_mask: (B, 1, suffix_len, prefix_len + suffix_len) float, additive.
        """
        B = lang_masks.shape[0]
        prefix_pad_masks = lang_masks  # (B, prefix_len) bool
        suffix_pad_masks = torch.ones(B, suffix_len, dtype=lang_masks.dtype, device=device)
        if lang_att_masks is None:
            prefix_att_masks_1d = torch.ones(B, prefix_len, dtype=torch.long, device=device)
        else:
            prefix_att_masks_1d = lang_att_masks.to(dtype=torch.long, device=device)
        suffix_att_masks_1d = torch.zeros(B, suffix_len, dtype=torch.long, device=device)
        suffix_att_masks_1d[:, 0] = 1

        pad_masks_full = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks_full = torch.cat([prefix_att_masks_1d, suffix_att_masks_1d], dim=1)
        full_att_2d = make_att_2d_masks(pad_masks_full, att_masks_full)
        full_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(dtype).min
        ).to(dtype=dtype)

        prefix_att_mask = full_att_mask_4d[:, :, :prefix_len, :prefix_len]
        suffix_att_mask = full_att_mask_4d[:, :, prefix_len:, :]
        return prefix_att_mask, suffix_att_mask

    @staticmethod
    def _detach_qwen3_5_cache(cache):
        """Detach all states in a Qwen3.5 cache for knowledge insulation."""
        for i in range(len(cache.key_cache)):
            if cache.key_cache[i] is not None:
                cache.key_cache[i] = cache.key_cache[i].detach()
            if cache.value_cache[i] is not None:
                cache.value_cache[i] = cache.value_cache[i].detach()
            if cache.conv_states[i] is not None:
                cache.conv_states[i] = cache.conv_states[i].detach()
            if cache.recurrent_states[i] is not None:
                cache.recurrent_states[i] = cache.recurrent_states[i].detach()
        return cache

    # ──────────────────────────────────────────────────────────────────────
    #  Qwen3.5 layer-wise cross-attention DiT expert (expert_type='cross_dit')
    # ──────────────────────────────────────────────────────────────────────

    def _qwen3_5_build_prefix(self, images, lang_tokens, lang_masks, image_grid_thw, lang_att_masks):
        """Shared cross-DiT prefix construction: fuse vision+language, build the
        prefix-LM 4D mask, M-RoPE ids, and the additive cross-attention key mask.

        Returns (prefix_embs, prefix_att_mask_4d, position_ids, enc_attn_mask).
        """
        from .qwen3_5_vlm import Qwen3_5VLMModel

        model = self.qwen3_5_vlm
        batch_size = lang_tokens.shape[0]

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3_5VLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)

        image_token_id = model.config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        if lang_att_masks is None:
            prefix_att_masks = torch.ones(batch_size, lang_masks.shape[1], dtype=torch.long, device=lang_masks.device)
        else:
            prefix_att_masks = lang_att_masks.to(dtype=torch.long)
        full_att_2d = make_att_2d_masks(lang_masks, prefix_att_masks)  # (B, Lp, Lp) bool
        prefix_att_mask_4d = torch.where(
            full_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1
        spatial_merge_size = model.config.vision_config.spatial_merge_size
        position_ids, _ = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size, pad_masks=lang_masks)

        # Additive key-padding mask for the DiT cross-attention over VLM tokens.
        enc_attn_mask = torch.where(
            lang_masks.to(torch.bool), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)  # (B, Lp)

        return prefix_embs, prefix_att_mask_4d, position_ids, enc_attn_mask

    def _forward_qwen3_5_crossdit(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Layer-wise cross-attention DiT forward (flow matching).

        Runs the pure VLM once over the prefix, exposing every decoder layer's
        hidden state, then a decoupled DiT cross-attends to those per-layer hiddens
        to predict the velocity field ``v_t``. The VLM's hybrid attention is
        transparent here — no state bridging, no cache. Knowledge insulation is a
        plain ``.detach()`` of the per-layer hiddens before the DiT.
        """
        model = self.qwen3_5_vlm

        prefix_embs, prefix_att_mask_4d, position_ids, enc_attn_mask = self._qwen3_5_build_prefix(
            images, lang_tokens, lang_masks, image_grid_thw, lang_att_masks)

        last_hidden, hidden_tuple = model.forward_prefix_hidden(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask_4d)
        num_dit_layers = self.qwen3_5_crossdit_expert.num_dit_layers
        vl_embs_list = list(hidden_tuple[-num_dit_layers:])
        if self.enable_knowledge_insulation:
            vl_embs_list = [h.detach() for h in vl_embs_list]

        # DiT expert → v_t (action_in/out_proj are embodiment-specific, float32)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(x_t, emb_ids=emb_ids).to(dtype=prefix_embs.dtype)
        dit_out = self.qwen3_5_crossdit_expert(
            vl_embs_list, action_emb, timestep, encoder_attention_mask=enc_attn_mask)
        dit_out = dit_out.to(dtype=self.action_out_proj.weight.dtype)
        result = {'v_t': self.action_out_proj(dit_out, emb_ids=emb_ids)}

        # Optional NTP from the VLM final hidden state.
        if self.enable_next_token_prediction:
            prefix_out = last_hidden.to(dtype=model.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            if compact_masks is not None and self.language_loss_chunk_size > 0:
                result.update(_chunked_language_token_loss(
                    model, prefix_out, lang_tokens, compact_masks,
                    self.language_loss_chunk_size, self.language_loss_checkpoint))
            else:
                result.update(_compact_language_logits(model, prefix_out, compact_masks))

        return result

    @torch.no_grad()
    def _sample_actions_qwen3_5_crossdit(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
    ) -> Tensor:
        """Cross-DiT inference: VLM prefix once → DiT Euler denoising loop.

        The per-layer hidden states are computed a single time; each Euler step only
        re-runs the (cheap) DiT. No cache snapshotting (unlike the dual-stream path).
        """
        model = self.qwen3_5_vlm
        B = lang_tokens.shape[0]
        device = lang_tokens.device

        prefix_embs, prefix_att_mask_4d, position_ids, enc_attn_mask = self._qwen3_5_build_prefix(
            images, lang_tokens, lang_masks, None, lang_att_masks)

        _, hidden_tuple = model.forward_prefix_hidden(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask_4d)
        num_dit_layers = self.qwen3_5_crossdit_expert.num_dit_layers
        vl_embs_list = list(hidden_tuple[-num_dit_layers:])

        x_t = self._prepare_action_noise(noise, B, device)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        for t in timesteps:
            t_batch = t.expand(B)
            action_emb = self.action_in_proj(x_t, emb_ids=emb_ids).to(dtype=prefix_embs.dtype)
            dit_out = self.qwen3_5_crossdit_expert(
                vl_embs_list, action_emb, t_batch, encoder_attention_mask=enc_attn_mask)
            v_t = self.action_out_proj(dit_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t.to(x_t.dtype))

        return x_t

    # ──────────────────────────────────────────────────────────────────────
    #  Qwen3.5 warm-started Gemma3-1B cross-attention expert (expert_type='gemma3')
    # ──────────────────────────────────────────────────────────────────────

    def _gemma3_select_vl_embs(self, hidden_tuple):
        """Evenly-spaced selection of `num_expert_layers` VLM layer hiddens.

        The Gemma3-1B expert has 26 blocks; Qwen3.5 has 32 layers. Block i
        cross-attends to the i-th evenly-spaced VLM layer (shallow→deep coverage).
        """
        n_expert = len(self.gemma3_expert.blocks)
        layer_outs = list(hidden_tuple[-self._gemma3_num_vlm_layers:])
        idx = torch.linspace(0, len(layer_outs) - 1, n_expert).round().long().tolist()
        return [layer_outs[i] for i in idx]

    def _forward_qwen3_5_gemma3(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Warm-started Gemma3-1B cross-attention expert forward (flow matching).

        Same decoupled structure as the cross-DiT path (one VLM pass exposing
        per-layer hiddens; the expert cross-attends to them), but the expert is an
        independent Gemma3-1B stack (self-attn + cross-attn + mlp), warm-started from
        gemma-3-1b-pt. KI is a plain ``.detach()`` of the per-layer hiddens.
        """
        model = self.qwen3_5_vlm
        prefix_embs, prefix_att_mask_4d, position_ids, enc_attn_mask = self._qwen3_5_build_prefix(
            images, lang_tokens, lang_masks, image_grid_thw, lang_att_masks)
        last_hidden, hidden_tuple = model.forward_prefix_hidden(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask_4d)
        vl_embs_list = self._gemma3_select_vl_embs(hidden_tuple)
        if self.enable_knowledge_insulation:
            vl_embs_list = [h.detach() for h in vl_embs_list]

        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(x_t, emb_ids=emb_ids).to(dtype=prefix_embs.dtype)
        expert_out = self.gemma3_expert(
            vl_embs_list, action_emb, timestep, encoder_attention_mask=enc_attn_mask)
        expert_out = expert_out.to(dtype=self.action_out_proj.weight.dtype)
        result = {'v_t': self.action_out_proj(expert_out, emb_ids=emb_ids)}

        if self.enable_next_token_prediction:
            prefix_out = last_hidden.to(dtype=model.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            if compact_masks is not None and self.language_loss_chunk_size > 0:
                result.update(_chunked_language_token_loss(
                    model, prefix_out, lang_tokens, compact_masks,
                    self.language_loss_chunk_size, self.language_loss_checkpoint))
            else:
                result.update(_compact_language_logits(model, prefix_out, compact_masks))
        return result

    @torch.no_grad()
    def _sample_actions_qwen3_5_gemma3(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
    ) -> Tensor:
        """Gemma3 cross-attention expert inference: VLM prefix once → Euler loop.

        The per-layer hidden states are computed once; each Euler step re-runs only
        the (cheap) expert. Mirrors the cross-DiT sampler.
        """
        model = self.qwen3_5_vlm
        B = lang_tokens.shape[0]
        device = lang_tokens.device
        prefix_embs, prefix_att_mask_4d, position_ids, enc_attn_mask = self._qwen3_5_build_prefix(
            images, lang_tokens, lang_masks, None, lang_att_masks)
        _, hidden_tuple = model.forward_prefix_hidden(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask_4d)
        vl_embs_list = self._gemma3_select_vl_embs(hidden_tuple)

        x_t = self._prepare_action_noise(noise, B, device).to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)
        for t in timesteps:
            t_batch = t.expand(B)
            action_emb = self.action_in_proj(x_t, emb_ids=emb_ids).to(dtype=prefix_embs.dtype)
            expert_out = self.gemma3_expert(
                vl_embs_list, action_emb, t_batch, encoder_attention_mask=enc_attn_mask)
            v_t = self.action_out_proj(expert_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = x_t + dt * v_t.to(x_t.dtype)
        return x_t

    # ──────────────────────────────────────────────────────────────────────
    #  Qwen3.5 Qwen-RobotManip-style DiT expert (expert_type='robomanip')
    # ──────────────────────────────────────────────────────────────────────

    def _qwen3_5_modality_key_masks(self, lang_tokens, lang_masks, image_token_id, dtype):
        """Additive (B, L) key masks for the RoboManip expert's cross-attention: 0 for the
        modality's tokens, -inf elsewhere. Returns (visual, text, full): visual = image
        tokens, text = valid non-image tokens, full = all valid prefix tokens (V+L together,
        used by the paper's variant-3 'Q/A jointly cross-attend V/L')."""
        neg = torch.finfo(dtype).min
        visual_pos = lang_tokens == image_token_id
        valid_pos = lang_masks.to(torch.bool)
        text_pos = valid_pos & (~visual_pos)
        visual_key_mask = torch.where(visual_pos, 0.0, neg).to(dtype)
        text_key_mask = torch.where(text_pos, 0.0, neg).to(dtype)
        full_key_mask = torch.where(valid_pos, 0.0, neg).to(dtype)
        return visual_key_mask, text_key_mask, full_key_mask

    def _forward_qwen3_5_robomanip(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        lang_loss_masks: Tensor | None = None,
        state: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Qwen-RobotManip-style forward: run the pure VLM once for its last-layer
        hidden, then a DiT expert with self-attn + modality-alternating cross-attn
        (even blocks attend visual tokens, odd blocks attend language tokens). The
        continuous proprioceptive state is MLP-encoded and prepended inside the expert."""
        model = self.qwen3_5_vlm

        prefix_embs, prefix_att_mask_4d, position_ids, _ = self._qwen3_5_build_prefix(
            images, lang_tokens, lang_masks, image_grid_thw, lang_att_masks)
        last_hidden = model.forward_prefix_last(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask_4d)

        # KI insulates the action gradient from the VLM, but NTP (if on) still trains it.
        expert_hidden = last_hidden.detach() if self.enable_knowledge_insulation else last_hidden
        visual_key_mask, text_key_mask, full_key_mask = self._qwen3_5_modality_key_masks(
            lang_tokens, lang_masks, model.config.image_token_id, prefix_embs.dtype)

        if state is not None:
            state = state.to(dtype=prefix_embs.dtype)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        action_emb = self.action_in_proj(x_t, emb_ids=emb_ids).to(dtype=prefix_embs.dtype)
        dit_out = self.qwen3_5_robomanip_expert(
            expert_hidden, action_emb, timestep, visual_key_mask, text_key_mask, full_key_mask, state=state)
        dit_out = dit_out.to(dtype=self.action_out_proj.weight.dtype)
        result = {'v_t': self.action_out_proj(dit_out, emb_ids=emb_ids)}

        if self.enable_next_token_prediction:
            prefix_out = last_hidden.to(dtype=model.lm_head.weight.dtype)
            compact_masks = lang_loss_masks if self.compact_language_logits else None
            if compact_masks is not None and self.language_loss_chunk_size > 0:
                result.update(_chunked_language_token_loss(
                    model, prefix_out, lang_tokens, compact_masks,
                    self.language_loss_chunk_size, self.language_loss_checkpoint))
            else:
                result.update(_compact_language_logits(model, prefix_out, compact_masks))
        return result

    @torch.no_grad()
    def _sample_actions_qwen3_5_robomanip(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state: Tensor | None = None,
    ) -> Tensor:
        """RoboManip-style inference: VLM prefix once (last hidden) -> Euler loop on DiT.

        The robomanip expert is trained with a continuous proprioceptive state token,
        so callers (rollout/pipeline) must pass ``state`` (batch['observation.state'])
        to avoid a train/inference mismatch; this raises if it is missing.
        """
        model = self.qwen3_5_vlm
        B = lang_tokens.shape[0]
        device = lang_tokens.device

        if self.qwen3_5_robomanip_expert.state_encoder is not None and state is None:
            raise ValueError(
                'robomanip expert was trained with a continuous state token; '
                'sample_actions(..., state=observation.state) is required.'
            )

        prefix_embs, prefix_att_mask_4d, position_ids, _ = self._qwen3_5_build_prefix(
            images, lang_tokens, lang_masks, None, lang_att_masks)
        last_hidden = model.forward_prefix_last(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask_4d)
        visual_key_mask, text_key_mask, full_key_mask = self._qwen3_5_modality_key_masks(
            lang_tokens, lang_masks, model.config.image_token_id, prefix_embs.dtype)
        if state is not None:
            state = state.to(dtype=prefix_embs.dtype)

        x_t = self._prepare_action_noise(noise, B, device)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        for t in timesteps:
            t_batch = t.expand(B)
            action_emb = self.action_in_proj(x_t, emb_ids=emb_ids).to(dtype=prefix_embs.dtype)
            dit_out = self.qwen3_5_robomanip_expert(
                last_hidden, action_emb, t_batch, visual_key_mask, text_key_mask, full_key_mask, state=state)
            v_t = self.action_out_proj(dit_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t.to(x_t.dtype))

        return x_t

    @torch.no_grad()
    def _sample_actions_qwen3_vl(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> Tensor:
        """Dual-stream inference: VLM prefill once -> Expert denoising loop.

        Uses standard DynamicCache (KV only). Simpler than Qwen3.5 version.
        """
        from .qwen3_vl_vlm import Qwen3VLVLMModel

        model = self.qwen3_vl_with_expert
        B = lang_tokens.shape[0]
        device = lang_tokens.device

        # 1. Build prefix embeddings
        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3VLVLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs, deepstack_features = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)

        image_token_id = model.vlm_config.image_token_id
        visual_pos_masks = (lang_tokens == image_token_id)
        special_image_mask = visual_pos_masks.unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1
        spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
        position_ids, _ = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
            pad_masks=lang_masks)

        if lang_att_masks is None:
            prefix_att_masks = torch.ones(B, lang_masks.shape[1], dtype=torch.long, device=device)
        else:
            prefix_att_masks = lang_att_masks.to(device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)
        if state_pad_masks is not None:
            state_visual_pos_masks = torch.zeros(
                B,
                state_pad_masks.shape[1],
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([state_visual_pos_masks, visual_pos_masks], dim=1)

        # Build attention mask matching training (prefix-LM via lang_att_masks).
        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = self.n_action_steps
        prefix_att_mask, suffix_att_mask = self._build_dual_stream_inference_masks(
            prefix_pad_masks, prefix_len, suffix_len, dtype=prefix_embs.dtype, device=device,
            lang_att_masks=prefix_att_masks)

        # 2. VLM Prefill -- DeepStack injection happens inside forward_prefix_only;
        # the cached K/V already reflect injected hidden states, so the Expert denoising
        # loop transparently picks up DeepStack features without per-step injection.
        prefix_out, cached_vlm_states = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_features)

        # KI: detach the cache once after prefill. Expert never writes to it, so the
        # same detached cache is reused across all denoising steps (no per-step copy).
        if self.enable_knowledge_insulation:
            cached_vlm_states = self._detach_dynamic_cache(cached_vlm_states)

        # 3. Expert Denoising Loop
        x_t = self._prepare_action_noise(noise, B, device)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)

        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        # Expert position IDs: start from prefix_max_pos (matches training)
        if position_ids.ndim == 3:
            # Reduce across T/H/W so expert tokens start from a single scalar; per-dim max would let
            # vision H/W grid push H/W beyond T, breaking 3D-equal RoPE and 1D degeneracy for expert.
            prefix_max_pos = position_ids.max(dim=-1).values.max(dim=0).values.unsqueeze(0).expand(3, -1)  # (3, B)
        else:
            prefix_max_pos = position_ids.max(dim=-1).values.unsqueeze(0).expand(3, -1)

        for t in timesteps:
            t_batch = t.expand(B)
            x_t_cast = x_t.to(dtype=self.action_in_proj.weight.dtype)
            suffix_embs, _, _, adarms_cond = self.embed_suffix(x_t_cast, t_batch, emb_ids=emb_ids)

            L_suffix = suffix_embs.shape[1]
            suffix_offsets = torch.arange(1, L_suffix + 1, device=device).view(1, 1, -1)
            expert_pos_ids = prefix_max_pos.unsqueeze(-1) + suffix_offsets
            expert_out = model.forward_expert_only(
                suffix_embs, cached_vlm_states, adarms_cond=adarms_cond, position_ids=expert_pos_ids,
                attention_mask=suffix_att_mask)

            v_t = self.action_out_proj(expert_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t)

        return x_t

    def _sample_actions_qwen2_5_vl(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        image_grid_thw: Tensor | None = None,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> Tensor:
        """Dual-stream inference for Qwen2.5-VL: VLM prefill once -> Expert denoising.

        Standard DynamicCache (KV only). No DeepStack injection (Qwen2.5-VL has none).
        """
        from .qwen2_5_vl_vlm import Qwen2_5VLVLMModel

        model = self.qwen2_5_vl_with_expert
        B = lang_tokens.shape[0]
        device = lang_tokens.device

        # 1. Build prefix embeddings
        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen2_5VLVLMModel.images_to_pixel_values(
            images_flat, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        )
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)

        image_token_id = model.vlm_config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        if model.use_1d_rope:
            position_ids = self._compute_1d_position_ids(lang_masks)
        else:
            mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
            mm_token_type_ids[lang_tokens == image_token_id] = 1
            spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
            position_ids, _ = self._compute_mrope_position_ids(
                lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
                pad_masks=lang_masks)

        if lang_att_masks is None:
            prefix_att_masks = torch.ones(B, lang_masks.shape[1], dtype=torch.long, device=device)
        else:
            prefix_att_masks = lang_att_masks.to(device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = self.n_action_steps
        prefix_att_mask, suffix_att_mask = self._build_dual_stream_inference_masks(
            prefix_pad_masks, prefix_len, suffix_len, dtype=prefix_embs.dtype, device=device,
            lang_att_masks=prefix_att_masks)

        # 2. VLM Prefill (no DeepStack injection)
        prefix_out, cached_vlm_states = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

        # KI: detach the cache once after prefill.
        if self.enable_knowledge_insulation:
            cached_vlm_states = self._detach_dynamic_cache(cached_vlm_states)

        # 3. Expert Denoising Loop
        x_t = self._prepare_action_noise(noise, B, device)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)

        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        for t in timesteps:
            t_batch = t.expand(B)
            x_t_cast = x_t.to(dtype=self.action_in_proj.weight.dtype)
            suffix_embs, _, _, adarms_cond = self.embed_suffix(x_t_cast, t_batch, emb_ids=emb_ids)

            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1])
            expert_out = model.forward_expert_only(
                suffix_embs, cached_vlm_states, adarms_cond=adarms_cond, position_ids=expert_pos_ids,
                attention_mask=suffix_att_mask)

            v_t = self.action_out_proj(expert_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t)

        return x_t

    def _sample_actions_gemma3(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> Tensor:
        """Dual-stream inference for Gemma3 (Phase 3): VLM prefill once -> Expert denoising.

        Standard DynamicCache (sliding/full layers share it). 1D position IDs only —
        no rope_deltas, no DeepStack injection.
        """
        model = self.gemma3_with_expert
        B = lang_tokens.shape[0]
        device = lang_tokens.device
        image_token_id = model.vlm_config.image_token_id

        # 1. Build prefix embeddings (vision + language) — same as training forward.
        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        img_embs = model.embed_image(images_flat)
        img_embs = img_embs.reshape(-1, img_embs.shape[-1])
        lang_embs = model.embed_language_tokens(lang_tokens)
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        position_ids = self._compute_1d_position_ids(lang_masks)
        if lang_att_masks is None:
            prefix_att_masks = torch.ones(B, lang_masks.shape[1], dtype=torch.long, device=device)
        else:
            prefix_att_masks = lang_att_masks.to(device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = self.n_action_steps
        prefix_att_mask, suffix_att_mask = self._build_dual_stream_inference_masks(
            prefix_pad_masks, prefix_len, suffix_len, dtype=prefix_embs.dtype, device=device,
            lang_att_masks=prefix_att_masks,
        )

        # 2. VLM Prefill — run once, cache K/V across all 34 layers.
        prefix_out, cached_vlm_states = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
        )

        # KI: detach the cache once after prefill (downstream Expert reads view it as
        # leaves, so action gradients won't flow back into the VLM pretrained weights).
        if self.enable_knowledge_insulation:
            cached_vlm_states = self._detach_dynamic_cache(cached_vlm_states)

        # 3. Expert Denoising Loop.
        x_t = self._prepare_action_noise(noise, B, device)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)

        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        for t in timesteps:
            t_batch = t.expand(B)
            x_t_cast = x_t.to(dtype=self.action_in_proj.weight.dtype)
            suffix_embs, _, _, adarms_cond = self.embed_suffix(x_t_cast, t_batch, emb_ids=emb_ids)

            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1])
            expert_out = model.forward_expert_only(
                suffix_embs, cached_vlm_states, adarms_cond=adarms_cond,
                position_ids=expert_pos_ids, attention_mask=suffix_att_mask,
                prefix_position_ids=position_ids,
            )

            v_t = self.action_out_proj(expert_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t)

        return x_t

    def _gemma4_embed_prefix(self, model, images, img_masks, lang_tokens, image_token_id) -> Tensor:
        """Build Gemma4 prefix embeddings (vision soft tokens scattered into the
        image-token slots of the language embedding).

        Shared by `_sample_actions_gemma4` (flow inference) and
        `_init_lang_generation_gemma4` (FAST autoregressive prefill) so both use
        identical vision encoding / 方案 B present-camera gathering.
        """
        from .gemma4_vlm import Gemma4VLMModel

        B = lang_tokens.shape[0]
        # 方案 B: scatter only present cameras' embeddings (mirrors the training forward).
        n_img_slots = int((lang_tokens == image_token_id).sum().item())
        present_cam = self._present_camera_mask(img_masks)  # (B, N_cam)
        if images[0].dim() == 5:
            # Multi-frame observation memory (observation_memory_size > 1).
            img_embs = self._gemma4_embed_images_temporal(model, images, img_masks)  # (B*N_cam*S, hidden)
            img_embs = self._gather_present_image_embs(
                self._reshape_gemma4_image_embs(img_embs, B, len(images)),
                present_cam, n_img_slots,
            )
        elif len({tuple(img.shape[-2:]) for img in images}) == 1:
            # Fast path: all cameras share resolution -> stack into one batch.
            images_stacked = torch.stack(images, dim=1)
            images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
            pixel_values, image_position_ids = Gemma4VLMModel.images_to_pixel_values(images_flat)
            img_embs = model.embed_image(pixel_values, image_position_ids)
            img_embs = self._gather_present_image_embs(
                self._reshape_gemma4_image_embs(img_embs, B, len(images)),
                present_cam, n_img_slots,
            )
        else:
            # Per-camera resolution (high_res_cam): cameras differ in H/W and can't be
            # stacked. Embed each separately, reshape the FLAT (B*soft_c, hidden) vision
            # output back to (B, soft_c, hidden) — Gemma4VisionModel strips padding via
            # boolean indexing so it returns a flattened tensor — then interleave in
            # (sample, camera) order to match the prompt's per-camera vision blocks.
            # Mirrors the training-forward branch in _forward_gemma4_with_expert.
            per_cam_embs = []
            for cam_img in images:
                pv, pos = Gemma4VLMModel.images_to_pixel_values(cam_img)
                embs_flat = model.embed_image(pv, pos)  # (B*soft_c, hidden)
                hidden = embs_flat.shape[-1]
                assert embs_flat.shape[0] % B == 0, (
                    f"embed_image returned {embs_flat.shape[0]} tokens, not divisible by "
                    f"batch_size={B}; vision tower output unexpected (padding?)"
                )
                soft_c = embs_flat.shape[0] // B
                per_cam_embs.append(embs_flat.reshape(B, soft_c, hidden))
            # 方案 B: drop absent cameras only when the prompt actually omitted blocks.
            omit_active = n_img_slots != sum(e.shape[0] * e.shape[1] for e in per_cam_embs)
            img_embs = torch.cat(
                [
                    per_cam_embs[c][b]
                    for b in range(B)
                    for c in range(len(per_cam_embs))
                    if not omit_active or bool(present_cam[b, c])
                ],
                dim=0,
            )
            anchor = self._zero_grad_graph_anchor(*per_cam_embs)
            if anchor is not None:
                img_embs = img_embs + anchor.to(dtype=img_embs.dtype)
            assert img_embs.shape[0] == n_img_slots, (
                f"方案 B image-token mismatch (high_res): {img_embs.shape[0]} present vs "
                f"{n_img_slots} prompt slots"
            )
        lang_embs = model.embed_language_tokens(lang_tokens)
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        return lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

    def _sample_actions_gemma4(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> Tensor:
        """Dual-stream inference for Gemma4 (Phase 3): VLM prefill once -> Expert denoising.

        Standard DynamicCache for layers 0-23 + a separate `shared_kv_states`
        dict for layers 24-41. KI requires detaching BOTH (the second is
        Gemma4-specific — without it, gradients leak via the shared-KV path).
        """
        from .gemma4_with_expert import detach_cross_kv, detach_kv_cache, detach_shared_kv_states

        model = self.gemma4_with_expert
        B = lang_tokens.shape[0]
        device = lang_tokens.device
        image_token_id = model.vlm_config.image_token_id

        # 1. Build prefix embeddings (vision + language).
        prefix_embs = self._gemma4_embed_prefix(model, images, img_masks, lang_tokens, image_token_id)

        # PLE compute (image tokens remapped to PAD).
        ple_remap_mask = self._gemma4_prefix_ple_remap_mask(lang_tokens, image_token_id)
        per_layer_inputs = model.compute_per_layer_inputs(
            lang_tokens, prefix_embs, ple_remap_mask=ple_remap_mask,
        )

        position_ids = self._compute_1d_position_ids(lang_masks)
        if lang_att_masks is None:
            prefix_att_masks = torch.ones(B, lang_masks.shape[1], dtype=torch.long, device=device)
        else:
            prefix_att_masks = lang_att_masks.to(device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        if state_pad_masks is not None:
            state_len = state_pad_masks.shape[1]
            state_per_layer_inputs = self._gemma4_continuous_per_layer_inputs(
                model,
                prefix_embs[:, :state_len, :],
            )
            if state_per_layer_inputs is not None:
                per_layer_inputs = torch.cat([state_per_layer_inputs, per_layer_inputs], dim=1)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_len = prefix_pad_masks.shape[1]
        suffix_len = self.n_action_steps
        prefix_att_mask, suffix_att_mask = self._build_dual_stream_inference_masks(
            prefix_pad_masks, prefix_len, suffix_len, dtype=prefix_embs.dtype, device=device,
            lang_att_masks=prefix_att_masks,
        )

        # 2. VLM Prefill — run once, cache K/V across all 42 layers AND populate
        # shared_kv_states from layer 22 (sliding) / layer 23 (full). When
        # expert_cross_attend_shared_layers is on, also cache the Expert's fresh
        # per-shared-layer cross K/V (empty dict otherwise).
        prefix_out, cached_vlm_states, cached_shared_kv, cached_cross_kv = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
            per_layer_inputs=per_layer_inputs, return_cross_kv=True,
        )

        # KI: detach BOTH cache and shared_kv_states (Gemma4-specific —
        # layers 24-41 leak gradients via shared_kv_states without this).
        if self.enable_knowledge_insulation:
            cached_vlm_states = detach_kv_cache(cached_vlm_states)
            cached_shared_kv = detach_shared_kv_states(cached_shared_kv)
            cached_cross_kv = detach_cross_kv(cached_cross_kv)

        # 3. Expert Denoising Loop.
        x_t = self._prepare_action_noise(noise, B, device)
        x_t = x_t.to(dtype=self.action_in_proj.weight.dtype)

        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        for t in timesteps:
            t_batch = t.expand(B)
            x_t_cast = x_t.to(dtype=self.action_in_proj.weight.dtype)
            suffix_embs, _, _, adarms_cond = self.embed_suffix(x_t_cast, t_batch, emb_ids=emb_ids)

            expert_pos_ids = self._compute_expert_pos_ids(position_ids, suffix_embs.shape[1])
            expert_out = model.forward_expert_only(
                suffix_embs, cached_vlm_states, cached_shared_kv,
                adarms_cond=adarms_cond,
                position_ids=expert_pos_ids, attention_mask=suffix_att_mask,
                prefix_position_ids=position_ids,
                cached_cross_kv=cached_cross_kv,
            )

            v_t = self.action_out_proj(expert_out.to(self.action_out_proj.weight.dtype), emb_ids=emb_ids)
            x_t = self._mask_action_denoise_dims(x_t + dt * v_t)

        return x_t

    @staticmethod
    def _compute_expert_pos_ids(position_ids: Tensor, L_suffix: int, fast_action_indicator: Tensor | None = None) -> Tensor:
        """Compute expert position IDs starting from prefix_max_pos.

        Expert positions continue from the max position of the prefix, keeping
        expert tokens close to VLM tokens in RoPE space.

        ``fast_action_indicator`` (B, L_prefix; True on FAST action target
        tokens) shifts the suffix back by the per-sample FAST count so the flow
        tokens reuse the FAST targets' positions. At inference the prompt
        carries no FAST tokens, so without this shift the expert's RoPE
        distances to the prefix would differ between KI training and inference
        (mirrors the paligemma2 path in ``forward``). KI training call sites
        pass it; inference call sites leave it None (prompt has no FAST).

        Polymorphic on ``position_ids.ndim``:
          - ndim==3 (M-RoPE):  in (3, B, L_prefix) -> out (3, B, L_suffix). Reduces
            across T/H/W so expert starts from a single scalar; per-dim max would
            let vision H/W push H/W beyond T, breaking 3D-equal RoPE and 1D
            degeneracy for expert. FAST tokens are text tokens (advance all three
            dims by 1 each), so the same count is subtracted from all dims.
          - ndim==2 (1D RoPE): in (B, L_prefix) -> out (B, L_suffix).
        """
        num_fast = fast_action_indicator.long().sum(dim=-1) if fast_action_indicator is not None else None  # (B,)
        if position_ids.ndim == 3:
            prefix_max_pos = position_ids.max(dim=-1).values.max(dim=0).values.unsqueeze(0).expand(3, -1)  # (3, B)
            if num_fast is not None:
                prefix_max_pos = prefix_max_pos - num_fast.unsqueeze(0)
            suffix_offsets = torch.arange(1, L_suffix + 1, device=position_ids.device).view(1, 1, -1)
            return prefix_max_pos.unsqueeze(-1) + suffix_offsets  # (3, B, L_suffix)
        # 1D RoPE
        prefix_max_pos = position_ids.max(dim=-1).values.unsqueeze(-1)  # (B, 1)
        if num_fast is not None:
            prefix_max_pos = prefix_max_pos - num_fast.unsqueeze(-1)
        suffix_offsets = torch.arange(1, L_suffix + 1, device=position_ids.device).view(1, -1)
        return prefix_max_pos + suffix_offsets  # (B, L_suffix)

    @staticmethod
    def _compute_1d_position_ids(pad_masks: Tensor) -> Tensor:
        """Build 1D RoPE position IDs from a padding mask (lingbot-vla / π0 style).

        Equivalent to ``cumsum(pad_masks, dim=1) - 1``: each non-padding token
        gets a contiguous index, padding tokens reuse their predecessor's value
        (irrelevant since the attention mask masks them out anyway).

        Args:
            pad_masks: (B, L) bool/int padding mask, True/1 = real token.

        Returns:
            (B, L) long tensor of 1D positions.
        """
        return (torch.cumsum(pad_masks.long(), dim=1) - 1).clamp_min(0)

    @staticmethod
    def _detach_dynamic_cache(cache):
        """Detach standard DynamicCache for knowledge insulation (KV only).

        Uses .layers[i].keys/.values API (transformers 5.x DynamicCache).
        """
        for i in range(len(cache.layers)):
            layer = cache.layers[i]
            if layer.keys is not None:
                layer.keys = layer.keys.detach()
            if layer.values is not None:
                layer.values = layer.values.detach()
        return cache

    def _compute_mrope_position_ids(
        self,
        input_ids: Tensor,
        mm_token_type_ids: Tensor,
        image_grid_thw: Tensor,
        spatial_merge_size: int,
        pad_masks: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Compute 3D M-RoPE position IDs for vision + text tokens.

        Simplified version of Qwen3_5Model.get_rope_index() that doesn't
        require a full Qwen3_5Model instance.

        Padding tokens (pad_masks=False) are excluded from position counting:
        their slots in position_ids stay at 0 and they don't advance current_pos.
        Otherwise prefix_max_pos would include padding contributions, which makes
        expert RoPE positions sample-dependent (varies with active prefix length)
        and shifts inference relative to training.
        """
        import itertools

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Padding tokens get a sentinel modality so groupby splits them out and
        # the loop below skips them entirely.
        PAD_TYPE = -1
        if pad_masks is not None:
            mm_token_type_ids = mm_token_type_ids.clone()
            mm_token_type_ids[~pad_masks] = PAD_TYPE

        position_ids = torch.zeros(3, batch_size, seq_len, dtype=torch.long, device=device)
        mrope_deltas = []

        grid_iter = iter(image_grid_thw) if image_grid_thw is not None else None

        for batch_idx in range(batch_size):
            token_types = mm_token_type_ids[batch_idx]

            # Group consecutive tokens by type
            groups = []
            for key, group in itertools.groupby(enumerate(token_types.tolist()), lambda x: x[1]):
                group_list = list(group)
                groups.append((key, group_list[0][0], group_list[-1][0] + 1))

            current_pos = 0
            max_pos = -1
            for modality_type, start_idx, end_idx in groups:
                if modality_type == PAD_TYPE:
                    # Padding: keep position_ids at 0 (attention_mask masks these out
                    # anyway); do not advance current_pos.
                    continue
                if modality_type == 0:
                    # Text: 1D positions (all 3 dims same)
                    text_len = end_idx - start_idx
                    pos = torch.arange(text_len, device=device).view(1, -1).expand(3, -1) + current_pos
                    position_ids[:, batch_idx, start_idx:end_idx] = pos
                    current_pos += text_len
                    max_pos = max(max_pos, current_pos - 1)
                else:
                    # Vision: 3D positions
                    grid_thw = next(grid_iter)
                    llm_grid_h = grid_thw[1].item() // spatial_merge_size
                    llm_grid_w = grid_thw[2].item() // spatial_merge_size
                    llm_grid_t = grid_thw[0].item()
                    seq_length = llm_grid_h * llm_grid_w * llm_grid_t

                    pos_w = torch.arange(current_pos, current_pos + llm_grid_w, device=device).repeat(llm_grid_h * llm_grid_t)
                    pos_h = torch.arange(current_pos, current_pos + llm_grid_h, device=device).repeat_interleave(llm_grid_w * llm_grid_t)
                    pos_t = torch.full((seq_length,), current_pos, device=device, dtype=torch.long)

                    pos = torch.stack([pos_t, pos_h, pos_w], dim=0)
                    position_ids[:, batch_idx, start_idx:end_idx] = pos
                    current_pos += max(llm_grid_h, llm_grid_w)
                    max_pos = max(max_pos, current_pos - 1)

            if max_pos >= 0:
                mrope_deltas.append(torch.tensor(max_pos + 1 - seq_len, device=device))
            else:
                mrope_deltas.append(torch.tensor(0, device=device))

        mrope_deltas = torch.stack(mrope_deltas).unsqueeze(1)
        return position_ids, mrope_deltas

    def embed_prefix(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_att_masks: Tensor | None = None,
        fast_action_indicator: Tensor | None = None,
        subtask_indicator: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, int | None]:
        """Embeds prefix inputs (images, optional trajectory tokens, language)
        into a single sequence.

        This method uses a vision model (SigLIP) for images and an embedding layer
        for language tokens, combining them into a unified embedding sequence for
        the transformer.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            lang_att_masks: Optional attention masks for language tokens.
            fast_action_indicator: Optional tensor indicating fast action token positions.
            subtask_indicator: Optional tensor indicating subtask token positions.

        Returns:
            A tuple containing:
            - embs: The combined prefix embeddings.
            - pad_masks: The corresponding padding masks.
            - att_masks: The corresponding attention masks.
            - fast_action_indicator: Updated fast action indicator with image/traj prefix padding.
            - subtask_indicator: Updated subtask indicator with image/traj prefix padding.
            - traj_token_start_idx: The starting index of trajectory tokens, if they exist.
        """
        num_images = len(images)

        # Stack images and masks for batch processing
        images_stacked = torch.stack(images, dim=0)  # (num_images, bsize, ...)
        img_masks_stacked = torch.stack(img_masks, dim=0)  # (num_images, bsize)

        # Batch embed all images at once
        # Reshape to (num_images * bsize, ...)
        orig_shape = images_stacked.shape
        images_flat = images_stacked.reshape(-1, *orig_shape[2:])
        img_masks_flat = None
        if img_masks_stacked.ndim == 3:
            img_masks_flat = img_masks_stacked.reshape(-1, img_masks_stacked.shape[-1])
        img_embs_flat = self.paligemma_with_expert.embed_image(images_flat, image_attention_mask=img_masks_flat)

        # Reshape back to (num_images, bsize, num_img_embs, emb_dim)
        bsize = orig_shape[1]
        img_embs = img_embs_flat.reshape(num_images, bsize, *img_embs_flat.shape[1:])

        # Normalize image embeddings
        img_emb_dim = img_embs.shape[-1]
        num_img_embs = img_embs.shape[2]

        if img_masks_stacked.ndim == 3:
            current_img_masks = img_masks_stacked[:, :, -1]
        else:
            current_img_masks = img_masks_stacked

        # Expand masks: (num_images, bsize) -> (num_images, bsize, num_img_embs)
        img_masks_expanded = current_img_masks[:, :, None].expand(num_images, bsize, num_img_embs)

        # Reshape to (bsize, num_images * num_img_embs, emb_dim)
        img_embs_concat = img_embs.transpose(0, 1).reshape(bsize, num_images * num_img_embs, img_emb_dim)
        img_masks_concat = img_masks_expanded.transpose(0, 1).reshape(bsize, num_images * num_img_embs)

        # Process language embeddings
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        lang_emb = lang_emb.to(dtype=img_embs_concat.dtype)
        lang_emb = self._scatter_proprioception_anchor(
            lang_tokens,
            lang_emb,
            proprioception,
            agent_pos_mask,
            proprioception_present,
        )

        num_lang_embs = lang_emb.shape[1]
        num_img_embs_total = num_images * num_img_embs
        num_traj_embs = self.num_traj_tokens if self.enable_learnable_traj_token else 0
        total_seq_len = num_img_embs_total + num_traj_embs + num_lang_embs

        # Pre-allocate final tensors
        embs = torch.empty(bsize, total_seq_len, img_emb_dim, dtype=img_embs_concat.dtype, device=img_embs_concat.device)
        pad_masks = torch.empty(bsize, total_seq_len, dtype=torch.bool, device=img_embs_concat.device)

        # Fill pre-allocated tensors
        embs[:, :num_img_embs_total] = img_embs_concat
        pad_masks[:, :num_img_embs_total] = img_masks_concat

        num_prefix_embs = num_img_embs_total
        traj_token_start_idx = None
        if self.enable_learnable_traj_token:
            traj_emb = self.traj_token[None].repeat(bsize, 1, 1)
            embs[:, num_img_embs_total : num_img_embs_total + num_traj_embs] = traj_emb
            pad_masks[:, num_img_embs_total : num_img_embs_total + num_traj_embs] = torch.ones(
                bsize, self.num_traj_tokens, dtype=torch.bool, device=traj_emb.device
            )
            traj_token_start_idx = num_prefix_embs
            num_prefix_embs += num_traj_embs

        embs[:, num_img_embs_total + num_traj_embs :] = lang_emb
        pad_masks[:, num_img_embs_total + num_traj_embs :] = lang_masks

        # Create attention masks (all zeros for full attention between image and language)
        att_masks = torch.zeros(total_seq_len, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, total_seq_len).clone()
        if lang_att_masks is not None:
            att_masks[:, -num_lang_embs:] = lang_att_masks
        if fast_action_indicator is not None:
            fast_action_indicator = F.pad(fast_action_indicator, (num_prefix_embs, 0))
        if subtask_indicator is not None:
            subtask_indicator = F.pad(subtask_indicator, (num_prefix_embs, 0))

        return embs, pad_masks, att_masks, fast_action_indicator, subtask_indicator, traj_token_start_idx

    def embed_suffix(self, noisy_actions: Tensor, timestep: Tensor, emb_ids: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Embeds suffix inputs (noisy actions and timestep) for the
        transformer.

        Args:
            noisy_actions: The noisy action tensor at the current timestep.
            timestep: The current timestep value.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.

        Returns:
            A tuple containing:
            - embs: The combined suffix embeddings.
            - pad_masks: The corresponding padding masks.
            - att_masks: The corresponding attention masks.
            - adarms_cond: The conditioning vector for AdaRMSNorm.
        """
        embs = []
        pad_masks = []
        att_masks = []

        if noisy_actions.ndim != 3:
            raise ValueError(f'noisy_actions must have shape [B, T, D], got {tuple(noisy_actions.shape)}')
        if noisy_actions.shape[1] != self.n_action_steps:
            raise ValueError(
                f'noisy_actions temporal length {noisy_actions.shape[1]} does not match '
                f'policy.n_action_steps={self.n_action_steps}'
            )

        action_emb = self.action_in_proj(noisy_actions, emb_ids=emb_ids)
        bsize, action_steps = action_emb.shape[:2]
        dtype = action_emb.dtype
        device = action_emb.device

        # `timestep` is [B] for standard flow matching, or [B, T] for TTRTC, where the
        # frozen-prefix tokens carry t=0 ("this slot is already decided") and the postfix
        # tokens carry the sampled t. The per-token form only works because the suffix is
        # exactly `n_action_steps` tokens (asserted above), so cond and tokens line up 1:1.
        per_token_time = timestep.ndim == 2
        if per_token_time:
            if tuple(timestep.shape) != (bsize, action_steps):
                raise ValueError(
                    f'per-token timestep must have shape [B, T] = {(bsize, action_steps)}, '
                    f'got {tuple(timestep.shape)}'
                )
            flat_time = timestep.reshape(-1)
        elif timestep.ndim == 1:
            flat_time = timestep
        else:
            raise ValueError(f'timestep must be [B] or [B, T], got {tuple(timestep.shape)}')

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(flat_time, self.proj_width, min_period=4e-3, max_period=4.0, device=device)
        time_emb = time_emb.type(dtype=dtype)

        # Time MLP (for adaRMS). Pointwise, so running it on the flattened [B*T] axis is
        # numerically identical to running it per token after the reshape.
        time_emb = self.time_mlp_in(time_emb)
        time_emb = F.silu(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = F.silu(time_emb)
        if per_token_time:
            time_emb = time_emb.reshape(bsize, action_steps, -1)
        adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_emb)

        action_time_mask = torch.ones(bsize, action_steps, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image and language do not attend to action tokens
        att_masks += [1] + ([0] * (action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def sample_noise(self, shape: tuple, device: torch.device | str) -> Tensor:
        """Samples noise from a standard normal distribution.

        Args:
            shape: The desired shape of the noise tensor.
            device: The device to create the tensor on.

        Returns:
            A tensor of the given shape filled with noise.
        """
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def set_action_denoise_mask(self, mask: Tensor | list[bool] | None) -> None:
        """Restrict flow denoising to configured action dimensions.

        The mask is applied to both the initial noise and every Euler update so
        padded action dimensions stay zero exactly as they did during masked-noise
        training. It is runtime-only and is not serialized into model config.
        """
        if mask is None:
            self._action_denoise_mask = None
            return

        resolved = torch.as_tensor(mask, dtype=torch.bool).reshape(-1)
        if resolved.numel() != self.max_action_dim:
            raise ValueError(
                f'action denoise mask must have {self.max_action_dim} entries, '
                f'got {resolved.numel()}'
            )
        self._action_denoise_mask = resolved

    def _mask_action_denoise_dims(self, value: Tensor) -> Tensor:
        mask = self._action_denoise_mask
        if mask is None:
            return value
        mask = mask.to(device=value.device).view(*([1] * (value.ndim - 1)), -1)
        return torch.where(mask, value, torch.zeros((), dtype=value.dtype, device=value.device))

    def _prepare_action_noise(
        self,
        noise: Tensor | None,
        batch_size: int,
        device: torch.device | str,
    ) -> Tensor:
        if noise is None:
            noise = self.sample_noise((batch_size, self.n_action_steps, self.max_action_dim), device)
        else:
            if noise.ndim != 3:
                raise ValueError(f'noise must have shape [B, T, D], got {tuple(noise.shape)}')
            if noise.shape[0] != batch_size:
                raise ValueError(f'noise batch size {noise.shape[0]} does not match input batch size {batch_size}')
            if noise.shape[1] != self.n_action_steps:
                raise ValueError(
                    f'noise temporal length {noise.shape[1]} does not match policy.n_action_steps={self.n_action_steps}'
                )
            if noise.shape[2] != self.max_action_dim:
                raise ValueError(
                    f'noise action dimension {noise.shape[2]} does not match policy.max_action_dim={self.max_action_dim}'
                )
        return self._mask_action_denoise_dims(noise)

    @torch.no_grad()
    def sample_actions(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        emb_ids: Tensor,
        enable_2d_traj_output: bool = False,
        noise: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
        state: Tensor | None = None,
        prev_action_chunk: Tensor | None = None,
        ttrtc_prefix_steps: int = 0,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Performs a full inference pass to sample actions.

        This involves creating a key-value cache from image and language inputs,
        then iteratively denoising an initial noise tensor to produce the final actions.

        Args:
            images: A list of image tensors.
            img_masks: A list of boolean masks for the images.
            lang_tokens: Language input token IDs.
            lang_masks: Boolean mask for the language tokens.
            emb_ids: Embodiment IDs for selecting embodiment-specific layers.
            enable_2d_traj_output: If True, also returns trajectory predictions.
            noise: Optional initial noise tensor. If None, it will be sampled.

        Returns:
            - The sampled action tensor.
            - If `enable_2d_traj_output` is True, a tuple of the action tensor
              and the trajectory prediction tensor.
        """
        # Route to Qwen3-VL dual-stream inference
        if self.vlm_type == 'qwen3_vl' and getattr(self, 'has_action_expert', False):
            return self._sample_actions_qwen3_vl(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks, state_memory=state_memory,
                state_memory_masks=state_memory_masks)

        # Route to Qwen2.5-VL dual-stream inference
        if self.vlm_type == 'qwen2_5_vl' and getattr(self, 'has_action_expert', False):
            return self._sample_actions_qwen2_5_vl(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks, state_memory=state_memory,
                state_memory_masks=state_memory_masks)

        # Route to Qwen3.5 cross-DiT inference
        if (
            self.vlm_type == 'qwen3_5'
            and getattr(self, 'has_action_expert', False)
            and getattr(self, 'expert_type', 'dual_stream') == 'cross_dit'
        ):
            return self._sample_actions_qwen3_5_crossdit(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks)

        # Route to Qwen3.5 RoboManip-style inference
        if (
            self.vlm_type == 'qwen3_5'
            and getattr(self, 'has_action_expert', False)
            and getattr(self, 'expert_type', 'dual_stream') == 'robomanip'
        ):
            return self._sample_actions_qwen3_5_robomanip(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks, state=state)

        # Route to Qwen3.5 warm-started Gemma3-1B cross-attention expert inference
        if (
            self.vlm_type == 'qwen3_5'
            and getattr(self, 'has_action_expert', False)
            and getattr(self, 'expert_type', 'dual_stream') == 'gemma3'
        ):
            return self._sample_actions_qwen3_5_gemma3(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks)

        # Route to Qwen3.5 dual-stream inference
        if self.vlm_type == 'qwen3_5' and getattr(self, 'has_action_expert', False):
            return self._sample_actions_qwen3_5(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks, state_memory=state_memory,
                state_memory_masks=state_memory_masks)

        # Route to Gemma3 dual-stream inference (Phase 3)
        if self.vlm_type == 'gemma3' and getattr(self, 'has_action_expert', False):
            return self._sample_actions_gemma3(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks, state_memory=state_memory,
                state_memory_masks=state_memory_masks)

        # Route to Gemma4 dual-stream inference (Phase 3)
        if self.vlm_type == 'gemma4' and getattr(self, 'has_action_expert', False):
            return self._sample_actions_gemma4(
                images, img_masks, lang_tokens, lang_masks, emb_ids, noise=noise,
                lang_att_masks=lang_att_masks, state_memory=state_memory,
                state_memory_masks=state_memory_masks)

        bsize = images[0].shape[0]
        device = images[0].device

        noise = self._prepare_action_noise(noise, bsize, device)

        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, traj_token_start_idx = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        if state_pad_masks is not None and traj_token_start_idx is not None:
            traj_token_start_idx += state_pad_masks.shape[1]
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)

        # Compute image and language key value cache
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.use_cache,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        if enable_2d_traj_output:
            traj_out = prefix_out[:, traj_token_start_idx : traj_token_start_idx + self.num_traj_tokens, :]
            traj_out = traj_out.to(dtype=self.traj_out_proj.weight.dtype)
            traj_out, _ = self.traj_decoder(traj_out)
            traj_pred = self.traj_out_proj(traj_out)

        x_t = noise.to(dtype=self.action_in_proj.weight.dtype)
        dt = -1.0 / self.num_steps
        timesteps = torch.arange(1.0, -dt / 2, dt, dtype=torch.float32, device=device)

        # ---- TTRTC 推理侧硬 inpaint（与 TTRTC 训练的 clean-prefix 契约逐字对应）----
        # 训练时 `add_noise` 采样 d ~ U{0..rtc_max_delay}，把 actions[:d] 原样保留（噪声不加）
        # 并令这 d 个 token 的 flow 时间为 0，loss 只算 postfix。推理时必须做同样的事：
        # 前 d 步钉死成上一次 chunk 的对应帧、时间置 0，模型只需要"续写"后面的部分。
        # 不钉死 = 模型没见过的输入分布；只钉输入不置时间 = adaRMS 条件错位。两者都会静默劣化。
        horizon = int(x_t.shape[1])
        prefix_steps = max(0, min(int(ttrtc_prefix_steps or 0), horizon))
        use_ttrtc = prev_action_chunk is not None and prefix_steps > 0
        if use_ttrtc:
            prev_chunk = prev_action_chunk.to(device=x_t.device, dtype=x_t.dtype)
            if prev_chunk.ndim == 2:
                prev_chunk = prev_chunk[None, ...]
            if prev_chunk.shape[-2:] != x_t.shape[-2:]:
                raise ValueError(
                    f'prev_action_chunk must be [B, {horizon}, {x_t.shape[-1]}], '
                    f'got {tuple(prev_chunk.shape)}'
                )
            pin_mask = (torch.arange(horizon, device=x_t.device) < prefix_steps)[None, :, None]
            time_zero = pin_mask[..., 0].expand(bsize, horizon)

        for timestep in timesteps:
            if use_ttrtc:
                x_in = torch.where(pin_mask, prev_chunk, x_t)
                # per-token 时间：prefix 恒为 0（已是干净动作），postfix 走正常 flow 时间。
                # embed_suffix 支持 [B] 与 [B, T] 两种形态，这里用后者。
                t_in = torch.where(
                    time_zero,
                    torch.zeros((), dtype=torch.float32, device=device),
                    timestep,
                ).expand(bsize, horizon)
            else:
                x_in = x_t
                t_in = timestep.expand(bsize)
            v_t = self.denoise_step(
                prefix_pad_masks,
                past_key_values,
                x_in,
                t_in,
                emb_ids,
            )
            x_t = self._mask_action_denoise_dims(x_in + dt * v_t)
            if use_ttrtc:
                x_t = torch.where(pin_mask, prev_chunk, x_t)

        if enable_2d_traj_output:
            return x_t, traj_pred

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks: Tensor,
        past_key_values: dict,
        x_t: Tensor,
        timestep: Tensor,
        emb_ids: Tensor,
    ) -> Tensor:
        """Applies one denoising step to `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(x_t, timestep, emb_ids=emb_ids)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=self.use_cache,
            fill_kv_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.n_action_steps :]
        suffix_out = suffix_out.to(dtype=self.action_out_proj.weight.dtype)
        v_t = self.action_out_proj(suffix_out, emb_ids=emb_ids)
        return v_t

    @torch.no_grad()
    def init_lang_generation(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        lang_att_masks: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
        proprioception: Tensor | None = None,
        agent_pos_mask: Tensor | None = None,
        proprioception_present: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Prefill stage: build KV cache from images + existing language prefix and
        return logits for the next token.

        ``lang_att_masks`` (make_att_2d_masks semantics: 1 = start a new causal
        block, 0 = continue the bidirectional block) lets the prefill reproduce
        the training prefix-LM structure (e.g. Gemma4 keeps vision tokens
        bidirectional). Only the Gemma4 path consumes it today; the Qwen paths
        use their backbone's own causal LM mask and ignore it.

        Returns:
            A tuple containing:
            - next_logits: Logits for the next token prediction, shape (batch_size, vocab_size).
            - state: A dictionary containing the state for autoregressive generation:
                - 'past_key_values': The key-value cache from the transformer.
                - 'position_cursor': The current position in the sequence for position embeddings.
                - 'batch_size': The batch size.
                - 'device': The device of the tensors.
                - 'prefix_pad_masks': The padding mask for the prefix.
        """
        if self.vlm_type == 'qwen3_vl':
            return self._init_lang_generation_qwen3_vl(
                images, img_masks, lang_tokens, lang_masks, image_grid_thw,
                emb_ids=emb_ids, state_memory=state_memory, state_memory_masks=state_memory_masks)
        if self.vlm_type == 'qwen2_5_vl':
            return self._init_lang_generation_qwen2_5_vl(
                images, img_masks, lang_tokens, lang_masks, image_grid_thw,
                emb_ids=emb_ids, state_memory=state_memory, state_memory_masks=state_memory_masks)
        if self.vlm_type == 'qwen3_5':
            return self._init_lang_generation_qwen3_5(
                images, img_masks, lang_tokens, lang_masks, image_grid_thw,
                emb_ids=emb_ids, state_memory=state_memory, state_memory_masks=state_memory_masks)
        if self.vlm_type == 'gemma4':
            return self._init_lang_generation_gemma4(
                images, img_masks, lang_tokens, lang_masks, lang_att_masks,
                emb_ids=emb_ids, state_memory=state_memory, state_memory_masks=state_memory_masks)
        if self.vlm_type == 'gemma3' and self.state_input_mode == 'proprio_memory':
            if not getattr(self, 'has_action_expert', False):
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._init_lang_generation_gemma3_with_expert(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                lang_att_masks=lang_att_masks,
                emb_ids=emb_ids,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
            )

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # embed prefix (images + lang)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = self.embed_prefix(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks=lang_att_masks,
            proprioception=proprioception,
            agent_pos_mask=agent_pos_mask,
            proprioception_present=proprioception_present,
        )
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, _ = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = self._compute_1d_position_ids(prefix_pad_masks)

        # forward to build cache and get language hidden states
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
            adarms_cond=[None, None],
        )

        # Take the hidden state of the last valid language token per sample as
        # the condition for the next-token distribution
        lang_length = lang_masks.shape[-1]
        lang_region = prefix_out[:, -lang_length:, :]
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = lang_region[batch_indices, -1, :].unsqueeze(1)
        last_hidden = last_hidden.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
        next_logits = self.paligemma_with_expert.language_out_proj(last_hidden)[:, 0, :]

        # Next-token position (0-indexed) equals the number of consumed tokens so far
        position_cursor = torch.sum(prefix_pad_masks, dim=-1).to(dtype=torch.long)

        state = {
            'past_key_values': past_key_values,
            'position_cursor': position_cursor,
            'batch_size': bsize,
            'device': device,
            'prefix_pad_masks': prefix_pad_masks[:, None, :],
        }
        if self.state_input_mode == 'proprio_memory':
            state = self._attach_full_prefill_generation_state(
                state,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                lang_att_masks,
                image_grid_thw,
                emb_ids,
                state_memory,
                state_memory_masks,
            )
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_qwen3_5(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen3.5 prefill: encode image+text with KV cache, return next_logits."""
        from .qwen3_5_vlm import Qwen3_5VLMModel

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if self.state_input_mode == 'proprio_memory':
            if not getattr(self, 'has_action_expert', False):
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._init_lang_generation_qwen3_5_with_expert(
                images, img_masks, lang_tokens, lang_masks, image_grid_thw,
                emb_ids=emb_ids, state_memory=state_memory, state_memory_masks=state_memory_masks)

        # Prepare pixel_values (batch-major for masked_scatter)
        images_stacked = torch.stack(images, dim=1)  # (B, num_cameras, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3_5VLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        # Forward with cache enabled
        hidden_states, past_key_values = self.qwen3_5_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=lang_masks.long(),
            use_cache=True,
        )

        # Get logits for the last valid token per sample
        valid_lengths = lang_masks.sum(dim=-1).long()
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = hidden_states[batch_indices, valid_lengths - 1, :]
        next_logits = self.qwen3_5_vlm.language_out_proj(last_hidden)

        # M-RoPE correction: vision tokens advance positions by max(grid_h, grid_w)/merge
        # rather than by their token count. rope_deltas captures this offset.
        # Correct decode position = valid_lengths + rope_deltas.
        rope_deltas = getattr(self.qwen3_5_vlm.model.model, 'rope_deltas', None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.squeeze(1).to(device)  # (B,)
        else:
            rope_deltas = torch.zeros(bsize, dtype=torch.long, device=device)

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths + rope_deltas,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'qwen3_5',
        }
        if self.state_input_mode == 'proprio_memory':
            state = self._attach_full_prefill_generation_state(
                state,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                None,
                image_grid_thw,
                emb_ids,
                state_memory,
                state_memory_masks,
            )
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_qwen3_vl(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen3-VL prefill: encode image+text with standard DynamicCache."""
        from .qwen3_vl_vlm import Qwen3VLVLMModel

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if self.state_input_mode == 'proprio_memory':
            if not getattr(self, 'has_action_expert', False):
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._init_lang_generation_qwen3_vl_with_expert(
                images, img_masks, lang_tokens, lang_masks, image_grid_thw,
                emb_ids=emb_ids, state_memory=state_memory, state_memory_masks=state_memory_masks)

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3VLVLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        hidden_states, past_key_values = self.qwen3_vl_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=lang_masks.long(),
            use_cache=True,
        )

        valid_lengths = lang_masks.sum(dim=-1).long()
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = hidden_states[batch_indices, valid_lengths - 1, :]
        next_logits = self.qwen3_vl_vlm.language_out_proj(last_hidden)

        rope_deltas = getattr(self.qwen3_vl_vlm.model.model, 'rope_deltas', None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.squeeze(1).to(device)
        else:
            rope_deltas = torch.zeros(bsize, dtype=torch.long, device=device)

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths + rope_deltas,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'qwen3_vl',
        }
        if self.state_input_mode == 'proprio_memory':
            state = self._attach_full_prefill_generation_state(
                state,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                None,
                image_grid_thw,
                emb_ids,
                state_memory,
                state_memory_masks,
            )
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_qwen3_5_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen3.5 expert-backed FAST prefill with continuous proprio prefix tokens."""
        from .qwen3_5_vlm import Qwen3_5VLMModel

        model = self.qwen3_5_with_expert
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3_5VLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)
        image_token_id = model.vlm_config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        prefix_att_masks = torch.ones(bsize, lang_masks.shape[1], dtype=torch.long, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_mask = torch.where(
            prefix_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1
        spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
        position_ids, _ = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
            pad_masks=lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_out, past_key_values = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

        valid_lengths = prefix_pad_masks.sum(dim=-1).long()
        last_valid_indices = self._last_valid_indices(prefix_pad_masks)
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = prefix_out[batch_indices, last_valid_indices, :]
        next_logits = model.language_out_proj(last_hidden.to(dtype=model.lm_head.weight.dtype))

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'qwen3_5_expert',
        }
        state = self._attach_full_prefill_generation_state(
            state,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            None,
            image_grid_thw,
            emb_ids,
            state_memory,
            state_memory_masks,
        )
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_qwen3_vl_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen3-VL expert-backed FAST prefill with continuous proprio prefix tokens."""
        from .qwen3_vl_vlm import Qwen3VLVLMModel

        model = self.qwen3_vl_with_expert
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen3VLVLMModel.images_to_pixel_values(images_flat)
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs, deepstack_features = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)
        image_token_id = model.vlm_config.image_token_id
        visual_pos_masks = lang_tokens == image_token_id
        special_image_mask = visual_pos_masks.unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        prefix_att_masks = torch.ones(bsize, lang_masks.shape[1], dtype=torch.long, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        if state_pad_masks is not None:
            visual_pos_masks = torch.cat([
                torch.zeros(
                    bsize,
                    state_pad_masks.shape[1],
                    dtype=visual_pos_masks.dtype,
                    device=visual_pos_masks.device,
                ),
                visual_pos_masks,
            ], dim=1)
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_mask = torch.where(
            prefix_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
        mm_token_type_ids[lang_tokens == image_token_id] = 1
        spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
        position_ids, _ = self._compute_mrope_position_ids(
            lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
            pad_masks=lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_out, past_key_values = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
            visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_features)

        valid_lengths = prefix_pad_masks.sum(dim=-1).long()
        last_valid_indices = self._last_valid_indices(prefix_pad_masks)
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = prefix_out[batch_indices, last_valid_indices, :]
        next_logits = model.language_out_proj(last_hidden.to(dtype=model.lm_head.weight.dtype))

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'qwen3_vl_expert',
        }
        state = self._attach_full_prefill_generation_state(
            state,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            None,
            image_grid_thw,
            emb_ids,
            state_memory,
            state_memory_masks,
        )
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_qwen2_5_vl_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen2.5-VL expert-backed FAST prefill with continuous proprio prefix tokens."""
        from .qwen2_5_vl_vlm import Qwen2_5VLVLMModel

        model = self.qwen2_5_vl_with_expert
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen2_5VLVLMModel.images_to_pixel_values(
            images_flat, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        )
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        img_embs = model.embed_image(pixel_values, image_grid_thw)
        lang_embs = model.embed_language_tokens(lang_tokens)
        image_token_id = model.vlm_config.image_token_id
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        prefix_att_masks = torch.ones(bsize, lang_masks.shape[1], dtype=torch.long, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_mask = torch.where(
            prefix_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)

        if model.use_1d_rope:
            position_ids = self._compute_1d_position_ids(lang_masks)
        else:
            mm_token_type_ids = torch.zeros_like(lang_tokens, dtype=torch.int)
            mm_token_type_ids[lang_tokens == image_token_id] = 1
            spatial_merge_size = model.vlm_config.vision_config.spatial_merge_size
            position_ids, _ = self._compute_mrope_position_ids(
                lang_tokens, mm_token_type_ids, image_grid_thw, spatial_merge_size,
                pad_masks=lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_out, past_key_values = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

        valid_lengths = prefix_pad_masks.sum(dim=-1).long()
        last_valid_indices = self._last_valid_indices(prefix_pad_masks)
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = prefix_out[batch_indices, last_valid_indices, :]
        next_logits = model.language_out_proj(last_hidden.to(dtype=model.lm_head.weight.dtype))

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'qwen2_5_vl_expert',
        }
        state = self._attach_full_prefill_generation_state(
            state,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            None,
            image_grid_thw,
            emb_ids,
            state_memory,
            state_memory_masks,
        )
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_gemma3_with_expert(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        lang_att_masks: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[Tensor, dict]:
        """Gemma3 expert-backed FAST prefill with continuous proprio prefix tokens."""
        model = self.gemma3_with_expert
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        image_token_id = model.vlm_config.image_token_id

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        img_embs = model.embed_image(images_flat)
        img_embs = img_embs.reshape(-1, img_embs.shape[-1])
        lang_embs = model.embed_language_tokens(lang_tokens)
        special_image_mask = (lang_tokens == image_token_id).unsqueeze(-1).expand_as(lang_embs)
        prefix_embs = lang_embs.masked_scatter(special_image_mask, img_embs.to(lang_embs.dtype))

        if lang_att_masks is None:
            prefix_att_masks = torch.ones(bsize, lang_masks.shape[1], dtype=torch.long, device=device)
        else:
            prefix_att_masks = lang_att_masks.to(dtype=torch.long, device=device)
        prefix_embs, prefix_pad_masks, prefix_att_masks, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_mask = torch.where(
            prefix_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min
        ).to(dtype=prefix_embs.dtype)
        position_ids = self._compute_1d_position_ids(lang_masks)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_out, past_key_values = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask)

        valid_lengths = prefix_pad_masks.sum(dim=-1).long()
        last_valid_indices = self._last_valid_indices(prefix_pad_masks)
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = prefix_out[batch_indices, last_valid_indices, :]
        next_logits = model.language_out_proj(last_hidden.to(dtype=model.lm_head.weight.dtype))

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'gemma3_expert',
        }
        state = self._attach_full_prefill_generation_state(
            state,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            lang_att_masks,
            None,
            emb_ids,
            state_memory,
            state_memory_masks,
        )
        return next_logits, state

    @torch.no_grad()
    def next_lang_logits(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Decode step: given the previously generated token, return logits for
        the next token and update the KV cache/state.

        Args:
            state: A dictionary containing the state from a previous generation step.
                   See `init_lang_generation` for details on its contents.
            input_token: The token generated in the previous step, shape (batch_size, 1).

        Returns:
            A tuple containing:
            - next_logits: Logits for the next token prediction, shape (batch_size, vocab_size).
            - state: The updated state dictionary for the next generation step.
        """
        if state.get('full_prefill_decode'):
            return self._next_lang_logits_full_prefill(state, input_token)
        if state.get('vlm_type') == 'qwen3_vl':
            return self._next_lang_logits_qwen3_vl(state, input_token)
        if state.get('vlm_type') == 'qwen2_5_vl':
            return self._next_lang_logits_qwen2_5_vl(state, input_token)
        if state.get('vlm_type') == 'qwen3_5':
            return self._next_lang_logits_qwen3_5(state, input_token)
        if state.get('vlm_type') == 'gemma4':
            return self._next_lang_logits_gemma4(state, input_token)

        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']
        bsize = state['batch_size']
        device = state['device']
        prefix_pad_masks = state['prefix_pad_masks']

        # Prepare single-step language token embedding
        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        token_emb = self.paligemma_with_expert.embed_language_tokens(input_token)
        # Match scaling used in embed_prefix to keep prefill/decoding consistent
        lang_emb_dim = token_emb.shape[-1]
        token_emb = token_emb * math.sqrt(lang_emb_dim)

        # Build attention mask: one-step query attends to all history and itself
        att_2d_masks = torch.cat([prefix_pad_masks, torch.ones((bsize, 1, 1), dtype=torch.bool, device=device)], dim=2)

        # Position id is the current cursor (0-indexed)
        position_ids = position_cursor.view(bsize, 1).to(device)

        (lang_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[token_emb, None],
            use_cache=True,
            fill_kv_cache=False,
            adarms_cond=[None, None],
            auto_regression_inference_mode=True,
        )

        # Single step only, shape: (b, 1, d)
        lang_out = lang_out[:, -1:, :]
        lang_out = lang_out.to(dtype=self.paligemma_with_expert.lm_head.weight.dtype)
        next_logits = self.paligemma_with_expert.language_out_proj(lang_out)[:, 0, :]

        # Advance the cursor; KV cache is persistently appended inside the model
        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        state['prefix_pad_masks'] = att_2d_masks
        return next_logits, state

    @torch.no_grad()
    def _next_lang_logits_qwen3_5(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen3.5 decode step: single-token forward with hybrid cache."""
        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']

        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        hidden_states, past_key_values = self.qwen3_5_vlm(
            input_ids=input_token,
            attention_mask=None,
            position_ids=position_cursor.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_logits = self.qwen3_5_vlm.language_out_proj(hidden_states[:, -1, :])

        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        return next_logits, state

    @torch.no_grad()
    def _next_lang_logits_qwen3_vl(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen3-VL decode step: standard KV cache incremental inference."""
        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']

        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        hidden_states, past_key_values = self.qwen3_vl_vlm(
            input_ids=input_token,
            attention_mask=None,
            position_ids=position_cursor.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_logits = self.qwen3_vl_vlm.language_out_proj(hidden_states[:, -1, :])

        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_qwen2_5_vl(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        image_grid_thw: Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen2.5-VL prefill: encode image+text with standard DynamicCache."""
        from .qwen2_5_vl_vlm import Qwen2_5VLVLMModel

        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        if self.state_input_mode == 'proprio_memory':
            if not getattr(self, 'has_action_expert', False):
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._init_lang_generation_qwen2_5_vl_with_expert(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                image_grid_thw,
                emb_ids=emb_ids,
                state_memory=state_memory,
                state_memory_masks=state_memory_masks,
            )

        images_stacked = torch.stack(images, dim=1)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, computed_grid_thw = Qwen2_5VLVLMModel.images_to_pixel_values(
            images_flat, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        )
        if image_grid_thw is None:
            image_grid_thw = computed_grid_thw
        elif image_grid_thw.dim() == 3:
            image_grid_thw = image_grid_thw.reshape(-1, 3)

        hidden_states, past_key_values = self.qwen2_5_vl_vlm(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            attention_mask=lang_masks.long(),
            use_cache=True,
        )

        valid_lengths = lang_masks.sum(dim=-1).long()
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = hidden_states[batch_indices, valid_lengths - 1, :]
        next_logits = self.qwen2_5_vl_vlm.language_out_proj(last_hidden)

        rope_deltas = getattr(self.qwen2_5_vl_vlm.model.model, 'rope_deltas', None)
        if rope_deltas is not None:
            rope_deltas = rope_deltas.squeeze(1).to(device)
        else:
            rope_deltas = torch.zeros(bsize, dtype=torch.long, device=device)

        state = {
            'past_key_values': past_key_values,
            'position_cursor': valid_lengths + rope_deltas,
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'qwen2_5_vl',
        }
        if self.state_input_mode == 'proprio_memory':
            state = self._attach_full_prefill_generation_state(
                state,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                None,
                image_grid_thw,
                emb_ids,
                state_memory,
                state_memory_masks,
            )
        return next_logits, state

    @torch.no_grad()
    def _next_lang_logits_qwen2_5_vl(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Qwen2.5-VL decode step: standard KV cache incremental inference."""
        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']

        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)

        hidden_states, past_key_values = self.qwen2_5_vl_vlm(
            input_ids=input_token,
            attention_mask=None,
            position_ids=position_cursor.unsqueeze(1),
            past_key_values=past_key_values,
            use_cache=True,
        )

        next_logits = self.qwen2_5_vl_vlm.language_out_proj(hidden_states[:, -1, :])

        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_gemma4(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        lang_att_masks: torch.Tensor | None = None,
        emb_ids: Tensor | None = None,
        state_memory: Tensor | None = None,
        state_memory_masks: Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """Gemma4 FAST prefill: encode image+text, build the VLM KV cache + the
        cross-layer shared-KV dict, and return logits for the first action token.

        The prefix attention structure follows training: with ``lang_att_masks``
        supplied (make_att_2d_masks semantics) it is reproduced bit-exactly; when
        omitted we fall back to vision-bidirectional + text-causal (the
        ``prefix_lm_text=False`` training layout) by treating every image-token
        position as a single bidirectional block. Sliding-window restriction is
        applied inside ``forward_prefix_only`` per layer type. Decoding is pure
        VLM stream — the flow-matching expert is not involved.

        When the policy has NO action expert (pure-FAST training,
        has_action_expert=False), the backbone is ``gemma4_vlm`` (Gemma4VLMModel)
        which has its own KV-cache forward — route to the vlm-only path instead.
        """
        if not getattr(self, 'has_action_expert', False):
            if self.state_input_mode == 'proprio_memory':
                raise NotImplementedError("state_input_mode='proprio_memory' requires an expert-backed VLA path")
            return self._init_lang_generation_gemma4_vlm(images, img_masks, lang_tokens, lang_masks)

        model = self.gemma4_with_expert
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        image_token_id = model.vlm_config.image_token_id

        # Prefix embeddings (vision + language) — shared with flow inference.
        prefix_embs = self._gemma4_embed_prefix(model, images, img_masks, lang_tokens, image_token_id)

        # PLE compute. Prefill normally has no FAST tokens, but decode steps
        # still remap generated tail FAST ids below.
        ple_remap_mask = self._gemma4_prefix_ple_remap_mask(
            lang_tokens,
            image_token_id,
        ) | self._gemma4_tail_fast_token_mask(lang_tokens)
        per_layer_inputs = model.compute_per_layer_inputs(
            lang_tokens, prefix_embs, ple_remap_mask=ple_remap_mask,
        )

        position_ids = self._compute_1d_position_ids(lang_masks)  # (B, P)

        # Prefix attention mask (+ padding), per make_att_2d_masks semantics
        # (a `1` opens a new causal block; `0` stays in the bidirectional block).
        prefix_len = lang_masks.shape[1]
        if lang_att_masks is not None:
            prefix_att_masks_1d = lang_att_masks.to(dtype=torch.long, device=device)
        else:
            # Fallback: vision tokens bidirectional (0), all other tokens causal (1).
            prefix_att_masks_1d = (lang_tokens != image_token_id).to(dtype=torch.long, device=device)

        prefix_embs, prefix_pad_masks, prefix_att_masks_1d, _, _, state_pad_masks = self._prepend_proprio_prefix_tokens(
            prefix_embs,
            lang_masks,
            prefix_att_masks_1d,
            state_memory,
            state_memory_masks,
            emb_ids,
        )
        if state_pad_masks is not None:
            state_len = state_pad_masks.shape[1]
            state_per_layer_inputs = self._gemma4_continuous_per_layer_inputs(
                model,
                prefix_embs[:, :state_len, :],
            )
            if state_per_layer_inputs is not None:
                per_layer_inputs = torch.cat([state_per_layer_inputs, per_layer_inputs], dim=1)
        position_ids = self._prepend_proprio_position_ids(position_ids, state_pad_masks)

        prefix_att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks_1d)
        prefix_att_mask = torch.where(
            prefix_att_2d.unsqueeze(1), 0.0, torch.finfo(prefix_embs.dtype).min,
        ).to(dtype=prefix_embs.dtype)

        # VLM prefill: cache K/V for layers 0-23 + shared_kv_states for 24-41.
        prefix_out, cached_vlm_states, cached_shared_kv = model.forward_prefix_only(
            prefix_embs, position_ids=position_ids, attention_mask=prefix_att_mask,
            per_layer_inputs=per_layer_inputs,
        )

        # Logits for the first generated token: last valid prefix token per sample.
        valid_lengths = prefix_pad_masks.sum(dim=-1).long()
        last_valid_indices = self._last_valid_indices(prefix_pad_masks)
        batch_indices = torch.arange(bsize, device=device)
        last_hidden = prefix_out[batch_indices, last_valid_indices, :]
        last_hidden = last_hidden.to(dtype=model.lm_head.weight.dtype)
        next_logits = model.language_out_proj(last_hidden)

        state = {
            'past_key_values': cached_vlm_states,
            'shared_kv_states': cached_shared_kv,
            'position_cursor': valid_lengths,        # next-token RoPE position per sample
            'key_positions': position_ids,           # (B, P) RoPE positions of cached keys
            'key_pad_mask': prefix_pad_masks.bool(),       # (B, P) valid-key mask
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'gemma4',
        }
        if self.state_input_mode == 'proprio_memory':
            state = self._attach_full_prefill_generation_state(
                state,
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                lang_att_masks,
                None,
                emb_ids,
                state_memory,
                state_memory_masks,
            )
        return next_logits, state

    @torch.no_grad()
    def _next_lang_logits_gemma4(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Gemma4 FAST decode step: embed the new token, append its K/V onto the
        VLM cache + shared-KV dict, and return logits for the next token."""
        if not getattr(self, 'has_action_expert', False):
            return self._next_lang_logits_gemma4_vlm(state, input_token)

        model = self.gemma4_with_expert
        past_cache = state['past_key_values']
        past_shared_kv = state['shared_kv_states']
        position_cursor = state['position_cursor']   # (B,)
        key_positions = state['key_positions']       # (B, K)
        key_pad_mask = state['key_pad_mask']         # (B, K)
        bsize = state['batch_size']
        device = state['device']

        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)   # (B, 1)

        # ScaledWordEmbedding already applies the sqrt(hidden) scaling.
        token_emb = model.embed_language_tokens(input_token)  # (B, 1, hidden)
        # PLE for the new token. In tail-vocab mode FAST ids are inside the
        # base vocab, so explicitly remap them to PAD for PLE.
        per_layer_inputs = model.compute_per_layer_inputs(
            input_token,
            token_emb,
            ple_remap_mask=self._gemma4_tail_fast_token_mask(input_token),
        )

        pos_new = position_cursor.view(bsize, 1).to(device)
        vlm_out, past_cache, shared_kv_states = model.forward_vlm_decode(
            token_emb, pos_new, past_cache, past_shared_kv,
            key_positions, key_pad_mask, per_layer_inputs=per_layer_inputs,
        )

        last_hidden = vlm_out[:, -1, :].to(dtype=model.lm_head.weight.dtype)
        next_logits = model.language_out_proj(last_hidden)

        new_valid = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        state['past_key_values'] = past_cache
        state['shared_kv_states'] = shared_kv_states
        state['key_positions'] = torch.cat([key_positions, pos_new], dim=1)
        state['key_pad_mask'] = torch.cat([key_pad_mask, new_valid], dim=1)
        state['position_cursor'] = position_cursor + 1
        return next_logits, state

    @torch.no_grad()
    def _init_lang_generation_gemma4_vlm(
        self,
        images: list[Tensor],
        img_masks: list[Tensor],
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Pure-VLM Gemma4 FAST prefill (has_action_expert=False -> gemma4_vlm).

        Mirrors the training forward ``_forward_gemma4``: the inner Gemma4Model
        handles vision encoding, masked_scatter, image-bidirectional masking,
        sliding/full attention, PLE and KV cross-layer share internally, and
        exposes a native KV cache for incremental decode (same pattern as the
        Qwen vlm paths).

        We TRIM the prompt to its valid length before prefill so the KV cache
        holds only real tokens. Otherwise the trailing right-padding would be
        cached and the incremental decode (which passes no padding mask) would
        attend to that padding KV and corrupt every step. B=1 rollout only.
        """
        from .gemma4_vlm import Gemma4VLMModel

        model = self.gemma4_vlm
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device
        assert bsize == 1, 'gemma4 pure-VLM FAST decode currently supports batch_size=1 (rollout).'

        valid_len = int(lang_masks[0].sum().item())
        lang_tokens = lang_tokens[:, :valid_len]  # drop trailing right-padding

        images_stacked = torch.stack(images, dim=1)  # (B, N_cam, C, H, W)
        images_flat = images_stacked.reshape(-1, *images_stacked.shape[2:])
        pixel_values, image_position_ids = Gemma4VLMModel.images_to_pixel_values(images_flat)

        hidden_states, past_key_values = model(
            input_ids=lang_tokens,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            attention_mask=torch.ones_like(lang_tokens),
            use_cache=True,
        )

        last_hidden = hidden_states[:, -1, :]  # last valid prompt token (position valid_len-1)
        next_logits = model.language_out_proj(last_hidden.to(model.lm_head.weight.dtype))

        position_cursor = torch.full((bsize,), valid_len, dtype=torch.long, device=device)
        state = {
            'past_key_values': past_key_values,
            'position_cursor': position_cursor,   # next-token 1D RoPE position
            'batch_size': bsize,
            'device': device,
            'vlm_type': 'gemma4',
        }
        return next_logits, state

    @torch.no_grad()
    def _next_lang_logits_gemma4_vlm(
        self,
        state: dict,
        input_token: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """Pure-VLM Gemma4 FAST decode step — single-token forward on gemma4_vlm's
        native KV cache (Gemma4Model handles the cross-layer KV share / sliding)."""
        model = self.gemma4_vlm
        past_key_values = state['past_key_values']
        position_cursor = state['position_cursor']

        if input_token.dtype != torch.long:
            input_token = input_token.to(torch.long)
        if input_token.dim() == 1:
            input_token = input_token.unsqueeze(1)  # (B, 1)

        hidden_states, past_key_values = model(
            input_ids=input_token,
            position_ids=position_cursor.view(-1, 1),
            past_key_values=past_key_values,
            ple_remap_mask=self._gemma4_tail_fast_token_mask(input_token),
            use_cache=True,
        )

        next_logits = model.language_out_proj(hidden_states[:, -1, :].to(model.lm_head.weight.dtype))
        state['past_key_values'] = past_key_values
        state['position_cursor'] = position_cursor + 1
        return next_logits, state


class EmbodimentSpecificLinear(nn.Module):
    """A linear layer with weights and biases specific to an embodiment
    category."""

    def __init__(self, input_dim: int, output_dim: int, num_categories: int = 1, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.num_categories = num_categories
        self.weight = nn.Parameter(torch.empty(num_categories, input_dim, output_dim, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(num_categories, output_dim, dtype=dtype))

        # Initialize using Linear's default initialization: U(-k, k) where k = 1/sqrt(in_features)
        k = 1.0 / math.sqrt(input_dim)
        nn.init.uniform_(self.weight, -k, k)
        nn.init.uniform_(self.bias, -k, k)

    def forward(self, x: Tensor, emb_ids: Tensor) -> Tensor:
        """Applies the embodiment-specific linear transformation.

        Args:
            x: The input tensor.
            emb_ids: A tensor of embodiment IDs, used to select the appropriate
                     weights and biases.

        Returns:
            The transformed tensor.
        """
        # Use one-hot to aggregate per-category weights/biases to avoid dynamic indexing (compile-friendly)
        if emb_ids.dtype != torch.long:
            emb_ids = emb_ids.to(torch.long)
        emb_ids = emb_ids.reshape(-1).to(device=self.weight.device)

        one_hot = F.one_hot(emb_ids, num_classes=self.num_categories).to(dtype=self.weight.dtype)

        selected_weight = torch.einsum('bc,cij->bij', one_hot, self.weight)
        selected_bias = torch.einsum('bc,cj->bj', one_hot, self.bias)

        return torch.bmm(x, selected_weight) + selected_bias.unsqueeze(1)

    def extra_repr(self):
        return f'EmbodimentSpecificLinear(num_categories={self.num_categories}, input_dim={self.weight.shape[1]}, output_dim={self.weight.shape[2]})'


def get_safe_dtype(dtype: torch.dtype, device: str | torch.device):
    """MPS is currently not compatible with float64."""
    if isinstance(device, torch.device):
        device = device.type
    if device == 'mps' and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def create_sinusoidal_pos_embedding(time: torch.Tensor, dimension: int, min_period: float, max_period: float, device='cpu') -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar
    positions."""
    if dimension % 2 != 0:
        raise ValueError(f'dimension ({dimension}) must be divisible by 2')

    if time.ndim != 1:
        raise ValueError('The time tensor is expected to be of shape `(batch_size, )`.')

    dtype = get_safe_dtype(torch.float64, device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks: Tensor, att_masks: Tensor) -> Tensor:
    """Creates a 2D attention mask from 1D padding and attention-type masks.

    This function is copied from big_vision.

    The `att_masks` allow for different attention patterns:
    - `[[1, 1, 1, 1, 1, 1]]`: Purely causal attention.
    - `[[0, 0, 0, 1, 1, 1]]`: Prefix-LM attention. The first 3 tokens attend
      to each other, and the last 3 tokens are causal.
    - `[[1, 0, 1, 0, 1, 0]]`: Causal attention between blocks. Tokens can
      attend to previous blocks and tokens within the same block.

    Args:
        pad_masks: bool[B, N] indicating valid (true) vs. padding (false) tokens.
        att_masks: int[B, N] defining attention type. A `1` at a position
                   indicates the start of a new causal block.

    Returns:
        A 2D boolean attention mask of shape (B, N, N).
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    # att_masks arrive as bf16 under mixed-precision training (embed_suffix builds them
    # with embs.dtype); bf16 has a 7-bit mantissa, so cumsum values above 256 round to
    # even integers and the block comparison below corrupts the mask for fully-causal
    # prefixes longer than 256 real tokens (train-only; inference builds masks in long).
    cumsum = torch.cumsum(att_masks.long(), dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks

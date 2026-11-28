"""Request-local exact joint Text/IVAVA cache. No weights or global state.

Only clean, causally earlier tokens are cached, in post-RoPE [B,H,S,D]
storage. Rebuild on every rolling window (Text attends to the current I).
Prefix Memory/history validity is captured with Text validity. State-FiLM
and cross-attention memory banks remain unsupported. Cache lifetime is ONE
action sampling call, never an episode or rolling window sequence.
Capture uses a full reference forward. Decode preserves reference GEMM and
attention shapes using inactive scratch rows while reusing clean prefix K/V.
"""

from dataclasses import dataclass
import torch
from wall_x.model.vap.layout import ActionStageLayout, InterleaveLayout


@dataclass(frozen=True)
class JointActionPrefixCache:
    layers: tuple
    positions: torch.Tensor
    key_valid: torch.Tensor
    owner: int
    full_positions: torch.Tensor
    active_start: int
    joint_layout: object


@torch.no_grad()
def prefill(
    core,
    video,
    action,
    *,
    active_action_chunk,
    video_timestep,
    action_timestep,
    video_position_ids,
    action_position_ids,
    condition_video_tokens,
    video_camera_ids,
    text_context,
    text_valid_mask,
    state_condition=None,
    first_frame_memory=None,
    memory_prefix_valid=None,
):
    if core.training or torch.is_grad_enabled():
        raise RuntimeError("prefix cache is inference-only")
    if (
        core.state_film is not None
        or state_condition is not None
        or first_frame_memory is not None
    ):
        raise NotImplementedError(
            "VAPv2 prefix cache supports prefix Memory, not banks/State-FiLM"
        )
    future = video.shape[1] - condition_video_tokens
    if future <= 0 or future % core.num_chunks or action.shape[1] % core.num_chunks:
        raise ValueError("prefix cache needs a full equal-chunk IVAVA window")
    layout = ActionStageLayout(
        core.num_chunks,
        active_action_chunk,
        condition_video_tokens,
        future // core.num_chunks,
        action.shape[1] // core.num_chunks,
    )
    nv, na = layout.num_video_tokens, layout.num_action_tokens
    start = na - layout.action_tokens_per_chunk
    if (
        video_timestep.shape != video.shape[:2]
        or action_timestep.shape != action.shape[:2]
    ):
        raise ValueError("cache requires explicit per-token timesteps")
    if torch.count_nonzero(video_timestep[:, :nv]) or torch.count_nonzero(
        action_timestep[:, :start]
    ):
        raise ValueError("cached prefix must be clean (timestep zero)")
    output, cache = core(
        video,
        action,
        video_timestep=video_timestep,
        action_timestep=action_timestep,
        video_position_ids=video_position_ids,
        action_position_ids=action_position_ids,
        condition_video_tokens=condition_video_tokens,
        video_camera_ids=video_camera_ids,
        text_context=text_context,
        text_valid_mask=text_valid_mask,
        memory_prefix_valid=memory_prefix_valid,
        # Preserve the reference forward's GEMM/attention shapes for capture.
        # Future tokens are computed once but are never retained in the cache.
        layout=InterleaveLayout(core.num_chunks, condition_video_tokens,
                                future // core.num_chunks,
                                action.shape[1] // core.num_chunks),
        capture_action_prefix_rank=layout.active_rank,
    )
    return output.action_velocity[:, start:na], cache


@torch.no_grad()
def decode(
    core, active_actions, *, action_timestep, prefix_cache, state_condition=None
):
    if core.training:
        raise RuntimeError("prefix cache is inference-only")
    if not isinstance(prefix_cache, JointActionPrefixCache) or prefix_cache.owner != id(
        core
    ):
        raise ValueError("cache belongs to a different model or cache format")
    if state_condition is not None or core.state_film is not None:
        raise NotImplementedError("VAPv2 prefix cache does not support State-FiLM")
    if (
        active_actions.ndim != 3
        or active_actions.shape[-1] != core.action_input_dim
        or active_actions.shape[:2] != prefix_cache.positions.shape[:2]
        or len(prefix_cache.layers) != len(core.blocks)
    ):
        raise ValueError("active action shape/layers differ from cache")
    if action_timestep.shape not in (
        active_actions.shape[:1],
        active_actions.shape[:2],
    ):
        raise ValueError("active action timestep must be [B] or [B,S]")
    accelerator = getattr(core, "_v2_action_accelerator", None)
    if accelerator is not None:
        return accelerator.decode(active_actions, action_timestep, prefix_cache)
    return decode_impl(core, active_actions, action_timestep, prefix_cache)


def decode_impl(core, active_actions, action_timestep, prefix_cache,
                *, time_pair=None, action_rope=None):
    """Same reference shapes/operators; optional precomputed time and RoPE.

    Graph capture calls this only after public decode validation. No RNG,
    window/history construction, or physical action conversion occurs here.
    """
    # Retain reference Action GEMM row counts as well as attention tile shapes.
    # Inactive rows are scratch only: their new K/V never replace cached prefix.
    start = prefix_cache.active_start
    end = start + active_actions.shape[1]
    full = active_actions.new_zeros(
        active_actions.shape[0], prefix_cache.full_positions.shape[1],
        active_actions.shape[2])
    full[:, start:end] = active_actions
    full_time = action_timestep.new_zeros(full.shape[:2])
    full_time[:, start:end] = (action_timestep[:, None] if action_timestep.ndim == 1
                              else action_timestep)
    action = core.action_identity(core.action_encoder(full))
    if time_pair is None:
        time = core.action_time.embed(full_time, dtype=action.dtype)
        mod = core.action_time.project(time, dtype=action.dtype)
    else:
        time, mod = time_pair
    for block, cache in zip(core.blocks, prefix_cache.layers):
        action = block.forward_cached_action(
            action,
            time_mod=mod,
            positions=prefix_cache.full_positions,
            cache=cache,
            key_valid=prefix_cache.key_valid,
            reference_layout=prefix_cache.joint_layout,
            active_slice=(start, end),
            action_rope=action_rope,
        )
    return core.action_head(action, time)[:, start:end]

"""History packing and joint video/action flow sampling.

Each call handles a compatible group of environments using bundle sampling defaults.
"""

import math
from dataclasses import dataclass, field

import torch
from torch.nn import functional as F


def time_features(timestep, cfg):
    half = cfg.m["time_width"] // 2
    frequency = torch.pow(
        cfg.sampler["frequency_base"],
        -torch.arange(half, device=timestep.device, dtype=torch.float64) / half,
    )
    phase = torch.outer(timestep.flatten().double(), frequency)
    return torch.cat((phase.cos(), phase.sin()), -1).to(timestep.dtype)


def axis_rope(positions, width, base):
    frequency = 1 / (
        base ** (torch.arange(0, width, 2, device=positions.device).float() / width)
    )
    phase = positions.float()[..., None] * frequency
    return torch.polar(torch.ones_like(phase), phase)


def video_rope(times, height, width, cfg):
    axes, base = cfg.rope["video_axes"], cfg.rope["base"]
    time = axis_rope(times, axes[0], base)[:, None, None].expand(-1, height, width, -1)
    vertical = axis_rope(torch.arange(height, device=times.device), axes[1], base)[
        None, :, None
    ].expand(len(times), -1, width, -1)
    horizontal = axis_rope(torch.arange(width, device=times.device), axes[2], base)[
        None, None
    ].expand(len(times), height, -1, -1)
    return torch.cat((time, vertical, horizontal), -1)


def rotate(value, frequencies, cfg):
    shaped = value.unflatten(-1, (cfg.m["heads"], cfg.m["head_dim"]))
    complex_value = torch.view_as_complex(
        shaped.float().unflatten(-1, (cfg.m["head_dim"] // 2, 2))
    )
    rotated = torch.view_as_real(complex_value * frequencies).flatten(-2)
    return rotated.to(value.dtype).transpose(1, 2).contiguous()


def pad_kernel(value, kernel):
    time, height, width = value.shape[-3:]
    kt, kh, kw = kernel
    padding = (0, -width % kw, 0, -height % kh, 0, -time % kt)
    return F.pad(value, padding, mode="replicate") if any(padding) else value


def pack_history(projection, latent, times, cfg):
    """Single-environment convenience wrapper; output retains its batch axis."""
    if latent.ndim != 4:
        raise ValueError("Expected history latent [C,T,H,W]")
    return pack_history_batch(projection, (latent[None], times[None]), cfg)


def pack_history_batch(projection, history, cfg):
    """Batch compatible history projections and pooled RoPE without temporal padding across environments."""
    latent, times = history
    if latent.ndim == 4:
        latent, times = latent[None], times[None]
    if (
        latent.ndim != 5
        or latent.shape[1] != cfg.channels
        or tuple(latent.shape[-2:])
        != (
            cfg.observation.height // cfg.spatial_stride,
            cfg.observation.width // cfg.spatial_stride,
        )
    ):
        raise ValueError("History latent shape disagrees with model config")
    batch, length = latent.shape[0], latent.shape[2]
    if (
        batch < 1
        or not 1 <= length <= cfg.history["max_frames"] // cfg.temporal_stride + 1
        or times.shape != (batch, length)
    ):
        raise ValueError("History frame/time count disagrees with model config")
    groups = [(projection.proj, [slice(0, 1)])]
    if length > 1:
        groups[0][1].append(slice(length - 1, length))
    near = max(1, length - 1 - cfg.history["near_latents"])
    if length > 2:
        groups.append((projection.proj_2x, [slice(near, length - 1)]))
    if near > 1:
        groups.append((projection.proj_4x, [slice(1, near)]))
    tokens, rotations = [], []
    for layer, selections in groups:
        kt, kh, kw = layer.kernel_size
        values = torch.cat(
            [pad_kernel(latent[:, :, span], (kt, 1, 1)) for span in selections]
        )
        projected = (
            layer(pad_kernel(values, layer.kernel_size)).flatten(2).transpose(1, 2)
        )
        # Conv batch is selection-major; token order within each environment
        # must remain oldest, newest, then the coarser history groups.
        projected = projected.reshape(len(selections), batch, *projected.shape[1:])
        tokens.append(projected.permute(1, 0, 2, 3).flatten(1, 2))
        for span in selections:
            selected = times[:, span]
            selected = torch.cat(
                (selected, selected[:, -1:].expand(-1, kt - selected.shape[1])), 1
            )
            gh, gw = cfg.head_grid
            freqs = video_rope(selected.flatten(), gh, gw, cfg).reshape(
                batch, kt, gh, gw, cfg.m["head_dim"] // 2
            )
            components = torch.view_as_real(freqs).flatten(-2).permute(0, 4, 1, 2, 3)
            kernel = (kt, kh // cfg.m["patch_size"][1], kw // cfg.m["patch_size"][2])
            pooled = F.avg_pool3d(pad_kernel(components, kernel), kernel, stride=kernel)
            pooled = (
                pooled.permute(0, 2, 3, 4, 1)
                .contiguous()
                .unflatten(-1, (cfg.m["head_dim"] // 2, 2))
            )
            rotations.append(torch.view_as_complex(pooled).flatten(1, 3))
    return torch.cat(tokens, 1), torch.cat(rotations, 1)[:, :, None]


def joint_mask(history_tokens, device, cfg):
    head = math.prod(cfg.head_grid)
    wrist = cfg.tokens_per_frame - head
    # Rows/columns: history, current head, current wrist, future head,
    # future wrist, action. In particular, video never reads action, and
    # action never directly reads history tokens.
    visible = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 0],
            [0, 1, 1, 1, 1, 1],
        ],
        device=device,
        dtype=torch.bool,
    )
    groups = (
        [0] * history_tokens
        + [1] * head
        + [2] * wrist
        + ([3] * head + [4] * wrist) * (cfg.grid[0] - 1)
        + [5] * cfg.m["action_horizon"]
    )
    groups = torch.tensor(groups, device=device)
    return visible[groups[:, None], groups[None, :]][None, None]


def draw_noise(first_frame, cfg):
    """Preserve the serial RNG order: video0, action0, video1, action1, ... ."""
    videos, actions = [], []
    for _ in range(first_frame.shape[0]):
        videos.append(
            torch.randn(1, *cfg.video_shape, device="cpu", dtype=torch.float32).to(
                first_frame
            )
        )
        actions.append(
            torch.randn(
                1,
                cfg.m["action_horizon"],
                cfg.m["action_dim"],
                device="cpu",
                dtype=torch.float32,
            ).to(first_frame)
        )
    return torch.cat(videos), torch.cat(actions)


@dataclass
class DenoisingCache:
    """One environment group, one action chunk; never persisted on the model.

    History and the current head/wrist tokens have timestep zero and cannot
    attend to future video/action. Their per-layer KV is therefore invariant
    across denoising steps. The first full pass populates it; subsequent passes
    evaluate only future video/action queries, preserving the full key layout.
    """

    packed: tuple[torch.Tensor, torch.Tensor] | None
    video_freqs: torch.Tensor
    action_freqs: torch.Tensor
    mask: torch.Tensor
    video_kv: list[tuple[torch.Tensor, torch.Tensor]]
    action_kv: list[tuple[torch.Tensor, torch.Tensor]]
    world_kv: list[tuple[torch.Tensor, torch.Tensor]]
    static_kv: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


def prepare_cache(network, text, world_queries, action_context, history):
    cfg = network.config
    vn, an = network.video, network.action
    packed = (
        None
        if history is None
        else pack_history_batch(vn.history_packing, history, cfg)
    )
    count = 0 if packed is None else packed[0].shape[1]
    device = text.device
    vf = (
        video_rope(torch.arange(cfg.grid[0]), *cfg.grid[1:], cfg)
        .reshape(1, cfg.video_tokens, 1, cfg.m["head_dim"] // 2)
        .to(device)
    )
    if packed is not None:
        vf = torch.cat((packed[1], vf.expand(text.shape[0], -1, -1, -1)), 1)
    vc, ac = vn.text_embedding(text), an.text_embedding(action_context)
    return DenoisingCache(
        packed=packed,
        video_freqs=vf,
        action_freqs=axis_rope(
            torch.arange(cfg.m["action_horizon"]), cfg.m["head_dim"], cfg.rope["base"]
        )[None, :, None].to(device),
        mask=joint_mask(count, device, cfg),
        video_kv=[b.cross_attn.prepare_kv(vc) for b in vn.blocks],
        action_kv=[b.cross_attn.prepare_kv(ac) for b in an.blocks],
        world_kv=[b.prepare_kv(world_queries) for b in vn.vlm_cross_attentions],
    )


def joint_velocity(
    network,
    video,
    action,
    timestep,
    text,
    text_mask,
    world_queries,
    action_context,
    history,
    cache: DenoisingCache | None = None,
):
    """Predict both velocities in one directed-attention pass per layer."""
    cfg = network.config
    batch = video.shape[0]
    if tuple(video.shape) != (batch, *cfg.video_shape) or tuple(action.shape) != (
        batch,
        cfg.m["action_horizon"],
        cfg.m["action_dim"],
    ):
        raise ValueError("Joint sampler input shapes disagree with model config")
    video_net, action_net = network.video, network.action
    video_tokens = (
        video_net.patch_embedding(video).flatten(2).transpose(1, 2).contiguous()
    )
    token_times = timestep.reshape(1, 1).expand(1, cfg.video_tokens).clone()
    token_times[:, : cfg.tokens_per_frame] = 0
    video_time = video_net.time_embedding(time_features(token_times, cfg)).reshape(
        1, cfg.video_tokens, cfg.m["video_width"]
    )
    video_modulation = video_net.time_projection(video_time).unflatten(
        -1, (6, cfg.m["video_width"])
    )
    video_time = video_time.expand(batch, -1, -1)
    video_modulation = video_modulation.expand(batch, -1, -1, -1)
    video_freqs = (
        cache.video_freqs
        if cache is not None
        else (
            video_rope(torch.arange(cfg.grid[0]), *cfg.grid[1:], cfg)
            .reshape(1, cfg.video_tokens, 1, cfg.m["head_dim"] // 2)
            .to(video.device)
        )
    )
    history_count = 0
    if history is not None:
        history_tokens, history_freqs = (
            cache.packed
            if cache is not None
            else pack_history_batch(video_net.history_packing, history, cfg)
        )
        history_count = history_tokens.shape[1]
        video_tokens = torch.cat((history_tokens, video_tokens), 1)
        if cache is None:
            video_freqs = torch.cat(
                (history_freqs, video_freqs.expand(batch, -1, -1, -1)), 1
            )
        video_time = torch.cat(
            (video_time[:, :1].expand(-1, history_count, -1), video_time), 1
        )
        video_modulation = torch.cat(
            (
                video_modulation[:, :1].expand(-1, history_count, -1, -1),
                video_modulation,
            ),
            1,
        )
    action_tokens = action_net.action_encoder(action)
    action_time = action_net.time_embedding(time_features(timestep, cfg))
    action_modulation = (
        action_net.time_projection(action_time)
        .reshape(1, 6, cfg.m["action_width"])
        .expand(batch, -1, -1)
    )
    action_freqs = (
        cache.action_freqs
        if cache is not None
        else axis_rope(
            torch.arange(cfg.m["action_horizon"]), cfg.m["head_dim"], cfg.rope["base"]
        )[None, :, None].to(action.device)
    )
    if cache is None:
        video_context = video_net.text_embedding(text)
        action_context = action_net.text_embedding(action_context)
    else:
        video_context = action_context = None
    mask = (
        cache.mask
        if cache is not None
        else joint_mask(history_count, video.device, cfg)
    )
    skip = (
        history_count + cfg.tokens_per_frame
        if cache is not None and cache.static_kv
        else 0
    )
    if skip:
        video_tokens = video_tokens[:, skip:].contiguous()
        video_modulation = video_modulation[:, skip:].contiguous()
        video_freqs = video_freqs[:, skip:]
        mask = mask[:, :, skip:, :]
    video_length = video_tokens.shape[1]
    for index, (video_block, action_block) in enumerate(
        zip(video_net.blocks, action_net.blocks, strict=True)
    ):
        video_qkv, video_gates = video_block.prepare_attention(
            video_tokens, video_modulation
        )
        action_qkv, action_gates = action_block.prepare_attention(
            action_tokens, action_modulation
        )
        joined = []
        for column in range(3):
            if column < 2:
                v_part = rotate(video_qkv[column], video_freqs, cfg)
                a_part = rotate(action_qkv[column], action_freqs, cfg)
            else:
                v_part = (
                    video_qkv[column]
                    .unflatten(-1, (cfg.m["heads"], cfg.m["head_dim"]))
                    .transpose(1, 2)
                    .contiguous()
                )
                a_part = (
                    action_qkv[column]
                    .unflatten(-1, (cfg.m["heads"], cfg.m["head_dim"]))
                    .transpose(1, 2)
                    .contiguous()
                )
            if skip and column > 0:
                v_part = torch.cat((cache.static_kv[index][column - 1], v_part), dim=2)
            joined.append(torch.cat((v_part, a_part), dim=2))
        if cache is not None and not skip:
            cache.static_kv.append(
                tuple(
                    x[:, :, : history_count + cfg.tokens_per_frame].clone()
                    for x in joined[1:]
                )
            )
        attended = (
            F.scaled_dot_product_attention(*joined, attn_mask=mask)
            .transpose(1, 2)
            .flatten(2)
        )
        video_tokens = video_block.finish_attention(
            video_tokens,
            attended[:, :video_length],
            video_gates,
            video_context,
            text_mask,
            visual=(
                video_net.vlm_norms[index],
                video_net.vlm_cross_attentions[index],
                world_queries,
            ),
            context_kv=cache.video_kv[index] if cache is not None else None,
            visual_kv=cache.world_kv[index] if cache is not None else None,
        )
        action_tokens = action_block.finish_attention(
            action_tokens,
            attended[:, video_length:],
            action_gates,
            action_context,
            context_kv=cache.action_kv[index] if cache is not None else None,
        )
    output = video_net.head(
        video_tokens if skip else video_tokens[:, history_count:],
        video_time[:, skip:] if skip else video_time[:, history_count:],
    )
    frames = cfg.grid[0] - 1 if skip else cfg.grid[0]
    output = output.reshape(
        batch, frames, *cfg.grid[1:], *cfg.m["patch_size"], cfg.channels
    ).permute(0, 7, 1, 4, 2, 5, 3, 6)
    output = output.reshape(batch, cfg.channels, frames, *cfg.video_shape[-2:])
    if skip:
        # The sampler re-clamps the first frame after every update; its
        # velocity is unused, so no static-token output head is needed here.
        output = torch.cat((torch.zeros_like(video[:, :, :1]), output), 2)
    return output, action_net.head(action_tokens)


def validate_flow_shift(flow_shift):
    if (
        type(flow_shift) not in (int, float)
        or not math.isfinite(flow_shift)
        or flow_shift <= 0
    ):
        raise ValueError("flow_shift must be a finite positive number")
    return float(flow_shift)


def flow_schedule(
    device, dtype, num_inference_steps: int, flow_shift: float, timestep_scale: float
):
    if type(num_inference_steps) is not int or num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer")
    flow_shift = validate_flow_shift(flow_shift)
    u = torch.linspace(
        1, 0, num_inference_steps + 1, device=device, dtype=torch.float32
    )
    sigma = flow_shift * u / (1 + (flow_shift - 1) * u)
    return (sigma[:-1] * timestep_scale).to(dtype), (sigma[1:] - sigma[:-1]).to(dtype)


@torch.no_grad()
def sample_actions(
    network,
    first_frame,
    text,
    text_mask,
    world_queries,
    action_queries,
    normalized_state,
    history=None,
    *,
    num_inference_steps: int | None = None,
    flow_shift: float | None = None,
    use_kv_cache: bool = True,
    noise: tuple[torch.Tensor, torch.Tensor] | None = None,
):
    """Return the configured horizon of normalized absolute joint actions and denoised video latents.

    Use the caller's CPU RNG; no reseeding per request or episode. Video noise
    is drawn before action noise, in FP32, then converted to the model dtype.
    """
    cfg = network.config
    if num_inference_steps is None:
        num_inference_steps = cfg.inference_defaults["num_inference_steps"]
    if flow_shift is None:
        flow_shift = cfg.inference_defaults["flow_shift"]
    batch = first_frame.shape[0]
    if batch < 1 or tuple(first_frame.shape) != (
        batch,
        cfg.channels,
        1,
        *cfg.video_shape[-2:],
    ):
        raise ValueError("First-frame latent shape disagrees with model config")
    if any(
        x.shape[0] != batch for x in (text, text_mask, world_queries, action_queries)
    ):
        raise ValueError(
            "Each environment must have its own text and planner conditions"
        )
    if normalized_state.numel() != batch * cfg.m["state_dim"]:
        raise ValueError("State count does not match the environment batch")
    if history is not None and (
        (history[0].ndim == 4 and batch != 1)
        or (history[0].ndim == 5 and history[0].shape[0] != batch)
    ):
        raise ValueError("History count does not match the environment batch")
    times, deltas = flow_schedule(
        first_frame.device,
        first_frame.dtype,
        num_inference_steps,
        flow_shift,
        cfg.sampler["timestep_scale"],
    )
    video, action = (
        draw_noise(first_frame, cfg)
        if noise is None
        else tuple(value.to(first_frame).clone() for value in noise)
    )
    if video.shape != (batch, *cfg.video_shape) or action.shape != (
        batch,
        cfg.m["action_horizon"],
        cfg.m["action_dim"],
    ):
        raise ValueError("Noise shapes do not match the environment batch")
    video[:, :, :1] = first_frame
    state_token = network.state_projection(
        normalized_state.reshape(batch, 1, cfg.m["state_dim"]).to(first_frame)
    )
    action_context = torch.cat((action_queries, state_token), dim=1)
    cache = (
        prepare_cache(network, text, world_queries, action_context, history)
        if use_kv_cache and num_inference_steps > 1
        else None
    )
    for timestep, delta in zip(times, deltas, strict=True):
        video_velocity, action_velocity = joint_velocity(
            network,
            video,
            action,
            timestep,
            text,
            text_mask,
            world_queries,
            action_context,
            history,
            cache,
        )
        video = video + video_velocity * delta
        video[:, :, :1] = first_frame
        action = action + action_velocity * delta
    return action, video

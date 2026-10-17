"""Training-only foresight supervision for FastWAM (Video-Mirai style).

A frozen bidirectional copy of the pretrained video DiT encodes the clean
ground-truth clip at t=0 ("foresight encoder"). A lightweight projector maps
the policy's mid-depth observation-token hidden states to the encoder's
feature space, pulled toward the fused future-frame features with a cosine
loss. Both the encoder and the projector are discarded at inference, so the
deployed model and its latency are unchanged.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn

from fastwam.utils.logging_config import get_logger

from .wan_video_dit import WanVideoDiT, flash_attention

logger = get_logger(__name__)


class ForesightBlock(nn.Module):
    """Pre-LN transformer block (token-mixing self-attention + MLP)."""

    def __init__(self, hidden_dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got {hidden_dim} and {num_heads}"
            )
        self.num_heads = num_heads
        self.norm1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn = flash_attention(q=self.q(h), k=self.k(h), v=self.v(h), num_heads=self.num_heads)
        x = x + self.o(attn)
        x = x + self.ffn(self.norm2(x))
        return x


class ForesightProjector(nn.Module):
    """Maps observation-token hidden states into the foresight encoder's feature space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 24,
        ffn_dim: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(in_dim)
        ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_dim * 4
        self.in_proj = nn.Identity() if hidden_dim == int(in_dim) else nn.Linear(int(in_dim), hidden_dim)
        self.blocks = nn.ModuleList(
            [ForesightBlock(hidden_dim, int(num_heads), ffn_dim, eps=eps) for _ in range(int(num_layers))]
        )
        self.norm_out = nn.LayerNorm(hidden_dim, eps=eps)
        self.proj_out = nn.Linear(hidden_dim, int(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.proj_out(self.norm_out(x))


class SharedForesightProjector(nn.Module):
    """One shared transformer backbone + lightweight per-target linear heads.

    For per-frame multi-layer foresight there are (num_layers x num_offsets) targets; giving
    each its own full projector explodes params (~12 x 0.34B). Instead a single 3-block backbone
    does the shared token-mixing, and a cheap Linear head per target key does the final
    projection — ~0.45B total instead of ~4B. `keys` are e.g. "{layer}_{offset}" (per-frame) or
    "{layer}" (fused).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        keys: Sequence[str],
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        num_heads: int = 24,
        ffn_dim: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        hidden_dim = int(hidden_dim) if hidden_dim is not None else int(in_dim)
        ffn_dim = int(ffn_dim) if ffn_dim is not None else hidden_dim * 4
        self.in_proj = nn.Identity() if hidden_dim == int(in_dim) else nn.Linear(int(in_dim), hidden_dim)
        self.blocks = nn.ModuleList(
            [ForesightBlock(hidden_dim, int(num_heads), ffn_dim, eps=eps) for _ in range(int(num_layers))]
        )
        self.norm = nn.LayerNorm(hidden_dim, eps=eps)
        self.heads = nn.ModuleDict({k: nn.Linear(hidden_dim, int(out_dim)) for k in keys})

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Shared token-mixing trunk; run once per student layer, reuse across its frame heads."""
        x = self.in_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def head(self, feat: torch.Tensor, key: str) -> torch.Tensor:
        return self.heads[key](feat)


class MLPForesightProjector(nn.Module):
    """REPA-style shared MLP projector: a WIDE 3-layer MLP (no token-mixing, no attention).

    Faithful to REPA's projection head -- `Linear -> SiLU -> Linear -> SiLU -> Linear`, hidden width
    ~2048 (REPA's projector_dim), NOT a tight bottleneck -- so information is NOT compressed before
    alignment. This matches the "we align to rich FUTURE-frame features, do not throttle them"
    intuition (a narrow bottleneck would cap the predictable signal a single token can carry).

    To stay cheap with our (num_layers x num_offsets) targets, the first 2 wide layers are SHARED
    across all keys (REPA has one target, so no sharing); each target keeps its own 3rd-layer
    `Linear(hidden -> out)` readout. So per target it is a 3-layer MLP, with the trunk amortized.
    Cosine is still vs the FROZEN teacher feature (out_dim space) -> no learnable target, no collapse.

    Params (in=out=3072, 12 keys): hidden=2048 -> ~86M, 1024 -> ~42M, 3072 -> ~132M (all << 0.45B).
    Same `.backbone()/.head()` interface as SharedForesightProjector, so the loss path is unchanged.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        keys: Sequence[str],
        hidden: int = 2048,
        eps: float = 1e-6,
    ):
        super().__init__()
        # Shared wide trunk: 2 linear layers + SiLU (REPA-style), no compression.
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim), eps=eps),
            nn.Linear(int(in_dim), int(hidden)),
            nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)),
            nn.SiLU(),
        )
        # Per-key 3rd-layer readout to the teacher feature space.
        self.heads = nn.ModuleDict(
            {k: nn.Linear(int(hidden), int(out_dim)) for k in keys}
        )

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Shared wide MLP trunk; run once per student layer, reused across that layer's frame heads."""
        return self.net(x)

    def head(self, feat: torch.Tensor, key: str) -> torch.Tensor:
        return self.heads[key](feat)


class ActionForesightHead(nn.Module):
    """Maps action-expert hidden states into the foresight encoder's feature space.

    Action tokens have no spatial grid, so a pointwise MLP suffices (the temporal/group
    structure is handled outside by reshaping into per-transition groups).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_mult: int = 4, eps: float = 1e-6):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim), eps=eps),
            nn.Linear(int(in_dim), int(in_dim) * hidden_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(in_dim) * hidden_mult, int(out_dim)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpatialActionForesightHead(nn.Module):
    """Per-token variant of ActionForesightHead.

    Instead of predicting one pooled vector per future frame, a single grouped action vector is
    broadcast over the future frame's spatial tokens and combined with a learned positional
    embedding, then an MLP predicts EACH spatial token's target feature. This mirrors the
    per-token richness that made the video-stream foresight work (no spatial mean-pooling of the
    target). `max_tokens` sizes the positional table; the forward slices it to the actual
    tokens_per_frame (data-dependent), so the parameter lives in the optimizer from construction.
    """

    def __init__(self, in_dim: int, out_dim: int, max_tokens: int = 4096, hidden_mult: int = 4, eps: float = 1e-6):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(int(max_tokens), int(in_dim)))
        nn.init.normal_(self.pos, std=0.02)
        self.net = nn.Sequential(
            nn.LayerNorm(int(in_dim), eps=eps),
            nn.Linear(int(in_dim), int(in_dim) * hidden_mult),
            nn.GELU(approximate="tanh"),
            nn.Linear(int(in_dim) * hidden_mult, int(out_dim)),
        )

    def forward(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        # x: [B, in_dim] (one grouped action vector) -> [B, num_tokens, out_dim]
        if int(num_tokens) > self.pos.shape[0]:
            raise ValueError(
                f"num_tokens {num_tokens} exceeds SpatialActionForesightHead max_tokens {self.pos.shape[0]}"
            )
        h = x.unsqueeze(1) + self.pos[: int(num_tokens)].unsqueeze(0)  # [B, num_tokens, in_dim]
        return self.net(h)


FORESIGHT_TEACHER_MODES = frozenset(
    {"frozen_pretrained", "online_ema", "frozen_random"}
)

FORESIGHT_TARGET_TYPES = frozenset(
    {"encoder_hidden", "patch_embedding", "vae_direct"}
)


def normalize_foresight_target_type(target_type: Optional[str]) -> str:
    """Return the canonical representation used as the foresight target.

    ``encoder_hidden`` preserves the historical teacher-block target.  The two
    ablation modes deliberately remove teacher computation while keeping the
    student/projector/loss path unchanged:

    - ``patch_embedding`` ("Random Layer0"): frozen teacher patch embedding.
    - ``vae_direct``: deterministic spatial patches of a frozen VAE latent; the
      VAE source is selected by ``teacher_mode`` (pretrained or random).
    """
    canonical = "encoder_hidden" if target_type is None else str(target_type).strip().lower()
    aliases = {
        "hidden": "encoder_hidden",
        "layer0": "patch_embedding",
        "pre_dit": "patch_embedding",
        "vae": "vae_direct",
        "latent": "vae_direct",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in FORESIGHT_TARGET_TYPES:
        raise ValueError(
            f"Unsupported `foresight.target_type` {target_type!r}; expected one of "
            f"{sorted(FORESIGHT_TARGET_TYPES)}."
        )
    return canonical


def normalize_foresight_teacher_mode(mode: Optional[str]) -> str:
    """Return the canonical target-encoder mode used by Wan foresight."""
    canonical = "frozen_pretrained" if mode is None else str(mode).strip().lower()
    aliases = {
        "pretrained": "frozen_pretrained",
        "fixed": "frozen_pretrained",
        "ema": "online_ema",
        "random": "frozen_random",
    }
    canonical = aliases.get(canonical, canonical)
    if canonical not in FORESIGHT_TEACHER_MODES:
        raise ValueError(
            f"Unsupported `foresight.teacher_mode` {mode!r}; expected one of "
            f"{sorted(FORESIGHT_TEACHER_MODES)}."
        )
    return canonical


def _freeze_foresight_encoder(encoder: nn.Module, *, source: str) -> nn.Module:
    encoder.video_attention_mask_mode = "bidirectional"
    encoder.use_gradient_checkpointing = False
    encoder.eval()
    encoder.requires_grad_(False)
    num_params = sum(p.numel() for p in encoder.parameters())
    logger.info(
        "Built %s foresight encoder: %.2f B params (bidirectional, t=0).",
        source,
        num_params / 1e9,
    )
    return encoder


def build_random_foresight_vae(
    reference_vae: nn.Module,
    *,
    seed: int,
    device: Union[str, torch.device],
    torch_dtype: torch.dtype,
) -> nn.Module:
    """Build an independently random, frozen VAE for teacher targets only.

    The policy/student keeps ``reference_vae`` and its pretrained weights.  A
    fresh instance of the exact same class supplies random future-image
    targets.  Construction is isolated from the run RNG so projector
    initialization and training noise remain identical to the pretrained-VAE
    control.  The module is returned frozen/eval and is kept outside the
    FastWAM module registry by the caller, so it is neither optimized nor
    checkpointed.
    """
    vae_cls = type(reference_vae)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        random_vae = vae_cls()

    reference_state = reference_vae.state_dict()
    random_state = random_vae.state_dict()
    if reference_state.keys() != random_state.keys():
        raise ValueError(
            "Random teacher VAE must have the same state-dict structure as the "
            f"policy VAE ({vae_cls.__name__})."
        )
    mismatched = [
        name
        for name in reference_state
        if tuple(reference_state[name].shape) != tuple(random_state[name].shape)
    ]
    if mismatched:
        raise ValueError(
            "Random teacher VAE parameter shapes differ from the policy VAE: "
            f"{mismatched[:8]}"
        )

    random_vae = random_vae.to(device=device, dtype=torch_dtype)
    random_vae.eval()
    random_vae.requires_grad_(False)
    num_params = sum(parameter.numel() for parameter in random_vae.parameters())
    logger.info(
        "Built random frozen teacher VAE: class=%s seed=%d params=%.2f M; "
        "policy VAE remains pretrained.",
        vae_cls.__name__,
        int(seed),
        num_params / 1e6,
    )
    return random_vae


class _PatchEmbeddingOnlyEncoder(nn.Module):
    """Minimal exact Layer0 subset of WanVideoDiT.

    Wan constructs ``patch_embedding`` before every other parameterized
    submodule.  Constructing the same Conv3d first under the same forked seed is
    therefore weight-identical, while avoiding roughly 90M never-used
    text/time/head parameters in the zero-block ablation.
    """

    def __init__(self, *, in_dim: int, hidden_dim: int, patch_size: Sequence[int]):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.patch_embedding = nn.Conv3d(
            int(in_dim),
            self.hidden_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.blocks = nn.ModuleList()

    def patchify(self, x: torch.Tensor):
        return self.patch_embedding(x)


def build_foresight_encoder(
    video_expert: nn.Module,
    *,
    max_layers: Optional[int] = None,
    copy_model: bool = True,
    source: str = "pretrained",
) -> nn.Module:
    """Frozen bidirectional copy of the (pretrained) video expert used as target encoder.

    Must be called before any fine-tuning step so the copy holds the pretrained weights.
    ``copy_model=False`` adopts an independently loaded teacher without making a
    second multi-billion-parameter copy.
    """
    encoder = copy.deepcopy(video_expert) if copy_model else video_expert
    if max_layers is not None:
        max_layers = int(max_layers)
        if not 1 <= max_layers <= len(encoder.blocks):
            raise ValueError(
                f"`max_layers` must be in [1, {len(encoder.blocks)}], got {max_layers}."
            )
        # The target path exits immediately after max(encoder_layers), so later
        # blocks can never affect a target.  EMA teachers use this exact prefix to
        # make room for the fp32 shadow without changing target numerics.
        encoder.blocks = encoder.blocks[:max_layers]
    return _freeze_foresight_encoder(encoder, source=source)


def build_random_foresight_encoder(
    video_dit_config: dict,
    *,
    max_layers: int,
    seed: int,
    device: Union[str, torch.device],
    torch_dtype: torch.dtype,
) -> nn.Module:
    """Build an independently random, frozen Wan prefix without consuming run RNG.

    Constructing a fresh module (instead of recursively calling reset_parameters)
    is important because Wan blocks own parameters such as ``modulation`` that do
    not have a reset hook.
    """
    config = dict(video_dit_config)
    configured_layers = int(config["num_layers"])
    max_layers = int(max_layers)
    if not 0 <= max_layers <= configured_layers:
        raise ValueError(
            f"Random teacher depth must be in [0, {configured_layers}], got {max_layers}."
        )
    config["num_layers"] = max_layers
    config["video_attention_mask_mode"] = "bidirectional"
    config["use_gradient_checkpointing"] = False

    # WanVideoDiT is constructed on CPU.  Fork only the CPU generator so teacher
    # initialization cannot shift projector initialization or training noise.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(int(seed))
        if max_layers == 0:
            encoder = _PatchEmbeddingOnlyEncoder(
                in_dim=int(config["in_dim"]),
                hidden_dim=int(config["hidden_dim"]),
                patch_size=config["patch_size"],
            )
        else:
            encoder = WanVideoDiT(**config)
    encoder = encoder.to(device=device, dtype=torch_dtype)
    return _freeze_foresight_encoder(encoder, source=f"random(seed={int(seed)})")


@torch.no_grad()
def encode_foresight_patch_embedding(
    encoder: nn.Module,
    clean_latents: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Encode a clip with only the frozen teacher's patch embedding (Layer0).

    This is exactly the token tensor returned by ``WanVideoDiT.pre_dit`` before
    its first transformer block, but it intentionally avoids computing unused
    text, timestep, RoPE, and context tensors.
    """
    patches = encoder.patchify(clean_latents)
    if not isinstance(patches, torch.Tensor) or patches.ndim != 5:
        raise ValueError(
            "Wan patch embedding must return [B, D, F, H, W], got "
            f"{type(patches).__name__} shape={getattr(patches, 'shape', None)}."
        )
    batch, dim, frames, height, width = patches.shape
    hidden = patches.permute(0, 2, 3, 4, 1).reshape(
        batch, frames * height * width, dim
    ).contiguous()
    return hidden, int(height * width)


@torch.no_grad()
def patch_foresight_vae_latents(
    clean_latents: torch.Tensor,
    *,
    patch_size: Sequence[int],
) -> tuple[torch.Tensor, int]:
    """Patch clean VAE latents in the exact order used by Wan's Conv3d.

    Latent patches follow the exact Conv3d kernel order ``C, pt, ph, pw`` used
    by Wan's patch embedding. The raw target is 48*1*2*2=192-D for the current
    model. Keeping this operation separate from the fixed 192->3072 lift lets
    the cache store the compact raw representation without changing the loss.
    """
    if clean_latents.ndim != 5:
        raise ValueError(
            f"`clean_latents` must be [B, C, F, H, W], got {tuple(clean_latents.shape)}."
        )
    if len(patch_size) != 3:
        raise ValueError(f"`patch_size` must contain (pt, ph, pw), got {patch_size}.")
    pt, ph, pw = (int(value) for value in patch_size)
    if min(pt, ph, pw) <= 0:
        raise ValueError(f"Patch dimensions must be positive, got {(pt, ph, pw)}.")

    batch, channels, frames, height, width = clean_latents.shape
    if frames % pt or height % ph or width % pw:
        raise ValueError(
            "VAE latent shape must be divisible by the target patch size: "
            f"shape={(frames, height, width)} patch={(pt, ph, pw)}."
        )

    # unfold appends each kernel dimension at the end, producing
    # [B,C,F',H',W',pt,ph,pw].  Permuting places tokens in Wan's F/H/W order
    # and values in Conv3d's C/pt/ph/pw order.
    patches = (
        clean_latents.unfold(2, pt, pt)
        .unfold(3, ph, ph)
        .unfold(4, pw, pw)
    )
    target_frames = frames // pt
    target_height = height // ph
    target_width = width // pw
    raw_dim = channels * pt * ph * pw
    patches = patches.permute(0, 2, 3, 4, 1, 5, 6, 7).reshape(
        batch, target_frames * target_height * target_width, raw_dim
    ).contiguous()

    return patches, int(target_height * target_width)


@torch.no_grad()
def lift_foresight_vae_patches(
    patches: torch.Tensor,
    *,
    output_dim: int,
) -> torch.Tensor:
    """Apply the deterministic isometric raw-VAE-patch feature lift.

    Repeating each raw vector and scaling by ``1/sqrt(repeats)`` exactly
    preserves target norms, dot products and cosine geometry. It contains no
    learned or random parameters and is therefore safe to apply after reading
    compact raw patches from the target cache.
    """
    if patches.ndim != 3:
        raise ValueError(
            f"`patches` must be [B, tokens, raw_dim], got {tuple(patches.shape)}."
        )

    raw_dim = int(patches.shape[-1])
    output_dim = int(output_dim)
    if output_dim < raw_dim or output_dim % raw_dim != 0:
        raise ValueError(
            "`output_dim` must be an integer multiple of the raw VAE patch dim "
            f"{raw_dim}, got {output_dim}."
        )
    repeats = output_dim // raw_dim
    if repeats > 1:
        patches = patches.repeat(1, 1, repeats) / (repeats ** 0.5)
    return patches


@torch.no_grad()
def encode_foresight_vae_direct(
    clean_latents: torch.Tensor,
    *,
    patch_size: Sequence[int],
    output_dim: int,
) -> tuple[torch.Tensor, int]:
    """Turn clean pretrained-VAE latents into deterministic spatial targets.

    This is the historical online VAE-direct path. It is deliberately composed
    from the compact raw patching operation and the fixed isometric lift so the
    online and cached paths share exactly the same implementation.
    """
    patches, tokens_per_frame = patch_foresight_vae_latents(
        clean_latents,
        patch_size=patch_size,
    )
    return (
        lift_foresight_vae_patches(patches, output_dim=output_dim),
        tokens_per_frame,
    )


def build_foresight_ema_shadow(teacher: nn.Module) -> dict[str, torch.Tensor]:
    """Create unregistered fp32 EMA master weights for a bf16/fp16 teacher."""
    return {
        name: parameter.detach().to(dtype=torch.float32).clone()
        for name, parameter in teacher.named_parameters()
    }


@torch.no_grad()
def update_foresight_ema_teacher_(
    teacher: nn.Module,
    student: nn.Module,
    shadow: dict[str, torch.Tensor],
    *,
    decay: float,
) -> None:
    """Update fp32 EMA masters from ``student`` and refresh inference weights."""
    decay = float(decay)
    if not 0.0 <= decay < 1.0:
        raise ValueError(f"EMA decay must be in [0, 1), got {decay}.")

    student_parameters = dict(student.named_parameters())
    teacher_parameters = dict(teacher.named_parameters())
    if set(shadow) != set(teacher_parameters):
        missing = sorted(set(teacher_parameters) - set(shadow))
        unexpected = sorted(set(shadow) - set(teacher_parameters))
        raise ValueError(
            "EMA shadow/teacher parameter mismatch: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}."
        )

    for name, teacher_parameter in teacher_parameters.items():
        student_parameter = student_parameters.get(name)
        if student_parameter is None or student_parameter.shape != teacher_parameter.shape:
            raise ValueError(
                f"EMA student parameter {name!r} is missing or shape-mismatched: "
                f"student={None if student_parameter is None else tuple(student_parameter.shape)}, "
                f"teacher={tuple(teacher_parameter.shape)}."
            )
        master = shadow[name]
        if master.dtype != torch.float32 or master.shape != teacher_parameter.shape:
            raise ValueError(
                f"EMA shadow {name!r} must be fp32 with shape {tuple(teacher_parameter.shape)}, "
                f"got dtype={master.dtype} shape={tuple(master.shape)}."
            )
        master.mul_(decay).add_(student_parameter.detach(), alpha=1.0 - decay)
        teacher_parameter.copy_(master)


@torch.no_grad()
def encode_foresight_hidden_multi(
    encoder: nn.Module,
    clean_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    capture_layers: Sequence[int],
    fuse_vae_embedding_in_latents: bool = True,
) -> tuple[dict[int, torch.Tensor], int]:
    """Single encoder pass capturing hidden states at MULTIPLE depths.

    Bit-identical to calling `encode_foresight_hidden` once per layer (same ops in
    the same order up to each capture point), but runs blocks[:max(capture_layers)]
    ONCE instead of re-running the prefix per layer (e.g. layers [12,14,16,18]:
    18 block passes instead of 12+14+16+18=60).

    Returns:
        hidden_by_layer: {layer: [B, f * tokens_per_frame, D]} after that many blocks.
        tokens_per_frame: token count per latent frame.
    """
    wanted = sorted({int(L) for L in capture_layers})
    num_blocks = len(encoder.blocks)
    if not wanted or not (1 <= wanted[0] and wanted[-1] <= num_blocks):
        raise ValueError(f"`capture_layers` must be within [1, {num_blocks}], got {capture_layers}")

    timestep = torch.zeros(
        (clean_latents.shape[0],), dtype=clean_latents.dtype, device=clean_latents.device
    )
    pre = encoder.pre_dit(
        x=clean_latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
    )
    x = pre["tokens"]
    hidden_by_layer: dict[int, torch.Tensor] = {}
    for depth, block in enumerate(encoder.blocks[: wanted[-1]], start=1):
        x = block(
            x,
            pre["context"],
            pre["t_mod"],
            pre["freqs"],
            context_mask=pre["context_mask"],
            self_attn_mask=None,
        )
        if depth in wanted:
            hidden_by_layer[depth] = x
    return hidden_by_layer, int(pre["meta"]["tokens_per_frame"])


def encode_foresight_hidden(
    encoder: nn.Module,
    clean_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    capture_layer: int,
    fuse_vae_embedding_in_latents: bool = True,
) -> tuple[torch.Tensor, int]:
    """Run the frozen encoder on the clean clip at t=0; return states after `capture_layer` blocks.

    Args:
        clean_latents: Clean VAE latents of the full clip, [B, C, f, H', W'].
        capture_layer: Number of encoder blocks applied before capture (1-indexed depth).

    Returns:
        hidden: [B, f * tokens_per_frame, D] hidden states after `capture_layer` blocks.
        tokens_per_frame: token count per latent frame.
    """
    num_blocks = len(encoder.blocks)
    if not (1 <= int(capture_layer) <= num_blocks):
        raise ValueError(f"`capture_layer` must be in [1, {num_blocks}], got {capture_layer}")

    timestep = torch.zeros(
        (clean_latents.shape[0],), dtype=clean_latents.dtype, device=clean_latents.device
    )
    pre = encoder.pre_dit(
        x=clean_latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=fuse_vae_embedding_in_latents,
    )
    x = pre["tokens"]
    # Bidirectional attention over the full clean clip (self_attn_mask=None).
    for block in encoder.blocks[: int(capture_layer)]:
        x = block(
            x,
            pre["context"],
            pre["t_mod"],
            pre["freqs"],
            context_mask=pre["context_mask"],
            self_attn_mask=None,
        )
    return x, int(pre["meta"]["tokens_per_frame"])


def fuse_future_targets(
    hidden: torch.Tensor,
    tokens_per_frame: int,
    offsets: Optional[Sequence[int]],
    frame_valid: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average encoder hidden states over the foresight window.

    Args:
        hidden: [B, f * tokens_per_frame, D] encoder hidden states.
        offsets: Latent-frame indices to fuse (0 = current obs frame). None = all frames.
        frame_valid: Optional [B, f] bool mask, False for padded latent steps.

    Returns:
        target: [B, tokens_per_frame, D] fused target.
        sample_valid: [B] bool, False when no selected frame is valid for that sample.
    """
    batch_size, seq_len, dim = hidden.shape
    if seq_len % tokens_per_frame != 0:
        raise ValueError(
            f"Hidden seq length {seq_len} not divisible by tokens_per_frame {tokens_per_frame}"
        )
    num_frames = seq_len // tokens_per_frame
    hidden = hidden.view(batch_size, num_frames, tokens_per_frame, dim)

    if offsets is None:
        offsets = list(range(num_frames))
    offsets = [int(o) for o in offsets]
    if len(offsets) == 0:
        raise ValueError("`offsets` must contain at least one frame index.")
    for o in offsets:
        if not (0 <= o < num_frames):
            raise ValueError(f"Foresight offset {o} out of range for {num_frames} latent frames.")

    selected = hidden[:, offsets]  # [B, K, tpf, D]
    if frame_valid is None:
        weights = torch.ones(
            (batch_size, len(offsets)), dtype=hidden.dtype, device=hidden.device
        )
    else:
        if frame_valid.shape != (batch_size, num_frames):
            raise ValueError(
                f"`frame_valid` must be [B, {num_frames}], got {tuple(frame_valid.shape)}"
            )
        weights = frame_valid[:, offsets].to(dtype=hidden.dtype)

    weight_sum = weights.sum(dim=1)  # [B]
    target = (selected * weights[:, :, None, None]).sum(dim=1) / weight_sum.clamp(min=1.0)[
        :, None, None
    ]
    sample_valid = weight_sum > 0
    return target, sample_valid


def per_frame_targets(
    hidden: torch.Tensor,
    tokens_per_frame: int,
    frame_valid: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-frame (un-fused) encoder targets, preserving temporal structure.

    Returns:
        targets: [B, num_frames, tokens_per_frame, D] — one spatial target map per latent frame.
        valid:   [B, num_frames] bool — per-frame validity (False for padded frames).
    """
    batch_size, seq_len, dim = hidden.shape
    num_frames = seq_len // tokens_per_frame
    targets = hidden.view(batch_size, num_frames, tokens_per_frame, dim)
    if frame_valid is None:
        valid = torch.ones((batch_size, num_frames), dtype=torch.bool, device=hidden.device)
    else:
        valid = frame_valid
    return targets, valid

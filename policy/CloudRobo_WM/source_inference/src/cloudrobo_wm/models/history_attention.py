"""Observed RGB history, sharing VideoDiT weights with an isolated state stream.

The old sequence never moves. Concatenation extends only its K/V axis; gated
attention leaves its softmax unchanged. No fourth expert, head or history loss.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .wan_video_dit import sinusoidal_embedding_1d


@dataclass(frozen=True)
class HistoryConfig:
    mode: str = "gated"
    rope: str = "slots"
    offsets: tuple = (-16, -12, -8, -4, 0)
    sampling: str = "uniform_stride4_v1"
    visibility: str = "causal_history_v1"

    @classmethod
    def parse(cls, value):
        value = dict(value or {})
        if "offsets" in value:
            value["offsets"] = tuple(value["offsets"])
        result = cls(**value)
        if result.mode not in ("none", "concat", "gated"):
            raise ValueError("history.mode must be none, concat or gated")
        if result.rope not in ("slots", "end_time"):
            raise ValueError("history.rope must be slots or end_time")
        expected = {"uniform_stride4_v1": (-16,-12,-8,-4,0),
                    "episode_first_multiscale_v2": ("episode_first",-32,-16,-8,-4),
                    "episode_anchor_recent_split_v3": ("episode_first",-20,-16,-12,-8,-4)}
        if result.sampling not in expected or result.offsets != expected[result.sampling]:
            raise ValueError("History sampling name and frame selection disagree")
        if result.visibility not in ("causal_history_v1", "observed_bidirectional_v2", "observed_clean_anchors_v3"):
            raise ValueError("Unknown observed-history visibility contract")
        return result

    def contract(self):
        if self.sampling == "episode_anchor_recent_split_v3":
            return dict(version=4 if self.visibility == "observed_clean_anchors_v3" else 3, **{**asdict(self), "offsets": list(self.offsets)},
                        offset_scale_steps=self.offset_scale_steps, current_in_history=False,
                        vae="independent_anchor_1_to_1_and_recent_5_to_2_keep_second; old_RGB_encode_unchanged",
                        recent_context="actual_t_minus20; first_latent_discarded_context_retained",
                        cache="reset_on_each_encode_call", history_latent_order=["anchor", "recent"],
                        history_latent_frames=2, padding="repeat_episode_first_only_before_episode_start",
                        time_unit="native_control_step", presence="tied_to_present_RGB",
                        prediction_loss=False, original_rope="unchanged", history_timestep=0,
                        time_features="anchor_age_repeated5; recent_actual5_offsets_without_anchor",
                        supervision="JG14_EEF12_flow_matching; no_FK")
        return dict(version=2 if self.sampling == "episode_first_multiscale_v2" else 1,
                    **{**asdict(self), "offsets": list(self.offsets)},
                    offset_scale_steps=self.offset_scale_steps,
                    current_in_history=self.sampling == "uniform_stride4_v1",
                    vae="independent_causal_5_to_2; old_RGB_encode_unchanged",
                    padding="repeat_episode_first", time_unit="native_control_step",
                    presence="tied_to_present_RGB", prediction_loss=False,
                    original_rope="unchanged", history_timestep=0,
                    supervision="JG14_EEF12_flow_matching; no_FK")

    @property
    def offset_scale_steps(self):
        return 32. if self.sampling in ("episode_first_multiscale_v2", "episode_anchor_recent_split_v3") else 16.


def history_positions(offsets, mode, scale=16.):
    if mode == "slots":
        return offsets.new_tensor([-2., -1.]).expand(offsets.shape[0], -1)
    return offsets[:, [0, -1]].float() / scale


def history_rope(expert, positions, height, width):
    """Explicit negative/fractional phases; never negative-index a lookup table."""
    device = positions.device
    # angle(table[1]) exactly recovers each axis' positive unit frequency.
    unit = torch.angle(expert.freqs[0][1]).to(device)
    phase = positions.double()[..., None] * unit
    temporal = torch.polar(torch.ones_like(phase), phase)
    B, T = positions.shape
    return torch.cat((
        temporal[:, :, None, None].expand(B, T, height, width, -1),
        expert.freqs[1][:height].to(device)[None, None, :, None].expand(B, T, height, width, -1),
        expert.freqs[2][:width].to(device)[None, None, None, :].expand(B, T, height, width, -1),
    ), -1).reshape(B, T*height*width, 1, -1)


class HistoryConditioning(nn.Module):
    def __init__(self, hidden_dim, num_layers, config):
        super().__init__()
        self.config = config
        self.role = nn.Parameter(torch.empty(hidden_dim))
        # Actual offsets for the first frame and all four frames in group two.
        # Repeated offsets explicitly represent episode-start padding.
        self.time_projection = nn.Linear(5, hidden_dim, bias=False)
        nn.init.normal_(self.role, std=.02)
        nn.init.normal_(self.time_projection.weight, std=.02)
        if config.mode == "gated":
            # One gate per layer and receiving tower, before the existing O.
            self.gates = nn.Parameter(torch.zeros(num_layers, 3))
        else:
            self.register_parameter("gates", None)

    def prepare(self, expert, latents, offsets, valid, present):
        # Calling forward directly intentionally bypasses the *main RGB* hook.
        # No main validity/time buffers are overwritten by this separate clip.
        x = expert.patch_embedding.forward(latents)
        B, D, T, H, W = x.shape
        if T != 2:
            raise ValueError("History conditioning requires exactly two latent frames")
        x = x.flatten(2).transpose(1, 2)
        first = offsets[:, :1].expand(-1, 5)
        recent = offsets[:, 1:] if self.config.sampling == "episode_anchor_recent_split_v3" else offsets
        groups = torch.stack((first, recent), 1).to(x) / self.config.offset_scale_steps
        extra = self.role.to(x.dtype) + self.time_projection(groups)
        x = x + extra.repeat_interleave(H*W, 1)
        valid = valid & present[:, None]
        x = torch.where(valid[..., None], x, 0)
        zero = torch.zeros(B*T*H*W, device=x.device, dtype=x.dtype)
        t = expert.time_embedding(sinusoidal_embedding_1d(expert.freq_dim, zero))
        mod = expert.time_projection(t).reshape(B, T*H*W, 6, D)
        frame = torch.arange(T, device=x.device).repeat_interleave(H*W)
        causal = (torch.ones_like(frame[:, None] >= frame[None, :])
                  if self.config.visibility in ("observed_bidirectional_v2", "observed_clean_anchors_v3")
                  else frame[:, None] >= frame[None, :])
        hh = causal[None] & valid[:, :, None] & valid[:, None, :]
        # Invalid query fallback is self-only; invalid keys never reach valid Q.
        hh = hh | ((~valid)[:, :, None] & torch.eye(T*H*W, device=x.device, dtype=torch.bool))
        return dict(tokens=x, valid=valid, t_mod=mod, self_mask=hh,
                    freqs=history_rope(expert, history_positions(offsets, self.config.rope, self.config.offset_scale_steps), H, W))


def _with_history(mot, q, k, v, original_mask, hk, hv, history_mask, gate):
    if mot.history_conditioning.config.mode == "concat":
        mask = original_mask
        if mask.ndim == 2:
            mask = mask[None].expand(q.shape[0], -1, -1)
        return mot._mixed_attention(q, torch.cat((k, hk), 1), torch.cat((v, hv), 1),
                                    attention_mask=torch.cat((mask, history_mask), -1))
    old = mot._mixed_attention(q, k, v, attention_mask=original_mask)
    # A zero dummy guarantees a finite softmax even for fully absent history.
    empty = ~history_mask.any(-1, keepdim=True)
    h = mot._mixed_attention(q, torch.cat((hk, torch.zeros_like(hk[:, :1])), 1),
                             torch.cat((hv, torch.zeros_like(hv[:, :1])), 1),
                             attention_mask=torch.cat((history_mask, empty), -1))
    return old + torch.tanh(gate).to(h.dtype) * h


def observed_current_mask(mask, slots, current_valid):
    """Current RGB queries read only current RGB keys; H is added separately.

    No future RGB/DINO/pointmap/action state can re-enter this observed block
    through another layer. Invalid current rows use a self-only fallback.
    """
    if mask.ndim == 2:
        mask = mask[None].expand(slots.shape[0], -1, -1)
    safe = current_valid[:, None, :].expand(-1, slots.shape[1], -1)
    empty = ~safe.any(-1, keepdim=True)
    eye = torch.eye(slots.shape[1], device=slots.device, dtype=torch.bool)[None]
    safe = safe | (empty & eye)
    return torch.where(slots[:, :, None], safe, mask)


def history_layer(mot, layer, xs, h, freqs, mods, contexts, attention_mask,
                  lengths, lens, masks, payload):
    experts = (mot.mixtures["video"], mot.mixtures["action"], mot.mixtures["action_eef"])
    ios = [mot._build_expert_attention_io(expert=e, block=e.blocks[layer],
            x=xs[i], freqs=freqs[i], t_mod=mods[i]) for i, e in enumerate(experts)]
    hi = mot._build_expert_attention_io(expert=experts[0], block=experts[0].blocks[layer],
                                       x=h, freqs=payload["freqs"], t_mod=payload["t_mod"])
    q, k, v = (torch.cat([io[j] for io in ios], 1) for j in range(3))
    hk = torch.where(payload["valid"][..., None], hi[1], 0)
    hv = torch.where(payload["valid"][..., None], hi[2], 0)
    outer = mot.hbridge_enabled and mot._is_outer_layer(layer)
    row = payload["query_valid"]
    if outer:
        # Only the first (RGB) substream reads H in outer HBridge layers.
        row = row & (torch.arange(q.shape[1], device=q.device) < lens[0])[None]
    history_mask = row[:, :, None] & payload["valid"][:, None, :]
    gates = mot.history_conditioning.gates
    gate = (torch.cat([gates[layer, i].expand(n) for i, n in enumerate(lengths)])[None, :, None]
            if gates is not None else None)
    visibility = mot.history_conditioning.config.visibility
    bidirectional = visibility in ("observed_bidirectional_v2", "observed_clean_anchors_v3")
    # Native HBridge masks already keep C0/D0/valid P0 free of future/action
    # inputs at every layer. Preserve their original middle-layer edges.
    restrict_current_to_RGB = visibility == "observed_bidirectional_v2"
    if restrict_current_to_RGB:
        attention_mask = observed_current_mask(attention_mask, payload["current_slots"], payload["current_valid"])
    if outer:
        pieces = []
        start = 0
        for n, mask in zip(lens, masks):
            sl = slice(start, start+n)
            # Do not even extend softmax for non-reading outer substreams.
            if start == 0:
                if restrict_current_to_RGB:
                    mask = observed_current_mask(mask, payload["current_slots"][:, :n], payload["current_valid"][:, :n])
                pieces.append(_with_history(mot, q[:, sl], k[:, sl], v[:, sl], mask,
                    hk, hv, history_mask[:, sl], gate[:, sl] if gate is not None else None))
            else:
                pieces.append(mot._mixed_attention(q[:, sl], k[:, sl], v[:, sl], attention_mask=mask))
            start += n
        mixed = torch.cat(pieces, 1)
    else:
        mixed = _with_history(mot, q, k, v, attention_mask, hk, hv, history_mask, gate)
    pieces = mixed.split(lengths, 1)
    result = tuple(mot._apply_post_with_optional_checkpoint(
        block=e.blocks[layer], residual_x=ios[i][3], mixed_slice=pieces[i],
        gate_msa=ios[i][4], shift_mlp=ios[i][5], scale_mlp=ios[i][6], gate_mlp=ios[i][7],
        use_gradient_checkpointing=ios[i][8], context_payload=contexts[i]) for i, e in enumerate(experts))
    if layer < mot.num_layers-1:
        if bidirectional:
            # Current K/V is taken from this layer's input, exactly like H K/V.
            # Its own updates are restricted at every layer above.
            cm = payload["valid"][:, :, None] & payload["current_valid"][:, None, :]
            hm = mot._mixed_attention(hi[0], torch.cat((hk,k),1), torch.cat((hv,v),1),
                                      attention_mask=torch.cat((payload["self_mask"],cm),-1))
        else:
            hm = mot._mixed_attention(hi[0], hk, hv, attention_mask=payload["self_mask"])
        context = contexts[0]
        mask = context["mask"]
        if mask is not None and mask.ndim == 3:
            mask = mask[:, :1].expand(-1, h.shape[1], -1)
        h = mot._apply_post_with_optional_checkpoint(block=experts[0].blocks[layer],
            residual_x=hi[3], mixed_slice=hm, gate_msa=hi[4], shift_mlp=hi[5],
            scale_mlp=hi[6], gate_mlp=hi[7], use_gradient_checkpointing=hi[8],
            context_payload=dict(context=context["context"], mask=mask))
        h = torch.where(payload["valid"][..., None], h, 0)
    return result, h


def forward_history(mot, video_tokens, action_tokens, video_freqs, action_freqs,
                    video_t_mod, action_t_mod, video_context, action_context,
                    attention_mask, sub_stream_lens, sub_stream_self_masks, payload):
    from .dual_cooperative_action import _contexts, _split_hbridge, _visual_cannot_read_action
    horizon = action_tokens.shape[1] // 2
    lens, masks = _split_hbridge(sub_stream_lens, sub_stream_self_masks, horizon)
    if lens is None:
        raise ValueError("History v1 requires explicit HBridge substream masks")
    attention_mask = _visual_cannot_read_action(attention_mask, video_tokens.shape[1])
    xs = (video_tokens, *action_tokens.chunk(2, 1))
    freqs = (video_freqs, *action_freqs.chunk(2, 0))
    mods = (video_t_mod, *action_t_mod)
    contexts = (video_context, *_contexts(action_context))
    lengths = (video_tokens.shape[1], horizon, horizon)
    h = payload["tokens"]
    for layer in range(mot.num_layers):
        args = (mot, layer, xs, h, freqs, mods, contexts, attention_mask,
                lengths, lens, masks, payload)
        if mot.checkpoint_full_layers and mot.training and torch.is_grad_enabled():
            xs, h = checkpoint(history_layer, *args, use_reentrant=False)
        else:
            xs, h = history_layer(*args)
    return xs[0], torch.cat(xs[1:], 1)

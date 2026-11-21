from typing import Literal

from falcon.config import LazyConfig


class GVAConfig(LazyConfig):
    patch_size: int
    v_in_channels: int
    a_in_channels: int
    a_embed_in_channels: int
    frames: int
    hidden_size: int
    ffn_hidden_size: int
    text_dim: int
    caption_length: int
    num_layers: int
    num_attention_heads: int
    hidden_size_per_attention_head: int
    p_uncond: float
    rope_interpolate_factor: float
    # ND-RoPE axes (flat-CP + fused path). Each entry:
    # {"dim_range": (start, end), "base": float, "interpolate_factor": float}.
    video_rope_axes: list[dict]
    audio_rope_axes: list[dict]
    adaln_hidden_size: int

    def __init__(self, cfg: LazyConfig):
        super().__init__()
        self.patch_size = cfg.model.get("patch_size", 1)
        self.v_in_channels = cfg.model.get("v_in_channels", 48)
        self.a_in_channels = cfg.model.get("a_in_channels", 64)
        self.a_embed_in_channels = cfg.model.get(
            "a_embed_in_channels",
            self.a_in_channels,
        )
        self.frames = cfg.model.get("frames", 1)
        self.hidden_size = cfg.model.get("hidden_size", 4096)
        self.ffn_hidden_size = cfg.model.get("ffn_hidden_size", 10880)
        self.text_dim = cfg.model.get("text_dim", 3584)
        self.caption_length = cfg.model.get("caption_length", 4096)
        self.num_layers = cfg.model.get("num_layers", 24)
        self.num_attention_heads = cfg.model.get("num_attention_heads", 32)
        self.hidden_size_per_attention_head = cfg.model.get("hidden_size_per_attention_head", 128)
        self.p_uncond = cfg.model.get("p_uncond", 0.1)
        self.rope_interpolate_factor = cfg.model.get("rope_interpolate_factor", 50 / 6)
        # Video 3D (F, H, W) RoPE — head_dim split 2:3:3 across axes.
        # Default values are strictly equivalent to the old QKFHWRoPE hardcoded literals:
        #   f_rotary_dim=128*2//8=32  h_rotary_dim=128*3//8=48  w_rotary_dim=48  base=256
        b = self.hidden_size_per_attention_head // 8
        self.video_rope_axes = cfg.model.get(
            "video_rope_axes",
            [
                {"dim_range": (0, b * 2), "base": 256.0, "interpolate_factor": 1.0},
                {"dim_range": (b * 2, b * 5), "base": 256.0, "interpolate_factor": 1.0},
                {"dim_range": (b * 5, b * 8), "base": 256.0, "interpolate_factor": 1.0},
            ],
        )
        # Audio 1D RoPE — a_rotary_dim=32  base=256  interpolate_factor=rope_interpolate_factor
        self.audio_rope_axes = cfg.model.get(
            "audio_rope_axes",
            [
                {"dim_range": (0, 32), "base": 256.0, "interpolate_factor": self.rope_interpolate_factor},
            ],
        )
        self.adaln_hidden_size = cfg.model.get("adaln_hidden_size", None)

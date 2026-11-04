"""Inference-only bundle configuration and shared client observation contract."""

import copy
import hashlib
import json
import math
from pathlib import Path

import yaml


def _keys(value, names, location):
    if not isinstance(value, dict) or set(value) != set(names.split()):
        raise ValueError(f"Unexpected/missing fields in {location}: expected {names}")


def _positive(value, location, *, integer=False):
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{location} must be a finite positive {'integer' if integer else 'number'}"
        )


def bundle_path(root: Path, relative: str) -> Path:
    """Bundle resources cannot escape their root or depend on absolute paths."""
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Bundle resource paths must be nonempty relative paths")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Bundle resource escapes root: {relative}")
    return path


class ObservationConfig:
    """The small, validated subset sent to clients before any preprocessing."""

    def __init__(self, data):
        _keys(
            data,
            "preprocessing head_size_hw history_interval_steps history_max_frames control_clock_hz action_horizon",
            "observation contract",
        )
        if data["preprocessing"] != "tri_view_pad_resize_v1":
            raise ValueError("Unsupported image preprocessing")
        size = data["head_size_hw"]
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError("head_size_hw must be [height, width]")
        for key in (
            "history_interval_steps",
            "history_max_frames",
            "action_horizon",
            "control_clock_hz",
        ):
            _positive(data[key], key, integer=True)
        for value in size:
            _positive(value, "head_size_hw", integer=True)
            if value % 2:
                raise ValueError("Head dimensions must be even for the wrist mosaic")
        self.data = copy.deepcopy(data)
        self.height, self.width = size
        self.interval = data["history_interval_steps"]
        self.max_frames = data["history_max_frames"]
        self.horizon = data["action_horizon"]

    def offsets(self, step):
        return list(
            range(
                -min(step // self.interval, self.max_frames) * self.interval,
                0,
                self.interval,
            )
        )


class ModelConfig:
    """One source of truth; unsupported semantic versions fail rather than fall back."""

    def __init__(self, data):
        _keys(
            data,
            "format_version architecture weight_format weight_dtype assets backbones model observation history video rope conditioning normalization sampler inference_defaults",
            "config",
        )
        if type(data["format_version"]) is not int:
            raise ValueError("format_version must be an integer")
        if (
            data["format_version"],
            data["architecture"],
            data["weight_format"],
            data["weight_dtype"],
        ) != (1, "HWM0Preview", "hwm0_preview_v1", "bfloat16"):
            raise ValueError("Unsupported model bundle format/architecture/dtype")
        self.data = copy.deepcopy(data)
        for name in (
            "assets",
            "backbones",
            "model",
            "history",
            "video",
            "rope",
            "conditioning",
            "normalization",
            "sampler",
            "inference_defaults",
        ):
            setattr(self, name, self.data[name])
        _keys(self.assets, "weights statistics processor tokenizer", "assets")
        for relative in self.assets.values():
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).is_absolute()
                or ".." in Path(relative).parts
                or ":" in relative
            ):
                raise ValueError("Bundle assets must be portable relative paths")
        if len(set(self.assets.values())) != len(self.assets) or any(
            x in {"config.yaml", "manifest.json"} for x in self.assets.values()
        ):
            raise ValueError(
                "Bundle resource names must be distinct and not overwrite metadata"
            )
        _keys(self.backbones, "vlm text vae", "backbones")
        _keys(
            self.backbones["vlm"]["text_config"],
            "vocab_size hidden_size intermediate_size num_hidden_layers num_attention_heads num_key_value_heads head_dim hidden_act rms_norm_eps attention_bias attention_dropout attn_output_gate full_attention_interval layer_types linear_conv_kernel_dim linear_key_head_dim linear_num_key_heads linear_num_value_heads linear_value_head_dim mamba_ssm_dtype max_position_embeddings tie_word_embeddings eos_token_id bos_token_id pad_token_id rope_parameters",
            "backbones.vlm.text_config",
        )
        _keys(
            self.backbones["vlm"]["vision_config"],
            "depth hidden_size intermediate_size num_heads in_channels out_hidden_size hidden_act patch_size spatial_merge_size temporal_patch_size num_position_embeddings deepstack_visual_indexes",
            "backbones.vlm.vision_config",
        )
        _keys(
            self.backbones["vlm"]["text_config"]["rope_parameters"],
            "rope_type rope_theta partial_rotary_factor mrope_interleaved mrope_section",
            "backbones.vlm.rope_parameters",
        )
        _keys(
            self.backbones["vlm"],
            "image_token_id video_token_id vision_start_token_id vision_end_token_id tie_word_embeddings text_config vision_config",
            "backbones.vlm",
        )
        _keys(
            self.backbones["text"],
            "vocab_size d_model d_ff d_kv num_layers num_decoder_layers num_heads feed_forward_proj dense_act_fn relative_attention_num_buckets relative_attention_max_distance layer_norm_epsilon dropout_rate classifier_dropout tie_word_embeddings scalable_attention decoder_start_token_id pad_token_id eos_token_id",
            "backbones.text",
        )
        _keys(
            self.backbones["vae"],
            "base_dim decoder_base_dim z_dim dim_mult num_res_blocks attn_scales temperal_downsample dropout in_channels out_channels patch_size scale_factor_spatial scale_factor_temporal is_residual",
            "backbones.vae",
        )
        _keys(
            self.model,
            "action_dim state_dim action_horizon world_queries layers heads head_dim time_width norm_eps video_width video_ffn_width action_width action_ffn_width patch_size history_kernels",
            "model",
        )
        for name, value in self.model.items():
            if name not in ("patch_size", "history_kernels"):
                _positive(value, name, integer=name != "norm_eps")
        self.m = self.model
        _keys(data["observation"], "preprocessing head_size_hw", "observation")
        _keys(
            self.history,
            "sample_interval_steps max_frames control_clock_hz vlm_pool_grid packing near_latents",
            "history",
        )
        self.observation = ObservationConfig(
            {
                **data["observation"],
                "history_interval_steps": self.history["sample_interval_steps"],
                "history_max_frames": self.history["max_frames"],
                "control_clock_hz": self.history["control_clock_hz"],
                "action_horizon": self.m["action_horizon"],
            }
        )
        _keys(self.video, "rgb_frames frame_interval_steps", "video")
        for name, value in self.video.items():
            _positive(value, name, integer=True)
        _keys(self.rope, "base video_axes", "rope")
        _positive(self.rope["base"], "rope.base")
        axes = self.rope["video_axes"]
        if (
            not isinstance(axes, list)
            or len(axes) != 3
            or any(type(x) is not int or x <= 0 or x % 2 for x in axes)
            or sum(axes) != self.m["head_dim"]
        ):
            raise ValueError(
                "RoPE axes must be three positive even widths summing to head_dim"
            )
        if self.m["time_width"] % 2:
            raise ValueError("time_width must be even")
        if self.m["action_dim"] != self.m["state_dim"]:
            raise ValueError(
                "This policy requires matching joint state/action dimensions"
            )
        if self.history["packing"] != "hwm_historypack_v1":
            raise ValueError("Unsupported history packing semantics")
        _positive(self.history["near_latents"], "history.near_latents", integer=True)
        pool = self.history["vlm_pool_grid"]
        if not isinstance(pool, list) or len(pool) != 2:
            raise ValueError("vlm_pool_grid must have two dimensions")
        for value in pool:
            _positive(value, "vlm_pool_grid", integer=True)
        patch = self.m["patch_size"]
        if not isinstance(patch, list) or len(patch) != 3 or patch[0] != 1:
            raise ValueError("The causal sampler requires temporal patch size 1")
        for value in patch:
            _positive(value, "patch_size", integer=True)
        kernels = self.m["history_kernels"]
        if not isinstance(kernels, list) or len(kernels) != 3 or kernels[0] != patch:
            raise ValueError(
                "History packing requires three kernels; first equals video patch"
            )
        for kernel in kernels:
            if not isinstance(kernel, list) or len(kernel) != 3:
                raise ValueError("History kernels must be triplets")
            for value in kernel:
                _positive(value, "history kernel", integer=True)
            if any(kernel[i] % patch[i] for i in (1, 2)):
                raise ValueError(
                    "History spatial kernels must be multiples of video patch"
                )
        for key in ("z_dim", "scale_factor_spatial", "scale_factor_temporal"):
            _positive(self.backbones["vae"][key], key, integer=True)
        vae = self.backbones["vae"]
        if self.spatial_stride != vae["patch_size"] * 2 ** (
            len(vae["dim_mult"]) - 1
        ) or self.temporal_stride != 2 ** sum(vae["temperal_downsample"]):
            raise ValueError("Declared VAE compression must match its architecture")
        if (self.video["rgb_frames"] - 1) % self.temporal_stride or self.video[
            "rgb_frames"
        ] < 2:
            raise ValueError("Video frame count must be 1 + N * VAE temporal stride")
        if (self.video["rgb_frames"] - 1) * self.video[
            "frame_interval_steps"
        ] != self.m["action_horizon"]:
            raise ValueError("Video coverage must match the action horizon")
        if self.history["near_latents"] > kernels[1][0]:
            raise ValueError("Near history selection exceeds its temporal kernel")
        max_latents = self.history["max_frames"] // self.temporal_stride + 1
        if max(0, max_latents - 2 - self.history["near_latents"]) > kernels[2][0]:
            raise ValueError("History window exceeds this packing layout's far kernel")
        for size, patch_dim in zip(
            (self.observation.height, self.observation.width), patch[1:], strict=True
        ):
            if size % (2 * self.spatial_stride * patch_dim):
                raise ValueError(
                    "Head and wrist sizes must align with VAE and video patches"
                )
        _keys(
            self.conditioning,
            "version joint_attention action_context video_prompt_template t5_max_length t5_padding planner_prompt history_prompt current_prompt max_update_tokens retry_token_budgets special_tokens eos_token_ids",
            "conditioning",
        )
        if (
            self.conditioning["version"],
            self.conditioning["joint_attention"],
            self.conditioning["action_context"],
            self.conditioning["t5_padding"],
        ) != (
            "hwm_memory_v1",
            "fastwam_joint_v1",
            "vlm_queries_and_state",
            "zero_embeddings_visible",
        ):
            raise ValueError(
                "Unsupported conditioning semantics; cannot change the KV-cache invariant"
            )
        for name in ("t5_max_length", "max_update_tokens"):
            _positive(self.conditioning[name], name, integer=True)
        budgets = self.conditioning["retry_token_budgets"]
        if not isinstance(budgets, list) or len(budgets) != 2:
            raise ValueError("Memory/subtask retries require two token budgets")
        for value in budgets:
            _positive(value, "retry budget", integer=True)
        for key in (
            "planner_prompt",
            "history_prompt",
            "current_prompt",
            "video_prompt_template",
        ):
            if (
                not isinstance(self.conditioning[key], str)
                or not self.conditioning[key]
            ):
                raise ValueError(f"Missing prompt: {key}")
        # Validate template fields without executing code or reading external files.
        self.conditioning["planner_prompt"].format(
            instruction="task", memory="None.", subtask="None."
        )
        self.conditioning["history_prompt"].format(interval_seconds=1)
        self.conditioning["video_prompt_template"].format(
            task="task", subtask="subtask"
        )
        tokens = self.conditioning["special_tokens"]
        eos = self.conditioning["eos_token_ids"]
        if not isinstance(eos, list) or not eos or len(set(eos)) != len(eos):
            raise ValueError("eos_token_ids must be a nonempty unique list")
        if not isinstance(tokens, dict) or set(tokens) != {
            "<WORLD_PLAN>",
            "<ACTION_PLAN>",
            "<KEEP>",
            "<UPDATE>",
        }:
            raise ValueError("Unsupported planner token protocol")
        for value in [*tokens.values(), *self.conditioning["eos_token_ids"]]:
            if (
                type(value) is not int
                or not 0 <= value < self.backbones["vlm"]["text_config"]["vocab_size"]
            ):
                raise ValueError("Token IDs must be within the model vocabulary")
        if len(set(tokens.values())) != len(tokens):
            raise ValueError("Planner special token IDs must be unique")
        _keys(self.normalization, "key method epsilon state_clip", "normalization")
        if self.normalization["method"] != "mean_std":
            raise ValueError("Unsupported normalization")
        _positive(self.normalization["epsilon"], "normalization.epsilon")
        clip = self.normalization["state_clip"]
        if (
            not isinstance(clip, list)
            or len(clip) != 2
            or any(type(v) not in (float, int) or not math.isfinite(v) for v in clip)
            or clip[0] >= clip[1]
        ):
            raise ValueError("Invalid state clip")
        _keys(self.sampler, "version timestep_scale frequency_base", "sampler")
        if self.sampler["version"] != "cpu_fp32_euler_flow_v1":
            raise ValueError("Unsupported sampler/numerical semantics")
        _positive(self.sampler["timestep_scale"], "timestep_scale")
        _positive(self.sampler["frequency_base"], "frequency_base")
        _keys(
            self.inference_defaults,
            "num_inference_steps flow_shift replan_steps",
            "inference_defaults",
        )
        self.resolve_inference({})

    @classmethod
    def load(cls, path):
        with Path(path).open() as handle:
            return cls(yaml.safe_load(handle))

    def resolve_inference(self, deploy):
        values = {
            k: self.inference_defaults[k] if deploy.get(k) is None else deploy[k]
            for k in self.inference_defaults
        }
        for key in ("num_inference_steps", "replan_steps"):
            _positive(values[key], key, integer=True)
        _positive(values["flow_shift"], "flow_shift")
        if values["replan_steps"] > self.m["action_horizon"]:
            raise ValueError("replan_steps exceeds the model action horizon")
        return values

    @property
    def fingerprint(self):
        return hashlib.sha256(
            json.dumps(self.data, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @property
    def spatial_stride(self):
        return self.backbones["vae"]["scale_factor_spatial"]

    @property
    def temporal_stride(self):
        return self.backbones["vae"]["scale_factor_temporal"]

    @property
    def channels(self):
        return self.backbones["vae"]["z_dim"]

    @property
    def planner_width(self):
        return self.backbones["vlm"]["text_config"]["hidden_size"]

    @property
    def video_shape(self):
        return (
            self.channels,
            (self.video["rgb_frames"] - 1) // self.temporal_stride + 1,
            self.observation.height * 3 // 2 // self.spatial_stride,
            self.observation.width // self.spatial_stride,
        )

    @property
    def grid(self):
        return tuple(
            x // p
            for x, p in zip(self.video_shape[1:], self.m["patch_size"], strict=True)
        )

    @property
    def head_grid(self):
        return (
            self.observation.height // self.spatial_stride // self.m["patch_size"][1],
            self.grid[2],
        )

    @property
    def tokens_per_frame(self):
        return self.grid[1] * self.grid[2]

    @property
    def video_tokens(self):
        return math.prod(self.grid)

    @property
    def latent_time_steps(self):
        return self.video["frame_interval_steps"] * self.temporal_stride

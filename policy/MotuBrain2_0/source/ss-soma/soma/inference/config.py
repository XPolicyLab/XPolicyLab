"""Embodied policy inference configuration.

All config fields are defined here as dataclasses with defaults.
Config files instantiate these directly — typos are caught at import time.

Model architecture config is auto-loaded from the training config.yaml
alongside the checkpoint, so ModelConfig is not needed here.
"""

import dataclasses
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

from morph import ActionProfile
from morph import PromptBuilder
from morph.convs.image import ResizeConfig

from soma.data import conditions
from soma.data import visual_history
from soma.inference.diffusion.solvers import BaseSolver, EulerSolver

logger = logging.getLogger(__name__)

_TRAINING_ONLY_MODEL_FIELDS = (
    "action_loss_weight",
    "action_component_sizes",
    "action_component_weights",
    "action_norm_path",
    "action_norm_dims",
    "action_norm_eps",
    "a_in_channels",
)


# ---------------------------------------------------------------------------
# Action representation postprocessing
# ---------------------------------------------------------------------------


@dataclass
class ActionConfig:
    """Action config: data frequency + representation postprocessing.

    ``profile`` is the shared Morph contract used by training and inference.
    It converts model output to robot-ready actions and defines the static
    model-space mask.
      state_key:  key in state dict for ref_state (needed for relative/delta)
      camera_key: observation kwarg containing T_base_camera for camera-frame EEF
    """

    video_action_freq: int = 2
    state_key: str = "eef"
    camera_key: Optional[str] = None
    embodiment: str | None = None
    state_field_order: tuple[str, ...] | None = None
    output_field_order: tuple[str, ...] | None = None
    profile: ActionProfile | None = None


# ---------------------------------------------------------------------------
# Policy config
# ---------------------------------------------------------------------------


@dataclass
class VaeConfig:
    type: str = "Wan2_2_VAE"
    # Required: set from the policy config or the training snapshot.
    vae_pth: str = ""
    # Autocast precision for VAE encode/decode: "fp32" / "bf16" / "fp16".
    precision: str = "fp32"
    # Preserve causal encoder features across consecutive observation chunks.
    incremental_encode: bool = False


@dataclass
class TextEncoderConfig:
    type: str = "T5GemmaEmbedder"
    # Required: set from the policy config or the training snapshot.
    pretrained_model_name_or_path: str = ""
    max_length: int = 4096
    # CPU tensor bytes per fixed encoder runtime/deployment; zero disables LRU retention.
    cache_bytes: int = 4 * 1024**3

    def __post_init__(self) -> None:
        if self.cache_bytes < 0:
            raise ValueError("text encoder cache_bytes must be non-negative")


@dataclass(frozen=True)
class GeometryConfig:
    """Frozen DA3 inputs used to condition Motubrain actions."""

    config_path: str
    checkpoint_path: str
    views: Mapping[str, conditions.ConditionViewConfig]
    layer_map: Mapping[int, int]
    condition_name: str = "geometry"
    image_size: tuple[int, int] = (280, 504)
    required_image_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.config_path or not self.checkpoint_path:
            raise ValueError("geometry config and checkpoint paths are required")
        if not self.condition_name or not self.layer_map:
            raise ValueError("geometry condition_name and layer_map must not be empty")
        if not self.views:
            raise ValueError("geometry views must not be empty")
        if any(not isinstance(view, conditions.ConditionViewConfig) for view in self.views.values()):
            raise TypeError("geometry views values must be ConditionViewConfig")
        if len(set(self.required_image_names)) != len(self.required_image_names):
            raise ValueError("required_image_names must be unique")
        if len(self.image_size) != 2 or any(size <= 0 or size % 14 for size in self.image_size):
            raise ValueError("geometry image_size must contain positive multiples of 14")


@dataclass
class MultiviewConfig:
    num_views: int = 3
    view_names: list[str] = field(default_factory=lambda: ["cam_left_wrist", "cam_high", "cam_right_wrist"])
    rope_mode: str = "hw_offset"
    # Full per-view overrides. Views absent here use ObservationConfig.resize.
    view_resizes: dict[str, ResizeConfig] = field(default_factory=dict)
    # Observation key -> training camera name, used only for view embedding.
    view_embed_aliases: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = set(self.view_resizes) - set(self.view_names)
        if unknown:
            raise ValueError(f"view_resizes contains unknown views: {sorted(unknown)}")
        if any(not isinstance(resize, ResizeConfig) for resize in self.view_resizes.values()):
            raise TypeError("view_resizes values must be Morph ResizeConfig")


@dataclass
class ObservationConfig:
    resize: ResizeConfig = field(
        default_factory=lambda: ResizeConfig(
            width=320,
            height=384,
            mode="direct",
        )
    )
    num_frames: int = 13
    vae_latent_channels: int = 48

    # Wire config: how the controller should prepare and send Observations.
    # Controllers query this via the /config handshake endpoint on startup.
    #   video_length:  number of buffered frames to return per get_observation
    #                  (1=latest only, -1=all since last get, N=last N)
    #   image_format:  "raw" | "png" | "jpeg" — wire encoding
    #   jpeg_quality:  JPEG quality (only used when image_format="jpeg")
    video_length: int = 1
    image_format: str = "jpeg"
    jpeg_quality: int = 95

    # Optional codec conditioning after view composition and spatial resize.
    # None passes pixels directly to the VAE; an integer selects H.264 CRF.
    post_resize_h264_crf: int | None = None
    post_resize_h264_repeat_frames: int = 73
    post_resize_h264_threads: int = 16

    # Torso-camera vertical stitching before the final observation resize.
    use_cam_torso: bool = False
    torso_resize: ResizeConfig | None = None
    main_resize: ResizeConfig | None = None
    torso_camera_keys: tuple = ("cam_torso", "torso_camera", "torso")
    main_camera_keys: tuple = ("cam_high", "head_camera", "head", "video")

    def __post_init__(self) -> None:
        if not isinstance(self.resize, ResizeConfig):
            raise TypeError("observation resize must be a Morph ResizeConfig")
        if self.post_resize_h264_crf is not None and (
            type(self.post_resize_h264_crf) is not int or not 0 <= self.post_resize_h264_crf <= 51
        ):
            raise ValueError("post_resize_h264_crf must be None or an integer in [0, 51]")
        for name in ("post_resize_h264_repeat_frames", "post_resize_h264_threads"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.use_cam_torso and (self.torso_resize is None or self.main_resize is None):
            raise ValueError("use_cam_torso requires torso_resize and main_resize")
        for name, resize in (
            ("torso_resize", self.torso_resize),
            ("main_resize", self.main_resize),
        ):
            if resize is not None and not isinstance(resize, ResizeConfig):
                raise TypeError(f"{name} must be a Morph ResizeConfig")


@dataclass(frozen=True)
class SparseMemoryConfig:
    """Select causal history using a stride measured in captured observations."""

    enabled: bool = False
    frame_stride: int = 3

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise TypeError("sparse memory enabled must be a boolean")
        if self.enabled:
            visual_history.history_offsets(self.frame_stride)


@dataclass
class PolicyConfig:
    type: object = None

    # DIT factory function (dotted path or callable)
    dit_factory: object = None

    # Encoders
    vae: VaeConfig = field(default_factory=VaeConfig)
    text_encoder: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    geometry: GeometryConfig | None = None

    # Observation / action
    observation: ObservationConfig = field(default_factory=ObservationConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    prompt_builder: PromptBuilder | None = None
    sparse_memory: SparseMemoryConfig = field(default_factory=SparseMemoryConfig)

    # Common policy params
    patch_size: Optional[int] = None  # override model patch_size if set
    head_split_num: int = 0

    # Multiview
    multiview: MultiviewConfig = field(default_factory=MultiviewConfig)


# ---------------------------------------------------------------------------
# Inference config
# ---------------------------------------------------------------------------


@dataclass
class CompileConfig:
    dit: bool = True
    vae: bool = True
    mode: str = "reduce-overhead"
    fullgraph: bool = False
    dynamic: bool = False
    cache_dir: Optional[str] = None


@dataclass
class SimpleRTCConfig:
    enabled: bool = False
    type: str = "simple"  # "simple" | "train"
    auto_delay: bool = True
    delay: int = 0
    fusion_window: Optional[int] = None
    decay: str = "exponential"

    def __post_init__(self) -> None:
        if self.type not in ("simple", "train"):
            raise ValueError("SimpleRTCConfig.type must be 'simple' or 'train', got " f"{self.type!r}")


@dataclass
class InferenceConfig:
    # ODE sampling
    steps: int = 30
    timeshift: float = 6.0
    action_timeshift: Optional[float] = None  # None = same as timeshift
    solver: type[BaseSolver] = EulerSolver
    decode_video: bool = False

    # Video denoising steps. None = same as `steps` (joint denoising).
    # 0 = skip video entirely (action-only). N = denoise video for N steps then
    # freeze and continue action-only.
    video_steps: Optional[int] = None

    # Modes
    # 0 disables smoothing; values > 0 enable Savitzky-Golay smoothing.
    action_smoothing_window: int = 0

    # CFG
    cfg_scale: Optional[float] = None

    # DiT caching
    dit_cache_threshold: Optional[float] = None
    dit_cache_skip_steps: int = 4
    dit_cache_threshold_2: Optional[float] = None
    dit_cache_skip_steps_2: int = 2
    dit_cache_damping: float = 0.25

    # Sub-configs
    simple_rtc: SimpleRTCConfig = field(default_factory=SimpleRTCConfig)
    compile: CompileConfig = field(default_factory=CompileConfig)

    # This release requires direct inference on one GPU.
    context_parallel_size: int = 1

    # Only BF16 inference is supported; enabling INT8 is rejected.
    int8: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------


@dataclass
class EmbodiedConfig:
    checkpoint_path: Optional[str] = None
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Model config dict — auto-loaded from training config.yaml.
    # Not a dataclass because it varies by model architecture.
    # Passed as **kwargs to build_model().
    model_kwargs: dict = field(default_factory=dict)

    @staticmethod
    def load(config_path: str, checkpoint: str = None) -> "EmbodiedConfig":
        """Load config from a Python file.

        The file should define `policy`, `inference` as dataclass instances,
        and optionally `checkpoint_path`.
        """
        import importlib.util

        spec = importlib.util.spec_from_file_location("_cfg", os.path.abspath(config_path))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        policy = getattr(mod, "policy", PolicyConfig())

        cfg = EmbodiedConfig(
            checkpoint_path=getattr(mod, "checkpoint_path", None),
            policy=policy,
            inference=getattr(mod, "inference", InferenceConfig()),
        )
        if checkpoint is not None:
            cfg.checkpoint_path = checkpoint
        return cfg

    def load_model_config(self, config_path: str | None = None):
        """Load an explicit model snapshot, or discover config.yaml by the checkpoint.

        Searches parent directories of checkpoint_path for config.yaml.
        Populates model_kwargs, encoder defaults, action config, and dit_factory.
        """
        if not self.checkpoint_path:
            raise ValueError("checkpoint_path is required to auto-load model config")

        config_yaml = config_path if config_path is not None else _find_training_config(self.checkpoint_path)
        if config_yaml is None:
            raise FileNotFoundError(f"No config.yaml found in parent directories of {self.checkpoint_path}")
        logger.info(f"Auto-loading model config from {config_yaml}")

        train_cfg = load_training_config(config_yaml)

        model_type = train_cfg.get("model", {}).get("type")
        if model_type and self.policy.dit_factory is None:
            self.policy.dit_factory = _resolve_falcon_model_type(model_type)
        self.model_kwargs = inference_model_kwargs(train_cfg)

        # Auto-load encoder settings from the online training preprocessor.
        # Explicit inference-config values take precedence over these defaults.
        batch_preps = train_cfg.get("data", {}).get("batch_preprocessors", [])
        vae_train_cfg = {}
        text_encoder_train_cfg = {}
        for bp in batch_preps:
            if not isinstance(bp, dict):
                continue
            if not vae_train_cfg and isinstance(bp.get("vae_cfg"), dict):
                vae_train_cfg = bp["vae_cfg"]
            if not text_encoder_train_cfg and isinstance(bp.get("text_encoder_cfg"), dict):
                text_encoder_train_cfg = bp["text_encoder_cfg"]

        _apply_training_defaults(
            self.policy.vae,
            VaeConfig(),
            vae_train_cfg,
            ("type", "vae_pth"),
            "VAE",
        )
        _apply_training_defaults(
            self.policy.text_encoder,
            TextEncoderConfig(),
            text_encoder_train_cfg,
            ("type", "pretrained_model_name_or_path", "max_length"),
            "text encoder",
        )

        # Auto-load data params from sampler config
        ds_cfg = train_cfg.get("data", {}).get("dataloader", {}).get("dataset", {})
        sampler_cfg = ds_cfg.get("sampler")
        if isinstance(sampler_cfg, dict):
            _alias = {"vaf": "video_action_freq", "vdr": "video_downsample_rate", "lvf": "latent_video_freq"}
            for short, long in _alias.items():
                if short in sampler_cfg:
                    sampler_cfg[long] = sampler_cfg[short]

            vaf = sampler_cfg.get("video_action_freq")
            if vaf is not None:
                self.policy.action.video_action_freq = vaf
                logger.info(f"  video_action_freq={vaf} (from sampler)")

            nf = sampler_cfg.get("num_frames")
            if nf is not None:
                self.policy.observation.num_frames = nf
                logger.info(f"  num_frames={nf} (from sampler)")

        # Auto-load noise scheduler params
        ns = train_cfg.get("noise_scheduler", {})
        if "timeshift" in ns:
            self.inference.timeshift = ns["timeshift"]
        if "action_timeshift" in ns:
            self.inference.action_timeshift = ns["action_timeshift"]
        if "timeshift" in ns or "action_timeshift" in ns:
            logger.info(f"  timeshift={self.inference.timeshift}, action_timeshift={self.inference.action_timeshift}")

        self.validate_action_conditions()
        self._load_history_config(train_cfg)

        return self

    def _load_history_config(self, train_cfg: Mapping[str, Any]) -> None:
        """Resolve history encoding and captured-observation spacing from ARIO."""
        sources = train_cfg.get("data", {}).get("dataloader", {}).get("sources", {})
        samplings = [source.get("dataset", {}).get("config", {}).get("sampling", {}) for source in sources.values()]
        histories = [sampling.get("sparse_history") for sampling in samplings]
        enabled = self.policy.sparse_memory.enabled
        if not enabled and not any(history is not None for history in histories):
            return
        if not enabled:
            raise ValueError("memory-trained checkpoints require policy.sparse_memory.enabled=True")
        if not histories or any(history is None for history in histories):
            raise ValueError("memory inference requires compressed history in every training source")
        windows = set()
        for sampling, history in zip(samplings, histories, strict=True):
            encoding = history.get("encoding")
            if encoding != "causal_9_to_3":
                raise ValueError("this release requires checkpoint history encoding causal_9_to_3")
            sampler = sampling["index_sampler"]
            num_frames = sampler["num_frames"]
            frame_stride = history.get("frame_stride")
            visual_history.history_offsets(frame_stride)
            video_action_freq = sampler["video_action_freq"]
            video_downsample_rate = sampler["video_downsample_rate"]
            windows.add((num_frames, video_action_freq, frame_stride, video_downsample_rate))
        if len(windows) != 1:
            raise ValueError("memory training sources must share prediction and history sampling settings")
        num_frames, video_action_freq, frame_stride, video_downsample_rate = windows.pop()
        if self.policy.observation.video_length != -1:
            raise ValueError("compressed memory requires observation.video_length=-1")
        if any(type(value) is not int or value < 1 for value in (video_downsample_rate, video_action_freq)):
            raise ValueError("nine-frame memory requires positive integer video and action sampling rates")
        frame_stride, remainder = divmod(frame_stride * video_action_freq, video_downsample_rate)
        if remainder or frame_stride < 1:
            raise ValueError(
                "raw-frame history stride cannot be represented exactly on the action observation grid"
            )
        self.policy.observation.num_frames = num_frames
        self.policy.action.video_action_freq = video_action_freq
        self.policy.sparse_memory = dataclasses.replace(
            self.policy.sparse_memory,
            frame_stride=frame_stride,
        )

    def validate_action_conditions(self) -> None:
        """Require policy encoders to supply the declared model inputs."""
        configured = self.model_kwargs.get("action_conditions", {})
        geometry = self.policy.geometry
        supplied = {geometry.condition_name} if geometry is not None else set()
        if set(configured) != supplied:
            raise ValueError("policy condition encoders must match model.action_conditions")
        if geometry is not None:
            if set(geometry.layer_map) != set(configured[geometry.condition_name]["layers"]):
                raise ValueError("geometry layer_map must cover exactly the configured action layers")


def inference_model_kwargs(
    train_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract inference model constructor kwargs from a training config."""
    model_kwargs = dict(train_config.get("model", {}))
    model_type = model_kwargs.pop("type", None)
    if model_type is not None:
        _resolve_falcon_model_type(str(model_type))
    for name in _TRAINING_ONLY_MODEL_FIELDS:
        model_kwargs.pop(name, None)
    return model_kwargs


def _apply_training_defaults(
    target,
    defaults,
    source: dict,
    names: tuple[str, ...],
    label: str,
) -> None:
    """Apply training values only where inference config retains its defaults."""
    for name in names:
        if name not in source or getattr(target, name) != getattr(defaults, name):
            continue
        setattr(target, name, source[name])
        logger.info("  %s.%s=%s (from training config)", label, name, source[name])


def load_training_config(config_path: str) -> dict[str, Any]:
    """Read a training YAML snapshot without constructing its Python objects."""
    if "://" in config_path:
        import megfile

        config_file = megfile.smart_open(config_path, "r")
    else:
        config_file = open(config_path, encoding="utf-8")
    with config_file:
        yaml.add_multi_constructor("tag:yaml.org,2002:python/", _yaml_ignore_python, Loader=yaml.SafeLoader)
        return yaml.safe_load(config_file)


def _find_training_config(checkpoint_path: str) -> Optional[str]:
    """Walk parent directories to find config.yaml."""
    if "://" in checkpoint_path:
        from megfile import smart_isfile, smart_path_join

        path = checkpoint_path.rstrip("/")
        if path.rsplit("/", 1)[-1].startswith("step-"):
            path = path.rsplit("/", 1)[0]
        for _ in range(5):
            candidate = smart_path_join(path, "config.yaml")
            if smart_isfile(candidate):
                return candidate
            path = path.rsplit("/", 1)[0]
        return None

    path = os.path.abspath(checkpoint_path)
    if os.path.isfile(path):
        path = os.path.dirname(path)
    if os.path.basename(path).startswith("step-"):
        path = os.path.dirname(path)
    for _ in range(5):  # max 5 levels up
        candidate = os.path.join(path, "config.yaml")
        if os.path.isfile(candidate):
            return candidate
        path = os.path.dirname(path)
    return None


def _yaml_ignore_python(loader, tag_suffix, node):
    """Handle !!python/object tags in yaml: extract kwargs as dict instead of constructing Python objects."""
    if hasattr(node, "value") and isinstance(node.value, list):
        # !!python/object with kwargs → return as dict
        try:
            return loader.construct_mapping(node)
        except Exception:
            pass
    return None


def _resolve_falcon_model_type(model_type: str):
    """Resolve the supported Motubrain model type to the inference builder."""
    from falcon.utils.locate import locate

    class_name = model_type.rsplit(".", 1)[-1] if "." in model_type else model_type
    if class_name != "MotubrainModel":
        raise ValueError("Soma inference only supports Falcon MotubrainModel, got " f"{model_type!r}")
    return locate("soma.inference.runtime.motubrain.build_model")

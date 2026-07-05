from __future__ import annotations

from dataclasses import asdict, dataclass, field
from glob import glob
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from omegaconf import DictConfig
from omegaconf import OmegaConf as om
from omegaconf.errors import OmegaConfBaseException
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

# from a1.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

from .aliases import PathOrStr
from .exceptions import OLMoConfigurationError
from .tokenizer import build_tokenizer
from .util import StrEnum

__all__ = [
    "ActivationType",
    "ActivationCheckpointingStrategy",
    "BlockType",
    "LayerNormType",
    "VisionBackboneType",
    "VisionBackboneConfig",
    "InitFnType",
    "ModelConfig",
    "OptimizerType",
    "OptimizerConfig",
    "SchedulerType",
    "SchedulerConfig",
    "DataConfig",
    "DatasetEvaluatorConfig",
    "TokenizerConfig",
    "TrainConfig",
    "PaddingDirection",
    "SpeedMonitorConfig",
    "WandbConfig",
    "CompilerConfig",
    "WandbConfig",
    "FSDPPrecision",
    "FSDPWrapStrategy",
    "FSDPConfig",
    "CheckpointType",
]

C = TypeVar("C", bound="BaseConfig")
D = TypeVar("D", bound="DictConfig|ListConfig")


class BaseConfig:
    @classmethod
    def _register_resolvers(cls, validate_paths: bool = True):
        # Expands path globs into a list.
        def path_glob(*paths) -> List[str]:
            out = []
            for path in paths:
                matches = sorted(glob(path))
                if not matches and validate_paths:
                    raise FileNotFoundError(f"{path} does not match any files or dirs")
                out.extend(matches)
            return out

        # Chooses the first path in the arguments that exists.
        def path_choose(*paths) -> str:
            from .util import is_url

            for path in paths:
                if is_url(path) or Path(path).exists():
                    return path
            if validate_paths:
                raise FileNotFoundError(", ".join(paths))
            else:
                return ""

        # Finds the latest checkpoint in a folder.
        def path_last_checkpoint(path) -> str:
            from .util import find_latest_checkpoint

            latest_checkpoint = find_latest_checkpoint(path)
            if latest_checkpoint is None:
                if validate_paths:
                    raise FileNotFoundError(f"Could not find a latest checkpoint at {path}")
                else:
                    return ""
            else:
                return str(latest_checkpoint)

        om.register_new_resolver("path.glob", path_glob, replace=True)
        om.register_new_resolver("path.choose", path_choose, replace=True)
        om.register_new_resolver("path.last_checkpoint", path_last_checkpoint, replace=True)

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """
        Update the legacy config settings whose schemas have undergone backwards-incompatible changes.
        """
        return config

    @classmethod
    def new(cls: Type[C], **kwargs) -> C:
        cls._register_resolvers()
        conf = om.structured(cls)
        try:
            if kwargs:
                conf = om.merge(conf, kwargs)
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    @classmethod
    def load(
        cls: Type[C],
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> C:
        """Load from a YAML file."""
        cls._register_resolvers(validate_paths=validate_paths)
        schema = om.structured(cls)
        try:
            raw = om.load(str(path))

            # Backwards compatibility hack, we need this here not in `update_legacy_settings`
            # since it has to be applied before selecting with `key`
            if "tokenizer" in raw and "model" in raw:
                raw["model"]["tokenizer"] = raw.pop("tokenizer")

            if key is not None:
                raw = raw[key]  # type: ignore
            raw = cls.update_legacy_settings(raw)
            conf = om.merge(schema, raw)
            if overrides:
                conf = om.merge(conf, om.from_dotlist(overrides))
            return cast(C, om.to_object(conf))
        except OmegaConfBaseException as e:
            raise OLMoConfigurationError(str(e))

    def save(self, path: PathOrStr) -> None:
        """Save to a YAML file."""
        om.save(config=self, f=str(path))

    def asdict(self, exclude: Optional[Iterable[str]] = None) -> Dict[str, Any]:
        out = asdict(self)  # type: ignore
        if exclude is not None:
            for name in exclude:
                if name in out:
                    del out[name]
        return out


class LayerNormType(StrEnum):
    default = "default"
    """
    The default LayerNorm implementation, equivalent to PyTorch's built-in version.
    """

    low_precision = "low_precision"
    """
    A low-precision version of the default LayerNorm.
    """

    rms = "rms"
    """
    An RMSNorm implementation. When using ``torch.compile`` this is
    probably the fastest implementation.
    """


class ActivationType(StrEnum):
    quick_gelu = "quick_gelu"
    gelu = "gelu"
    gelu_pytorch_tanh = "gelu_pytorch_tanh"
    relu = "relu"
    silu = "silu"
    llama_swiglu = "llama_swiglu"
    swiglu = "swiglu"


class BlockType(StrEnum):
    sequential = "sequential"

    llama = "llama"
    """
    A block similar to the sequential block with slightly different
    implementations of operations like attention to imitate the behavior of Llama.
    """

    moe = "moe"
    
    qwen = "qwen"


class InitFnType(StrEnum):
    mitchell = "mitchell"
    """
    The strategy suggested to us by Mitchell Wortsman from UW.
    This uses a truncated normal distribution with an adaptive standard deviation that depends
    on the size of the weights as well as the depth of the layer.
    """

    normal = "normal"
    """
    All weights are initialized from the same normal distribution.
    """

    kaiming_normal = "kaiming_normal"
    """
    All weights are initialized with the Kaiming method from a normal distribution.
    Note this currently won't work with FSDP.
    """

    fan_in = "fan_in"
    """
    "Fan-in variance scaling", i.e. normal with a standard deviation of ``1/sqrt(d_in)`` where ``d_in``
    is the input dimensionality of the kernel.
    """

    full_megatron = "full_megatron"
    """
    This is what metaseq calls "full megatron init". It is the init used for Llama 2.
    """


class VisionBackboneType(StrEnum):
    openai = "openai"
    siglip = "siglip"
    dino = "dino"


class ImagePaddingEmbed(StrEnum):
    """How to embed image padding information"""
    pad_and_partial_pad = "pad_and_partial_pad"
    pad_embed = "pad_embed"
    regress = "regress"


class ImagePooling2DType(StrEnum):
    """How to pool patch features"""
    attention = "attention"
    attention_meanq = "attention_meanq"
    attention_2wide = "attention_2wide"
    none = "none"
    stack = "stack"


class ImageProjectType(StrEnum):
    """How to project the pooled features into the LLM embedding space"""
    mlp = "mlp"
    mlpx2 = "2mlp"
    linear = "linear"


class AttentionType(StrEnum):
    """Attention to use"""
    sdpa = "sdpa"
    direct = "direct"
    flash = "flash"


@dataclass
class VisionBackboneConfig(BaseConfig):
    image_model_type: VisionBackboneType = VisionBackboneType.openai
    image_default_input_size: Tuple[int, int] = (336, 336)
    image_patch_size: int = 14
    image_pos_patch_size: int = 14
    image_emb_dim: int = 1024
    image_num_heads: int = 16
    image_num_key_value_heads: int = 16
    image_num_layers: int = 24
    image_head_dim: int = 64
    image_mlp_dim: int = 4096
    image_mlp_activations: ActivationType = ActivationType.gelu
    image_dropout_rate: float = 0.0
    image_num_pos: int = 577
    image_norm_eps: float = 1e-5
    attention_dropout: float = 0.0
    residual_dropout: float = 0.0
    initializer_range: float = 0.02
    fsdp_wrap: bool = False

    # how to preprocess imagse for this ViT
    resize_mode: str = "default"

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        if "fix_image_mask" in config:
            del config.fix_image_mask
        return config

    def __post_init__(self):
        self.image_default_input_size = tuple(self.image_default_input_size)  # type: ignore[assignment]

    @property
    def image_num_patch(self):
        h, w = self.image_default_input_size
        return h // self.image_patch_size, w // self.image_patch_size


@dataclass
class TokenizerConfig(BaseConfig):
    identifier: str = "gpt2"
    tokenizer_dir: Optional[str] = None

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = config.copy()
        # if tokenizer_cfg.identifier.startswith
        if config.identifier[:3] == "mm:":
            config.identifier = config.identifier[3:]
        if config.identifier[:3] == "hf-":
            config.identifier = config.identifier[3:]

        if "tokenizer_adds_space" in config:
            assert not config["tokenizer_adds_space"]
            del config.tokenizer_adds_space

        for k in ["olmo_eos_token_id", "olmo_bos_token_id", "truncate_direction"]:
            if k in config:
                del config[k]
        return config

@dataclass
class ActionTokenizerConfig(BaseConfig):
    identifier: str = "physical-intelligence/fast"
    tokenizer_dir: Optional[str] = None

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        return config

@dataclass
class ModelConfig(BaseConfig):
    """
    OLMo (model) configuration.
    """

    # Note that the defaults for these attributes are equivalent to the base GPT2 model.

    d_model: int = 768
    """
    The hidden size of the model.
    """

    n_heads: int = 12
    """
    The number of self-attention heads.
    """

    n_kv_heads: Optional[int] = None
    """
    The number of heads to use for keys and values. Defaults to `n_heads`.
    Set this to ``None`` or ``n_heads`` for normal multi-head attention.
    Set this to 1 for multi-query attention.
    Set it to some in-between value for Llama2-style grouped query attention.
    """

    qkv_bias: bool = False  # qwen models use bias in kvq layers
    """
    Do QKV projection a bias
    """

    clip_qkv: Optional[float] = None
    """
    Clip QKV to this value when set.
    """

    n_layers: int = 12
    """
    The number of layers/blocks.
    """

    mlp_ratio: int = 4
    """
    The ratio of the inner MLP dimensionality to ``d_model``.
    This is only used when ``mlp_hidden_size`` is not set.
    """

    mlp_hidden_size: Optional[int] = None
    """
    Set the exact hidden size for the MLP. Otherwise the inner MLP hidden size will be set to `mlp_ratio * d_model`.
    """

    activation_type: ActivationType = ActivationType.swiglu
    """
    The activation function to use within the MLP layers.
    """

    block_type: BlockType = BlockType.sequential
    """
    The transformer block implementation.
    """

    block_group_size: int = 1
    """
    The number of blocks to group together into a single parent block.
    This has no affect on the number of parameters in the model and is only used to wrap groups
    of blocks together with a single FSDP wrapper during training.
    """

    rope: bool = False
    """
    Use rotary positional embeddings (RoPE). Mutually exclusive with ``alibi``.
    """

    rope_full_precision: bool = True
    """
    If ``True``, apply RoPE embeddings at full precision regardless of the input type. Otherwise,
    apply RoPE at the precision of the input.
    """

    rope_theta: float = 10000.
    """
    RoPE theta parameter.
    """

    vision_backbone: Optional[VisionBackboneConfig] = None
    """
    Vision backbone settings for multi-modal models.
    """

    vit_load_path: Optional[str] = None
    """
    Use this to load the vit model.
    """

    llm_load_path: Optional[str] = None
    """
    Use this to partially load the llm transformer.
    """

    low_cpu_fsdp: bool = True
    """
    If ``True``, we save cpu memory by loading the pretrained vision model on randk0 only
    when init_device is `meta`.
    If TrainConfig.load_path is set, this should be set to ``False`` (default: True)
    """

    attention_type: AttentionType = AttentionType.sdpa
    """
    Attention implementation to use.
    """

    float32_attention: bool = True
    """
    Compute attention in float32
    """

    attention_dropout: float = 0.1
    """
    The dropout probability within the attention modules.
    """

    attention_layer_norm: bool = False
    """
    Apply layer norm to the keys and queries within the attention mechanism.
    This can help stabilize training.
    """

    residual_dropout: float = 0.1
    """
    The dropout probability for the MLP and attention output within each block.
    """

    response_residual_dropout: float = 0.0
    """
    Dropout applied only to loss/response tokens
    """

    embedding_dropout: float = 0.1
    """
    The dropout probability for embeddings.
    """

    layer_norm_type: LayerNormType = LayerNormType.default
    """
    The layernorm implementation to use.
    """

    layer_norm_with_affine: bool = True
    """
    Whether to include bias and weight parameters for the layer norms.
    This only affects layer norms that are immediately followed by a linear layer in the forward pass,
    so everything except QK-norms. To turn off affines for QK norms as well, set :attr:`attention_layer_norm_with_affine`
    to ``False``.
    """

    layer_norm_eps: Optional[float] = None
    """
    epsilon for layer norms
    """

    attention_layer_norm_with_affine: bool = True
    """
    Toggle affine transform for the QK norms.
    """

    max_sequence_length: int = 1024
    """
    The maximum input sequence length supported by the model.
    """

    max_position_embeddings: Optional[int] = None
    """
    Max positional embeddings to use in RoPE cache
    """

    include_bias: bool = True
    """
    Whether or not to include bias parameters in linear layers.
    """

    bias_for_layer_norm: Optional[bool] = None
    """
    Whether or not to include bias parameters in layer norm.
    When this is None (the default), it inherits the setting from include_bias.
    """

    scale_logits: bool = False
    """
    If ``True``, scale the output logits by ``1 / sqrt(d_model)``.
    """

    vocab_size: int = 50257
    """
    Vocabulary size of the model.
    """

    embedding_size: Optional[int] = 50304
    """
    The number of embeddings, i.e. the number of tokens. If set to ``None`` it will default
    to ``vocab_size``. If ``vocab_size`` is not a multiple of 128, setting this to the
    next multiple of 128 that's greater than ``vocab_size`` can improve throughput
    substantially.
    """

    ff_out_size: Optional[int] = None
    """
    The number of output features of the feedforward network. If set to ``None`` it will default
    to ``embedding_size``.
    """

    additional_vocab_size: Optional[int] = None
    """
    New tokens to add to the embeddings as part of the vision/language connector
    """

    new_embedding_init_range: float = 0.02
    """
    How to initialize embedding for new 
    """

    weight_tying: bool = True
    """
    Whether to tie output linear weights to the input embedding.
    """

    init_device: Optional[str] = None
    """
    The torch device to use when initializing the model parameters, e.g. "cpu", "cuda:0", "meta".
    """

    init_fn: InitFnType = InitFnType.normal
    """
    The weight initialization strategy.
    """

    init_std: float = 0.02
    """
    The standard deviation to use when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal".
    """

    init_cutoff_factor: Optional[float] = None
    """
    A positive factor used to scale the cutoff values when initializing weights with a "fixed distribution" ``init_fn``, such
    as "normal". Setting this to None means values are not cutoff.
    """

    norm_after: bool = False
    """
    Apply norm after the attention/feedforward layers rather than before, as introduced in the Swin transformer paper (Liu et al).
    """

    precision: Optional[str] = None
    """
    Precision used to train/evaluate with. You shouldn't set this directly.
    See :data:`TrainConfig.precision` instead.
    """

    max_crops: int = 12
    """
    Max crops to use, exluding the low-res overview crop
    """

    crop_mode: str = "patchify-v2-and-resize-c2"
    """
    How to divide the image into crop
    """

    use_col_tokens: bool = True
    """
    Place column or row-start tokens within the patch featuers 
    """

    prompt_type: str = "none"
    """
    How to construct prompts for the model
    """

    system_prompt_kind: str = "style"
    """
    How to construct the system prompt for the model
    """

    message_formatting: str = "none"
    """
    How to format messages (e.g., how to add user/assistant tags)
    """

    always_start_with_space: bool = True
    """
    Always add a space between the image and the initial text
    """

    multi_annotation_weighting: Optional[str] = None
    """
    How to automatically re-weight the loss for multiply annotated images
    """

    default_inference_len: Optional[int] = 65
    """
    For length condition, what length use for inference
    """

    overlap_margins: Tuple[int, int] = (4, 4)
    """
    For overlapping crops, how large the (left, right) overlap margins should be
    """

    pad_value: float = 0
    """
    Value to pad images 
    """

    image_padding_embed: Optional[ImagePaddingEmbed] = None
    """
    Image padding model, use in the September release but no long used in favour of modifying `pad_value`
    """

    fix_image_padding: bool = True
    """
    Use a version of the image padding mask that fixes the an off-by-one error how the embeddings
    are computed, should only be false for legacy models 
    """

    vit_layers: Tuple = (-1,)  # TODO should we fix the offset?
    """
    What layers to use from the VIT
    """

    image_pooling_h: int = 2
    """
    Pooling patch features height
    """

    image_pooling_w: int = 2
    """
    Pooling patch features width
    """

    image_pooling_2d: ImagePooling2DType = ImagePooling2DType.attention
    """
    Pooling layer
    """

    image_projector: ImageProjectType = ImageProjectType.mlp
    """
    Projector layer
    """

    image_feature_dropout: float = 0.0
    """
    Dropout for image patch features
    """

    initializer_range: float = 0.02
    """
    standard deviation to for initializing the weight models
    """

    normalize_input_embeds: bool = False
    """
    Normalize input embeddings (both for text and images) before 
    """

    use_position_ids: bool = True
    """
    Whether to use position IDs in the model.
    The model operation regarding positional embeddings changes depending on this variable.
    """

    head_dim: Optional[int] = None
    """
    The head dimensionality for the attention mechanism.
    """

    action_tokenizer: ActionTokenizerConfig = field(default_factory=ActionTokenizerConfig)

    """
    Action tokenizer configuration.
    """

    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    """
    Tokenizer configuration.
    """

    pad_tokenizer: bool = True

    moe_num_experts: Optional[int] = 8
    """
    The number of experts to use in the MoE block.
    """

    moe_top_k: Optional[int] = 2
    """
    The number of experts to select for each token.
    """

    moe_mlp_impl: Optional[str] = "sparse"
    """
    Choose "grouped" for grouped GEMM installable via `pip install git+https://git@github.com/tgale96/grouped_gemm.git@66c7195e35e8c4f22fa6a014037ef511bfa397cb`.
    """

    moe_log_expert_assignment: Optional[bool] = False
    """
    Whether to log the expert assignment.
    """

    moe_shared_expert: Optional[bool] = False
    """
    Whether to have an always-used expert like in [DeepSeekMoE](https://arxiv.org/abs/2401.06066).
    """

    moe_lbl_in_fp32: Optional[bool] = False
    """
    Whether to perform load balancing in FP32.
    """

    moe_interleave: Optional[bool] = False
    """
    Interleave sequential with MoE blocks starting with sequential.
    """

    moe_loss_weight: Optional[float] = 0.1
    """
    The weight to use for the MoE load balancing loss.
    """

    moe_zloss_weight: Optional[float] = None
    """
    Weight for MoE router z-loss where None means no router z-loss. 0.001 is a common value.
    """

    moe_dropless: Optional[bool] = True
    """
    Whether to use [dMoE](https://arxiv.org/abs/2211.15841).
    """

    moe_capacity_factor: Optional[float] = 1.25
    """
    The capacity factor to use in the MoE block. Only applies if not using dMoE.
    """

    # for action heads
    action_head: Optional[str] = "l1_regression"
    """ Action head to use, e.g. "l1_regression" or "diffusion" """

    action_dim: int = 7 #ACTION_DIM
    fixed_action_dim: int = 7
    right_end_effector_dim: int = 7
    left_end_effector_dim: int = 7
    mobile_base_dim: int = 3
    
    num_actions_chunk: int = 8 #NUM_ACTIONS_CHUNK
    proprio_dim: int = 8 #PROPRIO_DIM

    
    num_diffusion_steps: Optional[int] = 1000 #1000
    """ Number of diffusion steps to use for the diffusion action head """
    num_diffusion_inference_steps: Optional[int] = 30 #10
    """ Number of inference steps to use for the diffusion action head """
    use_proprio: bool = False
    """ Whether to use proprioceptive features in the input tokens"""
    action_head_dit_hidden_size: int = 1024
    """ The hidden size of the DIT action head."""
    action_head_dit_depth: int = 14
    """ The depth of the DIT action head."""
    action_head_dit_num_heads: int = 16
    """ The number of heads in the DIT action head."""
    """1B: num_head 32 hidden_size 2048, depth:28
       170M: num_head 32, hidden size: 1024, depth: 14 """

    action_head_flow_matching_dim: int = 896
    """ The hidden size of the flow matching action head."""
    action_head_flow_matching_layers: int = 18
    """ The number of layers of the flow matching action head."""
    action_head_flow_matching_heads: int = 8
    """ The number of heads of the flow matching action head."""
    action_head_flow_matching_intermediate_size: int = 4096
    action_head_flow_matching_kv_heads: int = 4
    
    """ The intermediate size of the flow matching action head."""
    action_head_flow_matching_pvf_function: str = "2d_attn_mask"

    
    llm_causal_attention: bool = False
    """ Whether to use causal attention in the LLM transformer """
    
    action_use_left_eef: bool = False
    """ Whether to use the left end-effector in the action head input tokens """
    action_use_mobile_base: bool = False
    """ Whether to use the mobile base in the action head input tokens """


    # ff_out_size: int =0
    """
    from kaidong's pretrained model
    """
    


    def get_action_tokenizer(self):
        tokenizer_cfg = self.action_tokenizer
        identifier = tokenizer_cfg.identifier
        tokenizer = AutoProcessor.from_pretrained(identifier, trust_remote_code=True)
        return tokenizer

    def get_tokenizer(self):
        tokenizer_cfg = self.tokenizer
        kargs = {}
        return build_tokenizer(
            tokenizer_cfg.identifier,
            tokenizer_dir=tokenizer_cfg.tokenizer_dir,
            pad_tokenizer_to=self.vocab_size if self.pad_tokenizer else None,
            **kargs
        )
    def __post_init__(self):
        self.vit_layers = tuple(self.vit_layers)  # type: ignore[assignment]

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        """
        Update the legacy config settings whose schemas have undergone backwards-incompatible changes.
        """
        # This handles a variety of OLMo configs options we don't support in Molmo, and some
        # legacy config options we have since abandoned
        config.tokenizer = TokenizerConfig.update_legacy_settings(config.tokenizer)
        if config.vision_backbone is not None:
            config.vision_backbone = VisionBackboneConfig.update_legacy_settings(config.vision_backbone)

        if config.image_pooling_2d == "attention-meanq":
            config.image_pooling_2d = ImagePooling2DType.attention_meanq

        if "flash_attention" in config:
            is_flash = config.flash_attention
            del config.flash_attention
            config.attention_type = AttentionType.flash if is_flash else AttentionType.sdpa

        if "query_pre_attn_scalar" in config:
            del config.query_pre_attn_scalar

        if "pad_token_id" in config:
            del config.pad_token_id

        if "prompt_override" in config:
            assert config.prompt_override is None
            del config.prompt_override

        if "bos_token_id" in config:
            config.tokenizer.olmo_bos_token_id = config.pop("bos_token_id")
            config.tokenizer.olmo_eos_token_id = config.pop("eos_token_id")

        if "response_attention_dropout" in config:
            assert config.response_attention_dropout == 0
            del config.response_attention_dropout

        if "alibi" in config:
            assert not config.alibi
            del config.alibi
            del config.alibi_bias_max

        if "rope_impl" in config:
            assert config.rope_impl == "llama"
            del config.rope_impl

        if "attn_logit_softcapping" in config:
            assert config.attn_logit_softcapping is None
            del config.attn_logit_softcapping

        if "final_logit_softcapping" in config:
            assert config.final_logit_softcapping is None
            del config.final_logit_softcapping

        if "image_padding_mask" in config:
            assert not config["image_padding_mask"]
            del config["image_padding_mask"]
            config["image_padding_embed"] = None
        elif "image_padding_embed" not in config:
            config["image_padding_embed"] = None

        if "multi_query_attention" in config:
            assert config.multi_query_attention is None
            del config.multi_query_attention

        if "do_random_scale" in config:
            assert not config.do_random_scale
            del config.do_random_scale

        if "fix_image_input_idx" in config:
            assert config.fix_image_input_idx == 2
            del config.fix_image_input_idx

        if "unconditioned" in config:
            assert not config.unconditioned
            del config.unconditioned

        if "use_cls_feature" in config:
            assert not config.use_cls_feature
            del config.use_cls_feature

        if "pad_to" in config:
            assert not config.pad_to
            del config.pad_to

        if "loss_token_weighting" in config:
            config.multi_annotation_weighting = config.loss_token_weighting
            del config.loss_token_weighting

        if "gin_bindings" in config:
            assert not config.gin_bindings
            del config.gin_bindings

        return config

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            return self.n_heads
        else:
            return self.n_kv_heads

    @property
    def image_num_patch(self):
        if self.vision_backbone is None:
            raise ValueError("No vision backbone")
        return self.vision_backbone.image_num_patch

    @property
    def image_patch_size(self):
        if self.vision_backbone is None:
            raise ValueError("No vision backbone")
        return self.vision_backbone.image_patch_size

    def llm_patches_per_crop(self):
        h, w = self.image_num_patch
        # Round up in case we need to pad the image features for pooling
        h = (h + self.image_pooling_h - 1) // self.image_pooling_h
        w = (w + self.image_pooling_w - 1) // self.image_pooling_w
        return h, w

    def get_max_crops(self) -> int:
        """Max numbers of that can be built for one image"""
        if self.crop_mode == "resize":
            # return 1 #, bug for vla?
            return self.max_crops
        elif "resize" in self.crop_mode:
            return 1 + self.max_crops
        else:
            return self.max_crops


class OptimizerType(StrEnum):
    lionw = "lionw"
    adamw = "adamw"


@dataclass
class OptimizerConfig(BaseConfig):
    name: OptimizerType = OptimizerType.lionw

    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1.0e-5
    """
    Default optimizer settings, used if the settings below are None
    """

    connector_learning_rate: Optional[float] = 1.0e-4
    vit_learning_rate: Optional[float] = 1.0e-4
    llm_learning_rate: Optional[float] = 1.0e-4
    action_head_learning_rate: Optional[float] = 1.0e-4
    """
    Separate learning_rate values for the connector, vision backbone, and llm transformer.
    """

    connector_weight_decay: Optional[float] = 0.01
    vit_weight_decay: Optional[float] = 0.01
    llm_weight_decay: Optional[float] = 0.01
    action_head_weight_decay: Optional[float] = 0.01
    """
    Separate weight decay values for the connector, vision backbone, and llm transformer.
    """

    connector_betas: Tuple[float, float] = (0.9, 0.95)
    vit_betas: Tuple[float, float] = (0.9, 0.95)
    llm_betas: Tuple[float, float] = (0.9, 0.95)
    action_head_betas: Tuple[float, float] = (0.9, 0.95)
    """
    Separate betas values for the connector, vision backbone, and llm transformer.
    """

    connector_eps: Optional[float] = 1.0e-6
    vit_eps: Optional[float] = 1.0e-6
    llm_eps: Optional[float] = 1.0e-6
    action_head_eps: Optional[float] = 1.0e-6
    """
    Separate weight decay values for the connector, vision backbone, and llm transformer.
    """

    metrics_log_interval: Optional[int] = None
    """
    The interval with which to collect and log detailed parameter-specific metrics.
    This only applies when logging to W&B, since these metrics won't be logged to the console.
    If not set, defaults to the wandb `log_interval`.
    """

    def __post_init__(self):
        self.betas = tuple(self.betas)  # type: ignore[assignment]
        self.connector_betas = tuple(self.connector_betas)  # type: ignore[assignment]
        self.vit_betas = tuple(self.vit_betas)  # type: ignore[assignment]
        self.llm_betas = tuple(self.llm_betas)  # type: ignore[assignment]
        self.action_head_betas = tuple(self.action_head_betas)  # type: ignore[assignment]

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()
        if "no_decay_norm_and_bias" in config:
            del new_config.no_decay_norm_and_bias
        if "decay_norm_and_bias" in config:
            del new_config.decay_norm_and_bias
        if "decay_embeddings" in config:
            del new_config.decay_embeddings

        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "name") and new_config.name == "decoupled_lionw":
                new_config.name = "lionw"

        return new_config


class SchedulerType(StrEnum):
    cosine_with_warmup = "cosine_with_warmup"
    linear_with_warmup = "linear_with_warmup"
    inverse_sqrt_with_warmup = "inverse_sqrt_with_warmup"
    max_scheduler = "max_scheduler"
    constant = "constant"
    multimodal = "multimodal"


class SchedulerUnits(StrEnum):
    steps = "steps"
    tokens = "tokens"


@dataclass
class SchedulerConfig(BaseConfig):
    name: SchedulerType = SchedulerType.cosine_with_warmup
    units: SchedulerUnits = SchedulerUnits.steps
    t_warmup: Union[int, float] = 100
    t_max: Optional[Union[int, float]] = None
    alpha_f: float = 0.1

    connector_t_warmup: Union[int, float] = 200
    vit_t_warmup: Union[int, float] = 200
    action_head_t_warmup: Union[int, float] = 200
    llm_t_warmup: Union[int, float] = 200
    connector_freeze_steps: Union[int, float] = 0
    vit_freeze_steps: Union[int, float] = 0
    action_head_freeze_steps: Union[int, float] = 0
    llm_freeze_steps: Union[int, float] = 0
    """
    Per-parameter group warmups
    """

    grad_clip_warmup_steps: Optional[Union[int, float]] = None
    """
    The warmup period for which the max grad norm (or norm ratio) will be set to its
    warmup value of `max_grad_norm * grad_clip_warmup_factor`.
    """

    grad_clip_warmup_factor: Optional[float] = None
    """
    The ratio of the max allowed gradient norm (or norm ratio) for clipping during the warmup period
    vs after the warmup period.
    """

    warmup_min_lr: Optional[float] = None
    """
    The starting LR during the warmup period. If not set this defaults to 10% of
    the target LR.
    """


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class EvaluatorConfig(BaseConfig):
    """Config for `Evaluator` objects that compute metrics"""

    n_to_log: int = 10
    """Num examples to log to console"""

    num_wandb_examples: int = 0
    """Num examples to log to Wandb as a HTML table"""

    save_predictions: Optional[str] = "_default"  # saves with default name to checkpoint dir
    """Where to save predictions files"""

    save_tokens: bool = False
    """If save predictions, should the tokens be saved"""

    vqa_eval: str = ''
    """name(s) of VQA-style eval to run, can be a comma seperated list"""

    # Other individual types of eval
    multi_threshold_box_eval: bool = False
    coordinate_eval: bool = False
    pointing_eval: bool = False
    count_eval: bool = False
    point_count_eval: bool = False
    trajectory_eval: bool = False
    android_eval: bool = False
    clock_eval: bool = False
    clock_bench_eval: bool = False # Clock reading benchmark, coco/openimg/movies
    math_vista_eval: bool = False
    action_eval: bool = False


@dataclass
class RootSizeMixture(BaseConfig):
    rate: float
    mixture: Dict[str, Optional[float]]


@dataclass
class DataConfig(BaseConfig):
    """Configuration for a dataset to train or evaluate on"""

    dataset: Optional[str] = None
    """Dataset name, will be used int `get_dataset_by_name`"""

    mixture: Optional[Dict[str, float]] = None
    """Mixture of dataset names and sampling rates"""

    root_size_mixture: Optional[List[RootSizeMixture]] = None
    """Mixture-of-mixtures where sub-mixtures rates are determined by the root dataset size"""

    split: Optional[str] = None  # default to train or validation
    """Dataset split to load"""

    seed: Optional[int] = None
    """Dataset seed, defaults to the global seed if None"""

    shuffle_messages: bool = True
    """For multi-annotated images, should we shuffle the messages"""

    pad: Optional[str]="to_max"
    """How pad array in the collator"""

    sequence_length: Optional[int] = None
    """Max sequence length to truncate examples to in the Collator"""

    shuffle: Optional[bool] = True
    """Shuffle the data"""

    for_inference: Optional[bool] = None
    """Inference mode where the response will not be present in the batch"""

    multi_modal: Optional[str] = "torch"
    """Kind of dataset to load, in this repo only 'torch' is supported"""

    # DataLoader args
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = False
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0

    # vla dataset settings
    use_wrist_image: bool = False
    use_proprio: bool = False
    

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = config.copy()
        for k in ["pad_direction", "label_mask_paths", "generate_attention_mask",
                  "use_memory_cache", "shuffle_buffer_size"]:
            if k in config:
                del config[k]
        if "num_epochs" in config:
            del config.num_epochs
        if "instance_filter" in config:
            assert config.instance_filter is None
            del config.instance_filter
        if "per_node_data_loader" in config:
            assert not config.per_node_data_loader
            del config.per_node_data_loader
        return config


@dataclass
class DatasetEvaluatorConfig(BaseConfig):
    """Configuration for a datset and metrics to use to evaluate a model"""

    label: str
    """Label to use when logging results"""

    data: DataConfig = field(default_factory=DataConfig)
    """Data to evaluate on"""

    device_eval_batch_size: Optional[int] = None
    """Batch size, can default to the eval batch set set in the global config"""

    subset_num_batches: Optional[int] = None
    """Number of matches to run on, if None use the entire dataset"""

    max_examples: Optional[int] = None
    """Max number of examples to run on, overrides `subset_num_batches`"""

    max_new_tokens: Optional[int] = 448
    """Max number of tokens to generate"""

    mm_evaluator: Optional[EvaluatorConfig] = None
    """Specifies how to compute metrics and save the predictions"""

    save_dir: Optional[str] = None
    """Where to save prediction, metrics, and visualizations"""

    save_to_checkpoint_dir: Optional[bool] = False
    """Use the checkpoint directory as `self.save_dir`"""

    eval_name: Optional[str] = None
    """Name to post-fix the evaluation outputs with"""

    skip_if_metrics_cached: bool = True
    """Skip a the metric file already exists in the save location, otherwise override it"""

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        config = config.copy()
        if getattr(config, "mm_evaluator", None):
            config.mm_evaluator = EvaluatorConfig.update_legacy_settings(config.mm_evaluator)
        if getattr(config, "data", None):
            config.data = DataConfig.update_legacy_settings(config.data)
        return config

    def __post_init__(self):
        if self.save_to_checkpoint_dir and self.save_dir:
            raise ValueError("Cannot set both save_to_dir and save_to_checkpoint_dir")


@dataclass
class WandbConfig(BaseConfig):
    project: Optional[str] = None
    entity: Optional[str] = "ai2-llm"
    group: Optional[str] = None
    name: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=lambda: ["watching"])
    log_artifacts: bool = False
    rank_zero_only: bool = True
    log_interval: int = 1


@dataclass
class SpeedMonitorConfig(BaseConfig):
    window_size: int = 100
    gpu_flops_available: Optional[Union[float, int]] = None


@dataclass
class CompilerConfig(BaseConfig):
    mode: Optional[str] = None
    """
    The mode to compile the model in. At the moment this can be "default",
    "reduce-overhead" (useful for smaller models/batches), or "max-autotune"
    (the fastest for larger models, but takes a long time to compile).
    """

    fullgraph: bool = False
    """
    Whether it is OK to break model into several subgraphs when compiling.
    Note that this is not compatible with FSDP.
    """

    backend: str = "inductor"
    """
    The backend to use.
    """


class FSDPWrapStrategy(StrEnum):
    by_block = "by_block"
    """
    Wrap each OLMo block with its own FSDP instance.
    """

    by_block_and_size = "by_block_and_size"
    """
    Like 'by_block' but `wte` and `ff_out` will be wrapped separately as well.
    """

    by_block_group = "by_block_group"
    """
    Wrap each block group together into its own FSDP instance.
    This requires :attr:`~ModelConfig.block_group_size` to be bigger than 1.
    """

    by_block_group_and_size = "by_block_group_and_size"
    """
    Like 'by_block_group' but `wte` and `ff_out` will be wrapped separately as well.
    """

    size_based = "size_based"
    """
    Used PyTorch's default size-based auto wrap policy.
    """

    one_in_two = "one_in_two"
    one_in_three = "one_in_three"
    one_in_four = "one_in_four"
    one_in_five = "one_in_five"


class FSDPPrecision(StrEnum):
    pure = "pure"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, ``reduce_dtype``,
    and ``buffer_dtype`` all set to the autocast precision data type.
    """

    mixed = "mixed"
    """
    Equivalent to :class:`torch.distributed.fsdp.MixedPrecision` with ``param_dtype``, and ``buffer_dtype``
    set to the autocast precision data type, while ``reduce_dtype`` is set to fp32.
    """

    float = "float"


@dataclass
class FSDPConfig(BaseConfig):
    use_orig_params: bool = True
    """
    This must be ``True`` if using ``compile`` or you want to track the parameter norm during training.
    """

    sharding_strategy: ShardingStrategy = ShardingStrategy.FULL_SHARD

    wrapping_strategy: Optional[FSDPWrapStrategy] = None
    """
    The wrapping strategy to use. If ``None``, the default, the model is wrapped with a single top-level
    FSDP instance.
    """

    precision: FSDPPrecision = FSDPPrecision.pure

    hybrid_sharding_num_model_replicas: Optional[int] = None
    """
    The number of model instances, when using a hybrid sharding strategy.
    If not ``None``, this must divide the total number of nodes. If ``None``, the default,
    a model instance is used per node (as determined by ``get_world_size() // get_local_world_size()``).
    PyTorch's default HSDP behavior matches this default behavior.
    """


class CheckpointType(StrEnum):
    sharded = "sharded"
    unsharded = "unsharded"
    sharded_ephemeral = "sharded_ephemeral"
    action_head = "action_head" # no shared


class ShardedCheckpointerType(StrEnum):
    torch_new = "torch_new"
    torch_legacy = "torch_legacy"
    local = "local"
    olmo_core = "olmo_core"


class BatchDivisor(StrEnum):
    global_batch = "global_batch"
    device_batch = "device_batch"
    instance = "instance"


class ActivationCheckpointingStrategy(StrEnum):
    whole_layer = "whole_layer"
    """
    Checkpoint every transformer layer.
    """

    one_in_two = "one_in_two"
    """
    Checkpoint one in two transformer layers.
    """

    one_in_three = "one_in_three"
    """
    Checkpoint one in three transformer layers.
    """

    one_in_four = "one_in_four"
    """
    Checkpoint one in four transformer layers.
    """

    two_in_three = "two_in_three"
    """
    Checkpoint two out of every three transformer layers.
    """

    three_in_four = "three_in_four"
    """
    Checkpoint three out of four of every transformer layers.
    """

    fine_grained = "fine_grained"
    """
    Focus checkpointing on where it is cheap to recompute and saves most memory.
    """

    vit_only = "vit_only"
    """
    Checkpoint every vit transformer layer.
    """


@dataclass
class TrainConfig(BaseConfig):
    """
    OLMo training configuration.
    """

    run_name: Optional[str] = None
    """
    Run name, used when logging 
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    epoch: Optional[int] = None
    """
    Increment this when starting a new epoch.
    """

    dry_run: bool = False
    """
    If ``True``, don't actually train.
    """

    model: ModelConfig = field(default_factory=ModelConfig)
    """
    OLMo Model configuration.
    """

    allow_resume: bool = False
    """
    Try to resume training if a checkpoint already exists in the checkpoint directory
    """

    ft_llm: bool = True
    """
    Tune the LLM parameters
    """

    ft_vit: bool = True
    """
    Tune the image encoder parameters
    """

    ft_connector: bool = True
    """
    Tune the V/L connector parameters
    """

    # Do we fine-tune the input/output embeddings
    ft_embedding: str = "lm_head"
    """
    Tune the embedding layers
    """

    early_exit: bool = False
    """
    Use early exit model during training and inference
    """

    train_exit_random_layer: bool = False
    """
    Train the early exit model with a random layer
    """

    ## add by jian zhang

    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    """
    Optimizer configuration.
    """

    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    """
    Learning rate scheduler configuration.
    """

    data: DataConfig = field(default_factory=DataConfig)
    """
    Training data configuration.
    """

    restore_dataloader: bool = True
    """
    When resuming, restore the data loader to where it left off.
    If you restarting in order to train on a different dataset, set this to ``False``.
    """

    fast_forward_batches: Optional[int] = None
    """
    When resuming, use this to fast-forward the dataloader beyond the last checkpoint.
    """

    evaluators: List[DatasetEvaluatorConfig] = field(default_factory=list)
    """
    Evaluation configurations.
    """

    eval_interval: int = 1000
    """
    How often (in terms of batches) to run evaluations.
    """

    inf_eval_interval: Optional[int] = -1
    """
    How often (in terms of batches) to run inference evaluations
    """

    inf_evaluators: List[DatasetEvaluatorConfig] = field(default_factory=list)
    """
    Inference Evaluation configurations.
    """

    save_folder: str = "./"
    """
    The directory to save checkpoints to.
    """

    remote_save_folder: Optional[str] = None
    """
    A folder in a cloud bucket to upload saved checkpoints to.
    """

    canceled_check_interval: int = 50
    """
    How often (in batches) to check if the run has been canceled or reached its time limit.
    """

    save_interval: int = 1000
    """
    How often (in terms of steps) to save sharded training state checkpoints.
    """

    save_interval_unsharded: Optional[int] = None
    """
    How often (if at all) to save unsharded training state checkpoint.
    For large models it can be costly to save these, so it usually makes sense to save
    these less often than regular (sharded) training checkpoints.
    """

    save_interval_ephemeral: Optional[int] = None
    """
    How often (if at all) to save ephemeral sharded checkpoints. These checkpoints are the same
    as those saved every `save_interval` except that at most only the most recent one of these is kept.
    This is useful when you want to checkpoint often for restarts in case of failures, but don't
    want to keep the majority of these checkpoints.

    For example, suppose you want to keep your checkpoints at every 1000 steps, but you also want to save
    a temporary checkpoint every 100 steps in case your job fails. In that case you would
    set `save_interval=1000` and `save_interval_ephemeral=100`.
    """

    save_interval_action_head: Optional[int] = None
    """
    How often (if at all) to save action head checkpoints.
    This is useful when you want to save the action head model separately from the main model.
    """

    save_num_checkpoints_to_keep: int = -1
    """
    How many sharded checkpoints to keep.
    """

    save_num_unsharded_checkpoints_to_keep: int = -1
    """
    How many unsharded checkpoints to keep.
    """

    save_num_action_head_checkpoints_to_keep: int = -1
    """
    How many action head checkpoints to keep.
    """

    save_overwrite: bool = False
    """
    If ``True``, overwrite any conflicting checkpoint files.
    """

    force_save_unsharded: bool = False
    """
    Save an unsharded checkpoint before training (even during a dry run).
    Use this option with `--load-path={PATH}` and `--dry_run` to convert a sharded
    checkpoint into an unsharded checkpoint.
    """

    no_pre_train_checkpoint: bool = True
    """
    Skip saving pre-train checkpoint.
    """

    initial_model_checkpoint: Optional[str] = None
    """
    Path to a model to use to initialize the model at the step 0
    
    Unlike `load_path` this will be used only at step0 and only loads the parameters    
    """

    load_model_config: Optional[str] = None
    """
    Load the model config the from a defaults setting, then  
    
    """

    checkpoint_dir: Optional[str] = None
    """
    The directory to load checkpoints from.
    """

    load_path: Optional[str] = None
    """
    The path to a training checkpoint to restore/resume from.

    Note that you can make use of the "path.last_checkpoint" Omegaconfig YAML resolver here, which takes
    a local or remote directory and resolves to the latest checkpoint (sharded or unsharded) in that directory.
    For example,

    ```bash
    --load_path='${path.last_checkpoint:s3://ai2-llm/checkpoints/7b/v1_5-mix-run-001}'
    ```
    """

    load_path_sharded_checkpointer: Optional[ShardedCheckpointerType] = None
    """
    The sharded checkpointer type to use to load the initial checkpoint from ``load_path``.
    """

    reset_optimizer_state: bool = False
    """
    When this is set, we restore the model from a checkpoint (if given), but we leave the optimizer uninitialized.
    We also set a new learning rate schedule that does a new warmup, such that it intercepts the original learning
    curve (according to the current learning rate schedule settings), and continues from there.
    """

    reset_trainer_state: bool = False
    """
    When this is set we don't restore the trainer state from a checkpoint.
    """

    save_dataloader_state: bool = False
    """
    When this is set we save restore the dataloader state for multimodal training.
    """

    reset_dataloader_state: bool = False
    """
    When this is set we don't restore the dataloader state from a checkpoint for multimodal training.
    """

    keep_lr_on_load: bool = False
    """
    When resuming from a checkpoint, keep the learning rate and weight decay values
    stored in the optimizer state from the checkpoint instead of resetting them
    according to the current scheduler/config.
    """
    
    sharded_checkpointer: ShardedCheckpointerType = ShardedCheckpointerType.torch_legacy
    """
    The name of the sharded checkpointer to use to save (sharded) checkpoints throughout training.
    """

    max_duration: Union[int, str] = 10000
    """
    How long to train for.

    If specified without a unit (the default), the units are assumed to be steps.
    You can also specify this in terms of tokens, for example: `max_duration="2e12T"` means train until
    2 trillion tokens.
    """

    global_train_batch_size: int = 512
    """
    The effective global batch size.
    """

    device_train_batch_size: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``global_train_batch_size // world_size``.
    """

    device_train_microbatch_size: int = 16
    """
    The number of instances passed to the model in a single forward-backward pass. You should set
    this as large as you can based on available GPU memory.
    """

    device_eval_batch_size: int = 16
    """
    The number of evaluation instances passed to the model in a single forward pass on each device.
    """

    eval_subset_num_batches: int = -1
    """
    The number of batches to use for downstream evaluation from each dataset.
    """

    eval_on_load: bool = False
    """
    When resuming from a checkpoint, run the evaluation loop right away.
    """

    device_inf_eval_batch_size: int = 16
    """
    The number of inference evaluation instances passed to the model in a single forward pass on each device.
    """

    inf_eval_subset_num_batches: int = -1
    """
    The number of batches to use for inference evaluation from each dataset.
    """

    device_train_grad_accum: Optional[int] = None  # calculated automatically
    """
    Don't set this manually. This will be set to ``device_train_batch_size // device_train_microbatch_size``.
    """

    max_grad_norm: Optional[float] = None
    """
    Clip gradient norms to this value if set.
    """

    multi_component_grad_norm: bool =True
    """
    Use separate grad norm for each component in multi-modal model
    """

    batch_divisor: Optional[BatchDivisor] = BatchDivisor.global_batch
    """
    How loss is normalized in distributed settings
    """

    max_grad_norm_ratio: Optional[float] = None
    """
    If set, gradient norms will be clipped to `max_grad_norm_ratio * exp_avg(norm(grad))`.
    This takes priority over `max_grad_norm` when set.
    """

    precision: Optional[str] = None
    """
    Precision to train with (e.g. "amp_bf16", "amp_fp16", or "fp32").
    """

    wandb: Optional[WandbConfig] = None
    """
    Weights & Biases configuration.
    """

    speed_monitor: SpeedMonitorConfig = field(default_factory=SpeedMonitorConfig)
    """
    Speed monitor configuration.
    """

    console_log_interval: int = 1
    """
    How often to log to the console.
    """

    gen1_gc_interval: Optional[int] = 1
    """
    How often (in steps) to run generation 1 garbage collection.
    Set to ``None`` to use automatic garbage collection (i.e. we don't mess with it).
    """

    compile: Optional[CompilerConfig] = None
    """
    Settings for compiling the model with ``torch.compile()``.
    """

    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    """
    Fully sharded data parallel settings.
    """

    softmax_auxiliary_loss: bool = False
    """
    If ``True``, we add the auxiliary loss function from PaLM that encourages the softmax
    normalizing term to be close to 0 (z-loss).
    """

    softmax_auxiliary_loss_scale: float = 1e-4
    """
    The scale of the auxiliary loss function (z-loss).
    """

    time_limit: Optional[float] = None
    """
    The maximum amount of time to train for before saving a checkpoint and ending early.
    """

    extra_steps_after_cancel: int = 10
    """
    Under certain conditions when a run is canceled we train for a few extra steps after saving
    the final checkpoint so that when the run is restarted from the latest checkpoint we have some
    overlap in metrics.
    """

    python_profiling: bool = False
    """
    Whether to run the Python profiler on batches 6, 7, and 8.
    """

    torch_profiling: bool = False
    """
    Whether to run the PyTorch profiler on batches 6, 7, and 8.
    """

    stop_at: Optional[int] = None
    """
    Stop at a specific step.
    """

    stop_after: Optional[int] = None
    """
    Stop after a specific number of steps.
    """

    activation_checkpointing: Optional[ActivationCheckpointingStrategy] = None
    """
    The activation checkpointing strategy to use.
    """

    fused_loss: Optional[bool] = None
    """
    Whether to use the fused CE loss function from `flash-attn`.
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")

    @property
    def fsdp_precision(self) -> MixedPrecision:
        if self.fsdp.precision == FSDPPrecision.pure:
            return MixedPrecision(
                param_dtype=self.autocast_precision,
                reduce_dtype=self.autocast_precision,
                buffer_dtype=self.autocast_precision,
            )
        elif self.fsdp.precision == FSDPPrecision.mixed:
            return MixedPrecision(
                param_dtype=self.autocast_precision,
                reduce_dtype=torch.float32,
                buffer_dtype=self.autocast_precision,
            )
        elif self.fsdp.precision == FSDPPrecision.float:
            return MixedPrecision(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.float32,
            )
        else:
            raise NotImplementedError(f"{self.fsdp.precision}")

    @classmethod
    def update_legacy_settings(cls, config: D) -> D:
        new_config = config.copy()

        if "new_style_checkpoints" in new_config:
            assert new_config.new_style_checkpoints is None
            del new_config.new_style_checkpoints
        if "early_stopping_factor" in new_config:
            assert new_config.early_stopping_factor is None
            del new_config.early_stopping_factor
        if "save_data_indices" in new_config:
            del new_config["save_data_indices"]

        if om.is_dict(new_config):
            assert isinstance(new_config, DictConfig)

            if hasattr(new_config, "activation_checkpointing"):
                if new_config.activation_checkpointing is False:
                    new_config.activation_checkpointing = None
                if new_config.activation_checkpointing is True:
                    new_config.activation_checkpointing = ActivationCheckpointingStrategy.whole_layer

            if hasattr(new_config, "optimizer"):
                new_config.optimizer = OptimizerConfig.update_legacy_settings(new_config.optimizer)

            if hasattr(new_config, "data"):
                new_config.data = DataConfig.update_legacy_settings(new_config.data)

            if hasattr(new_config, "model"):
                new_config.model = ModelConfig.update_legacy_settings(new_config.model)

            for k in ["inf_evaluators", "evaluators"]:
                if hasattr(new_config, k):
                    setattr(new_config, k, [DatasetEvaluatorConfig.update_legacy_settings(x)
                                            for x in getattr(new_config, k)])

        return new_config


@dataclass
class DiTActionConfig(BaseConfig):
    text_model: str = "Qwen3-1.7B"
    """
    The name of text model.
    """
    # text_model_path_url: str = "google/siglip-so400m-patch14-384"
    text_model_path_url: str = "/mnt/data/zhangjian/Qwen3/Qwen3-1.7B"
    """ 
    The path or url of the text model.
    """

    vision_model_path_url: str = "/mnt/data/zhangjian/google/siglip-so400m-patch14-384"
    """
    The path or url of the vision model. "google/siglip-so400m-patch14-384"
    """
    # 基本训练配置
    run_name: str = "dit_action_train"
    save_folder: str = None
    seed: int = 42
    max_steps: int = 10000
    global_batch_size: int = 32
    device_batch_size: int = 4
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    use_fsdp: bool = False
    ft_vit: bool = False
    
    # 数据配置
    dataset_name: str = "libero_spatial_no_noops"
    data_root_dir: str = "/path/to/rlds/data"
    use_wrist_image: bool = True
    use_proprio: bool = True
    sequence_length: int = 768
    num_workers: int = 0
    
    # 模型配置
    # action_dim: int = 7
    # action_horizon: int = 8
    num_diffusion_steps: int = 1000
    num_diffusion_inference_steps: int = 30
    lang_cond_dim: int = 768
    img_cond_dim: int = 768
    num_patches: int = 196  
    
    # DiT模型参数
    dit_hidden_dim: int = 1024
    dit_depth: int = 14
    dit_num_heads: int = 16
    
    # 检查点配置
    save_interval: int = 500
    eval_interval: int = 500
    log_interval: int = 10

    save_num_checkpoints_to_keep: int = 5  # 保留的检查点数量，-1表示保留所有
    save_overwrite: bool = False  # 是否覆盖已存在的检查点
    
    # 分布式配置
    precision: str = "amp_bf16"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # W&B配置
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None
    
    @classmethod
    def load_yaml(cls, yaml_path: str) -> "DiTActionConfig":
        """从YAML文件加载配置"""
        import yaml
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    

    def set_text_model_path_or_url(self,):
        """设置文本模型的路径或URL"""
        if self.text_model == "Qwen3-1.7B":
            self.text_model_path_url = "/mnt/data/zhangjian/Qwen3/Qwen3-1.7B"
        elif self.text_model == "Qwen3-4B":
            self.text_model_path_url = "/mnt/data/zhangjian/Qwen3/Qwen3-4B"
        elif self.text_model == "Qwen3-0.6B":
            self.text_model_path_url = "/mnt/data/zhangjian/Qwen3/Qwen3-0.6B"
        elif self.text_model == "SmolLM3-3B":
            self.text_model_path_url = "/mnt/data/zhangjian/HuggingFaceTB/SmolLM3-3B"
        elif self.text_model == "SigLip":
            self.text_model_path_url = "/mnt/data/zhangjian/google/siglip-so400m-patch14-384"
        else:
            raise ValueError(f"Unsupported text model: {self.text_model}")

@dataclass
class EvalConfig(BaseConfig):
    evaluations: List[DatasetEvaluatorConfig] = field(default_factory=list)
    """
    Inference Evaluation configurations.
    """

    load_path: str = "./"
    """
    The directory to load the
    """

    max_crops_override: Optional[int] = None
    """
    Override the max crops used in the model
    """

    seed: int = 6198
    """
    Used to seed all initial RNG states.
    """

    device_inf_eval_batch_size: int = 16
    """
    The number of inference evaluation instances passed to the model in a single forward pass on each device.
    """

    console_log_interval: int = 10
    """
    How often to log what step we are on to console
    """

    fsdp: Optional[FSDPConfig] = None
    """
    Runs models with FSPD, needed for large models or large sequence lengths to reduce memory
    """

    precision: Optional[str] = "fp32"
    """
    Autocase precision 
    """

    pbar: bool = True
    """
    Whether to show a tqdm progress bar
    """

    @property
    def autocast_precision(self) -> torch.dtype:
        if self.precision == "amp_bf16":
            return torch.bfloat16
        elif self.precision == "amp_fp16":
            return torch.float16
        elif self.precision == "fp32":
            return torch.float32
        else:
            raise ValueError(f"Unexpected precision type '{self.precision}'")


def config_to_moe_args(config: ModelConfig) -> Dict[str, Any]:
    from megablocks.layers.arguments import Arguments as MoEArgs

    from .model import Activation

    hidden_size = (
        config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
    )
    act = Activation.build(config)
    num_layers = config.n_layers // 2 if config.moe_interleave else config.n_layers
    kwargs = {
        "activation_fn": F.silu if "swiglu" in config.activation_type.lower() else Activation.build(config),
        "mlp_type": "glu" if "glu" in config.activation_type.lower() else "mlp",
        "mlp_impl": config.moe_mlp_impl,
        "hidden_size": config.d_model,
        "ffn_hidden_size": int(act.output_multiplier * hidden_size),
        "moe_num_experts": config.moe_num_experts,
        "num_layers": num_layers,
        # Handled by FSDP (https://github.com/databricks/megablocks/issues/57#issuecomment-1854594483)
        "moe_weight_parallelism": False,
        "moe_expert_model_parallelism": False,
        "moe_top_k": config.moe_top_k,
        "moe_capacity_factor": config.moe_capacity_factor,
        "moe_loss_weight": config.moe_loss_weight,
        "device": config.init_device,
        # Handled by FSDP
        "bf16": False,
        "fp16": False,
        "bias": config.include_bias,
        "return_bias": False,
        "shared_expert": config.moe_shared_expert,
        "moe_lbl_in_fp32": config.moe_lbl_in_fp32,
    }
    if config.moe_zloss_weight:
        kwargs["moe_zloss_weight"] = config.moe_zloss_weight

    return MoEArgs(**kwargs)

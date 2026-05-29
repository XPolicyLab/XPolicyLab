"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
from functools import partial
from os.path import join
from pathlib import Path
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
    Union,
)

import einops
import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    LayerNormType,
    ModelConfig,
    VisionBackboneType,
    ImagePooling2DType,
    ImageProjectType, AttentionType, InitFnType,
)
from .exceptions import OLMoConfigurationError
from .image_vit import ViTMultiHeadDotProductAttention, ResidualAttentionBlock, VisionTransformer, \
    SiglipVisionTransformer, DinoVisionTransformer
from .initialization import ModuleType, init_weights, init_normal
from .safetensors_util import safetensors_file_to_state_dict
from .torch_util import ensure_finite_
from .util import resource_path

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OLMoBlock",
    "OLMoSequentialBlock",
    "Molmo",
    "OLMoOutput",
    "OLMoGenerateOutput",
    "get_causal_attention_bias",
    "should_checkpoint_block",
    "BufferCache",
    "MolmoVisionBackbone",
    "OLMoBlockGroup",
]


log = logging.getLogger(__name__)


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = not (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and
        (cfg.residual_dropout == 0.0) and (cfg.response_residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=True,
        use_reentrant=False,
    )


def should_checkpoint_block(strategy: Optional[ActivationCheckpointingStrategy], block_idx: int) -> bool:
    if strategy is None:
        return False
    elif (
        (strategy == ActivationCheckpointingStrategy.whole_layer)
        or (strategy == ActivationCheckpointingStrategy.one_in_two and block_idx % 2 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_three and block_idx % 3 == 0)
        or (strategy == ActivationCheckpointingStrategy.one_in_four and block_idx % 4 == 0)
        or (strategy == ActivationCheckpointingStrategy.two_in_three and block_idx % 3 != 0)
        or (strategy == ActivationCheckpointingStrategy.three_in_four and block_idx % 4 != 0)
    ):
        return True
    else:
        return False


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        num_new_embeddings: int,
        features: int,
        device: Union[str, torch.device],
        initializer_range: float = 0.02,
        new_embed_initializer_range: float = 0.02,
    ):
        super().__init__()
        self.initializer_range = initializer_range
        self.new_embed_initializer_range = new_embed_initializer_range
        self.embedding = nn.Parameter(
            torch.zeros(num_embeddings, features, device=device),
        )
        self.new_embedding = nn.Parameter(
            torch.zeros(num_new_embeddings, features, device=device),
        )

    def reset_parameters(self):
        nn.init.normal_(self.embedding, std=self.initializer_range)
        nn.init.normal_(self.new_embedding, std=self.new_embed_initializer_range)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.embedding(x, torch.cat([self.embedding, self.new_embedding], dim=0))
        return x


class Dropout(nn.Dropout):
    def __init__(
        self,
        p: float = 0.5,
        inplace: bool = False,
        mask_p: float = 0,
        broadcast_dims: Sequence[int] = (),
    ):
        super().__init__(p, inplace)
        self.mask_p = mask_p
        self.broadcast_dims = broadcast_dims

    def forward(self, input: torch.Tensor, drop_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        :param input: A tensor of shape `(batch_size, seq_len, embed_dim)`
        :param drop_mask: A tensor of shape `(batch_size, seq_len)` with values of zero or one.
        """
        if self.p == 0.0 and (self.mask_p is None or self.mask_p == 0.0):
            return input
        else:
            if self.mask_p > 0. and self.training:
                assert drop_mask is not None
                drop_mask = drop_mask.to(input.dtype)
                keep_prob = 1.0 - self.p
                keep_prob2 = 1.0 - self.mask_p
                keep_prob = drop_mask * keep_prob2 + (1 - drop_mask) * keep_prob
                keep_prob = keep_prob.unsqueeze(-1)
                dropout_shape = list(input.shape)
                keep_prob = keep_prob.broadcast_to(dropout_shape)
                multiplier = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier.div_(keep_prob)
                return input * multiplier
            elif self.p > 0. and len(self.broadcast_dims) > 0 and self.training:
                keep_prob = 1.0 - self.p
                dropout_shape = list(input.shape)
                for dim in self.broadcast_dims:
                    dropout_shape[dim] = 1
                keep = input.new_empty(dropout_shape).bernoulli_(keep_prob)
                multiplier = keep.broadcast_to(input.shape)
                multiplier.div_(keep_prob)
                input = input * multiplier
            else:
                return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
        weight_initializer: Optional[Callable] = torch.ones,
        bias_initializer: Optional[Callable] = torch.zeros,
    ):
        super().__init__()
        self.config = config
        self.eps = self.config.layer_norm_eps or eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(weight_initializer(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(bias_initializer(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
            return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(
            config.max_position_embeddings or config.max_sequence_length,
            _non_meta_init_device(config)
        )

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.head_dim if self.config.head_dim is not None else self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def rotate_every_two(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, hs // 2, 2)
        x1, x2 = x.unbind(dim=-1)
        x = torch.stack((-x2, x1), dim=-1)
        return x.view(B, nh, T, hs)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            batch_size = q_.shape[0]
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            if position_ids is not None:
                freqs_cis_len = (self.config.max_position_embeddings or self.config.max_sequence_length)
            else:
                freqs_cis_len = key_len
            pos_sin, pos_cos = self.get_rotary_embedding(freqs_cis_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            if position_ids is not None:
                assert query_len == key_len, "Query and key lengths must be equal when using position IDs."
                pos_sin = pos_sin[0, 0][position_ids].view(
                    (batch_size, 1, key_len, pos_sin.shape[-1])
                )
                pos_cos = pos_cos[0, 0][position_ids].view(
                    (batch_size, 1, key_len, pos_cos.shape[-1])
                )
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.quick_gelu:
            return QuickGELU(config)
        elif config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.gelu_pytorch_tanh:
            return cast(Activation, GELU(approximate="tanh"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.silu:
            return cast(Activation, SiLU(inplace=False))
        elif config.activation_type == ActivationType.llama_swiglu:
            return LlamaSwiGLU(config)
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{config.activation_type}'")


class QuickGELU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)

    @property
    def output_multiplier(self) -> float:
        return 1.0


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SiLU(nn.SiLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class LlamaSwiGLU(Activation):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return F.silu(x1) * x2

    @property
    def output_multiplier(self) -> float:
        return 0.5


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


class OLMoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        if config.head_dim is None:
            assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout, mask_p=config.response_residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            assert config.effective_n_kv_heads is not None
            self.k_norm = LayerNormBase.build(
                config,
                size=(config.d_model // config.n_heads) * config.effective_n_kv_heads,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Make sure QKV clip coefficient is positive, otherwise it's not well-defined.
        if config.clip_qkv is not None:
            assert config.clip_qkv > 0

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        input_dim = config.head_dim * config.n_heads if config.head_dim is not None else config.d_model
        self.attn_out = nn.Linear(
            input_dim, config.d_model,
            bias=config.include_bias,
            device=config.init_device
        )

        if self.config.block_type != BlockType.moe:
            # Feed-forward output projection.
            self.ff_out = nn.Linear(
                int(self.act.output_multiplier * self.hidden_size),
                config.d_model,
                bias=config.include_bias,
                device=config.init_device,
            )
            self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

        self.flash_attn_func = None
        if config.attention_type == AttentionType.flash:
            try:
                from flash_attn import flash_attn_func  # type: ignore

                self.flash_attn_func = flash_attn_func
            except ModuleNotFoundError:
                pass

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.
        """
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        if self.flash_attn_func is not None and attn_mask is None:
            r = self.flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=dropout_p, causal=is_causal
            )
            return r.transpose(1, 2)
        else:
            # torch's sdpa doesn't support GQA, so we're doing this
            assert k.size(1) == v.size(1)
            num_kv_heads = k.size(1)
            num_q_heads = q.size(1)
            if num_q_heads != num_kv_heads:
                assert num_q_heads % num_kv_heads == 0
                k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
                v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)

            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        k = k.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)
        # shape: (B, n_kv_h, T, hs)
        v = v.view(B, T, self.config.effective_n_kv_heads, C // self.config.n_heads).transpose(1, 2)

        if self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            v = torch.cat((past_value.to(v.device), v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if not self.config.use_position_ids and self.config.rope:
            # Apply rotary embeddings
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
            drop_mask=drop_mask,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> OLMoBlock:
        if config.block_type == BlockType.sequential:
            return OLMoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OLMoLlamaBlock(layer_id, config, cache)
        elif config.block_type == BlockType.moe:
            return OLMoEBlock(layer_id, config, cache)
        elif config.block_type == BlockType.qwen:
            return OLMoQwenBlock(layer_id, config, cache) # add for qwen3
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OLMoQwenBlock(OLMoBlock):
    """
    This is a transformer block that mimics the Qwen3 architecture.
    The key difference is the per-head RMS normalization applied to queries and keys
    within the attention mechanism.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms for the block inputs
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache
        
        # Per-head Layer norms for query and key, which is a key feature of Qwen3.
        # self.head_dim = config.d_model // config.n_heads
        self.head_dim = config.head_dim 
        self.q_norm = LayerNorm.build(config, size=self.head_dim)
        self.k_norm = LayerNorm.build(config, size=self.head_dim)

        # Attention input projection. Projects x -> (q, k, v)
        # q_proj_out_dim = config.d_model
        q_proj_out_dim = self.head_dim * config.n_heads
        k_proj_out_dim = config.effective_n_kv_heads * self.head_dim
        v_proj_out_dim = config.effective_n_kv_heads * self.head_dim

        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.qkv_bias, device=config.init_device
        )
        mlp_intermediate_size = self.hidden_size
        # Feed-forward input projection (for SwiGLU).
        self.ff_proj1 = nn.Linear(
            config.d_model, mlp_intermediate_size, bias=False, device=config.init_device
        )
        self.ff_proj2 = nn.Linear(
            config.d_model, mlp_intermediate_size, bias=False, device=config.init_device
        )
        self.ff_out = nn.Linear(
            mlp_intermediate_size,
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True
        if self.config.norm_after:
            raise NotImplementedError("Qwen-style blocks do not support post-normalization")

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        self.q_norm.reset_parameters()
        self.k_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj1, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj2, d=self.config.d_model, layer_id=None)

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Reshape for per-head operations.
        # q shape: (B, n_heads, T, head_dim)
        q = q.view(B, T, self.config.n_heads, self.head_dim).transpose(1, 2)
        # k,v shape: (B, n_kv_heads, T, head_dim)
        k = k.view(B, T, self.config.effective_n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.config.effective_n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply per-head LayerNorm, the key feature of Qwen3.
        q = self.q_norm(q).to(dtype=dtype)
        k = self.k_norm(k).to(dtype=dtype)

        # The rest of the attention logic is similar to the base implementation.
        # Apply rotary embeddings.
        if self.config.rope:
            q, k = self.rotary_emb(q, k, position_ids=position_ids)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key.to(k.device), k), dim=-2)
            v = torch.cat((past_value.to(v.device), v), dim=-2)

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]

        if attention_bias is not None:
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Scaled dot product attention.
        att = self._scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_bias,
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        # att = att.transpose(1, 2).contiguous().view(B, T, self.config.d_model)
        
        # reshape 回 (B, T, num_heads * head_dim) 后再做投影
        head_output_dim = self.head_dim * self.config.n_heads
        att = att.transpose(1, 2).contiguous().view(B, T, head_output_dim)

        # Apply output projection.
        return self.attn_out(att), present

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-normalization for attention
        x_normed = self.attn_norm(x)
        
        # Projections
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Attention computation
        att, cache = self.attention(
            q, k, v, attention_bias, position_ids=position_ids,
            drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache
        )

        # First residual connection
        x = x + self.dropout(att, drop_mask=drop_mask)

        # Keep a reference for the second residual connection
        og_x = x
        
        # Pre-normalization for MLP
        x = self.ff_norm(x)
        
        # MLP (SwiGLU)
        x1 = self.ff_proj1(x)
        x2 = self.ff_proj2(x)
        x = self.act(x1, x2)
        x = self.ff_out(x)
        
        # Second residual connection
        x = og_x + self.dropout(x, drop_mask=drop_mask)

        return x, cache

class OLMoEBlock(OLMoBlock):
    """
    This is a transformer MoE block where the output is computed as ``MoE(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        try:
            from megablocks.layers.dmoe import dMoE
            from megablocks.layers.moe import MoE
        except ImportError:
            raise ImportError(
                "To train MoEs, run `pip install git+https://github.com/Muennighoff/megablocks.git@olmoe`"
            )
        from .config import config_to_moe_args

        super().__init__(layer_id, config, cache)

        self.moe_args = config_to_moe_args(config)
        self.ffn = dMoE(self.moe_args) if self.config.moe_dropless else MoE(self.moe_args)

        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)

        # Attention input projection. Projects x -> (q, k, v)
        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()

        if self.config.init_fn == InitFnType.normal:
            attn_out_std = ff_out_std = in_std = self.config.init_std
            cutoff_factor = self.config.init_cutoff_factor
        elif self.config.init_fn == InitFnType.mitchell:
            in_std = 1 / math.sqrt(self.config.d_model)
            attn_out_std = 1 / (math.sqrt(2 * self.config.d_model * (self.layer_id + 1)))
            ff_out_std = 1 / (math.sqrt(2 * self.ff_out.in_features * (self.layer_id + 1)))
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        elif self.config.init_fn == InitFnType.full_megatron:
            in_std = self.config.init_std
            attn_out_std = ff_out_std = self.config.init_std / math.sqrt(2.0 * self.config.n_layers)
            cutoff_factor = self.config.init_cutoff_factor or 3.0
        else:
            raise NotImplementedError(self.config.init_fn)

        init_normal(self.att_proj, std=in_std, init_cutoff_factor=cutoff_factor)
        init_normal(self.attn_out, std=attn_out_std, init_cutoff_factor=cutoff_factor)
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        init_normal(self.ffn.experts.mlp.w1, std=in_std, init_cutoff_factor=cutoff_factor)
        init_normal(self.ffn.experts.mlp.w2, std=ff_out_std, init_cutoff_factor=cutoff_factor)
        if hasattr(self.ffn.experts.mlp, "v1"):
            init_normal(self.ffn.experts.mlp.v1, std=in_std, init_cutoff_factor=cutoff_factor)
        if self.ffn.experts.bias is not None:
            torch.nn.init.zeros_(self.ffn.experts.bias)
        init_normal(self.ffn.router.layer, std=in_std, init_cutoff_factor=cutoff_factor)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        max_doc_len: Optional[int] = None,
        cu_doc_lens: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)
        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                qkv = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x))
            else:
                qkv = self.att_proj(self.attn_norm(x))
        else:
            qkv = self.att_proj(x)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention,
                q,
                k,
                v,
                attention_bias,
                position_ids=position_ids,
                drop_mask=drop_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                # max_doc_len=max_doc_len,
                # cu_doc_lens=cu_doc_lens,
            )
        else:
            att, cache = self.attention(
                q,
                k,
                v,
                attention_bias,
                position_ids=position_ids,
                drop_mask=drop_mask,
                layer_past=layer_past,
                use_cache=use_cache,
                # max_doc_len=max_doc_len,
                # cu_doc_lens=cu_doc_lens,
            )
        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)
        # Add attention scores.
        # shape: (B, T, C)
        
        x = x + self.dropout(att, drop_mask=drop_mask)
        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if self.config.norm_after:
            x = self.ffn(x)
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)
            return og_x + self.dropout(x, drop_mask=drop_mask), cache
        else:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)
            # Activation checkpointing for the MoE FFN is not supported
            x = self.ffn(x)
            x =  og_x + self.dropout(x, drop_mask=drop_mask)
            return x, cache


class OLMoSequentialBlock(OLMoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims),
            bias=config.include_bias or config.qkv_bias,
            device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #  - for group query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_kv_heads)

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                atten_in = self._activation_checkpoint_fn(self.attn_norm, x)
            else:
                atten_in = self.attn_norm(x)
        else:
            atten_in = x
        qkv = self.att_proj(atten_in)

        if self.config.clip_qkv is not None:
            qkv.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        q, k, v = qkv.split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, position_ids=position_ids, drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, position_ids=position_ids, drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                att = self._activation_checkpoint_fn(self.attn_norm, att)
            else:
                att = self.attn_norm(att)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att, drop_mask=drop_mask)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x

        if not self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)

        if self.config.norm_after:
            if self._activation_checkpoint_fn is not None:
                x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
            else:
                x = self.ff_norm(x)

        x = self.dropout(x, drop_mask=drop_mask)
        x = og_x + x

        return x, cache


class OLMoLlamaBlock(OLMoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OLMoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        q_proj_out_dim = config.d_model
        k_proj_out_dim = config.effective_n_kv_heads * (config.d_model // config.n_heads)
        v_proj_out_dim = config.effective_n_kv_heads * (config.d_model // config.n_heads)

        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.qkv_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.qkv_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.qkv_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj1 = nn.Linear(
            config.d_model, self.hidden_size // 2, bias=False, device=config.init_device
        )
        self.ff_proj2 = nn.Linear(
            config.d_model, self.hidden_size // 2, bias=False, device=config.init_device
        )
        if self.config.norm_after:
            raise NotImplementedError()

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj1, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj2, d=self.config.d_model, layer_id=None)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        response_dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        # For GQA
        assert k.size(1) == v.size(1)
        num_kv_heads = k.size(1)
        num_q_heads = q.size(1)
        if num_q_heads != num_kv_heads:
            assert num_q_heads % num_kv_heads == 0
            k = k.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
            v = v.repeat_interleave(num_q_heads // num_kv_heads, dim=1, output_size=num_q_heads)
        
        og_dtype = q.dtype
        k = k.to(q.device)
        v = v.to(q.device)
        if attn_mask is not None:
            attn_mask = attn_mask.to(q.device)

        assert response_dropout_p == 0.0, "Response dropout is not supported in Llama."

        if self.config.float32_attention:
            q, k = q.to(torch.float), k.to(torch.float)

        if self.config.attention_type == AttentionType.direct:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)

            if is_causal:
                assert attn_mask is None

                query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
                attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
            elif attn_mask is not None:
                attn_bias = attn_mask
            else:
                attn_bias = torch.zeros_like(attn_weights)

            attn_weights += attn_bias

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=self.training).to(v.dtype)

            att = torch.matmul(attn_weights, v)
        elif self.config.attention_type == AttentionType.sdpa:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=is_causal,
            )
        else:
            raise NotImplementedError(self.config.attention_type)
        att = att.to(og_dtype)
        return att

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        if self.config.clip_qkv is not None:
            q.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            k.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)
            v.clamp_(min=-self.config.clip_qkv, max=self.config.clip_qkv)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, position_ids=position_ids, drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, position_ids=position_ids, drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att, drop_mask=drop_mask)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x1 = self.ff_proj1(x)
        x2 = self.ff_proj2(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x1, x2)  # type: ignore
        else:
            x = self.act(x1, x2)
        x = self.ff_out(x)
        x = self.dropout(x, drop_mask=drop_mask)
        x = og_x + x

        return x, cache


class OLMoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """

    hidden_states: Optional[Tuple[torch.Tensor]]
    """
    Hidden states from each block.
    """


class OLMoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


class OLMoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        drop_mask: Optional[torch.Tensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block, x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=drop_mask, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class ImageProjectorMLP(nn.Module):
    """MLP used for the image projector"""

    def __init__(self, config: ModelConfig, input_dim: int, dropout: float = 0.0):
        super().__init__()
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.initializer_range = config.initializer_range

        self.w1 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=config.init_device,
        )
        self.w2 = nn.Linear(
            self.hidden_size // 2,
            config.d_model,
            bias=False,
            device=config.init_device,
        )
        self.w3 = nn.Linear(
            input_dim,
            self.hidden_size // 2,
            bias=False,
            device=config.init_device,
        )
        # Activation function.
        self.act = Activation.build(config)
        self.dropout = Dropout(dropout)
    
    def reset_parameters(self):
        nn.init.normal_(self.w1.weight, std=self.initializer_range)
        nn.init.normal_(self.w2.weight, std=self.initializer_range)
        nn.init.normal_(self.w3.weight, std=self.initializer_range)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.w2(self.act(self.w1(x), self.w3(x)))
        x = self.dropout(x)
        return x


class Residual(nn.Module):
    def __init__(self, submodule: nn.Module):
        super().__init__()
        self.submodule = submodule
    
    def reset_parameters(self):
        self.submodule.reset_parameters()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.submodule(x)


class MolmoVisionBackbone(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        input_dim: int = None
        self.image_pooling_2d: nn.Module = None
        if config.image_pooling_2d in {ImagePooling2DType.attention, ImagePooling2DType.attention_meanq}:
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(config, is_vit_layer=False)
            input_dim = config.vision_backbone.image_emb_dim
        elif config.image_pooling_2d == ImagePooling2DType.attention_2wide:
            cfg = deepcopy(config)
            cfg.vision_backbone.image_emb_dim *= 2
            cfg.vision_backbone.image_head_dim *= 2
            self.image_pooling_2d = ViTMultiHeadDotProductAttention(cfg, is_vit_layer=False)
            input_dim = cfg.vision_backbone.image_emb_dim
        elif config.image_pooling_2d in [ImagePooling2DType.none, ImagePooling2DType.stack]:
            self.image_pooling_2d = None
            nlayers = 1 if config.vit_layers is None else len(config.vit_layers)
            input_dim = nlayers * config.vision_backbone.image_emb_dim
            if config.image_pooling_2d == ImagePooling2DType.stack:
                input_dim *= 4
        else:
            raise NotImplementedError(f"Unknown image pooling 2D method: {config.image_pooling_2d}")
        
        self.input_dim = input_dim

        # `MLP` assume the activation takes two inputs, so it must be a 'llama' version
        if config.activation_type == ActivationType.swiglu:
            mlp_config = replace(config, activation_type=ActivationType.llama_swiglu)
        else:
            mlp_config = config
        if config.image_projector == ImageProjectType.mlpx2:
            self.image_projector = nn.ModuleList(
                [ImageProjectorMLP(mlp_config, input_dim), Residual(ImageProjectorMLP(config, input_dim))]
            )
        elif config.image_projector == ImageProjectType.mlp:
            self.image_projector = ImageProjectorMLP(mlp_config, input_dim)
        elif config.image_projector == ImageProjectType.linear:
            self.image_projector = nn.Linear(
                input_dim,
                config.d_model,
                bias=False,
                device=config.init_device,
            )
        else:
            raise NotImplementedError(f"Unknown image projector: {config.image_projector}")

        self.image_feature_dropout = Dropout(config.image_feature_dropout)
        self.grad_checkpointing = False

        v_cfg = self.config.vision_backbone
        if v_cfg.image_model_type == VisionBackboneType.openai:
            self.image_vit = VisionTransformer(config)
        elif v_cfg.image_model_type == VisionBackboneType.siglip:
            self.image_vit = SiglipVisionTransformer(config)
        elif v_cfg.image_model_type == VisionBackboneType.dino:
            self.image_vit = DinoVisionTransformer(config)
        else:
            raise NotImplementedError(f"Unknown image model type: {v_cfg.image_model_type}")

        self.num_prefix_tokens = self.image_vit.num_prefix_tokens
        assert self.num_prefix_tokens in {0, 1}, "Only 0 or 1 prefix tokens are supported"

        self.pad_embed = None
        if config.image_padding_embed:
            image_dim = v_cfg.image_emb_dim*len(self.config.vit_layers)
            if config.image_padding_embed in ["pad_embed", "regress"]:
                self.pad_embed = nn.Parameter(
                    torch.zeros((image_dim,), device=config.init_device))
            elif config.image_padding_embed == "pad_and_partial_pad":
                self.pad_embed = nn.Parameter(
                    torch.zeros((2, image_dim), device=config.init_device))
            else:
                raise ValueError(config.image_padding_embed)

    @classmethod
    def build(cls, config: ModelConfig) -> 'MolmoVisionBackbone':
        v_cfg = config.vision_backbone
        assert v_cfg is not None
        return MolmoVisionBackbone(config)
    
    def reset_connector_parameters(self):
        if self.image_pooling_2d is not None:
            self.image_pooling_2d.reset_parameters()
        if self.config.image_projector == "2mlp":
            for module in self.image_projector:
                module.reset_parameters()
        elif self.config.image_projector == "linear":
            nn.init.xavier_uniform_(self.image_projector.weight)
        else:
            self.image_projector.reset_parameters()

    def reset_parameters(self):
        self.reset_connector_parameters()
        self.image_vit.reset_parameters()

    def reset_with_pretrained_weights(self):
        self.reset_connector_parameters()  # resets the connector
        if self.config.vit_load_path:
            vit_load_path = Path(self.config.vit_load_path)
            state_dict_path = resource_path(
                vit_load_path.parent, vit_load_path.name,
                local_cache=vit_load_path.parent,
            )
            assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
            state_dict = torch.load(state_dict_path, map_location="cpu")
            self.image_vit.load_state_dict(state_dict)
        else:
            self.image_vit.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.grad_checkpointing = True
        if strategy in (ActivationCheckpointingStrategy.whole_layer, ActivationCheckpointingStrategy.vit_only):
            self.image_vit.set_grad_checkpointing()
    
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        : param images: (batch_size, num_crops, num_patch, n_pixels)
        """
        cfg = self.config
        v_cfg = self.config.vision_backbone
        B, T, N, D = images.shape

        mask = ~torch.all(images.view(B * T, N, D) == -1, dim=(1, 2), keepdim=True)

        # Output all hidden states
        # n_layers x (batch_num_crops, (1+)n_tokens, image_emb_dim)
        images = images.view(B * T, N, D)
        image_features = self.image_vit(images)

        if cfg.vit_layers is not None:
            features = []
            for layer in cfg.vit_layers:
                features.append(image_features[layer])
            image_features = torch.cat(features, dim=-1)
        else:
            image_features = image_features[-1]

        cls_embed: torch.Tensor = None
        if self.num_prefix_tokens > 0:
            cls_embed = image_features[:, 0]
            image_features = image_features[:, 1:]
        
        image_features = image_features * mask
        image_features = image_features.view(B, T, N, -1)

        cls_embed = cls_embed.view(B, T, -1) if cls_embed is not None else None

        return image_features
    
    def forward(self, images: torch.Tensor, image_masks: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        cfg = self.config

        # image_features: (batch_size, num_crops(=num_image), num_patch, nximage_emb_dim)
        batch_size, num_image = images.shape[:2]
        image_features = self.encode_image(images)

        if cfg.image_padding_embed:
            assert image_masks is not None
            if cfg.image_padding_embed == "pad_embed":
                all_pad = (image_masks == 0).to(dtype=torch.float32)
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(all_pad, -1)
            elif cfg.image_padding_embed == "regress":
                pad_embed = self.pad_embed[None, None, None, :]
                image_features = image_features + pad_embed * torch.unsqueeze(torch.maximum(image_masks, torch.zeros_like(image_masks)), -1)
            elif cfg.image_padding_embed == "pad_and_partial_pad":
                pad_embed = self.pad_embed[:, None, None, None, :]
                all_pad = image_masks == 0
                partial_pad = torch.logical_and(image_masks < 1, torch.logical_not(all_pad)).to(dtype=torch.float32)
                all_pad = all_pad.to(dtype=torch.float32)

                # print(f"****MolmoVisionBackbone forward,image_features: {image_features.shape}, partial_pad: {partial_pad.shape}, pad_embed: {pad_embed.shape}")
                image_features = image_features + pad_embed[0] * torch.unsqueeze(all_pad, -1)
                image_features = image_features + pad_embed[1] * torch.unsqueeze(partial_pad, -1)
            else:
                raise ValueError(cfg.image_padding_embed)

        image_features = self.image_feature_dropout(image_features)

        image_features = image_features.reshape(
            (batch_size, num_image) + cfg.image_num_patch + (-1,),
        )

        if cfg.image_num_patch[0] % cfg.image_pooling_h != 0 or cfg.image_num_patch[1] % cfg.image_pooling_w != 0:
            pad_h = cfg.image_num_patch[0] % cfg.image_pooling_h
            pad_w = cfg.image_num_patch[1] % cfg.image_pooling_w
            # Pad so we can still pool mxn patches
            image_features = F.pad(
                image_features,
                (0, 0, 0, pad_w, 0, pad_h, 0, 0, 0, 0),
            )

        # image pooling
        image_features = einops.rearrange(
            image_features,
            'b n (h dh) (w dw) c -> (b n h w) (dh dw) c',
            dh=cfg.image_pooling_h,
            dw=cfg.image_pooling_w,
        )

        if cfg.image_pooling_2d == ImagePooling2DType.attention_meanq:
            query = image_features.mean(-2, keepdim=True)
            image_features = self.image_pooling_2d(query, image_features)
        elif cfg.image_pooling_2d not in {ImagePooling2DType.none, ImagePooling2DType.stack}:
            if self.grad_checkpointing:
                from torch.utils.checkpoint import checkpoint
                image_features = checkpoint(self.image_pooling_2d, image_features[:, :1, :], image_features, use_reentrant=False)
            else:
                image_features = self.image_pooling_2d(image_features[:, :1, :], image_features)

        h, w = cfg.llm_patches_per_crop()
        image_features = image_features.reshape(batch_size, num_image, h * w, -1)

        # MLP layer to map the feature.
        if cfg.image_projector == ImageProjectType.mlpx2:
            for module in self.image_projector:
                image_features = module(image_features)
        else:
            if self.grad_checkpointing:
                from torch.utils.checkpoint import checkpoint
                image_features = checkpoint(self.image_projector, image_features, use_reentrant=False)
            else:
                image_features = self.image_projector(image_features)
        
        # image_features: (batch_size, num_image, num_patch, d_model)
        # cls_embed: (batch_size, num_image, d_model)
        return image_features


class Molmo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

        # Validate config.
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OLMoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        wte = None
        if self.config.additional_vocab_size is not None:
            wte = Embedding(
                config.embedding_size or config.vocab_size,
                config.additional_vocab_size,
                config.d_model,
                device=config.init_device,
                initializer_range=config.initializer_range,
                new_embed_initializer_range=config.new_embedding_init_range
            )
        else:
            wte=nn.Embedding(
                config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
            )

        self.transformer = nn.ModuleDict(
            dict(
                wte=wte,
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [OLMoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OLMoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not self.config.rope:
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.ff_out_size or config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )
        
        self.vision_backbone: Optional[MolmoVisionBackbone] = None
        if config.vision_backbone is not None:
            self.vision_backbone = MolmoVisionBackbone.build(config)

        self.__num_fwd_flops: Optional[int] = None

    def reset_with_pretrained_weights(self):
        if self.config.llm_load_path is None:
            self.reset_non_vision_parameters()
        else:
            state_dict_path = resource_path(
                Path(self.config.llm_load_path).parent, Path(self.config.llm_load_path).name,
                local_cache=Path(self.config.llm_load_path).parent,
            )
            assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
            if state_dict_path.name.endswith("safetensors"):
                state_dict = safetensors_file_to_state_dict(state_dict_path, map_location="cpu")
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")
            if all(x.startswith("transformer.") for x in state_dict.keys()):
                state_dict = {k[len("transformer."):]: v for k, v in state_dict.items()}
            if "wte.weight" in state_dict and self.config.additional_vocab_size:
                state_dict["wte.embedding"] = state_dict.pop("wte.weight")
            transformer_keys = set(x[0] for x in self.transformer.named_parameters())

            # print(f"****** Molmo reset_with_pretrained_weights: transformer_keys - set(state_dict.keys()): {transformer_keys - set(state_dict.keys())}")
            assert transformer_keys - set(state_dict.keys()) <= {"wte.new_embedding"}, \
                "Unexpected keys in the model file"
            self.transformer.load_state_dict(state_dict, strict=False)
            if hasattr(self.transformer.wte, "new_embedding"):
                # This is the only parameter not initialized from the LLM weights
                nn.init.normal_(self.transformer.wte.new_embedding, std=self.config.new_embedding_init_range)

        if self.vision_backbone is not None:
            self.vision_backbone.reset_with_pretrained_weights()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

        if self.vision_backbone is not None:
            self.vision_backbone.set_activation_checkpointing(strategy)
    
    @staticmethod
    def get_connector_parameters():
        return tuple(
            [
                "vision_backbone.image_pooling_2d",
                "vision_backbone.image_projector",
                "vision_backbone.cls_projector",
                "vision_backbone.pad_embed",
                "transformer.wte.new_embedding",
            ]
        )

    @staticmethod
    def get_vit_parameters():
        return tuple(
            [
                "vision_backbone.image_vit",
            ]
        )

    @staticmethod
    def get_llm_parameters():
        return tuple(
            [
                "transformer.wte.embedding", "transformer.wte.weight", "transformer.wpe",
                "transformer.blocks", "transformer.block_groups",
                "transformer.ln_f", "transformer.ff_out",
            ]
        )

    @staticmethod
    def get_weight_decay_exclusions():
        return tuple(
            [
                "wte", "attn_norm", "ff_norm",
                "pre_attn_norm", "post_attn_norm",
                "pre_ff_norm", "post_ff_norm",
                "ln_f",
                "pre_ln",
                "attention_norm", "ffn_norm",
                "lambda1", "lambda2",
                "positional_embedding", "class_embedding",
            ]
        )

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()
        self.reset_non_vision_parameters()

    def reset_non_vision_parameters(self):
        # Top-level embeddings / linear layers.
        if self.config.additional_vocab_size is not None:
            self.transformer.wte.reset_parameters()
            if hasattr(self.transformer.wte, "new_embedding"):
                nn.init.normal_(self.transformer.wte.new_embedding, std=self.config.new_embedding_init_range)
        else:
            init_weights(
                self.config,
                self.transformer.wte,  # type: ignore
                std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
                type_of_module=ModuleType.emb,
            )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()


    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        response_mask: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        subsegment_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
        output_hidden_states: Optional[bool] = None,
        append_last_valid_logits: Optional[torch.Tensor] = None,
    ) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param response_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            the response mask. A `1` value in the mask means that the corresponding token
            is a response token. A `0` means that the corresponding token is not
            a response token.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        output_hidden_states = output_hidden_states if output_hidden_states is not None else False

        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        has_image = images is not None

        assert not (has_image and input_embeddings is not None), "Cannot provide both images and input embeddings."
        assert not (has_image and past_key_values is not None), "Cached key and values should not be used with images."

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
        
        if subsegment_ids is not None:
            assert not use_cache, "Subsegment_ids cannot be used with cache."
            subsegment_mask = subsegment_ids.unsqueeze(2) <= subsegment_ids.unsqueeze(1)
            attention_mask = (
                subsegment_mask.to(attention_mask.dtype) *
                attention_mask.unsqueeze(2) *
                attention_mask.unsqueeze(1))
            if position_ids is None:
                raise ValueError(f"Positioned ids must be given if using subsegment_ids")
        else:
            if self.config.use_position_ids and position_ids is None:
                position_ids = torch.clamp(
                    torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                    min=0,
                ).broadcast_to((batch_size, attention_mask.shape[-1]))

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        if input_ids is not None:
            input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        num_image: Optional[int] = None
        if images is not None:
            # shape: (batch_size, num_image, num_patch, d_model)
            # cls_embed: (batch_size, num_image, d_model)
            image_features = self.vision_backbone(images, image_masks)
            num_image, num_patch = image_features.shape[1:3]
            assert image_input_idx.shape == (batch_size, num_image, num_patch)

            # inster the image feature into the embedding.
            image_features = image_features.view(batch_size, num_image * num_patch, -1)
            image_input_idx = image_input_idx.view(batch_size, num_image * num_patch)

            valid = image_input_idx >= 0
            batch_idx = torch.arange(batch_size, device=x.device)
            batch_idx = torch.tile(batch_idx[:, None], [1, image_features.shape[1]])

            # For hf demo/endpoint
            image_features = image_features.to(x.device)

            x[batch_idx[valid], image_input_idx[valid]] += image_features[valid]

        if not self.config.rope:
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x
        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore
        # normalized
        if self.config.normalize_input_embeds:
            x = x * (self.config.d_model ** 0.5)

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            if len(attention_mask.shape) == 2:
                attention_mask = attention_mask[:, :past_length + seq_len]
                attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            else:
                attention_mask = attention_mask.unsqueeze(1).to(dtype=torch.float)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)
            
            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # decoder layers
        all_hidden_states = []

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if should_checkpoint_block(self.activation_checkpointing_strategy, block_idx):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layer_past=layer_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                if output_hidden_states:
                    # add hidden states
                    all_hidden_states.append(x)

                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, position_ids=position_ids, drop_mask=response_mask, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)
        
        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            if append_last_valid_logits is not None:
                last_valid_output = x[
                    torch.arange(x.shape[0], device=x.device), append_last_valid_logits.to(x.device)]
                x = last_valid_output.unsqueeze(1)
            else:
                x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore
        if output_hidden_states:
            # add final hidden state post-final-layernorm, following HuggingFace's convention
            all_hidden_states.append(x)

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        if not last_logits_only and append_last_valid_logits is not None:
            last_valid_logit = logits[
                torch.arange(logits.shape[0], device=logits.device), append_last_valid_logits]
            logits = torch.cat([logits[:, :-1], last_valid_logit[:, None]], dim=1)
        return OLMoOutput(logits=logits, attn_key_values=attn_key_values, hidden_states=tuple(all_hidden_states) if output_hidden_states else None)  # type: ignore[arg-type]

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

        # The 'recurse' mode for the wrap function does not behave like you'd expect.
        # Even if we return False, it may still recurse because PyTorch does what it wants,
        # not what you want. This causes issues when, for example, we want to wrap 'ff_out' (a linear layer)
        # but not other linear layers within a block.
        # So we have to explicitly tell PyTorch which linear layers to wrap, and we also just
        # return True in 'recurse' mode for simplicity.
        size_based_module_to_wrap = {self.transformer.wte}
        if hasattr(self.transformer, "ff_out"):
            size_based_module_to_wrap.add(self.transformer.ff_out)
        if hasattr(self.transformer, "ln_f"):
            size_based_module_to_wrap.add(self.transformer.ln_f)
        if self.vision_backbone is not None and self.config.vision_backbone.fsdp_wrap:
            size_based_module_to_wrap.add(self.vision_backbone.image_pooling_2d)
            size_based_module_to_wrap.add(self.vision_backbone.image_projector)

        wrap_layer_names = (ResidualAttentionBlock, MolmoVisionBackbone, VisionTransformer)

        if wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names + (OLMoBlock,))
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names + (OLMoBlock,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names + (OLMoBlockGroup,))
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OLMoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, wrap_layer_names + (OLMoBlockGroup,)) or module in size_based_module_to_wrap
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return size_based_auto_wrap_policy
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, OLMoBlock) and module.layer_id % c == 0
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True, include_inactive_params: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        if not include_inactive_params:
            # Need to reduce blocks to the number of experts that are selected
            # If not dropless 'transformer.blocks.0.ffn.experts.mlp.w1' has shape (total_experts, in_dim, out_dim)
            # change to 'transformer.blocks.0.ffn.experts.mlp.w1' with shape (selected_experts, in_dim, out_dim)
            # If dropless, the total_experts & out_dim are combined into one dimension
            idx = self.config.moe_top_k
            if self.config.moe_dropless:
                idx *= self.transformer.blocks[1].moe_args.ffn_hidden_size
            params = [(np[0], np[1][:idx]) if "experts.mlp" in np[0] else np for np in params]  # type: ignore
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops
        n_params = self.num_params()
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.config.max_sequence_length
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * (self.config.max_sequence_length**2))
        )
        self.__num_fwd_flops = params_flops_per_seq + attn_flops_per_seq
        return self.__num_fwd_flops

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        image_masks: Optional[torch.Tensor] = None,
        image_input_idx: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
        is_distributed: bool=False
    ) -> OLMoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see :class:`BeamSearch`.
        """
        beam_search = BeamSearch(
            self.config.get_tokenizer().eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
            distributed_model=is_distributed
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        mask_len = seq_len + max_steps if self.config.use_position_ids else seq_len
        position_ids: Optional[torch.Tensor] = None
        append_last_valid_logits: Optional[torch.Tensor] = None
        if self.config.use_position_ids and attention_mask is None:
            attention_mask = input_ids != -1
            position_ids = torch.clamp(
                torch.cumsum(attention_mask.to(torch.int32), dim=-1) - 1,
                min=0
            )
            append_last_valid_logits = attention_mask.long().sum(dim=-1) - 1
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, max_steps))],
                dim=1,
            )
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, mask_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated
            nonlocal position_ids
            nonlocal images
            nonlocal image_input_idx
            nonlocal append_last_valid_logits

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if not self.config.use_position_ids and attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
                _images = None
                _image_input_idx = None
                if self.config.use_position_ids:
                    position_ids = position_ids[:, -1:] + 1
                    _, *last_dims = position_ids.size()
                    _position_ids = (
                        position_ids.unsqueeze(1)
                        .expand(batch_size, beam_size, *last_dims)
                        .reshape(batch_size * beam_size, *last_dims)
                    )
                else:
                    _position_ids = None
                
                _append_last_valid_logits = None

            else:
                past_key_values = None
                input_ids = state["input_ids"]
                _images = images
                _image_input_idx = image_input_idx
                _position_ids = position_ids
                _append_last_valid_logits = append_last_valid_logits

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                images=_images,
                image_masks=image_masks,
                image_input_idx=_image_input_idx,
                position_ids=_position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
                append_last_valid_logits=_append_last_valid_logits,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.no_grad():
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OLMoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu",
        checkpoint_type: Optional[CheckpointType] = None
    ) -> Molmo:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path
        if checkpoint_dir.startswith("hf:"):
            from .hf_molmo import load_hf_model
            return load_hf_model(checkpoint_dir[3:])

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        if Path(join(checkpoint_dir, "model.yaml")).exists():
            model_config = ModelConfig.load(Path(join(checkpoint_dir, "model.yaml")))
        else:
            config_path = resource_path(checkpoint_dir, "config.yaml")
            model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = Molmo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            dtype = state_dict[list(state_dict.keys())[0]].dtype
            log.info(f"Checkpoint weight dtype: {dtype}")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            from .checkpoint import load_model_state

            # Initialize model on target device. In this case the state dict is loaded in-place
            # so it's not necessary to start on CPU if the target device is a GPU.
            model_config.init_device = device
            model = Molmo(model_config)

            # Load state dict in place.
            load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new
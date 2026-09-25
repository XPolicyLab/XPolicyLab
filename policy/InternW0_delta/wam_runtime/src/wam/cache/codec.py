"""Shared value codecs for every artifact managed by :mod:`wam.cache`."""

from __future__ import annotations

from dataclasses import dataclass

import torch


RAW_CODEC = "raw"
INT8_SYMMETRIC_CODEC = "int8_symmetric"
VALUE_CODECS = frozenset({RAW_CODEC, INT8_SYMMETRIC_CODEC})
QUANTIZED_DTYPES = frozenset({torch.float16, torch.bfloat16, torch.float32})


def normalize_value_codec(value: object) -> str:
    codec = str(value or RAW_CODEC).strip().lower()
    if codec not in VALUE_CODECS:
        allowed = ", ".join(sorted(VALUE_CODECS))
        raise ValueError(f"Unsupported cache value codec {value!r}; allowed: {allowed}.")
    return codec


@dataclass(frozen=True)
class EncodedTensor:
    """An INT8 tensor kept compressed until its consumer materializes it."""

    quantized: torch.Tensor
    scale: torch.Tensor
    dtype: torch.dtype

    def __post_init__(self) -> None:
        if self.quantized.dtype != torch.int8:
            raise TypeError("EncodedTensor.quantized must use torch.int8.")
        if self.scale.dtype != torch.float32:
            raise TypeError("EncodedTensor.scale must use torch.float32.")
        if self.dtype not in QUANTIZED_DTYPES:
            raise TypeError(f"Unsupported EncodedTensor output dtype: {self.dtype}.")
        if tuple(self.scale.shape) != (*self.quantized.shape[:-1], 1):
            raise ValueError(
                "EncodedTensor scale shape must match quantized tensor vectors: "
                f"q={tuple(self.quantized.shape)} scale={tuple(self.scale.shape)}."
            )

    @property
    def shape(self) -> torch.Size:
        return self.quantized.shape

    @property
    def ndim(self) -> int:
        return self.quantized.ndim

    def __getitem__(self, index: object) -> "EncodedTensor":
        return EncodedTensor(self.quantized[index], self.scale[index], self.dtype)

    def pin_memory(self) -> "EncodedTensor":
        return EncodedTensor(
            self.quantized.pin_memory(),
            self.scale.pin_memory(),
            self.dtype,
        )

    def materialize(
        self,
        *,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        non_blocking: bool = True,
    ) -> torch.Tensor:
        quantized = self.quantized.to(device=device, non_blocking=non_blocking)
        scale = self.scale.to(device=device, non_blocking=non_blocking)
        return (quantized.float() * scale).to(dtype=dtype or self.dtype).contiguous()


def quantize_symmetric_int8(tensor: torch.Tensor) -> EncodedTensor:
    """Quantize each last-axis vector with its own FP32 symmetric scale."""

    source = tensor.detach().contiguous()
    if source.dtype not in QUANTIZED_DTYPES or source.ndim == 0 or source.numel() == 0:
        raise ValueError(
            "int8_symmetric requires a non-empty FP16/BF16/FP32 tensor with "
            "at least one dimension."
        )
    working = source.float()
    if not bool(torch.isfinite(working).all().item()):
        raise ValueError("int8_symmetric cannot encode non-finite cache values.")
    absmax = working.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(absmax > 0, absmax / 127.0, torch.ones_like(absmax))
    quantized = torch.round(working / scale).clamp_(-127, 127).to(torch.int8)
    return EncodedTensor(quantized.contiguous(), scale.contiguous(), source.dtype)


def materialize_cache_tensor(
    value: torch.Tensor | EncodedTensor,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    non_blocking: bool = True,
) -> torch.Tensor:
    if isinstance(value, EncodedTensor):
        return value.materialize(
            device=device,
            dtype=dtype,
            non_blocking=non_blocking,
        )
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected cached Tensor or EncodedTensor, got {type(value)}.")
    kwargs = {"device": device, "non_blocking": non_blocking}
    if dtype is not None:
        kwargs["dtype"] = dtype
    return value.to(**kwargs)

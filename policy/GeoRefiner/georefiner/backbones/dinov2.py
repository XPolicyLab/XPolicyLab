"""Hugging Face DINOv2 visual backbone with tensor-only preprocessing."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from georefiner.backbones.base import VisualBackboneBase


class DINOv2Backbone(VisualBackboneBase):
    """Return DINOv2 patch tokens for RGB tensors.

    Preprocessing is performed once, entirely with PyTorch tensors. Inputs may
    be ``uint8`` in ``[0, 255]`` or floating point in ``[0, 1]``. By default the
    CLS token is removed, matching the documented GeoRefiner engineering
    assumption.
    """

    def __init__(
        self,
        model: nn.Module,
        image_processor: Any,
        *,
        model_id: str,
        revision: str = "main",
        use_cls_token: bool = False,
        frozen: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_processor = image_processor
        self.model_id = model_id
        self.revision = revision
        self.use_cls_token = use_cls_token
        self.frozen = frozen
        hidden_size = getattr(model.config, "hidden_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("DINOv2 config must define a positive hidden_size.")
        self.output_dim = hidden_size

        settings = self._processor_settings(image_processor)
        self.preprocessing_config = settings
        self.register_buffer(
            "image_mean",
            torch.tensor(settings["image_mean"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(settings["image_std"], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self._set_frozen(frozen)

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        model_id: str | None = None,
        revision: str = "main",
        cache_dir: str | None = None,
        local_files_only: bool = False,
        use_cls_token: bool = False,
        frozen: bool = True,
    ) -> "DINOv2Backbone":
        """Load real DINOv2 weights; failures are never replaced by a mock."""

        try:
            from transformers import AutoImageProcessor, Dinov2Model

            processor = AutoImageProcessor.from_pretrained(
                model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_fast=False,
            )
            model = Dinov2Model.from_pretrained(
                model_name_or_path,
                revision=revision,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained DINOv2 model "
                f"{model_name_or_path!r} (revision={revision!r}, "
                f"local_files_only={local_files_only}). No mock fallback was used."
            ) from exc
        return cls(
            model,
            processor,
            model_id=model_id or model_name_or_path,
            revision=revision,
            use_cls_token=use_cls_token,
            frozen=frozen,
        )

    @classmethod
    def from_config(
        cls,
        config_dict: Mapping[str, Any],
        processor_path: str | Path,
        *,
        model_id: str,
        revision: str = "main",
        use_cls_token: bool = False,
        frozen: bool = True,
    ) -> "DINOv2Backbone":
        """Construct an uninitialized DINOv2 structure without model downloads."""

        try:
            from transformers import AutoImageProcessor, Dinov2Config, Dinov2Model

            config = Dinov2Config.from_dict(dict(config_dict))
            model = Dinov2Model(config)
            processor = AutoImageProcessor.from_pretrained(
                str(processor_path), local_files_only=True, use_fast=False
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to construct offline DINOv2 from checkpoint config and "
                f"processor artifact {str(processor_path)!r}."
            ) from exc
        return cls(
            model,
            processor,
            model_id=model_id,
            revision=revision,
            use_cls_token=use_cls_token,
            frozen=frozen,
        )

    def forward(self, images: Tensor) -> Tensor:
        """Preprocess RGB tensors and return ``[B, N, output_dim]`` tokens."""

        pixel_values = self.preprocess(images)
        parameter = next(self.model.parameters())
        pixel_values = pixel_values.to(device=parameter.device, dtype=parameter.dtype)
        context = torch.no_grad() if self.frozen else nullcontext()
        with context:
            output = self.model(pixel_values=pixel_values, return_dict=True)
        tokens = output.last_hidden_state
        if not self.use_cls_token:
            if tokens.shape[1] < 2:
                raise RuntimeError("DINOv2 output did not contain patch tokens.")
            tokens = tokens[:, 1:]
        if not torch.isfinite(tokens).all():
            raise FloatingPointError("DINOv2 produced NaN or Inf tokens.")
        return tokens

    def preprocess(self, images: Tensor) -> Tensor:
        """Apply the saved Hugging Face processor policy using tensor ops."""

        self._validate_images(images)
        settings = self.preprocessing_config
        output = images.to(dtype=torch.float32)
        if settings["do_rescale"]:
            if images.dtype == torch.uint8:
                output = output * float(settings["rescale_factor"])
            else:
                # Floating inputs are defined to already represent [0, 1].
                output = output * (float(settings["rescale_factor"]) * 255.0)

        if settings["do_resize"]:
            target = self._resize_shape(
                output.shape[-2], output.shape[-1], settings["size"]
            )
            if tuple(output.shape[-2:]) != target:
                output = F.interpolate(
                    output,
                    size=target,
                    mode=settings["interpolation"],
                    align_corners=False,
                    antialias=True,
                )

        if settings["do_center_crop"]:
            crop_h, crop_w = self._height_width(settings["crop_size"])
            output = self._center_crop_with_padding(output, crop_h, crop_w)

        if settings["do_normalize"]:
            mean = self.image_mean.to(device=output.device, dtype=output.dtype)
            std = self.image_std.to(device=output.device, dtype=output.dtype)
            output = (output - mean) / std
        return output

    def save_processor(self, path: str | Path) -> None:
        """Save the image processor needed by offline checkpoint restore."""

        self.image_processor.save_pretrained(str(path))

    def train(self, mode: bool = True) -> "DINOv2Backbone":
        """Keep a frozen backbone in eval mode when the parent model trains."""

        super().train(False if self.frozen else mode)
        return self

    def _set_frozen(self, frozen: bool) -> None:
        for parameter in self.model.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.model.eval()
            super().train(False)

    @staticmethod
    def _validate_images(images: Tensor) -> None:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "images must have shape [B, 3, H, W]; "
                f"got {tuple(images.shape)}."
            )
        if images.dtype != torch.uint8 and not images.is_floating_point():
            raise TypeError("images must be uint8 or floating point.")
        if images.numel() == 0:
            raise ValueError("images must be non-empty.")
        minimum = float(images.detach().min())
        maximum = float(images.detach().max())
        upper = 255.0 if images.dtype == torch.uint8 else 1.0
        if minimum < 0.0 or maximum > upper:
            raise ValueError(
                f"images values must be in [0, {upper:g}]; got [{minimum}, {maximum}]."
            )

    @staticmethod
    def _processor_settings(processor: Any) -> dict[str, Any]:
        interpolation = "bicubic"
        resample = getattr(processor, "resample", None)
        if resample is not None and int(resample) == 2:
            interpolation = "bilinear"
        return {
            "do_resize": bool(getattr(processor, "do_resize", True)),
            "size": dict(getattr(processor, "size", {"shortest_edge": 256})),
            "do_center_crop": bool(getattr(processor, "do_center_crop", True)),
            "crop_size": dict(
                getattr(processor, "crop_size", {"height": 224, "width": 224})
            ),
            "do_rescale": bool(getattr(processor, "do_rescale", True)),
            "rescale_factor": float(getattr(processor, "rescale_factor", 1.0 / 255.0)),
            "do_normalize": bool(getattr(processor, "do_normalize", True)),
            "image_mean": list(getattr(processor, "image_mean", (0.485, 0.456, 0.406))),
            "image_std": list(getattr(processor, "image_std", (0.229, 0.224, 0.225))),
            "interpolation": interpolation,
        }

    @classmethod
    def _resize_shape(
        cls, height: int, width: int, size: Mapping[str, int]
    ) -> tuple[int, int]:
        if "shortest_edge" in size:
            shortest = int(size["shortest_edge"])
            if height <= width:
                return shortest, int(shortest * width / height)
            return int(shortest * height / width), shortest
        return cls._height_width(size)

    @staticmethod
    def _height_width(size: Mapping[str, int]) -> tuple[int, int]:
        if "height" not in size or "width" not in size:
            raise ValueError(f"Processor size must define height/width; got {dict(size)}.")
        return int(size["height"]), int(size["width"])

    @staticmethod
    def _center_crop_with_padding(images: Tensor, height: int, width: int) -> Tensor:
        image_h, image_w = images.shape[-2:]
        pad_h = max(height - image_h, 0)
        pad_w = max(width - image_w, 0)
        if pad_h or pad_w:
            images = F.pad(
                images,
                (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
            )
            image_h, image_w = images.shape[-2:]
        top = (image_h - height) // 2
        left = (image_w - width) // 2
        return images[..., top:top + height, left:left + width]

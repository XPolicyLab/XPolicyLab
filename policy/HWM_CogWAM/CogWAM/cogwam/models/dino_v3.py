"""Frozen online DINOv3 feature extractor for dual-query training.

Training and policy inference extract features directly from source images,
eliminating an offline preprocessing and I/O dependency. ``dino.model_size``
selects the architecture, while ``dino.weights`` may point to local weights.
Both torchvision preprocessing and torch.hub/Transformers loaders are
supported.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import transforms

######### // code // ##########
DINOV3_PRESETS = {
    "vits16": {
        "name": "dinov3_vits16",
        "hf_model_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
        "embed_dim": 384,
        "patch_size": 16,
    },
    "vits16plus": {
        "name": "dinov3_vits16plus",
        "hf_model_id": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
        "embed_dim": 384,
        "patch_size": 16,
    },
    "vitb16": {
        "name": "dinov3_vitb16",
        "hf_model_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",
        "embed_dim": 768,
        "patch_size": 16,
    },
    "vitl16": {
        "name": "dinov3_vitl16",
        "hf_model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "embed_dim": 1024,
        "patch_size": 16,
    },
    "vith16plus": {
        "name": "dinov3_vith16plus",
        "hf_model_id": "facebook/dinov3-vith16plus-pretrain-lvd1689m",
        "embed_dim": 1280,
        "patch_size": 16,
    },
    "vit7b16": {
        "name": "dinov3_vit7b16",
        "hf_model_id": "facebook/dinov3-vit7b16-pretrain-lvd1689m",
        "embed_dim": 4096,
        "patch_size": 16,
    },
}
_SIZE_ALIASES = {
    "s": "vits16",
    "small": "vits16",
    "vits": "vits16",
    "b": "vitb16",
    "base": "vitb16",
    "vitb": "vitb16",
    "l": "vitl16",
    "large": "vitl16",
    "vitl": "vitl16",
    "h": "vith16plus",
    "huge": "vith16plus",
    "g": "vit7b16",
    "giant": "vit7b16",
    "7b": "vit7b16",
}


def _cfg_get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def normalize_dino_image_size(image_size: int | Sequence[int]) -> tuple[int, int]:
    """Return the DINO tensor size as ``(height, width)``.

    A scalar keeps the historical square-input behavior.  A two-element value
    preserves a rectangular image, which is required for the 384x320 FastWAM
    three-camera composite.
    """

    if isinstance(image_size, (int, np.integer)):
        height = width = int(image_size)
    elif isinstance(image_size, str):
        raise TypeError("dino.image_size must be an integer or [height, width], not a string")
    else:
        values = list(image_size)
        if len(values) != 2:
            raise ValueError(f"dino.image_size must contain [height, width], got {values!r}")
        height, width = (int(value) for value in values)
    if height <= 0 or width <= 0:
        raise ValueError(f"dino.image_size dimensions must be positive, got {(height, width)}")
    return height, width


def dino_patch_grid(image_size: int | Sequence[int], patch_size: int) -> tuple[int, int]:
    """Return the exact DINO patch grid as ``(rows, columns)``."""

    height, width = normalize_dino_image_size(image_size)
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError(f"dino.patch_size must be positive, got {patch_size}")
    if height % patch_size or width % patch_size:
        raise ValueError(f"DINO input {(height, width)} must be divisible by patch_size={patch_size} on both axes")
    return height // patch_size, width // patch_size


def dino_num_patches(image_size: int | Sequence[int], patch_size: int) -> int:
    rows, columns = dino_patch_grid(image_size, patch_size)
    return rows * columns


def resolve_dino_spec(dino_cfg) -> dict:
    """Resolve ``framework.dino`` into ``DINOv3Backbone`` arguments.

    ``model_size`` atomically selects name, model ID, embedding dimension, and
    patch size. ``weights`` accepts a local torch.hub checkpoint or HuggingFace
    snapshot; other explicitly configured loader and image fields are retained.
    """
    spec = {
        "name": _cfg_get(dino_cfg, "name", "dinov3_vits16"),
        "hf_model_id": _cfg_get(dino_cfg, "hf_model_id", "facebook/dinov3-vits16-pretrain-lvd1689m"),
        "repo_or_dir": _cfg_get(dino_cfg, "repo_or_dir", "facebookresearch/dinov3"),
        "weights": _cfg_get(dino_cfg, "weights", None),
        "loader": _cfg_get(dino_cfg, "loader", "auto"),
        "image_size": normalize_dino_image_size(_cfg_get(dino_cfg, "image_size", 224)),
        "patch_size": int(_cfg_get(dino_cfg, "patch_size", 16)),
        "embed_dim": int(_cfg_get(dino_cfg, "embed_dim", 384)),
    }
    size = _cfg_get(dino_cfg, "model_size", None)
    if size:
        key = str(size).lower()
        key = _SIZE_ALIASES.get(key, key)
        if key not in DINOV3_PRESETS:
            raise ValueError(
                f"Unknown dino.model_size={size!r}; valid keys: {sorted(DINOV3_PRESETS)} "
                f"(aliases: {sorted(_SIZE_ALIASES)})"
            )
        preset = DINOV3_PRESETS[key]
        spec["name"] = preset["name"]
        spec["hf_model_id"] = preset["hf_model_id"]
        spec["embed_dim"] = preset["embed_dim"]
        spec["patch_size"] = preset["patch_size"]
    # Fail during config resolution rather than halfway through the first
    # distributed training step.
    dino_num_patches(spec["image_size"], spec["patch_size"])
    return spec


######### // code // ##########


def _apply_transform(image: Image.Image, transform):
    return transform(image)


######### // code // ##########
class DINOv3Backbone(nn.Module):
    def __init__(
        self,
        name: str = "dinov3_vits16",
        hf_model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        repo_or_dir: str = "facebookresearch/dinov3",
        weights: str | None = None,
        loader: str = "auto",
        image_size: int | Sequence[int] = 224,
        patch_size: int = 16,
        embed_dim: int = 384,
        trust_repo: bool = True,
    ) -> None:
        super().__init__()
        self.name = name
        self.hf_model_id = hf_model_id
        self.repo_or_dir = repo_or_dir
        # Cluster jobs may inject the frozen teacher path without changing the
        # experiment YAML. An explicit YAML value still takes precedence.
        self.weights = weights or os.environ.get("DINOV3_WEIGHTS") or os.environ.get("DINOV3_VITL16_WEIGHTS")
        if self.weights:
            weights_path = Path(str(self.weights))
            if weights_path.is_absolute() and not weights_path.exists():
                raise FileNotFoundError(
                    f"Configured local DINOv3 weights do not exist: {weights_path}. "
                    "Refusing to fall back to a network download."
                )
        self.loader = loader
        self.image_size = normalize_dino_image_size(image_size)
        self.patch_size = int(patch_size)
        self.num_patches = dino_num_patches(self.image_size, self.patch_size)
        self.num_channels = int(embed_dim)
        self._hf_mode = False

        self.dino_transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.register_buffer("_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

        self.body = self._load_body(trust_repo=trust_repo)
        self.body.eval()
        for param in self.body.parameters():
            param.requires_grad_(False)

    def _load_body(self, trust_repo: bool):
        errors: list[str] = []
        weights = self.weights
        want_hf = self.loader == "hf" or (weights and Path(str(weights)).is_dir())
        if want_hf:
            from transformers import AutoModel

            ref = str(weights) if (weights and Path(str(weights)).is_dir()) else self.hf_model_id
            self._hf_mode = True
            return AutoModel.from_pretrained(
                ref,
                trust_remote_code=True,
                local_files_only=Path(str(ref)).is_dir(),
            )

        if self.loader in {"auto", "torchhub"}:
            try:
                kwargs = {}
                if weights:
                    kwargs["weights"] = str(weights)
                source = "local" if Path(self.repo_or_dir).exists() else "github"
                return torch.hub.load(self.repo_or_dir, self.name, source=source, trust_repo=trust_repo, **kwargs)
            except Exception as exc:  # pragma: no cover - depends on local weights/network
                errors.append(f"torch.hub failed: {exc}")
                if self.loader == "torchhub":
                    raise RuntimeError("; ".join(errors)) from exc

        try:
            from transformers import AutoModel

            ref = str(weights) if (weights and Path(str(weights)).is_dir()) else self.hf_model_id
            self._hf_mode = True
            return AutoModel.from_pretrained(
                ref,
                trust_remote_code=True,
                local_files_only=Path(str(ref)).is_dir(),
            )
        except Exception as exc:  # pragma: no cover - depends on installed transformers/model cache
            errors.append(f"HF AutoModel failed: {exc}")
            raise RuntimeError(
                "Unable to load DINOv3. Provide dino.weights (local .pth or HF snapshot dir), "
                "a local dino.repo_or_dir, or a cached HF model. " + " | ".join(errors)
            ) from exc

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        expected_patches = dino_num_patches(imgs.shape[-2:], self.patch_size)
        tokens = None
        if not self._hf_mode:
            if hasattr(self.body, "forward_features"):
                out = self.body.forward_features(imgs)
                if isinstance(out, dict) and "x_norm_patchtokens" in out:
                    tokens = out["x_norm_patchtokens"]
                elif isinstance(out, dict) and "patch_tokens" in out:
                    tokens = out["patch_tokens"]
            if tokens is None:
                out = self.body(imgs)
        else:
            out = self.body(pixel_values=imgs)

        if tokens is None and isinstance(out, dict):
            if "x_norm_patchtokens" in out:
                tokens = out["x_norm_patchtokens"]
            elif "last_hidden_state" in out:
                tokens = out["last_hidden_state"]
            else:
                raise RuntimeError(f"Unsupported DINOv3 output keys: {list(out.keys())}")
        elif tokens is None and hasattr(out, "feature_maps") and out.feature_maps:
            fmap = out.feature_maps[-1]
            tokens = fmap.flatten(2).transpose(1, 2)
        elif tokens is None and hasattr(out, "last_hidden_state"):
            tokens = out.last_hidden_state
        elif tokens is None and torch.is_tensor(out):
            tokens = out
        elif tokens is None:
            raise RuntimeError(f"Unsupported DINOv3 output type: {type(out)}")

        if tokens.ndim != 3:
            raise RuntimeError(f"Expected DINO tokens [B,T,C], got {tuple(tokens.shape)}")
        if tokens.shape[1] > expected_patches:
            tokens = tokens[:, -expected_patches:, :]
        if tokens.shape[1] != expected_patches:
            raise RuntimeError(
                f"DINO returned {tokens.shape[1]} patch tokens for input {tuple(imgs.shape[-2:])}; "
                f"expected {expected_patches}"
            )
        return tokens

    @torch.no_grad()
    def forward_multi_layer(
        self,
        imgs: torch.Tensor,
        layer_indices: Sequence[int],
    ) -> torch.Tensor:
        """Mean-fuse normalized patch tokens from several transformer layers.

        Layers are aggregated at matching spatial positions, so the patch grid
        is identical to ``forward``; only the per-position feature content
        differs. Normalization drops the affine parameters of the backbone's
        final norm: those were fit for the last layer and inflate shallower
        layers by up to 5x, which would turn the mean into a de facto readout
        of whichever selected layer sits earliest in the network.
        """

        expected_patches = dino_num_patches(imgs.shape[-2:], self.patch_size)
        eps = 1.0e-6
        if self._hf_mode:
            out = self.body(pixel_values=imgs, output_hidden_states=True)
            # hidden_states[0] is the embedding output; hidden_states[i + 1] is
            # the output of layer i, before the final norm.
            states = out.hidden_states
            raw = [states[index + 1] for index in layer_indices]
            eps = self.body.norm.eps
        else:
            raw = list(
                self.body.get_intermediate_layers(
                    imgs,
                    n=list(layer_indices),
                    reshape=False,
                    return_class_token=False,
                    norm=False,
                )
            )
        # Drop the CLS and register tokens so every layer contributes patch
        # tokens only.
        raw = [tokens[:, -expected_patches:, :] for tokens in raw]
        selected = [
            F.layer_norm(tokens, (tokens.shape[-1],), eps=eps) for tokens in raw
        ]
        fused = torch.stack(selected, dim=0).mean(dim=0)
        if fused.shape[1] != expected_patches:
            raise RuntimeError(
                f"DINO returned {fused.shape[1]} patch tokens for input "
                f"{tuple(imgs.shape[-2:])}; expected {expected_patches}"
            )
        return fused

    ######### // code // ##########
    @torch.no_grad()
    def preprocess_batch(
        self,
        images: Sequence,
        *,
        image_size: int | Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Normalize a batch at the requested DINO resolution.

        ``image_size=None`` preserves the checkpoint's canonical input size.
        An explicit size lets a shared frozen DINO backbone encode a rectangular
        current-observation composite and a square future target without
        stretching the future camera into the current composite geometry.
        """

        device = next(self.parameters()).device
        target_size = (
            self.image_size
            if image_size is None
            else normalize_dino_image_size(image_size)
        )
        tensors = []
        for img in images:
            if isinstance(img, Image.Image):
                arr = np.asarray(img.convert("RGB"))
            else:
                arr = np.asarray(img)
            # ``np.asarray(PIL.Image)`` commonly returns a read-only view.
            # torch.as_tensor warns because a tensor backed by that view could
            # be mutated; copy only that case and keep writable NumPy frames zero-copy.
            if not arr.flags.writeable:
                arr = arr.copy()
            t = torch.as_tensor(arr)
            if t.ndim == 2:
                t = t.unsqueeze(-1).repeat(1, 1, 3)
            if t.shape[-1] in (1, 3, 4):
                t = t[..., :3].permute(2, 0, 1)
            t = t.to(device=device, dtype=torch.float32)
            if float(t.max()) > 1.5:  # uint8 / [0,255] -> [0,1]
                t = t / 255.0
            t = t.unsqueeze(0)
            if tuple(t.shape[-2:]) != target_size:
                t = F.interpolate(
                    t,
                    size=target_size,
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
            tensors.append(t)
        x = torch.cat(tensors, dim=0)
        x = (x - self._imagenet_mean.to(device)) / self._imagenet_std.to(device)
        return x

    ######### // code // ##########

    def prepare_dino_input(self, img_list: Sequence[Sequence[Image.Image]]) -> torch.Tensor:
        with ThreadPoolExecutor() as executor:
            image_tensors = torch.stack(
                [
                    torch.stack(
                        list(
                            executor.map(lambda view: _apply_transform(view.convert("RGB"), self.dino_transform), views)
                        )
                    )
                    for views in img_list
                ]
            )

        bsz, num_view, channels, height, width = image_tensors.shape
        image_tensors = image_tensors.view(bsz * num_view, channels, height, width)
        return image_tensors.to(next(self.parameters()).device)


######### // code // ##########

"""Online StarVLA-distilled DINOv3 encoder for raw FastWAM visual latents."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from wam.utils.logging_config import get_logger
from .provenance import RUNTIME

logger = get_logger(__name__)


DEFAULT_STARVLA_SOURCE_PATH = str(RUNTIME)
DEFAULT_DINOV3_SOURCE_PATH = str(RUNTIME / "dinov3")
DEFAULT_DINOV3_SOURCE_COMMIT = "6876159a11b4df116f30f667f8c9888617df0751"
DEFAULT_DISTILLED_CHECKPOINT = ""


def _install_source_path(path_like: Optional[str], *, package: str) -> Path:
    if not path_like:
        raise ValueError(f"`{package}` source path must be provided.")
    root = Path(str(path_like)).expanduser().resolve()
    expected = RUNTIME if package == "starVLA" else RUNTIME / "dinov3"
    if root != expected.resolve():
        raise ValueError(f"Bundled encoder refuses external {package} source: {root}")
    if not (root / package).is_dir():
        raise FileNotFoundError(f"Expected package {package!r} under source path: {root}")
    for name, module in tuple(sys.modules.items()):
        if name == package or name.startswith(package + "."):
            filename = getattr(module, "__file__", None)
            if filename and not Path(filename).resolve().is_relative_to(root / package):
                raise RuntimeError(f"Conflicting {package} module already imported: {filename}")
    root_s = str(root)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    return root


def _verify_git_commit(root: Path, expected: Optional[str], *, label: str) -> Optional[str]:
    if not expected:
        return None
    git_metadata = root / ".git"
    source_commit = root / ".source-commit"
    if git_metadata.exists():
        try:
            observed = subprocess.run(
                ["git", "-C", str(root), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise RuntimeError(f"Could not verify {label} git commit under {root}") from exc
    elif source_commit.is_file():
        observed = source_commit.read_text(encoding="utf-8").strip()
    else:
        raise RuntimeError(
            f"Could not verify {label} source revision under {root}: "
            "neither nested .git metadata nor .source-commit exists."
        )
    if observed != str(expected):
        raise RuntimeError(f"{label} source commit mismatch: expected {expected}, got {observed}")
    return observed


def _load_checkpoint(path_like: str) -> dict[str, Any]:
    path = Path(str(path_like)).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"StarVLA distilled DINOv3 checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict checkpoint payload from {path}, got {type(payload)}")
    return payload


def _nested_get(mapping: dict[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = mapping
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _select_student_state(payload: dict[str, Any], checkpoint_state: str) -> dict[str, torch.Tensor]:
    state_name = str(checkpoint_state or "ema_student")
    training_model = payload.get("training_model")
    if isinstance(training_model, dict) and isinstance(training_model.get(state_name), dict):
        state = training_model[state_name]
    elif isinstance(payload.get(state_name), dict):
        state = payload[state_name]
    elif isinstance(payload.get("state_dict"), dict):
        state = payload["state_dict"]
    else:
        raise KeyError(
            "Could not find StarVLA student state in checkpoint. Expected "
            f"`training_model.{state_name}`, top-level `{state_name}`, or `state_dict`."
        )
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        key_s = str(key)
        key_s = key_s.removeprefix("module.").removeprefix("student.").removeprefix("ema_student.")
        cleaned[key_s] = value
    return cleaned


def _checkpoint_adapter_cfg(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = _nested_get(payload, ("config", "framework", "multi_teacher_distillation", "adapter"))
    return dict(cfg) if isinstance(cfg, dict) else {}


def _projection_mode(raw: Optional[str]) -> str:
    mode = str(raw or "raw").strip().lower().replace("-", "_")
    if mode in {"raw", "none", "identity", "no_projection"}:
        return "raw"
    raise ValueError(f"Unsupported DINOv3 projection_mode={raw!r}; only raw unprojected tokens are supported.")


def _output_source(raw: Optional[str]) -> str:
    source = str(raw or "adapter").strip().lower().replace("-", "_")
    if source in {"adapter", "student", "distilled", "dino_adapter", "dinov3_adapter"}:
        return "adapter"
    if source in {"backbone", "dino", "dinov3", "dino_only", "dinov3_only", "pure_dino"}:
        return "backbone"
    raise ValueError(
        f"Unsupported DINOv3 output_source={raw!r}; expected 'adapter' or 'backbone'."
    )


class OnlineDinov3DistilledEncoder(nn.Module):
    """Wraps the StarVLA EMA DINOv3 student as raw patch tokens.

    The transport tensor may be a horizontal multi-view stitch, but every view is
    split and encoded independently. Views are concatenated only after the
    DINOv3 feature grid is produced. By default this returns the distilled
    DINOv3+adapter tokens; set ``output_source='backbone'`` for a pure DINOv3
    patch-token ablation using the checkpoint's DINO backbone weights.
    """

    input_size_multiple = 16

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_DISTILLED_CHECKPOINT,
        starvla_source_path: str = DEFAULT_STARVLA_SOURCE_PATH,
        dinov3_source_path: str = DEFAULT_DINOV3_SOURCE_PATH,
        dinov3_source_commit: Optional[str] = DEFAULT_DINOV3_SOURCE_COMMIT,
        checkpoint_state: str = "ema_student",
        image_size: int = 256,
        patch_size: int = 16,
        torch_dtype: torch.dtype = torch.bfloat16,
        adapter_width: Optional[int] = None,
        adapter_depth: Optional[int] = None,
        adapter_num_heads: Optional[int] = None,
        adapter_mlp_ratio: Optional[float] = None,
        adapter_num_register_tokens: Optional[int] = None,
        adapter_supervision_taps: Optional[Sequence[int]] = None,
        adapter_layer_scale_init: Optional[float] = None,
        strict_load: bool = True,
        verify_source_commit: bool = False,
        projection_mode: str = "raw",
        output_source: str = "adapter",
        projector_state_path: Optional[str] = None,
        projector_out_dim: int = 1024,
        feature_norm_state_path: Optional[str] = None,
        require_feature_norm: bool = False,
        use_activation_checkpointing: bool = False,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        del projector_state_path, trust_remote_code
        self.checkpoint_path = str(checkpoint_path)
        self.checkpoint_state = str(checkpoint_state or "ema_student")
        self.torch_dtype = torch_dtype
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.projector_out_dim = int(projector_out_dim)
        self.projection_mode = _projection_mode(projection_mode)
        self.output_source = _output_source(output_source)
        if self.image_size % self.patch_size != 0:
            raise ValueError(f"DINOv3 image_size must be divisible by patch_size, got {self.image_size}/{self.patch_size}.")
        self.grid_size = self.image_size // self.patch_size
        self._embed_dim = 1024
        if self.projector_out_dim != self._embed_dim:
            logger.info(
                "Ignoring DINOv3 projector_out_dim=%d in raw mode; using %d unprojected channels.",
                self.projector_out_dim,
                self._embed_dim,
            )
            self.projector_out_dim = self._embed_dim
        if feature_norm_state_path is not None or require_feature_norm:
            raise ValueError(
                "DINOv3 raw mode is configured to use unnormalized features; "
                "leave feature_norm_state_path=null and require_feature_norm=false."
            )

        starvla_root = _install_source_path(starvla_source_path, package="starVLA")
        dinov3_root = _install_source_path(dinov3_source_path, package="dinov3")
        if verify_source_commit:
            observed = _verify_git_commit(dinov3_root, dinov3_source_commit, label="DINOv3")
            if observed:
                logger.info("Verified DINOv3 source commit %s under %s", observed, dinov3_root)

        from dinov3.hub import backbones as dinov3_backbones  # noqa: PLC0415
        from starVLA.model.modules.multi_teacher_distillation import (  # noqa: PLC0415
            SharedSpatiotemporalAdapter,
            StudentVideoEncoder,
        )

        payload = _load_checkpoint(self.checkpoint_path)
        adapter_cfg = _checkpoint_adapter_cfg(payload)
        layer_scale_init = (
            adapter_layer_scale_init
            if adapter_layer_scale_init is not None
            else adapter_cfg.get("layer_scale_init")
        )
        adapter_kwargs = {
            "input_dim": self._embed_dim,
            "width": int(adapter_width if adapter_width is not None else adapter_cfg.get("width", 1024)),
            "depth": int(adapter_depth if adapter_depth is not None else adapter_cfg.get("depth", 8)),
            "num_heads": int(
                adapter_num_heads if adapter_num_heads is not None else adapter_cfg.get("num_heads", 16)
            ),
            "mlp_ratio": float(
                adapter_mlp_ratio if adapter_mlp_ratio is not None else adapter_cfg.get("mlp_ratio", 4.0)
            ),
            "num_register_tokens": int(
                adapter_num_register_tokens
                if adapter_num_register_tokens is not None
                else adapter_cfg.get("num_register_tokens", 16)
            ),
            "patch_grid": self.grid_size,
            "supervision_taps": (
                adapter_supervision_taps
                if adapter_supervision_taps is not None
                else adapter_cfg.get("supervision_taps", None)
            ),
            "use_activation_checkpointing": bool(use_activation_checkpointing),
        }
        adapter_parameters = inspect.signature(SharedSpatiotemporalAdapter).parameters
        if "layer_scale_init" in adapter_parameters:
            adapter_kwargs["layer_scale_init"] = None if layer_scale_init is None else float(layer_scale_init)
        elif layer_scale_init is not None:
            raise RuntimeError(
                "The checkpoint requires adapter layer_scale_init, but the selected StarVLA source "
                "does not expose SharedSpatiotemporalAdapter(layer_scale_init=...)."
            )
        adapter = SharedSpatiotemporalAdapter(**adapter_kwargs)
        dino = dinov3_backbones.dinov3_vitl16(pretrained=False)
        self.student = StudentVideoEncoder(
            dino,
            adapter,
            use_activation_checkpointing=bool(use_activation_checkpointing),
        )
        state = _select_student_state(payload, self.checkpoint_state)
        missing, unexpected = self.student.load_state_dict(state, strict=bool(strict_load))
        if missing:
            allowed_missing: list[str] = []
            if self.output_source == "backbone" and not bool(strict_load):
                allowed_missing = [key for key in missing if str(key).startswith("adapter.")]
            disallowed_missing = [key for key in missing if key not in allowed_missing]
            if disallowed_missing:
                raise RuntimeError(f"StarVLA DINOv3 student load missing keys: {disallowed_missing[:8]}")
            logger.info(
                "Ignoring %d missing adapter keys for DINOv3 backbone-only output.",
                len(allowed_missing),
            )
        if unexpected:
            raise RuntimeError(f"StarVLA DINOv3 student load unexpected keys: {unexpected[:8]}")
        self.student.requires_grad_(False)
        self.student.eval()
        step = payload.get("step")
        logger.info(
            "Loaded StarVLA DINOv3 %s from %s step=%s (source=%s, dtype=%s, output_source=%s)",
            self.checkpoint_state,
            self.checkpoint_path,
            step,
            starvla_root,
            self.torch_dtype,
            self.output_source,
        )

    @property
    def embed_dim(self) -> int:
        return self._embed_dim

    def _student_device(self) -> torch.device:
        return next(self.student.parameters()).device

    def _student_dtype(self) -> torch.dtype:
        return next(self.student.parameters()).dtype

    @torch.no_grad()
    def _preprocess_view(self, strip_bcthw: torch.Tensor) -> torch.Tensor:
        if strip_bcthw.ndim != 5 or strip_bcthw.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W] rgb in [-1,1], got {tuple(strip_bcthw.shape)}")
        b, c, t, h, w = strip_bcthw.shape
        x01 = strip_bcthw.to(dtype=torch.float32).clamp(-1, 1) * 0.5 + 0.5
        x_bt = x01.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x_rs = F.interpolate(
            x_bt,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        ).clamp(0, 1)
        return x_rs.view(b, t, c, self.image_size, self.image_size).contiguous()

    @torch.no_grad()
    def _encode_view_grid(self, strip_bcthw: torch.Tensor, *, frame_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        frames = self._preprocess_view(strip_bcthw)
        b, t = int(frames.shape[0]), int(frames.shape[1])
        device = self._student_device()
        dtype = self._student_dtype()
        frames = frames.to(device=device, dtype=dtype)
        if frame_positions is None:
            frame_positions = torch.arange(t, device=device, dtype=torch.long).unsqueeze(0).expand(b, -1)
        else:
            frame_positions = frame_positions.to(device=device, dtype=torch.long)
            if frame_positions.shape != (b, t):
                raise ValueError(f"Expected frame_positions shape {(b, t)}, got {tuple(frame_positions.shape)}")
        patches, cls = self.student._dino_features(frames, patch_mask=None, require_grad=False)
        tokens = patches if self.output_source == "backbone" else self.student.adapter(patches, cls, frame_positions)
        if tokens.ndim != 4:
            raise ValueError(f"Expected DINOv3 student tokens [B,T,N,C], got {tuple(tokens.shape)}")
        expected_tokens = self.grid_size * self.grid_size
        if int(tokens.shape[2]) != expected_tokens:
            raise ValueError(
                f"DINOv3 token count {tokens.shape[2]} != {expected_tokens} "
                f"(image_size={self.image_size}, patch_size={self.patch_size})."
            )
        if int(tokens.shape[-1]) != self._embed_dim:
            raise ValueError(f"DINOv3 token dim {tokens.shape[-1]} != expected {self._embed_dim}.")
        return tokens.view(b, t, self.grid_size, self.grid_size, self._embed_dim).permute(0, 4, 1, 2, 3).contiguous()

    @torch.no_grad()
    def encode_raw_video_latents(
        self,
        torch_rgb_minus11_bcthw: torch.Tensor,
        *,
        num_views: int = 1,
    ) -> torch.Tensor:
        if torch_rgb_minus11_bcthw.ndim != 5 or torch_rgb_minus11_bcthw.shape[1] != 3:
            raise ValueError(f"Expected [B,3,T,H,W] rgb in [-1,1], got {tuple(torch_rgb_minus11_bcthw.shape)}")
        b, _c, t, _h, win = torch_rgb_minus11_bcthw.shape
        if t < 1:
            raise ValueError("DINOv3 raw encoding requires at least one frame.")
        nv = max(int(num_views), 1)
        if win % nv != 0:
            raise ValueError(f"Input width {win} must be divisible by num_views={nv} for raw DINOv3 encoding.")
        w_stride = win // nv
        frame_positions = torch.arange(t, device=torch_rgb_minus11_bcthw.device, dtype=torch.long).unsqueeze(0).expand(b, -1)

        view_grids: list[torch.Tensor] = []
        for view_idx in range(nv):
            start = view_idx * w_stride
            end = (view_idx + 1) * w_stride
            strip = torch_rgb_minus11_bcthw[:, :, :, :, start:end].contiguous()
            view_grids.append(self._encode_view_grid(strip, frame_positions=frame_positions))

        raw = view_grids[0] if len(view_grids) == 1 else torch.cat(view_grids, dim=-1)
        if raw.shape[0] != b or raw.shape[1] != self._embed_dim or raw.shape[2] != t:
            raise ValueError(
                "Unexpected DINOv3 raw latent shape: "
                f"got {tuple(raw.shape)}, expected B={b}, C={self._embed_dim}, T={t}."
            )
        return raw.to(dtype=torch_rgb_minus11_bcthw.dtype).contiguous()

    @torch.no_grad()
    def collect_raw_patch_tokens(
        self,
        torch_rgb_minus11_bcthw: torch.Tensor,
        *,
        num_views: int = 1,
    ) -> torch.Tensor:
        raw = self.encode_raw_video_latents(torch_rgb_minus11_bcthw, num_views=num_views)
        return raw.permute(0, 2, 3, 4, 1).reshape(-1, raw.shape[1]).detach().cpu().float()

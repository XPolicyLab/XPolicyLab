"""Optional-dependency shims for the direct Inspire-head training path.

The public EgoVLA release imports MANO and PyTorch3D at module import time,
even when the MANO keypoint loss is disabled.  XPolicyLab's direct Inspire-12
adapter does not use either dependency.  This module installs narrowly scoped
fallback modules only when those packages are unavailable; a real MANO install
continues to take precedence.
"""

from __future__ import annotations

import importlib.util
import importlib.machinery
import contextlib
import math
import os
import sys
import types
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np


def _has_module(name: str) -> bool:
    """Return whether an importable module exists without raising on stubs."""

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return name in sys.modules


def _install_legacy_mano_compat() -> None:
    """Keep the legacy MANO pickle dependency usable on Python 3.11/NumPy 1.26.

    MANO v1.2 pickles reference the old ``chumpy`` package.  The released
    chumpy 0.70 code imports removed NumPy aliases and calls
    ``inspect.getargspec``; neither is provided by current Python/NumPy.
    Install narrowly scoped aliases before ``smplx`` unpickles a MANO file.
    """

    # Avoid ``hasattr(np, name)`` here: NumPy emits a deprecation warning when
    # probing several of these names through its module ``__getattr__``.
    for name, value in (
        ("bool", bool),
        ("int", int),
        ("float", float),
        ("complex", complex),
        ("object", object),
        ("unicode", str),
        ("str", str),
    ):
        if name not in np.__dict__:
            setattr(np, name, value)

    import inspect

    if not hasattr(inspect, "getargspec"):
        ArgSpec = namedtuple("ArgSpec", "args varargs keywords defaults")

        def getargspec(function: Any) -> Any:
            spec = inspect.getfullargspec(function)
            return ArgSpec(spec.args, spec.varargs, spec.varkw, spec.defaults)

        inspect.getargspec = getargspec  # type: ignore[attr-defined]


def _find_mano_models() -> Path | None:
    """Locate a pair of licensed MANO model files without relying on cwd."""

    roots: list[Path] = []
    configured = os.environ.get("EGOVLA_MANO_ROOT", "").strip()
    if configured:
        root = Path(configured).expanduser()
        roots.extend((root, root / "models", root / "mano_v1_2" / "models"))

    adapter_dir = Path(__file__).resolve().parents[3]
    roots.extend(
        (
            adapter_dir / "mano_v1_2" / "models",
            adapter_dir / "EgoVLA_Release" / "mano_v1_2" / "models",
            Path.cwd() / "mano_v1_2" / "models",
            Path.cwd() / "EgoVLA_Release" / "mano_v1_2" / "models",
        )
    )
    seen: set[Path] = set()
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        if (root / "MANO_LEFT.pkl").is_file() and (root / "MANO_RIGHT.pkl").is_file():
            return root
    return None


def _patch_smplx_model_path(model_root: Path) -> None:
    """Resolve the upstream release's cwd-relative MANO paths."""

    try:
        import smplx
    except ImportError:
        return
    original = getattr(smplx, "_egovla_original_create", None)
    if original is None:
        original = smplx.create

        def create(model_path: Any, *args: Any, **kwargs: Any) -> Any:
            candidate = Path(os.fspath(model_path)).expanduser()
            if not candidate.is_absolute() and not candidate.is_file():
                local = model_root / candidate.name
                if local.is_file():
                    model_path = str(local)
            return original(model_path, *args, **kwargs)

        smplx._egovla_original_create = original
        smplx.create = create


def _spec(name: str, *, package: bool = False) -> importlib.machinery.ModuleSpec:
    return importlib.machinery.ModuleSpec(name, loader=None, is_package=package)


def _install_loguru_fallback() -> None:
    """Keep the release logger importable on minimal XPolicyLab images."""

    if _has_module("loguru"):
        return
    module = types.ModuleType("loguru")

    class _Logger:
        def __getattr__(self, name: str):
            if name in {"add", "remove", "configure", "bind", "patch", "opt"}:
                return lambda *args, **kwargs: self

            def emit(*args: Any, **kwargs: Any) -> None:
                if args:
                    print(*args)

            return emit

    module.logger = _Logger()
    module.Logger = _Logger
    module.__spec__ = _spec("loguru")
    sys.modules.setdefault("loguru", module)


def _install_s2wrapper_fallback() -> None:
    """Provide the single-scale path when NVIDIA's optional s2wrapper is absent."""

    if _has_module("s2wrapper"):
        return
    module = types.ModuleType("s2wrapper")

    def forward(model: Any, images: Any, *args: Any, **kwargs: Any) -> Any:
        # The released checkpoint uses a single SigLIP scale.  For a genuine
        # S2 checkpoint this conservative fallback still gives a valid feature
        # tensor at the requested base scale; installing s2wrapper is required
        # only when multi-scale features are explicitly enabled.
        return model(images)

    module.forward = forward
    module.__spec__ = _spec("s2wrapper")
    sys.modules.setdefault("s2wrapper", module)


def _scaled_dot_product(q: Any, k: Any, v: Any, *, causal: bool = False, scale: Any = None) -> Any:
    """Small torch-only replacement used solely if an optional flash kernel is called."""

    import torch
    import torch.nn.functional as F

    # Inputs are conventionally [B, S, H, D].  SDPA consumes [B, H, S, D].
    q_t = q.transpose(-3, -2)
    k_t = k.transpose(-3, -2)
    v_t = v.transpose(-3, -2)
    try:
        out = F.scaled_dot_product_attention(
            q_t, k_t, v_t, is_causal=causal,
            scale=float(scale) if scale is not None else None,
        )
    except TypeError:  # older torch has no explicit ``scale`` keyword
        out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=causal)
    return out.transpose(-3, -2)


def _flash_qkvpacked(qkv: Any, *args: Any, **kwargs: Any) -> Any:
    import torch

    if qkv.ndim == 5:  # [B, S, 3, H, D]
        q, k, v = qkv.unbind(dim=2)
        return _scaled_dot_product(
            q, k, v,
            causal=bool(kwargs.get("causal", False)),
            scale=kwargs.get("softmax_scale"),
        )
    if qkv.ndim != 4:  # pragma: no cover - defensive for future kernels
        raise ValueError(f"unsupported flash qkv shape: {tuple(qkv.shape)}")

    # Varlen kernels receive [NNZ, 3, H, D] and cumulative sequence lengths.
    cu = kwargs.get("cu_seqlens")
    if cu is None and len(args) >= 1 and torch.is_tensor(args[0]):
        cu = args[0]
    if cu is None:
        q, k, v = qkv.unbind(dim=1)
        return _scaled_dot_product(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0)
    cu = cu.detach().cpu().tolist()
    outputs = []
    for start, end in zip(cu[:-1], cu[1:]):
        chunk = qkv[int(start):int(end)]
        q, k, v = chunk.unbind(dim=1)
        outputs.append(_scaled_dot_product(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)).squeeze(0))
    return torch.cat(outputs, dim=0) if outputs else qkv.new_empty((0, qkv.shape[-2], qkv.shape[-1]))


def _flash_attention_func(q: Any, k: Any, v: Any, *args: Any, **kwargs: Any) -> Any:
    return _scaled_dot_product(
        q, k, v,
        causal=bool(kwargs.get("causal", False)),
        scale=kwargs.get("softmax_scale"),
    )


def _install_flash_attention_fallback() -> None:
    """Make import-only flash-attention references safe on the A800 env.

    The adapter forces the VILA language and SigLIP stacks to eager attention;
    these functions are therefore a last-resort CPU/SDPA fallback, not a
    performance replacement for the optional CUDA extension.
    """

    if _has_module("flash_attn"):
        return
    package = types.ModuleType("flash_attn")
    package.__path__ = []
    package.__spec__ = _spec("flash_attn", package=True)
    interface = types.ModuleType("flash_attn.flash_attn_interface")
    interface.__spec__ = _spec("flash_attn.flash_attn_interface")
    for name, fn in {
        "flash_attn_unpadded_qkvpacked_func": _flash_qkvpacked,
        "flash_attn_varlen_qkvpacked_func": _flash_qkvpacked,
        "flash_attn_func": _flash_attention_func,
        "flash_attn_varlen_func": _flash_attention_func,
    }.items():
        setattr(interface, name, fn)
        setattr(package, name, fn)

    def _unsupported(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "the optional flash-attn backward kernel is unavailable; "
            "set EGOVLA_USE_FLASH_ATTN=0 to use eager attention"
        )

    for name in (
        "_flash_attn_forward",
        "_flash_attn_backward",
        "_flash_attn_varlen_forward",
        "_flash_attn_varlen_backward",
    ):
        setattr(interface, name, _unsupported)

    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = _spec("flash_attn.bert_padding")

    def unpad_input(value: Any, mask: Any):
        import torch

        batch, seq = mask.shape
        indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
        unpadded = value.reshape(batch * seq, *value.shape[2:])[indices]
        lengths = mask.sum(dim=-1, dtype=torch.int32)
        cu = torch.zeros(batch + 1, dtype=torch.int32, device=value.device)
        cu[1:] = torch.cumsum(lengths, dim=0)
        return unpadded, indices, cu, int(lengths.max().item()) if lengths.numel() else 0

    def pad_input(value: Any, indices: Any, batch_size: int, seqlen: int):
        import torch

        result = value.new_zeros((batch_size * seqlen,) + tuple(value.shape[1:]))
        result[indices] = value
        return result.reshape((batch_size, seqlen) + tuple(value.shape[1:]))

    padding.unpad_input = unpad_input
    padding.pad_input = pad_input
    sys.modules.update(
        {
            "flash_attn": package,
            "flash_attn.flash_attn_interface": interface,
            "flash_attn.bert_padding": padding,
        }
    )


def _install_deepspeed_fallback() -> None:
    """Expose only the communication/no-op pieces imported by VILA.

    VILA imports ``deepspeed.comm`` even when DeepSpeed training is disabled.
    Installing the full package would compile CUDA extensions against a
    different torch build, so the adapter provides a tiny compatibility layer;
    the training script deliberately omits ``--deepspeed`` unless the user
    supplies a real installation.
    """

    if _has_module("deepspeed"):
        return
    import torch

    def _dist_ready() -> bool:
        return bool(torch.distributed.is_available() and torch.distributed.is_initialized())

    comm = types.ModuleType("deepspeed.comm")
    comm.__spec__ = _spec("deepspeed.comm")

    def get_rank(group: Any = None) -> int:
        return int(torch.distributed.get_rank(group=group)) if _dist_ready() else 0

    def get_world_size(group: Any = None) -> int:
        return int(torch.distributed.get_world_size(group=group)) if _dist_ready() else 1

    def new_group(ranks: Any, *args: Any, **kwargs: Any) -> Any:
        return torch.distributed.new_group(ranks=ranks, *args, **kwargs) if _dist_ready() else None

    def init_distributed(*args: Any, **kwargs: Any) -> None:
        if _dist_ready():
            return
        if torch.distributed.is_available() and os.environ.get("RANK") is not None:
            backend = kwargs.get("dist_backend", "nccl")
            torch.distributed.init_process_group(backend=backend)

    comm.is_initialized = _dist_ready
    comm.get_rank = get_rank
    comm.get_world_size = get_world_size
    comm.new_group = new_group
    comm.init_distributed = init_distributed
    comm.barrier = lambda *a, **k: torch.distributed.barrier(*a, **k) if _dist_ready() else None
    comm.destroy_process_group = lambda: torch.distributed.destroy_process_group() if _dist_ready() else None
    comm.get_local_rank = lambda: int(os.environ.get("LOCAL_RANK", 0))
    comm.get_local_world_size = lambda: int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    comm.get_global_rank = lambda group, rank: rank
    comm.has_all_gather_into_tensor = lambda: hasattr(torch.distributed, "all_gather_into_tensor")
    for name in ("broadcast", "all_reduce", "reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter"):
        setattr(
            comm,
            name,
            lambda *args, _name=name, **kwargs: getattr(torch.distributed, _name)(*args, **kwargs)
            if _dist_ready()
            else None,
        )

    @contextlib.contextmanager
    def _noop_context(*args: Any, **kwargs: Any):
        yield

    zero = types.ModuleType("deepspeed.zero")
    zero.__spec__ = _spec("deepspeed.zero")
    zero.GatheredParameters = _noop_context
    zero.Init = _noop_context
    zero.MiCS_Init = _noop_context
    zero.register_external_parameter = lambda *a, **k: None

    ds = types.ModuleType("deepspeed")
    ds.__path__ = []
    ds.__spec__ = _spec("deepspeed", package=True)
    ds.__version__ = "0.0.0-egovla-compat"
    ds.comm = comm
    ds.zero = zero
    ds.initialize = lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("a real DeepSpeed installation is required for --deepspeed")
    )

    runtime = types.ModuleType("deepspeed.runtime")
    runtime.__path__ = []
    runtime.__spec__ = _spec("deepspeed.runtime", package=True)
    runtime_zero = types.ModuleType("deepspeed.runtime.zero")
    runtime_zero.__path__ = []
    runtime_zero.__spec__ = _spec("deepspeed.runtime.zero", package=True)
    partition = types.ModuleType("deepspeed.runtime.zero.partition_parameters")
    partition.__spec__ = _spec("deepspeed.runtime.zero.partition_parameters")

    class ZeroParamStatus:
        AVAILABLE = "available"
        NOT_AVAILABLE = "not_available"

    partition.ZeroParamStatus = ZeroParamStatus
    partition.Init = _noop_context

    config = types.ModuleType("deepspeed.runtime.config")
    config.__spec__ = _spec("deepspeed.runtime.config")
    config.DeepSpeedConfig = lambda value, *a, **k: value
    runtime.config = config
    runtime.zero = runtime_zero

    accelerator = types.ModuleType("deepspeed.accelerator")
    accelerator.__spec__ = _spec("deepspeed.accelerator")

    class _Accelerator:
        communication_backend_name = "nccl"

        def __getattr__(self, name: str):
            if name == "device_name":
                return lambda device_index=None: f"cuda:{device_index or 0}"
            if name == "current_device":
                return lambda: torch.cuda.current_device() if torch.cuda.is_available() else 0
            if name in {"synchronize", "empty_cache", "reset_peak_memory_stats"}:
                return lambda *a, **k: None
            return lambda *a, **k: None

    accelerator.get_accelerator = lambda: _Accelerator()
    utils = types.ModuleType("deepspeed.utils")
    utils.__spec__ = _spec("deepspeed.utils")
    utils.instrument_w_nvtx = lambda fn: fn
    utils.log_dist = lambda *a, **k: None

    sys.modules.update(
        {
            "deepspeed": ds,
            "deepspeed.comm": comm,
            "deepspeed.zero": zero,
            "deepspeed.runtime": runtime,
            "deepspeed.runtime.zero": runtime_zero,
            "deepspeed.runtime.zero.partition_parameters": partition,
            "deepspeed.runtime.config": config,
            "deepspeed.accelerator": accelerator,
            "deepspeed.utils": utils,
        }
    )


def _torch_transforms_module() -> types.ModuleType:
    import torch

    module = types.ModuleType("pytorch3d.transforms")

    def axis_angle_to_matrix(axis_angle: Any) -> torch.Tensor:
        value = torch.as_tensor(axis_angle)
        shape = value.shape
        v = value.reshape(-1, 3)
        theta = torch.linalg.norm(v, dim=-1, keepdim=True)
        # Skew matrix for Rodrigues' formula.
        x, y, z = v.unbind(-1)
        zero = torch.zeros_like(x)
        skew = torch.stack(
            [zero, -z, y, z, zero, -x, -y, x, zero], dim=-1
        ).reshape(-1, 3, 3)
        eye = torch.eye(3, dtype=v.dtype, device=v.device).expand(v.shape[0], -1, -1)
        theta2 = theta * theta
        # Stable Taylor expansions around zero.
        a = torch.where(theta2 < 1e-8, 1.0 - theta2 / 6.0, torch.sin(theta) / torch.clamp(theta, min=1e-8))
        b = torch.where(theta2 < 1e-8, 0.5 - theta2 / 24.0, (1.0 - torch.cos(theta)) / torch.clamp(theta2, min=1e-8))
        result = eye + a[..., None] * skew + b[..., None] * (skew @ skew)
        return result.reshape(shape[:-1] + (3, 3))

    def so3_exp_map(log_rot: Any) -> torch.Tensor:
        return axis_angle_to_matrix(log_rot)

    def axis_angle_to_quaternion(axis_angle: Any) -> torch.Tensor:
        value = torch.as_tensor(axis_angle)
        theta = torch.linalg.norm(value, dim=-1, keepdim=True)
        half = theta * 0.5
        scale = torch.where(theta < 1e-8, 0.5 - theta * theta / 48.0, torch.sin(half) / torch.clamp(theta, min=1e-8))
        return torch.cat([torch.cos(half), value * scale], dim=-1)

    def matrix_to_axis_angle(matrix: Any) -> torch.Tensor:
        value = torch.as_tensor(matrix)
        trace = value[..., 0, 0] + value[..., 1, 1] + value[..., 2, 2]
        cos_theta = torch.clamp((trace - 1.0) * 0.5, -1.0, 1.0)
        theta = torch.acos(cos_theta)
        skew = torch.stack(
            [value[..., 2, 1] - value[..., 1, 2],
             value[..., 0, 2] - value[..., 2, 0],
             value[..., 1, 0] - value[..., 0, 1]], dim=-1
        )
        sin_theta = torch.linalg.norm(skew, dim=-1, keepdim=True) * 0.5
        scale = torch.where(sin_theta < 1e-6, 0.5 * torch.ones_like(theta)[..., None], theta[..., None] / torch.clamp(2.0 * sin_theta, min=1e-6))
        return skew * scale

    module.axis_angle_to_matrix = axis_angle_to_matrix
    module.so3_exp_map = so3_exp_map
    module.axis_angle_to_quaternion = axis_angle_to_quaternion
    module.matrix_to_axis_angle = matrix_to_axis_angle
    return module


def _mano_modules() -> dict[str, types.ModuleType]:
    import torch

    model = types.ModuleType("human_plan.utils.mano.model")

    class _UnavailableMANO:
        """Marker object; MANO-only code raises a useful error if reached."""

        dtype = torch.float32
        hand_mean = torch.zeros(15)
        np_hand_components = np.eye(15, dtype=np.float32)

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError(
                "MANO assets are not installed. Set hand_kp_loss_coeff=0 for "
                "the direct Inspire-12 EgoVLA adapter."
            )

    model.mano_left = _UnavailableMANO()
    model.mano_right = _UnavailableMANO()

    forward = types.ModuleType("human_plan.utils.mano.forward")

    def mano_forward(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(
            "MANO assets are not installed; direct Inspire-12 mode does not use mano_forward"
        )

    forward.mano_forward = mano_forward
    forward.mano_forward_retarget = mano_forward
    forward.mano_forward_retarget_isaaclab = mano_forward

    constants = types.ModuleType("human_plan.utils.mano.constants")
    eye = torch.eye(3)
    for name in (
        "LEFT_AXIS_TRANSFORMATION",
        "RIGHT_AXIS_TRANSFORMATION",
        "LEFT_AXIS_TRANSFORMATION_RETARGET",
        "RIGHT_AXIS_TRANSFORMATION_RETARGET",
        "LEFT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB",
        "RIGHT_AXIS_TRANSFORMATION_RETARGET_ISAACLAB",
        "RIGHT_RETARGET_MANO_TRANSFORMATION",
        "LEFT_RETARGET_MANO_TRANSFORMATION",
    ):
        setattr(constants, name, eye.clone())
    constants.LEFT_PELVIS = torch.zeros(1, 3)
    constants.RIGHT_PELVIS = torch.zeros(1, 3)
    constants.EMPTY_ROT = torch.zeros(3)
    constants.MANO_FINGERTIP_VERT_INDICES = {"thumb": 0, "index": 0, "middle": 0, "ring": 0, "pinky": 0}
    constants.mano_joint_mapping = np.arange(21)
    constants.holoassist_to_mano_joint_mapping = np.arange(21)
    return {
        "human_plan.utils.mano.model": model,
        "human_plan.utils.mano.forward": forward,
        "human_plan.utils.mano.constants": constants,
    }


def _install_smplx_fallback() -> None:
    """Stub the import-only SMPL-X API when MANO model files are absent.

    The released dataset module imports ``smplx`` unconditionally and creates
    two MANO layers at import time.  The XPolicyLab direct Inspire head never
    calls those layers, so importing the full package (and downloading the
    separately licensed MANO assets) would only make startup brittle.  Keep a
    deliberately loud runtime error for any code that accidentally reaches a
    MANO operation.
    """

    if "smplx" in sys.modules:
        return
    import torch

    class _UnavailableLayer:
        dtype = torch.float32
        hand_mean = torch.zeros(15)
        np_hand_components = np.eye(15, dtype=np.float32)
        v_template = torch.zeros((1, 3))
        shapedirs = torch.zeros((1, 3, 10))
        J_regressor = torch.zeros((1, 1))

        def to(self, *args: Any, **kwargs: Any):
            return self

        def __call__(self, *args: Any, **kwargs: Any):
            raise RuntimeError(
                "MANO/SMPL-X assets are not installed; the direct Inspire-12 "
                "EgoVLA path does not support MANO forward calls"
            )

    package = types.ModuleType("smplx")
    package.__path__ = []
    package.__spec__ = _spec("smplx", package=True)
    package.MANOLayer = _UnavailableLayer
    package.create = lambda *args, **kwargs: _UnavailableLayer()

    lbs = types.ModuleType("smplx.lbs")
    lbs.__spec__ = _spec("smplx.lbs")

    def _unsupported(*args: Any, **kwargs: Any):
        raise RuntimeError(
            "MANO/SMPL-X assets are not installed; direct Inspire-12 mode "
            "does not call smplx.lbs"
        )

    lbs.blend_shapes = _unsupported
    lbs.vertices2joints = _unsupported
    utils = types.ModuleType("smplx.utils")
    utils.__spec__ = _spec("smplx.utils")
    utils.MANOOutput = object
    utils.to_tensor = lambda value, *args, **kwargs: torch.as_tensor(value)
    vertex_ids = types.ModuleType("smplx.vertex_ids")
    vertex_ids.__spec__ = _spec("smplx.vertex_ids")
    vertex_ids.vertex_ids = {"mano": {}}
    sys.modules.update(
        {
            "smplx": package,
            "smplx.lbs": lbs,
            "smplx.utils": utils,
            "smplx.vertex_ids": vertex_ids,
        }
    )


def install_optional_compat() -> None:
    """Install fallbacks only for missing optional upstream dependencies."""

    # These imports are unconditional in the released VILA module graph, but
    # none is required for the direct Inspire-12 path.  Install the shims
    # before importing any ``llava`` or ``human_plan`` module.
    _install_loguru_fallback()
    _install_s2wrapper_fallback()
    _install_flash_attention_fallback()
    _install_deepspeed_fallback()

    try:
        pytorch3d_available = importlib.util.find_spec("pytorch3d") is not None
    except (ImportError, ValueError):
        pytorch3d_available = "pytorch3d.transforms" in sys.modules
    if not pytorch3d_available:
        transforms = _torch_transforms_module()
        package = types.ModuleType("pytorch3d")
        package.transforms = transforms
        sys.modules.setdefault("pytorch3d", package)
        sys.modules.setdefault("pytorch3d.transforms", transforms)

    # A MANO package without the licensed model pkl files is just as unusable
    # for this adapter as no package at all.  Detect the release's expected
    # files before deciding whether to import its eager modules.  The upstream
    # code uses cwd-relative paths, so also patch ``smplx.create`` to resolve
    # an adapter-local (or EGOVLA_MANO_ROOT) installation.
    mano_root = _find_mano_models()
    _install_legacy_mano_compat()
    try:
        smplx_available = importlib.util.find_spec("smplx") is not None
    except (ImportError, ValueError):
        smplx_available = "smplx" in sys.modules
    if not smplx_available or mano_root is None:
        # Install this before importing ``human_plan.dataset_preprocessing``;
        # that module performs ``smplx.create`` at module import time.
        _install_smplx_fallback()
        for name, module in _mano_modules().items():
            sys.modules.setdefault(name, module)
    else:
        _patch_smplx_model_path(mano_root)


def patch_attention_implementation() -> None:
    """Switch release hard-coded FlashAttention calls to eager attention.

    This is a no-op when ``EGOVLA_USE_FLASH_ATTN`` is explicitly enabled.  It
    is kept separate from dependency shims so callers can choose the same
    behavior for inference and training.
    """

    import os

    if os.environ.get("EGOVLA_USE_FLASH_ATTN", "0").strip().lower() in {"1", "true", "yes"}:
        return
    try:
        from llava.model.language_model.llava_llama import LlavaLlamaModel

        original_init = getattr(LlavaLlamaModel, "_egovla_original_init", None)
        if original_init is None:
            original_init = LlavaLlamaModel.__init__

            def safe_init(self, *args: Any, **kwargs: Any):
                kwargs["attn_implementation"] = "eager"
                return original_init(self, *args, **kwargs)

            LlavaLlamaModel._egovla_original_init = original_init
            LlavaLlamaModel.__init__ = safe_init
    except (ImportError, AttributeError):
        pass
    # The release moved this class between ``siglip.py`` and
    # ``modeling_siglip.py`` across commits; support both layouts.
    for module_name in (
        "llava.model.multimodal_encoder.siglip.siglip",
        "llava.model.multimodal_encoder.siglip.modeling_siglip",
    ):
        try:
            module = __import__(module_name, fromlist=["SiglipVisionModel"])
            SiglipVisionModel = module.SiglipVisionModel
            original = getattr(SiglipVisionModel, "_egovla_original_from_pretrained", None)
            if original is None:
                original = SiglipVisionModel.from_pretrained.__func__

                @classmethod
                def safe_from_pretrained(cls, *args: Any, _original=original, **kwargs: Any):
                    kwargs["attn_implementation"] = "eager"
                    return _original(cls, *args, **kwargs)

                SiglipVisionModel._egovla_original_from_pretrained = original
                SiglipVisionModel.from_pretrained = safe_from_pretrained
            break
        except (ImportError, AttributeError, KeyError):
            # Python 3.10 raises KeyError (not ImportError) when a regular
            # package is nested under a namespace directory that has no
            # __init__.py. Skip that layout instead of killing the server.
            continue

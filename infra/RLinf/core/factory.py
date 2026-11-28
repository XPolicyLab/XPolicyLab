# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""XPolicyLab unified embodied adapter for RLinf."""

from __future__ import annotations

import importlib
import os
import sys
from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

from .capabilities import RL_MODE_EVAL, infer_required_rl_mode
from .extensions import policy_extension

if TYPE_CHECKING:
    from .action_model import XPolicyLabForRLActionPrediction


# Generic ``actor.model`` fields forwarded to the XPolicyLab policy config.
_RLINF_FIELDS_TO_POLICY = (
    "model_path",
    "action_dim",
    "num_action_chunks",
    "add_value_head",
    "precision",
)


def _copy_rlinf_fields_into_policy_cfg(
    cfg: DictConfig, model_cfg: dict[str, Any]
) -> None:
    """Copy actor.model fields the XPolicyLab policy needs to build an RL backbone."""
    if getattr(cfg, "model_path", None):
        model_cfg.setdefault("model_path", str(cfg.model_path))
        model_cfg.setdefault("ckpt_path", str(cfg.model_path))
    for key in _RLINF_FIELDS_TO_POLICY:
        if key == "model_path":
            continue
        if key in model_cfg or not hasattr(cfg, key):
            continue
        value = getattr(cfg, key)
        if value is None:
            continue
        if isinstance(value, DictConfig):
            value = OmegaConf.to_container(value, resolve=True)
        model_cfg[key] = value


def _parent_of_xpolicylab_package(path: str) -> str | None:
    """Return the sys.path entry that makes ``import XPolicyLab`` work."""
    if os.path.isdir(os.path.join(path, "XPolicyLab", "policy")):
        return path
    if os.path.basename(path) == "XPolicyLab" and os.path.isdir(
        os.path.join(path, "policy")
    ):
        return os.path.dirname(path)
    return None


def _ensure_xpolicylab_importable() -> None:
    """Import ``XPolicyLab``.

    The adapter lives inside the XPolicyLab package. ``XPOLICYLAB_PATH`` still
    selects a checkout when set; otherwise the package already on ``sys.path``
    is used.
    """
    candidates = []
    env_path = os.environ.get("XPOLICYLAB_PATH")
    if env_path:
        candidates.append(os.path.abspath(env_path))
    package_root = next(
        (
            root
            for root in map(_parent_of_xpolicylab_package, candidates)
            if root is not None
        ),
        None,
    )

    if package_root is not None:
        expected = os.path.join(package_root, "XPolicyLab")
        loaded = sys.modules.get("XPolicyLab")
        if loaded is not None:
            loaded_dir = os.path.dirname(os.path.abspath(loaded.__file__))
            if os.path.realpath(loaded_dir) != os.path.realpath(expected):
                raise ImportError(
                    f"XPolicyLab is already imported from {loaded_dir}, "
                    f"expected {expected}. Set XPOLICYLAB_PATH or remove the "
                    "other checkout from PYTHONPATH."
                )
            return
        if package_root in sys.path:
            sys.path.remove(package_root)
        sys.path.insert(0, package_root)

    try:
        importlib.import_module("XPolicyLab")
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import XPolicyLab. Put the RoboTwin checkout on PYTHONPATH, "
            "or set XPOLICYLAB_PATH to the XPolicyLab checkout (or its parent)."
        ) from e


def get_model(
    cfg: DictConfig,
    torch_dtype=None,
) -> XPolicyLabForRLActionPrediction:
    """Load ``XPolicyLab.policy.<name>.model.Model`` and wrap it as a BasePolicy."""
    from .action_model import XPolicyLabForRLActionPrediction

    del torch_dtype
    xcfg = getattr(cfg, "xpolicylab", None)
    if xcfg is None:
        raise ValueError(
            "model_type 'xpolicylab' requires actor.model.xpolicylab.policy_name."
        )
    policy_name = getattr(xcfg, "policy_name", None)
    if not policy_name:
        raise ValueError("actor.model.xpolicylab.policy_name is required.")

    _ensure_xpolicylab_importable()

    module = importlib.import_module(f"XPolicyLab.policy.{policy_name}.model")
    model_cls = getattr(module, "Model")
    model_cfg: dict[str, Any] = OmegaConf.to_container(xcfg, resolve=True)  # type: ignore[assignment]
    if not isinstance(model_cfg, dict):
        raise TypeError("actor.model.xpolicylab must be a mapping.")
    model_cfg.setdefault("policy_name", str(policy_name))
    _copy_rlinf_fields_into_policy_cfg(cfg, model_cfg)
    extension = policy_extension(cfg)
    if hasattr(extension, "configure_model"):
        extension.configure_model(cfg, model_cfg)
    required_mode = infer_required_rl_mode(cfg)
    if "rl" not in model_cfg and required_mode != RL_MODE_EVAL:
        model_cfg["rl"] = {"modes": [RL_MODE_EVAL, required_mode]}

    xpolicy = model_cls(model_cfg)
    return XPolicyLabForRLActionPrediction(
        xpolicy,
        action_dim=int(cfg.action_dim),
        num_action_chunks=int(cfg.num_action_chunks),
        required_mode=required_mode,
    )

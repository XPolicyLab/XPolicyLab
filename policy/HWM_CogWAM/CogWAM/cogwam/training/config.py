"""
Shared configuration helpers:
- NamespaceWithGet: lightweight namespace behaving like a dict
- OmegaConf conversion helpers
- Framework default/YAML merging
- Checkpoint config/statistics loading
- The version "0.21" schema compatibility layer
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf

from cogwam.training.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class NamespaceWithGet(SimpleNamespace):
    def get(self, key, default=None):
        """
        Return attribute value if present, else default (dict-like API).

        Args:
            key: Attribute name.
            default: Fallback if attribute missing.

        Returns:
            Any: Stored value or default.
        """
        return getattr(self, key, default)

    def items(self):
        """
        Iterate (key, value) pairs like dict.items().

        Returns:
            Generator[Tuple[str, Any], None, None]
        """
        return ((key, getattr(self, key)) for key in self.__dict__)

    def __iter__(self):
        """
        Return iterator over attribute keys (enables dict unpacking **obj).

        Returns:
            Iterator[str]
        """
        return iter(self.__dict__)

    def to_dict(self):
        """
        Recursively convert nested NamespaceWithGet objects into plain dicts.

        Returns:
            dict: Fully materialized dictionary structure.
        """
        return {key: value.to_dict() if isinstance(value, NamespaceWithGet) else value for key, value in self.items()}


def dict_to_namespace(d):
    """
    Create an OmegaConf config from a plain dictionary.

    Args:
        d: Input dictionary.

    Returns:
        OmegaConf: DictConfig instance.
    """
    return OmegaConf.create(d)


def _to_omegaconf(x: Any):
    """
    Convert diverse input types into an OmegaConf object.

    Accepted types:
        - None -> empty DictConfig
        - str path -> load YAML/JSON via OmegaConf.load
        - dict -> DictConfig
        - DictConfig / ListConfig -> returned unchanged
        - NamespaceWithGet / SimpleNamespace -> converted via vars()/to_dict()

    Args:
        x: Input candidate.

    Returns:
        OmegaConf: Normalized configuration node.
    """
    if x is None:
        return OmegaConf.create({})
    if isinstance(x, OmegaConf.__class__):  # fallback, typically not hit
        return x
    try:
        # OmegaConf node detection
        from omegaconf import DictConfig, ListConfig

        if isinstance(x, (DictConfig, ListConfig)):
            return x
    except Exception:
        pass

    if isinstance(x, str):
        # treat as path
        return OmegaConf.load(x)
    if isinstance(x, dict):
        return OmegaConf.create(x)
    if isinstance(x, NamespaceWithGet) or isinstance(x, SimpleNamespace):
        # convert to plain dict
        try:
            d = x.to_dict() if hasattr(x, "to_dict") else vars(x)
        except Exception:
            d = vars(x)
        return OmegaConf.create(d)
    # fallback: try to create
    return OmegaConf.create(x)


def merge_framework_config(default_config_cls, cfg):
    """
    Merge a framework's default config (dataclass) with the incoming YAML config.

    Rules:
        - default_config_cls provides documented defaults for `cfg.framework`
        - YAML values (cfg.framework) override matching defaults
        - Extra YAML keys not in defaults are preserved (Config-as-API flexibility)
        - Missing YAML keys fall back to defaults (less YAML boilerplate)

    The merge only touches the `cfg.framework` sub-tree; datasets / trainer / etc.
    are left untouched.

    Args:
        default_config_cls: A dataclass **class** (not instance) whose fields() define
                            the default framework config with type hints and comments.
        cfg: The full OmegaConf config (must contain cfg.framework).

    Returns:
        cfg: The same config object with cfg.framework replaced by the merged result.
    """
    import dataclasses

    from omegaconf import DictConfig, OmegaConf

    # 1. Instantiate defaults and convert to OmegaConf
    defaults_instance = default_config_cls()
    defaults_dict = dataclasses.asdict(defaults_instance)
    defaults_omega = OmegaConf.create(defaults_dict)

    # 2. Extract the YAML framework section
    if hasattr(cfg, "framework"):
        # Unwrap AccessTrackedConfig if needed
        yaml_fw = cfg.framework
        if hasattr(yaml_fw, "_cfg"):
            yaml_fw = yaml_fw._cfg
        if not isinstance(yaml_fw, DictConfig):
            yaml_fw = OmegaConf.create(yaml_fw if isinstance(yaml_fw, dict) else {})
    else:
        yaml_fw = OmegaConf.create({})

    # 3. Merge: defaults first, YAML overrides (YAML wins on conflicts)
    merged_fw = OmegaConf.merge(defaults_omega, yaml_fw)

    # 4. Write back into the original cfg
    #    Handle both OmegaConf and AccessTrackedConfig transparently
    if hasattr(cfg, "_cfg") and isinstance(cfg._cfg, DictConfig):
        # AccessTrackedConfig caches child wrappers in _children dict.
        # After replacing the underlying DictConfig node, the old child wrapper
        # still points to the pre-merge node (stale data).  We must invalidate
        # the cache so the next attribute access creates a fresh wrapper around
        # the merged node.
        #
        # However, the old child's _local_accessed set records which keys were
        # already read (e.g. "name" from build_model).  Deleting the child
        # would lose that tracking info, causing save_accessed_config to omit
        # those keys from config.yaml.  So we preserve and restore it.
        cfg._cfg.framework = merged_fw
        if hasattr(cfg, "_children") and "framework" in cfg._children:
            old_accessed = cfg._children["framework"]._local_accessed.copy()
            del cfg._children["framework"]  # invalidate stale cache
            new_child = cfg.framework  # re-create child around merged_fw
            new_child._local_accessed.update(old_accessed)  # restore tracking
    elif isinstance(cfg, DictConfig):
        cfg.framework = merged_fw
    else:
        # Fallback — try direct attribute setting
        try:
            cfg.framework = merged_fw
        except Exception:
            overwatch.warning("Could not write merged framework config back to cfg.")

    return cfg


def read_mode_config(pretrained_checkpoint):
    """
    Load a released checkpoint's config.yaml plus its dataset normalization statistics.

    Expected directory layout:
        <run_dir>/checkpoints/<name>.pt|.safetensors
        <run_dir>/config.yaml
        <run_dir>/dataset_statistics.json

    Args:
        pretrained_checkpoint: Path to a .pt checkpoint file.

    Returns:
        tuple:
            global_cfg (dict)
            norm_stats (dict)
    """
    if os.path.isfile(pretrained_checkpoint):
        overwatch.info(f"Loading from local checkpoint path `{(checkpoint_pt := Path(pretrained_checkpoint))}`")

        # [Validate] Checkpoint Path should look like
        # `.../<RUN_ID>/checkpoints/<CHECKPOINT_PATH>.pt|.safetensors`
        assert checkpoint_pt.suffix in {".pt", ".safetensors"}
        run_dir = checkpoint_pt.parents[1]

        # Get paths for `config.json`, `dataset_statistics.json` and pretrained checkpoint
        config_yaml, dataset_statistics_json = run_dir / "config.yaml", run_dir / "dataset_statistics.json"
        assert config_yaml.exists(), f"Missing `config.yaml` for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"

        # Otherwise =>> try looking for a match on `model_id_or_path` on the HF Hub (`model_id_or_path`)
        # Load VLA Config (and corresponding base VLM `ModelConfig`) from `config.json`
        try:
            ocfg = OmegaConf.load(str(config_yaml))
            # Normalise legacy / pre-v0.21 configs to current schema (idempotent).
            apply_config_compat(ocfg)
            global_cfg = OmegaConf.to_container(ocfg, resolve=True)
        except Exception as e:
            overwatch.error(f"❌ Failed to load YAML config `{config_yaml}`: {e}")
            raise

        # Load Dataset Statistics for Action Denormalization
        with open(dataset_statistics_json, "r") as f:
            norm_stats = json.load(f)
    else:
        overwatch.error(f"❌ Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
        raise FileNotFoundError(f"Pretrained checkpoint `{pretrained_checkpoint}` does not exist.")
    return global_cfg, norm_stats


# =============================================================================
# Config compatibility / "tightening" layer (introduced in version_id "0.21").
#
# Goal: keep user-facing YAMLs short and unambiguous while preserving full
# the design rationale.
#
# This function is *idempotent* — calling it multiple times yields the same
# result. It does NOT touch framework class signatures; instead it normalises
# the OmegaConf tree so that downstream framework __init__ code (which still
# reads e.g. `future_action_window_size`) keeps working unchanged.
# =============================================================================

CONFIG_VERSION = "0.21"


def apply_config_compat(cfg, *, strict: bool = False):
    """
    Normalise an arbitrary (old or new) cogwam training config into the
    current `version_id == "0.21"` schema.

    Performed transformations (each applied only when needed):

      1.  `framework.action_model.action_horizon` ↔ `future_action_window_size`
          - `action_horizon` is canonical (preferred user-facing name).
          - `future_action_window_size = action_horizon - 1` is auto-filled so
            framework code that still reads the old key keeps working.
          - If both are present and inconsistent, a warning is emitted and
            `action_horizon` wins.

      2.  `framework.action_model.diffusion_model_cfg.output_dim`
          - Auto-filled from `framework.action_model.hidden_size` when missing.

      3.  `framework.action_model.diffusion_model_cfg.cross_attention_dim`
          - Auto-filled from `framework.qwenvl.vl_hidden_dim` when missing.
            Frameworks that further override this at runtime (e.g. QwenGR00T)
            are unaffected.

      4.  `framework.action_model.action_hidden_dim`
          - Auto-filled from `hidden_size` when missing. OFT-family frameworks
            still overwrite this from VLM hidden_size at runtime.

      5.  `framework.action_model.past_action_window_size`
          - Auto-filled to `0` when missing. All released cogwam frameworks
            run with past=0; the field is therefore dropped from user YAMLs
            and only re-materialised here for legacy code that still reads it.

      6.  `cfg.version_id` is stamped to `"0.21"`.

    Args:
        cfg: An OmegaConf DictConfig (or anything _to_omegaconf can wrap).
        strict: If True, raise on inconsistent values instead of warning.

    Returns:
        The same `cfg` object (mutated in place) for chaining convenience.
    """
    from omegaconf import OmegaConf

    if cfg is None:
        return cfg

    src_version = OmegaConf.select(cfg, "version_id", default=None)

    # ---- 1. action_horizon ↔ future_action_window_size ----
    am_path = "framework.action_model"
    am = OmegaConf.select(cfg, am_path, default=None)
    if am is not None:
        ah = OmegaConf.select(am, "action_horizon", default=None)
        fw = OmegaConf.select(am, "future_action_window_size", default=None)

        if ah is None and fw is not None:
            ah = int(fw) + 1
            OmegaConf.update(cfg, f"{am_path}.action_horizon", ah, force_add=True)
        elif ah is not None and fw is None:
            fw = int(ah) - 1
            OmegaConf.update(cfg, f"{am_path}.future_action_window_size", fw, force_add=True)
        elif ah is not None and fw is not None and int(ah) != int(fw) + 1:
            msg = (
                f"[apply_config_compat] inconsistent action_horizon={ah} vs "
                f"future_action_window_size={fw}; expected action_horizon == future + 1. "
                "Using action_horizon as canonical."
            )
            if strict:
                raise ValueError(msg)
            overwatch.warning(msg)
            OmegaConf.update(cfg, f"{am_path}.future_action_window_size", int(ah) - 1, force_add=True)

        # ---- 2 & 3. diffusion_model_cfg auto-fill ----
        dm_path = f"{am_path}.diffusion_model_cfg"
        dm = OmegaConf.select(cfg, dm_path, default=None)
        if dm is not None:
            hidden_size = OmegaConf.select(am, "hidden_size", default=None)
            if OmegaConf.select(dm, "output_dim", default=None) is None and hidden_size is not None:
                OmegaConf.update(cfg, f"{dm_path}.output_dim", int(hidden_size), force_add=True)

            if OmegaConf.select(dm, "cross_attention_dim", default=None) is None:
                vl_hidden = OmegaConf.select(cfg, "framework.qwenvl.vl_hidden_dim", default=None)
                if vl_hidden is not None:
                    OmegaConf.update(cfg, f"{dm_path}.cross_attention_dim", int(vl_hidden), force_add=True)
                # else: leave None — framework __init__ may auto-bind it

        # ---- 4. action_hidden_dim fallback ----
        if OmegaConf.select(am, "action_hidden_dim", default=None) is None:
            hidden_size = OmegaConf.select(am, "hidden_size", default=None)
            if hidden_size is not None:
                OmegaConf.update(cfg, f"{am_path}.action_hidden_dim", int(hidden_size), force_add=True)

        # ---- 5. past_action_window_size default ----
        if OmegaConf.select(am, "past_action_window_size", default=None) is None:
            OmegaConf.update(cfg, f"{am_path}.past_action_window_size", 0, force_add=True)

    # ---- 6. stamp version ----
    if src_version != CONFIG_VERSION:
        try:
            OmegaConf.update(cfg, "version_id", CONFIG_VERSION, force_add=True)
        except Exception:
            try:
                cfg.version_id = CONFIG_VERSION
            except Exception:
                pass
        overwatch.info(f"[apply_config_compat] normalised config from version_id={src_version!r} to {CONFIG_VERSION!r}")

    return cfg


__all__ = [
    "CONFIG_VERSION",
    "NamespaceWithGet",
    "apply_config_compat",
    "dict_to_namespace",
    "merge_framework_config",
    "read_mode_config",
]

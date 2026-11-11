"""Configuration helpers used by the inference framework."""
from omegaconf import OmegaConf
from starVLA.inference_utils import initialize_overwatch

overwatch = initialize_overwatch(__name__)

def dict_to_namespace(d):
    """
    Create an OmegaConf config from a plain dictionary.

    Args:
        d: Input dictionary.

    Returns:
        OmegaConf: DictConfig instance.
    """
    return OmegaConf.create(d)


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
        # AccessTrackedConfig path — write to underlying cfg AND invalidate
        # the cached child so subsequent attribute access sees the merged result.
        cfg._cfg.framework = merged_fw
        if hasattr(cfg, "_children") and "framework" in cfg._children:
            del cfg._children["framework"]
    elif isinstance(cfg, DictConfig):
        cfg.framework = merged_fw
    else:
        # Fallback — try direct attribute setting
        try:
            cfg.framework = merged_fw
        except Exception:
            overwatch.warning("Could not write merged framework config back to cfg.")

    return cfg

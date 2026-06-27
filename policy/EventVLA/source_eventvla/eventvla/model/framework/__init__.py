from eventvla.model.tools import FRAMEWORK_REGISTRY
from eventvla.model.legacy_compat import normalize_legacy_framework_config
        
def build_framework(cfg):
    """Build the public EventVLA model."""

    if not hasattr(cfg.framework, "name"):
        raise ValueError("Missing required config field `framework.name`.")

    compat_info = normalize_legacy_framework_config(cfg)
        
    if cfg.framework.name != "EventVLA":
        raise NotImplementedError("EventVLA open-source build only supports framework.name=EventVLA.")
    if compat_info.get("enabled", False):
        cfg.framework.compat_loaded_as = "EventVLA"

    from eventvla.model.framework.EventVLA import EventVLA

    return EventVLA(cfg)

__all__ = ["build_framework", "FRAMEWORK_REGISTRY"]

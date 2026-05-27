# Copyright (C) 2026 Xiaomi Corporation.
"""Model registry and public model symbols.

``MIMODEL`` is an mmengine-style registry used to look up and instantiate
model classes (e.g. runners, VLA heads) by name from config files.
"""

from mmengine import Registry

# Global model registry — modules decorated with @MIMODEL.register_module()
# are automatically registered and can be built via MIMODEL.build(cfg).
MIMODEL = Registry("MIMODEL")

from mibot.models.runner.base_runner import BaseRunner
from mibot.models.VLA.XR0 import XR0

__all__ = ["BaseRunner", "XR0"]

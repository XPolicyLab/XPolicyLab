# Copyright (C) 2026 Xiaomi Corporation.
from mmengine import Registry

MIMODEL = Registry("MIMODEL")

from mibot.models.runner.base_runner import BaseRunner
from mibot.models.VLA.xr1 import xr1

__all__ = ["BaseRunner", "xr1"]

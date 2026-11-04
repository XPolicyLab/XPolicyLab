from typing import Any, Optional, Sequence, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flexwam.utils.logging_config import get_logger
logger = get_logger(__name__)

import time

def _synchronize_for_timing(device: Any) -> None:
    if device is None:
        return
    device_obj = torch.device(device) if not isinstance(device, torch.device) else device
    if device_obj.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device_obj)

class _InferenceTimer:
    def __init__(self, device: Any) -> None:
        self.device = device
        self._started = time.perf_counter()

    def sync(self) -> None:
        _synchronize_for_timing(self.device)

    def elapsed_ms(self, start: float) -> float:
        self.sync()
        return (time.perf_counter() - start) * 1000.0

    def mark(self) -> float:
        self.sync()
        return time.perf_counter()

    def total_ms(self) -> float:
        return self.elapsed_ms(self._started)

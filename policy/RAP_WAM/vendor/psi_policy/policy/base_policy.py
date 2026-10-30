from __future__ import annotations

from typing import Dict

import torch

from psi_policy.model.common.module_attr_mixin import ModuleAttrMixin
from psi_policy.model.common.normalizer import LinearNormalizer


class BaseImagePolicy(ModuleAttrMixin):
    """v2 策略基类。"""

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def reset(self):
        return None

    def set_normalizer(self, normalizer: LinearNormalizer):
        raise NotImplementedError

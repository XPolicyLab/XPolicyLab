from typing import Dict, List, Optional

import torch


class SelectActionStateDims:
    def __init__(
        self,
        action_indices: Optional[Dict[str, List[int]]] = None,
        state_indices: Optional[Dict[str, List[int]]] = None,
    ):
        self.action_indices = action_indices or {}
        self.state_indices = state_indices or {}

    def forward(self, batch):
        if "action" in batch:
            self._select_category(batch["action"], self.action_indices)
        self._select_category(batch["state"], self.state_indices)
        return batch

    def backward(self, batch):
        raise NotImplementedError(
            "SelectActionStateDims is a lossy transform and cannot be inverted."
        )

    @staticmethod
    def _select_category(values: Dict[str, torch.Tensor], indices_by_key: Dict[str, List[int]]):
        for key, indices in indices_by_key.items():
            if key not in values:
                raise KeyError(f"Key '{key}' not found in batch.")
            idx = torch.as_tensor(indices, dtype=torch.long, device=values[key].device)
            values[key] = values[key].index_select(dim=-1, index=idx)

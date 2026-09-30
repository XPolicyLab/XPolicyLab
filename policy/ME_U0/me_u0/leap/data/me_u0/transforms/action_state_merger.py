from typing import Dict

import torch
from torch.nn.functional import pad


class ConcatLeftAlign:
    def __init__(
        self,
        action_target_dim: int | None = None,
        state_target_dim: int | None = None
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if "action" in batch:
            action_masks = batch.pop("action_dim_is_pad_by_key", None)
            batch["action"] = self._concat(batch["action"], self.action_meta)
            batch["action"], batch["action_dim_is_pad"] = self._pad(
                batch["action"],
                self.action_target_dim,
                self._concat_dim_mask(action_masks, self.action_meta, batch["action"].device),
            )

        state_masks = batch.pop("state_dim_is_pad_by_key", None)
        batch["state"] = self._concat(batch["state"], self.state_meta)
        batch["state"], batch["state_dim_is_pad"] = self._pad(
            batch["state"],
            self.state_target_dim,
            self._concat_dim_mask(state_masks, self.state_meta, batch["state"].device),
        )

        return batch

    def backward(self, batch):
        if self.state_target_dim is not None:
            assert batch["state"].shape[-1] == self.state_target_dim
        batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)

        if self.action_target_dim is not None:
            assert batch["action"].shape[-1] == self.action_target_dim
        batch["action"] = self._crop(batch["action"], self.action_meta)
        batch["action"] = self._split(batch["action"], self.action_meta)

        return batch

    @staticmethod
    def _pad(
        x: torch.Tensor,
        dim: int,
        dim_is_pad: torch.Tensor | None = None,
    ):
        if dim is None:
            dim = x.shape[-1]

        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        if dim_is_pad is None:
            mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device)
        else:
            mask = torch.as_tensor(dim_is_pad, dtype=torch.bool, device=x.device)
            assert mask.ndim == 1 and mask.shape[0] == x.shape[-1]
        mask = pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    @staticmethod
    def _concat_dim_mask(dim_masks, meta, device: torch.device) -> torch.Tensor:
        """Flatten optional per-field masks in the same order as ``_concat``."""
        pieces = []
        for m in meta:
            key, dim = m["key"], m["shape"]
            value = None if dim_masks is None else dim_masks.get(key)
            if value is None:
                mask = torch.zeros(dim, dtype=torch.bool, device=device)
            else:
                mask = torch.as_tensor(value, dtype=torch.bool, device=device)
                assert mask.ndim == 1 and mask.shape[0] == dim, (
                    f"Dynamic mask for {key!r} must have shape [{dim}], got {tuple(mask.shape)}"
                )
            pieces.append(mask)
        return torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.bool, device=device)

    @staticmethod
    def _crop(x: torch.Tensor, meta: int):
        assert x.ndim == 3
        dim = sum([m["shape"] for m in meta])
        x = x[:, :, :dim]
        return x

    @staticmethod
    def _concat(x: Dict[str, torch.Tensor], meta: Dict[str, Dict]):
        x = torch.cat([x[m["key"]] for m in meta], dim=-1)
        assert x.ndim == 2
        return x

    @staticmethod
    def _split(x: torch.Tensor, meta: Dict[str, Dict]):
        assert x.ndim == 3
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = x[:, :, idx: idx + dim]
            idx += dim

        return y

from typing import Any, Dict

import torch
from torch.nn.functional import pad


class ConcatLeftAlign:
    def __init__(
        self,
        action_target_dim: int | None = None,
        state_target_dim: int | None = None,
        action_target_slices: list[dict[str, Any]] | None = None,
        state_target_slices: list[dict[str, Any]] | None = None,
    ):
        self.action_target_dim = action_target_dim
        self.state_target_dim = state_target_dim
        self.action_target_slices = action_target_slices
        self.state_target_slices = state_target_slices

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

    def forward(self, batch):
        if "action" in batch:
            batch["action"] = self._concat(batch["action"], self.action_meta)
            if self.action_target_slices is not None:
                batch["action"], batch["action_dim_is_pad"] = self._scatter(
                    batch["action"], self.action_target_dim, self.action_target_slices
                )
            else:
                batch["action"], batch["action_dim_is_pad"] = self._pad(batch["action"], self.action_target_dim)

        batch["state"] = self._concat(batch["state"], self.state_meta)
        if self.state_target_slices is not None:
            batch["state"], batch["state_dim_is_pad"] = self._scatter(
                batch["state"], self.state_target_dim, self.state_target_slices
            )
        else:
            batch["state"], batch["state_dim_is_pad"] = self._pad(batch["state"], self.state_target_dim)

        return batch

    def backward(self, batch):
        if self.state_target_slices is not None:
            if self.state_target_dim is not None:
                assert batch["state"].shape[-1] == self.state_target_dim
            batch["state"] = self._gather(batch["state"], self.state_target_slices)
        else:
            if self.state_target_dim is not None:
                assert batch["state"].shape[-1] == self.state_target_dim
            batch["state"] = self._crop(batch["state"], self.state_meta)
        batch["state"] = self._split(batch["state"], self.state_meta)

        if self.action_target_slices is not None:
            if self.action_target_dim is not None:
                assert batch["action"].shape[-1] == self.action_target_dim
            batch["action"] = self._gather(batch["action"], self.action_target_slices)
        else:
            if self.action_target_dim is not None:
                assert batch["action"].shape[-1] == self.action_target_dim
            batch["action"] = self._crop(batch["action"], self.action_meta)
        batch["action"] = self._split(batch["action"], self.action_meta)

        return batch

    @staticmethod
    def _pad(x: torch.Tensor, dim: int):
        if dim is None:
            dim = x.shape[-1]
        
        assert x.ndim == 2 and x.shape[-1] <= dim
        pad_dim = dim - x.shape[-1]
        x_padded = pad(x, (0, pad_dim))
        mask = torch.zeros_like(x[0]).bool()
        mask = pad(mask, (0, pad_dim), value=True)
        return x_padded, mask

    @staticmethod
    def _slice_pair(entry: dict[str, Any], key: str) -> tuple[int, int]:
        value = entry.get(key)
        if value is None or isinstance(value, (str, bytes)):
            raise ValueError(f"{key} must be [start, end], got {value!r}")
        try:
            if len(value) != 2:
                raise ValueError
            start, end = int(value[0]), int(value[1])
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"{key} must be [start, end], got {value!r}") from exc
        if start < 0 or end <= start:
            raise ValueError(f"Invalid {key}: {value!r}")
        return start, end

    @classmethod
    def _scatter(cls, x: torch.Tensor, dim: int | None, target_slices: list[dict[str, Any]]):
        if dim is None:
            raise ValueError("target dim is required when target_slices is set")
        assert x.ndim == 2
        out = x.new_zeros((x.shape[0], int(dim)))
        mask = torch.ones(int(dim), dtype=torch.bool, device=x.device)
        used = torch.zeros(int(dim), dtype=torch.bool, device=x.device)
        for entry in target_slices:
            src_start, src_end = cls._slice_pair(entry, "source_slice")
            tgt_start, tgt_end = cls._slice_pair(entry, "target_slice")
            if src_end > x.shape[-1]:
                raise ValueError(f"source_slice {[src_start, src_end]} exceeds input dim {x.shape[-1]}")
            if tgt_end > int(dim):
                raise ValueError(f"target_slice {[tgt_start, tgt_end]} exceeds target dim {dim}")
            if (src_end - src_start) != (tgt_end - tgt_start):
                raise ValueError(
                    f"source_slice {[src_start, src_end]} and target_slice {[tgt_start, tgt_end]} have different widths"
                )
            if bool(used[tgt_start:tgt_end].any().item()):
                raise ValueError(f"Overlapping target_slice {[tgt_start, tgt_end]}")
            out[:, tgt_start:tgt_end] = x[:, src_start:src_end]
            mask[tgt_start:tgt_end] = False
            used[tgt_start:tgt_end] = True
        return out, mask

    @classmethod
    def _gather(cls, x: torch.Tensor, target_slices: list[dict[str, Any]]):
        assert x.ndim == 3
        source_dim = 0
        for entry in target_slices:
            _, src_end = cls._slice_pair(entry, "source_slice")
            source_dim = max(source_dim, src_end)
        out = x.new_zeros((*x.shape[:-1], source_dim))
        for entry in target_slices:
            src_start, src_end = cls._slice_pair(entry, "source_slice")
            tgt_start, tgt_end = cls._slice_pair(entry, "target_slice")
            if tgt_end > x.shape[-1]:
                raise ValueError(f"target_slice {[tgt_start, tgt_end]} exceeds input dim {x.shape[-1]}")
            if (src_end - src_start) != (tgt_end - tgt_start):
                raise ValueError(
                    f"source_slice {[src_start, src_end]} and target_slice {[tgt_start, tgt_end]} have different widths"
                )
            out[..., src_start:src_end] = x[..., tgt_start:tgt_end]
        return out

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
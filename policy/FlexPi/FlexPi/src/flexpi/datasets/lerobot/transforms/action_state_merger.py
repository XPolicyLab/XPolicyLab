from typing import Dict, List, Optional

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
            batch["action"] = self._concat(batch["action"], self.action_meta)
            batch["action"], batch["action_dim_is_pad"] = self._pad(batch["action"], self.action_target_dim)

        batch["state"] = self._concat(batch["state"], self.state_meta)
        batch["state"], batch["state_dim_is_pad"] = self._pad(batch["state"], self.state_target_dim)

        return batch

    def backward(self, batch):
        # Process whichever of {action, state} is present — eval scripts pass
        # only action; training pipeline passes both. (Mirrors ScatterToChannels.backward.)
        if "action" in batch:
            if self.action_target_dim is not None:
                assert batch["action"].shape[-1] == self.action_target_dim
            batch["action"] = self._crop(batch["action"], self.action_meta)
            batch["action"] = self._split(batch["action"], self.action_meta)

        if "state" in batch:
            if self.state_target_dim is not None:
                assert batch["state"].shape[-1] == self.state_target_dim
            batch["state"] = self._crop(batch["state"], self.state_meta)
            batch["state"] = self._split(batch["state"], self.state_meta)

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


class ScatterToChannels:
    """LingBot-VA style per-embodiment channel projection into a universal layout.

    forward:  concat per-key vector of length sum(shape_meta[cat].shape), scatter
              into universal slots [total_dim] at `used_*_channel_ids`, zeros
              elsewhere. Emit `*_dim_is_pad[total_dim]` (True at unused ids).
    backward: gather from [total_dim] at `used_*_channel_ids`, then split back
              into per-key chunks using shape_meta.

    See lingbot-va/wan_va/configs/va_libero_cfg.py:33 for the channel-id pattern.
    """

    def __init__(
        self,
        total_dim: int,
        used_action_channel_ids: Optional[List[int]] = None,
        used_state_channel_ids: Optional[List[int]] = None,
    ):
        self.total_dim = total_dim
        self.used_action_channel_ids = list(used_action_channel_ids) if used_action_channel_ids is not None else None
        self.used_state_channel_ids = list(used_state_channel_ids) if used_state_channel_ids is not None else None

    def set_shape_meta(self, shape_meta):
        self.action_meta = shape_meta["action"]
        self.state_meta = shape_meta["state"]

        action_in_dim = sum(m["shape"] for m in self.action_meta)
        state_in_dim = sum(m["shape"] for m in self.state_meta)

        if self.used_action_channel_ids is not None:
            assert len(self.used_action_channel_ids) == action_in_dim, \
                f"used_action_channel_ids has {len(self.used_action_channel_ids)} entries but action shape_meta sums to {action_in_dim}"
            assert max(self.used_action_channel_ids) < self.total_dim
            assert min(self.used_action_channel_ids) >= 0
        if self.used_state_channel_ids is not None:
            assert len(self.used_state_channel_ids) == state_in_dim, \
                f"used_state_channel_ids has {len(self.used_state_channel_ids)} entries but state shape_meta sums to {state_in_dim}"
            assert max(self.used_state_channel_ids) < self.total_dim
            assert min(self.used_state_channel_ids) >= 0

    def forward(self, batch):
        # A per-SAMPLE pad mask set by an upstream transform (AlignTo34DByWidth
        # marks a 32-wide sample's absent 7th-joint slots) is OR-ed into this
        # merger's static mask instead of being overwritten.
        if "action" in batch:
            x = ConcatLeftAlign._concat(batch["action"], self.action_meta)
            prior = batch.get("action_dim_is_pad", None)
            batch["action"], mask = self._scatter(x, self.used_action_channel_ids)
            if prior is not None:
                mask = mask | prior.to(device=mask.device, dtype=torch.bool)
            batch["action_dim_is_pad"] = mask

        x = ConcatLeftAlign._concat(batch["state"], self.state_meta)
        prior = batch.get("state_dim_is_pad", None)
        batch["state"], mask = self._scatter(x, self.used_state_channel_ids)
        if prior is not None:
            mask = mask | prior.to(device=mask.device, dtype=torch.bool)
        batch["state_dim_is_pad"] = mask

        return batch

    def backward(self, batch):
        # Process whichever of {action, state} is present — eval scripts pass
        # only action; training pipeline passes both.
        if "action" in batch:
            assert batch["action"].shape[-1] == self.total_dim
            batch["action"] = self._gather_and_split(batch["action"], self.used_action_channel_ids, self.action_meta)

        if "state" in batch:
            assert batch["state"].shape[-1] == self.total_dim
            batch["state"] = self._gather_and_split(batch["state"], self.used_state_channel_ids, self.state_meta)

        return batch

    def _scatter(self, x: torch.Tensor, used_ids: Optional[List[int]]):
        assert x.ndim == 2, f"Expected 2D (T, dim), got {x.shape}"
        T = x.shape[0]
        out = x.new_zeros(T, self.total_dim)
        mask = torch.ones(self.total_dim, dtype=torch.bool, device=x.device)
        if used_ids is None:
            # identity: require input dim == total_dim
            assert x.shape[-1] == self.total_dim, f"No channel ids given but dim {x.shape[-1]} != total {self.total_dim}"
            out = x
            mask = torch.zeros(self.total_dim, dtype=torch.bool, device=x.device)
        else:
            ids = torch.tensor(used_ids, dtype=torch.long, device=x.device)
            out[:, ids] = x
            mask[ids] = False
        return out, mask

    def _gather_and_split(self, x: torch.Tensor, used_ids: Optional[List[int]], meta):
        assert x.ndim == 3, f"Expected 3D (B, T, dim), got {x.shape}"
        if used_ids is None:
            gathered = x
        else:
            ids = torch.tensor(used_ids, dtype=torch.long, device=x.device)
            gathered = x[:, :, ids]
        # split into per-key chunks
        y = {}
        idx = 0
        for m in meta:
            key, dim = m["key"], m["shape"]
            y[key] = gathered[:, :, idx: idx + dim]
            idx += dim
        return y

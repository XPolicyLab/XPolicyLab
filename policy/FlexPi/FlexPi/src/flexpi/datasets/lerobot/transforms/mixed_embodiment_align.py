"""Per-sample alignment of mixed yam/mm2 + AgiBot corpora into the shared 34D layout.

Checkpoints co-trained on mm2 + YAM + AgiBot chain this transform before
`Yam32DRelativeAction(anchor="first", state_dim=34)`. It routes each sample by
its own width:

  32-wide (yam/mm2 eef32): zero-scatter into 34D. Slots 26 and 33 (each arm's
           absent 7th joint) are marked pad via a per-sample
           `action_dim_is_pad` / `state_dim_is_pad`, which `ScatterToChannels`
           ORs into its own mask. The relative math is slot-local, so the pad
           slots come out as 0-0=0 deltas.
  34-wide (AgiBot, 7-DoF arms): current-state recording -> future-pose
           lookahead. Training-only; not ported to this deploy codebase.

`backward()` cannot know the target embodiment from a model output, so it is set
explicitly: `backward_mode="yam32"` (default) gathers the 32 real slots back
into the eef32 layout — the YAM deployment path. Deploy must also zero the pad
slots in the action latent (`FlexPiBackbone.set_action_dim_mask`), since
training never supervised them and they would otherwise sample as noise.
"""

from __future__ import annotations

from typing import Dict

import torch

_PAD_SLOTS = (26, 33)
_KEEP_IDS = list(range(26)) + list(range(27, 33))  # 32 entries: eef32 -> 34D


class AlignTo34DByWidth:
    def __init__(self, backward_mode: str = "yam32") -> None:
        if backward_mode not in ("yam32", "agibot34"):
            raise ValueError(f"backward_mode must be 'yam32' or 'agibot34', got {backward_mode!r}")
        self.backward_mode = backward_mode

    def set_shape_meta(self, shape_meta) -> None:
        pass

    def forward(self, batch: Dict) -> Dict:
        width = batch["state"]["default"].shape[-1]
        if width == 32:
            keep = torch.tensor(_KEEP_IDS, dtype=torch.long)
            for cat in ("state", "action"):
                if cat not in batch:
                    continue
                x = batch[cat]["default"]
                y = x.new_zeros(*x.shape[:-1], 34)
                y[..., keep] = x
                batch[cat]["default"] = y
            mask = torch.zeros(34, dtype=torch.bool)
            mask[list(_PAD_SLOTS)] = True
            batch["action_dim_is_pad"] = mask.clone()
            batch["state_dim_is_pad"] = mask.clone()
            return batch
        if width == 34:
            raise NotImplementedError(
                "AlignTo34DByWidth: 34-wide (AgiBot) samples need AgiBotAbsoluteLookahead, "
                "which is training-only and not part of this deploy codebase."
            )
        raise ValueError(
            f"AlignTo34DByWidth: sample width {width} is neither 32 (yam/mm2 "
            f"eef32) nor 34 (AgiBot keep-all-joints)."
        )

    def backward(self, batch: Dict) -> Dict:
        if self.backward_mode == "agibot34":
            raise NotImplementedError(
                "AlignTo34DByWidth backward_mode='agibot34' is not ported; deploy YAM with 'yam32'."
            )
        # yam32: gather the 32 real slots back into the eef32 layout.
        keep = torch.tensor(_KEEP_IDS, dtype=torch.long)
        for cat in ("action", "state"):
            if cat in batch:
                x = batch[cat]
                if isinstance(x, dict):
                    x["default"] = x["default"][..., keep]
                else:
                    batch[cat] = x[..., keep]
        return batch

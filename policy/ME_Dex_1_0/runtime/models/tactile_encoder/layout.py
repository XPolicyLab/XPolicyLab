"""Slot indices are zero-based; array-side and arm identity are separate."""

import torch

PARTS = ("thumb", "index", "middle", "ring")
SLOT_NAMES = tuple(
    f"{hand}/{part}/{local}"
    for hand in ("left", "right")
    for part, local in [(p, i) for p in PARTS for i in range(4)] + [("palm", 0)]
)
# Input DexJoCo order: PALM, TH_PROX/PAD/TIP, FF*, MF*, RF*.
DEX_LOCAL_TO_UNI = (16, 0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)


def surface_slots(kind: str, *, hand: str = "right", finger: str = "index") -> torch.Tensor:
    if kind == "robotwin":
        return torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7],
                             [17, 18, 19, 20], [21, 22, 23, 24]])
    if kind == "dexjoco":
        result = torch.full((32, 4), -1, dtype=torch.long)
        result[:, 0] = torch.tensor([h * 17 + i for h in range(2) for i in DEX_LOCAL_TO_UNI])
        return result
    if kind == "manifeel":
        if hand not in ("left", "right") or finger not in ("thumb", "index"):
            raise ValueError("ManiFeel needs an explicit arm slot and opposing-finger role")
        start = 17 * (hand == "right") + 4 * (finger == "index")
        return torch.arange(start, start + 4)[None]
    raise ValueError(f"unknown surface layout {kind}")


def pooling_regions(slots: torch.Tensor) -> torch.Tensor:
    """Return [surfaces,4,10,14] read masks: four quadrants or one full pad."""
    result = torch.zeros(len(slots), 4, 10, 14, dtype=torch.bool, device=slots.device)
    quad = slots[:, 1].ge(0)
    for k, (y, x) in enumerate(((0, 0), (0, 7), (5, 0), (5, 7))):
        result[quad, k, y:y + 5, x:x + 7] = True
    result[~quad, 0] = True
    return result & slots.ge(0)[..., None, None]

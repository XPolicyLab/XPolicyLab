from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME = 'tshape_robotwin_384x320_uniform'
TSHAPE_ROBOTWIN_384X320_NAME = TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME

@dataclass(frozen=True)
class Slot:
    key: str
    top: int
    left: int
    h: int
    w: int
    src_hw: Optional[Tuple[int, int]]
    tile_mode: str = 'resize'

@dataclass(frozen=True)
class LayoutSpec:
    name: str
    composite_hw: Tuple[int, int]
    slots: Tuple[Slot, ...]
    default_slot_key_map: Mapping[str, str]
    dino_cam_patches: Tuple[Tuple[int, int], ...]
    dino_cam_regions: Tuple[Tuple[int, int, int, int], ...]
    dino_grid_hw: Tuple[int, int]
    text_prefix: Optional[str] = None

    def cam_slots(self) -> Tuple[Slot, ...]:
        return tuple((s for s in self.slots if s.src_hw is not None))

    def black_slots(self) -> Tuple[Slot, ...]:
        return tuple((s for s in self.slots if s.src_hw is None))

    def slot_hw(self) -> Mapping[str, Tuple[int, int]]:
        return MappingProxyType({s.key: (s.h, s.w) for s in self.cam_slots()})

    def resolve_slot_key_map(self, override: Optional[Mapping[str, str]]=None) -> Mapping[str, str]:
        m = dict(override) if override else dict(self.default_slot_key_map)
        missing = [s.key for s in self.cam_slots() if s.key not in m]
        if missing:
            raise ValueError(f'Layout {self.name!r}: slot_key_map missing slot(s) {missing}. Got {dict(m)!r}; cam slots are {[s.key for s in self.cam_slots()]!r}. Provide them via the data-config `composite_layout.slot_key_map` block.')
        return MappingProxyType(m)

    def resolved_per_cam_hw(self, override: Optional[Mapping[str, str]]=None) -> Mapping[str, Tuple[int, int]]:
        kmap = self.resolve_slot_key_map(override)
        out = {}
        for s in self.cam_slots():
            assert s.src_hw is not None
            out[kmap[s.key]] = s.src_hw
        return MappingProxyType(out)

    def resolved_slot_hw(self, override: Optional[Mapping[str, str]]=None) -> Mapping[str, Tuple[int, int]]:
        kmap = self.resolve_slot_key_map(override)
        out = {kmap[s.key]: (s.h, s.w) for s in self.cam_slots()}
        return MappingProxyType(out)

    def with_dino_pool(self, factor: int) -> Tuple[Tuple[Tuple[int, int], ...], Tuple[Tuple[int, int, int, int], ...], Tuple[int, int]]:
        if factor < 1:
            raise ValueError(f'dino_pool_factor must be >= 1, got {factor}')
        if factor == 1:
            return (self.dino_cam_patches, self.dino_cam_regions, self.dino_grid_hw)
        for i, (h, w) in enumerate(self.dino_cam_patches):
            if h % factor != 0 or w % factor != 0:
                raise ValueError(f"Layout {self.name!r}: dino_pool_factor={factor} doesn't evenly divide cam {i}'s patch grid {(h, w)}. Use explicit `dino_cam_patches`/`dino_cam_regions` overrides or a different pool factor. (Layouts with asymmetric per-cam pool — e.g. RoboTwin's 14×14 head + 7×7 wrists — bake the asymmetry into the layout itself; pool_factor must stay 1.)")
        for i, (h0, h1, w0, w1) in enumerate(self.dino_cam_regions):
            if (h1 - h0) % factor != 0 or (w1 - w0) % factor != 0:
                raise ValueError(f"Layout {self.name!r}: dino_pool_factor={factor} doesn't evenly divide cam {i}'s region bbox {(h0, h1, w0, w1)}")
        gH, gW = self.dino_grid_hw
        if gH % factor != 0 or gW % factor != 0:
            raise ValueError(f"Layout {self.name!r}: dino_pool_factor={factor} doesn't evenly divide the RoPE grid {(gH, gW)}")
        patches = tuple(((h // factor, w // factor) for h, w in self.dino_cam_patches))
        regions = tuple(((h0 // factor, h1 // factor, w0 // factor, w1 // factor) for h0, h1, w0, w1 in self.dino_cam_regions))
        grid_hw = (gH // factor, gW // factor)
        return (patches, regions, grid_hw)

    def __post_init__(self):
        H_total, W_total = self.composite_hw
        seen_keys: set = set()
        for s in self.slots:
            if s.top < 0 or s.left < 0 or s.h <= 0 or (s.w <= 0):
                raise ValueError(f'Layout {self.name!r}: invalid slot {s!r}')
            if s.top + s.h > H_total or s.left + s.w > W_total:
                raise ValueError(f'Layout {self.name!r}: slot {s!r} exceeds composite {self.composite_hw}')
            if s.src_hw is not None:
                if s.key == '':
                    raise ValueError(f'Layout {self.name!r}: cam slot must have non-empty key; got {s!r}')
                if s.key in seen_keys:
                    raise ValueError(f'Layout {self.name!r}: duplicate slot key {s.key!r}')
                seen_keys.add(s.key)
        n_cam = len(self.cam_slots())
        if len(self.dino_cam_patches) != n_cam:
            raise ValueError(f'Layout {self.name!r}: dino_cam_patches has {len(self.dino_cam_patches)} entries; expected {n_cam} (one per cam slot)')
        if len(self.dino_cam_regions) != n_cam:
            raise ValueError(f'Layout {self.name!r}: dino_cam_regions has {len(self.dino_cam_regions)} entries; expected {n_cam} (one per cam slot)')
        gH, gW = self.dino_grid_hw
        for i, (h0, h1, w0, w1) in enumerate(self.dino_cam_regions):
            if not (0 <= h0 < h1 <= gH and 0 <= w0 < w1 <= gW):
                raise ValueError(f'Layout {self.name!r}: dino_cam_regions[{i}]={self.dino_cam_regions[i]} outside grid {self.dino_grid_hw}')
LAYOUTS: dict = {TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME: LayoutSpec(name=TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME, composite_hw=(384, 320), slots=(Slot('slot_top', top=0, left=0, h=256, w=320, src_hw=(256, 320)), Slot('slot_bl', top=256, left=0, h=128, w=160, src_hw=(224, 224)), Slot('slot_br', top=256, left=160, h=128, w=160, src_hw=(224, 224))), default_slot_key_map={'slot_top': 'cam_high', 'slot_bl': 'cam_left_wrist', 'slot_br': 'cam_right_wrist'}, dino_cam_patches=((14, 14), (14, 14), (14, 14)), dino_cam_regions=((0, 14, 0, 14), (14, 28, 0, 14), (14, 28, 14, 28)), dino_grid_hw=(28, 28), text_prefix=None)}
LAYOUT_ALIASES: dict = {'tshape_robotwin_384x320': TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME, 'robotwin': TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME}

def canonical_layout_name(name):
    return LAYOUT_ALIASES.get(name, name)

def get_layout(name_or_layout) -> LayoutSpec:
    if name_or_layout is None:
        return LAYOUTS[TSHAPE_ROBOTWIN_384X320_NAME]
    if isinstance(name_or_layout, LayoutSpec):
        return name_or_layout
    if isinstance(name_or_layout, str):
        resolved = canonical_layout_name(name_or_layout)
        if resolved not in LAYOUTS:
            raise KeyError(f'Unknown composite layout {name_or_layout!r}. Registered: {sorted(LAYOUTS)!r}')
        return LAYOUTS[resolved]
    raise TypeError(f'Expected str | LayoutSpec | None; got {type(name_or_layout).__name__}')

def is_layout_name(name) -> bool:
    return isinstance(name, str) and canonical_layout_name(name) in LAYOUTS

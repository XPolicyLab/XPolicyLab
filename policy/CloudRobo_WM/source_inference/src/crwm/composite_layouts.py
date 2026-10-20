from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple
TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME = 'tshape_robotwin_384x320_uniform'

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
LAYOUTS: dict = {TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME: LayoutSpec(name=TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME, composite_hw=(384, 320), slots=(Slot('slot_top', top=0, left=0, h=256, w=320, src_hw=(256, 320)), Slot('slot_bl', top=256, left=0, h=128, w=160, src_hw=(224, 224)), Slot('slot_br', top=256, left=160, h=128, w=160, src_hw=(224, 224))), default_slot_key_map={'slot_top': 'cam_high', 'slot_bl': 'cam_left_wrist', 'slot_br': 'cam_right_wrist'}, dino_cam_patches=((14, 14), (14, 14), (14, 14)), dino_cam_regions=((0, 14, 0, 14), (14, 28, 0, 14), (14, 28, 14, 28)), dino_grid_hw=(28, 28), text_prefix=None)}

def get_layout(name_or_layout) -> LayoutSpec:
    if isinstance(name_or_layout, LayoutSpec):
        return name_or_layout
    if name_or_layout is None or name_or_layout == TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME:
        return LAYOUTS[TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME]
    raise ValueError('CR-WM uses tshape_robotwin_384x320_uniform')

def is_layout_name(name) -> bool:
    return name == TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME

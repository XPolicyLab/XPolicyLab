"""Multi-view composite layout registry.

Single source of truth for "how cameras land in the [B, 3, T, H, W] composite
that the WAN VAE consumes". Every consumer (dataset, model build_inputs, DINO
encoder, pointmap encoder, validation visualization, deploy ObservationBuilder)
should read its slot table from here rather than hardcoding 384/320/256/128/160
or assuming RoboTwin head-top + wrists-bottom.

Each layout is a `LayoutSpec` with:
- a fixed composite size,
- an ordered list of `Slot`s describing the spatial bbox for each camera plus
  any black (zero-pixel) slots,
- a `default_slot_key_map` from the layout's generic placeholder slot keys
  (e.g. ``"slot_top"``) to dataset camera keys (e.g. ``"cam_high"``). Datasets
  may override this via the ``composite_layout.slot_key_map`` config block,
- explicit DINO patch / region tables. These are load-bearing for RoPE
  position encoding (`helpers/dino.py::get_dino_mesh_id`), not just visualization,
  so they live in the registry rather than being auto-derived.
- an optional T5 layout-describing prefix (off by default).

Layouts are embodiment-agnostic: any 3-cam dataset (RoboTwin included) can opt
into any registered layout by setting `concat_multi_camera` to its name and
supplying a `slot_key_map` if the layout's default is empty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

# The 3-cam T: head 256×320 on top + two wrists 128×160 on bottom = 384×320.
# RoboTwin names it explicitly; YAM, AgiBot and the model default reach the same
# spec through the `tshape_384x320` shorthand (see LAYOUT_ALIASES). Every cam
# gets a uniform 14×14 DINO grid, which is what `dino_pixel_unshuffle: 2` needs
# to fold evenly 14→7.
TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME = "tshape_robotwin_384x320_uniform"
# Pixel-identical to the above; differs ONLY in the DINO RoPE grid, which is the
# asymmetric 21×14 (head 14², wrists 7² each) that predates the uniform grid.
# Nothing selects it — the fold requires uniform patches, and every shipped
# config runs the fold. It remains the resolution of `robotwin` and of an
# unspecified layout, so pre-uniform runs still describe themselves correctly.
TSHAPE_ROBOTWIN_384X320_NAME = "tshape_robotwin_384x320"
TSHAPE_384X320_HEAD_TOP_NAME = "tshape_384x320_head_top"
# The two specs above are the SAME pixel composite and differ only in DINO
# grid, so anything asking "is this the 3-cam 384x320 T?" must accept either.
# Compare canonical names against this, never against one of them alone.
TSHAPE_384X320_FAMILY = (
    TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME,
    TSHAPE_ROBOTWIN_384X320_NAME,
)
# 16:9-FAITHFUL T: keeps EVERY tile inside the WAN prior's aspect range
# [0.55,1.82] AND at/near native 16:9 — the top wrist is a TRUE 16:9 tile
# (288×512), so the dominant cam is undistorted AND in-prior. Aims to win BOTH
# in-dist val (native aspect) and OOD deploy (in-prior) instead of trading one
# for the other. 224 video tok. The LIBERO default. Three cam slots like every
# other T — LIBERO fills the third with a synthetic black camera, since it has
# only two real ones.
TSHAPE_LIBERO_2CAM_448X512_NAME = "tshape_libero_2cam_448x512"


@dataclass(frozen=True)
class Slot:
    """A region in the composite. Camera slots have ``src_hw`` set; black
    slots have ``src_hw=None`` and are filled with zeros.

    ``key`` is a generic placeholder (e.g. ``"slot_top"``) that the
    `slot_key_map` resolves to a real dataset camera key. For black slots,
    ``key`` is the empty string.

    ``tile_mode`` controls how the per-cam tensor is fit into the slot when
    ``src_hw != (h, w)``:
      - ``"resize"`` (default): antialiased bilinear resample (smooth).
      - ``"repeat"``: nearest-neighbor — equivalent to ``np.repeat`` for
        integer scale factors. For a slot that pixel-doubles its camera
        (matches dreamzero's ``np.repeat(axis=-1)``) rather than blurring it
        via bilinear. No registered layout uses it today.
    """
    key: str
    top: int
    left: int
    h: int
    w: int
    src_hw: Optional[Tuple[int, int]]  # native per-cam HW; None for black slots
    tile_mode: str = "resize"          # "resize" | "repeat"


@dataclass(frozen=True)
class LayoutSpec:
    """Describes how cameras land in a composite tensor.

    Use ``get_layout(name)`` to resolve from the registry rather than
    constructing instances directly.
    """
    name: str
    composite_hw: Tuple[int, int]
    slots: Tuple[Slot, ...]                   # cam slots first, then any black slots
    default_slot_key_map: Mapping[str, str]   # placeholder -> dataset cam key
    # DINO patch tables. ``dino_cam_patches[i]`` and ``dino_cam_regions[i]`` apply
    # to the i-th *camera* slot (skipping black slots), in slot-table order.
    # Both are load-bearing for RoPE; see flexpi/models/helpers/dino.py.
    dino_cam_patches: Tuple[Tuple[int, int], ...]
    dino_cam_regions: Tuple[Tuple[int, int, int, int], ...]
    dino_grid_hw: Tuple[int, int]             # encompassing RoPE grid; for sanity-check
    text_prefix: Optional[str] = None         # optional T5 layout prefix; off by default

    # ----- Slot accessors -----

    def cam_slots(self) -> Tuple[Slot, ...]:
        return tuple(s for s in self.slots if s.src_hw is not None)

    def black_slots(self) -> Tuple[Slot, ...]:
        return tuple(s for s in self.slots if s.src_hw is None)

    def slot_hw(self) -> Mapping[str, Tuple[int, int]]:
        """slot placeholder -> (h, w) on the composite (cam slots only)."""
        return MappingProxyType({s.key: (s.h, s.w) for s in self.cam_slots()})

    # ----- Slot ↔ dataset-key binding -----

    def resolve_slot_key_map(
        self, override: Optional[Mapping[str, str]] = None,
    ) -> Mapping[str, str]:
        """Return the slot_key_map this layout should use for a given config.

        Order:
          1) ``override`` if given (config-supplied),
          2) else ``default_slot_key_map``.

        Raises if the resolved map doesn't cover every camera slot.
        """
        m = dict(override) if override else dict(self.default_slot_key_map)
        missing = [s.key for s in self.cam_slots() if s.key not in m]
        if missing:
            raise ValueError(
                f"Layout {self.name!r}: slot_key_map missing slot(s) {missing}. "
                f"Got {dict(m)!r}; cam slots are "
                f"{[s.key for s in self.cam_slots()]!r}. "
                f"Provide them via the data-config `composite_layout.slot_key_map` block."
            )
        return MappingProxyType(m)

    def resolved_per_cam_hw(
        self, override: Optional[Mapping[str, str]] = None,
    ) -> Mapping[str, Tuple[int, int]]:
        """dataset cam key -> native per-cam (H, W) (i.e. ``Slot.src_hw``)."""
        kmap = self.resolve_slot_key_map(override)
        out = {}
        for s in self.cam_slots():
            assert s.src_hw is not None
            out[kmap[s.key]] = s.src_hw
        return MappingProxyType(out)

    def resolved_slot_hw(
        self, override: Optional[Mapping[str, str]] = None,
    ) -> Mapping[str, Tuple[int, int]]:
        """dataset cam key -> composite-slot (H, W)."""
        kmap = self.resolve_slot_key_map(override)
        out = {kmap[s.key]: (s.h, s.w) for s in self.cam_slots()}
        return MappingProxyType(out)

    # ----- Pool factor convenience -----

    def with_dino_pool(
        self, factor: int,
    ) -> Tuple[
        Tuple[Tuple[int, int], ...],
        Tuple[Tuple[int, int, int, int], ...],
        Tuple[int, int],
    ]:
        """Apply an additional ``factor`` to every cam's DINO patch grid.

        Returns ``(dino_cam_patches, dino_cam_regions, dino_grid_hw)`` where
        every patch grid AND region bbox AND the encompassing grid have
        been divided by ``factor``. ``factor=1`` returns the layout's
        defaults verbatim.

        Used to wire a single ``dino_pool_factor`` model knob: ``factor=2``
        on a uniform-14×14 layout (e.g. ``tshape_libero_2cam_448x512``) produces
        7×7 per cam — saving 4× DINO compute while preserving RoPE
        position uniqueness.

        Raises if ``factor`` doesn't divide every patch dim and every
        region bbox dim — e.g. ``factor=2`` on RoboTwin would raise
        because the head's 14×14 halves cleanly but the wrists' 7×7 do
        not. Layouts with asymmetric per-cam pool must be configured via
        explicit ``dino_cam_patches``/``dino_cam_regions`` overrides
        instead of a uniform pool factor.
        """
        if factor < 1:
            raise ValueError(f"dino_pool_factor must be >= 1, got {factor}")
        if factor == 1:
            return self.dino_cam_patches, self.dino_cam_regions, self.dino_grid_hw
        for i, (h, w) in enumerate(self.dino_cam_patches):
            if h % factor != 0 or w % factor != 0:
                raise ValueError(
                    f"Layout {self.name!r}: dino_pool_factor={factor} doesn't "
                    f"evenly divide cam {i}'s patch grid {(h, w)}. "
                    f"Use explicit `dino_cam_patches`/`dino_cam_regions` overrides "
                    f"or a different pool factor. (Layouts with asymmetric per-cam "
                    f"pool — e.g. RoboTwin's 14×14 head + 7×7 wrists — bake the "
                    f"asymmetry into the layout itself; pool_factor must stay 1.)"
                )
        for i, (h0, h1, w0, w1) in enumerate(self.dino_cam_regions):
            if (h1 - h0) % factor != 0 or (w1 - w0) % factor != 0:
                raise ValueError(
                    f"Layout {self.name!r}: dino_pool_factor={factor} doesn't "
                    f"evenly divide cam {i}'s region bbox {(h0, h1, w0, w1)}"
                )
        gH, gW = self.dino_grid_hw
        if gH % factor != 0 or gW % factor != 0:
            raise ValueError(
                f"Layout {self.name!r}: dino_pool_factor={factor} doesn't "
                f"evenly divide the RoPE grid {(gH, gW)}"
            )
        patches = tuple((h // factor, w // factor) for h, w in self.dino_cam_patches)
        regions = tuple(
            (h0 // factor, h1 // factor, w0 // factor, w1 // factor)
            for h0, h1, w0, w1 in self.dino_cam_regions
        )
        grid_hw = (gH // factor, gW // factor)
        return patches, regions, grid_hw

    # ----- Sanity checks -----

    def __post_init__(self):
        H_total, W_total = self.composite_hw
        seen_keys: set = set()
        for s in self.slots:
            if s.top < 0 or s.left < 0 or s.h <= 0 or s.w <= 0:
                raise ValueError(f"Layout {self.name!r}: invalid slot {s!r}")
            if s.top + s.h > H_total or s.left + s.w > W_total:
                raise ValueError(
                    f"Layout {self.name!r}: slot {s!r} exceeds composite "
                    f"{self.composite_hw}"
                )
            if s.src_hw is not None:
                if s.key == "":
                    raise ValueError(
                        f"Layout {self.name!r}: cam slot must have non-empty key; got {s!r}"
                    )
                if s.key in seen_keys:
                    raise ValueError(
                        f"Layout {self.name!r}: duplicate slot key {s.key!r}"
                    )
                seen_keys.add(s.key)
        n_cam = len(self.cam_slots())
        if len(self.dino_cam_patches) != n_cam:
            raise ValueError(
                f"Layout {self.name!r}: dino_cam_patches has "
                f"{len(self.dino_cam_patches)} entries; expected {n_cam} (one per cam slot)"
            )
        if len(self.dino_cam_regions) != n_cam:
            raise ValueError(
                f"Layout {self.name!r}: dino_cam_regions has "
                f"{len(self.dino_cam_regions)} entries; expected {n_cam} (one per cam slot)"
            )
        gH, gW = self.dino_grid_hw
        for i, (h0, h1, w0, w1) in enumerate(self.dino_cam_regions):
            if not (0 <= h0 < h1 <= gH and 0 <= w0 < w1 <= gW):
                raise ValueError(
                    f"Layout {self.name!r}: dino_cam_regions[{i}]={self.dino_cam_regions[i]} "
                    f"outside grid {self.dino_grid_hw}"
                )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

LAYOUTS: dict = {
    # Legacy asymmetric-DINO variant of the 3-cam T. Pixel-identical to
    # TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME below; the DINO RoPE grid is the 21×14 "T-shape"
    # (wrists on top, head on bottom) that predates the uniform grid.
    TSHAPE_ROBOTWIN_384X320_NAME: LayoutSpec(
        name=TSHAPE_ROBOTWIN_384X320_NAME,
        composite_hw=(384, 320),
        slots=(
            Slot("slot_top", top=0,   left=0,   h=256, w=320, src_hw=(256, 320)),
            Slot("slot_bl",  top=256, left=0,   h=128, w=160, src_hw=(224, 224)),
            Slot("slot_br",  top=256, left=160, h=128, w=160, src_hw=(224, 224)),
        ),
        default_slot_key_map={
            "slot_top": "cam_high",
            "slot_bl":  "cam_left_wrist",
            "slot_br":  "cam_right_wrist",
        },
        # Reproduces today's defaults bit-equal:
        #   dino_cam_regions = [(7, 21, 0, 14), (0, 7, 0, 7), (0, 7, 7, 14)]
        #   dino_cam_patches = [(14, 14), (7, 7), (7, 7)]
        # Order matches cam_slots() = (slot_top, slot_bl, slot_br) → head, left, right.
        dino_cam_patches=((14, 14), (7, 7), (7, 7)),
        dino_cam_regions=((7, 21, 0, 14), (0, 7, 0, 7), (0, 7, 7, 14)),
        dino_grid_hw=(21, 14),
        text_prefix=None,
    ),

    # Head-TOP variant of `robotwin`: PIXEL-IDENTICAL composite (head 256×320 on
    # top + two 128×160 wrists below) and the SAME asymmetric DINO budget (head
    # 14×14, wrists 7×7 = 294 tok), but the DINO RoPE places the head in the TOP
    # rows (0–13) and the wrists BELOW (14–20) — mirroring the pixel layout —
    # instead of `robotwin`'s legacy ⊥ grid (head at the BOTTOM rows 7–20, a
    # lingbot-va porting convention; see the file header note). RoPE-only change.
    #
    # Use for FRESH runs that want the head-rich (294-tok) DINO with head-up RoPE.
    # NOT checkpoint-compatible with `robotwin` (different DINO positions). The
    # uniform-DINO path already gets head-up via `tshape_robotwin_384x320_uniform`; this is its
    # non-uniform counterpart.
    TSHAPE_384X320_HEAD_TOP_NAME: LayoutSpec(
        name=TSHAPE_384X320_HEAD_TOP_NAME,
        composite_hw=(384, 320),
        slots=(
            Slot("slot_top", top=0,   left=0,   h=256, w=320, src_hw=(256, 320)),
            Slot("slot_bl",  top=256, left=0,   h=128, w=160, src_hw=(224, 224)),
            Slot("slot_br",  top=256, left=160, h=128, w=160, src_hw=(224, 224)),
        ),
        default_slot_key_map={
            "slot_top": "cam_high",
            "slot_bl":  "cam_left_wrist",
            "slot_br":  "cam_right_wrist",
        },
        dino_cam_patches=((14, 14), (7, 7), (7, 7)),
        # head TOP-left (rows 0–13); left wrist BOTTOM-left, right wrist BOTTOM-right.
        dino_cam_regions=((0, 14, 0, 14), (14, 21, 0, 7), (14, 21, 7, 14)),
        dino_grid_hw=(21, 14),
        text_prefix=None,
    ),

    # The 3-cam T: head 256×320 + two 128×160 wrists, total 384×320, with
    # uniform 14×14 DINO patches per cam — no wrist pooling, and foldable by
    # `dino_pixel_unshuffle: 2` to 7×7/cam = 147 tokens/frame. RoPE grid is 28×28
    # mirroring the pixel topology: head in the top-left quadrant, wrists in the
    # bottom-left / bottom-right. The top-right quadrant is unused (no DINO
    # tokens land there).
    TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME: LayoutSpec(
        name=TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME,
        composite_hw=(384, 320),
        slots=(
            Slot("slot_top", top=0,   left=0,   h=256, w=320, src_hw=(256, 320)),
            Slot("slot_bl",  top=256, left=0,   h=128, w=160, src_hw=(224, 224)),
            Slot("slot_br",  top=256, left=160, h=128, w=160, src_hw=(224, 224)),
        ),
        default_slot_key_map={
            "slot_top": "cam_high",
            "slot_bl":  "cam_left_wrist",
            "slot_br":  "cam_right_wrist",
        },
        dino_cam_patches=((14, 14), (14, 14), (14, 14)),
        dino_cam_regions=(
            (0,  14, 0,  14),    # head:        TL quadrant
            (14, 28, 0,  14),    # left wrist:  BL quadrant
            (14, 28, 14, 28),    # right wrist: BR quadrant
        ),
        dino_grid_hw=(28, 28),
        text_prefix=None,
    ),

    # ── 16:9-faithful T ──────────────────────────────────────────────────────────
    # Same topology as the robotwin T (one full-width top cam + two half-width
    # bottom cams) with a PLAIN-resize top slot, but the top is a TRUE 16:9 tile
    # (288×512) rather than a short wide strip, so a 16:9 source view upscales with
    # ZERO aspect distortion and lands squarely in the WAN prior's native landscape
    # bucket (1280×704 = 1.82). Bottom slots stay near-native at 160×256 (1.6).
    # Every tile ∈ [1.6, 1.78] — none stretched past 16:9, none crushed to square.
    # Uniform 14×14/cam DINO (28×28 grid), foldable via dino_pixel_unshuffle, which
    # weights the bottom cams equally with the top (unlike robotwin's head-rich
    # asymmetric budget). Owns its DINO grid. 448×512 → 14×16 = 224 video tok/frame.
    TSHAPE_LIBERO_2CAM_448X512_NAME: LayoutSpec(
        name=TSHAPE_LIBERO_2CAM_448X512_NAME,
        composite_hw=(448, 512),
        slots=(
            Slot("slot_top", top=0,   left=0,   h=288, w=512, src_hw=(288, 512)),
            Slot("slot_bl",  top=288, left=0,   h=160, w=256, src_hw=(160, 256)),
            Slot("slot_br",  top=288, left=256, h=160, w=256, src_hw=(160, 256)),
        ),
        default_slot_key_map={},
        dino_cam_patches=((14, 14), (14, 14), (14, 14)),
        dino_cam_regions=((0, 14, 0, 14), (14, 28, 0, 14), (14, 28, 14, 28)),
        dino_grid_hw=(28, 28),
        text_prefix=(
            "A multi-view video shows the robot. Top: the wrist camera across the "
            "full width. Bottom-left and bottom-right: the two exterior cameras."
        ),
    ),

    # ── LIBERO 2-camera layout (no black slot) ───────────────────────────────────
}


# Superseded names, plus the one shorthand that is meant to be used.
#
# `tshape_384x320` is NOT legacy — it is the plain name for the 3-cam T that
# YAM, AgiBot and the model default all use, resolving to the same spec RoboTwin
# names explicitly. RoboTwin pins the long form in its task config so its two
# 384×320 variants (uniform and the legacy asymmetric grid) sit side by side
# under one prefix; everyone else says `tshape_384x320` and gets the uniform one.
#
# The rest are old strings still live in saved run `config.yaml` snapshots, which
# `deploy_policy.py` replays verbatim — they must resolve forever, so never
# delete an entry here. `robotwin` predates all of them and still means the
# asymmetric grid it always meant, which the test_robotwin_*_parity suite pins.
LAYOUT_ALIASES: dict = {
    "tshape_384x320": TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME,
    "robotwin": TSHAPE_ROBOTWIN_384X320_NAME,
    "robotwin_head_top": TSHAPE_384X320_HEAD_TOP_NAME,
    "robotwin_uniform14": TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME,
    "tshape_384x320_uniform14": TSHAPE_ROBOTWIN_384X320_UNIFORM_NAME,
    # The asymmetric spec answered to this for one day (2026-08-16 → 08-17).
    "tshape_384x320_asym": TSHAPE_ROBOTWIN_384X320_NAME,
    "tshape169_448x512": TSHAPE_LIBERO_2CAM_448X512_NAME,
    "tshape_448x512": TSHAPE_LIBERO_2CAM_448X512_NAME,
}


def canonical_layout_name(name):
    """Map a possibly-legacy layout name to its canonical one (pass-through
    otherwise). Compare layout NAMES through this, never by string literal —
    an old checkpoint's config still says ``"robotwin"``."""
    return LAYOUT_ALIASES.get(name, name)


def get_layout(name_or_layout) -> LayoutSpec:
    """Resolve a layout from a name string or pass through a LayoutSpec.

    Accepts:
      - a registered name (e.g. ``"tshape_robotwin_384x320_uniform"``)
      - a legacy alias (e.g. ``"robotwin"``) -- see ``LAYOUT_ALIASES``
      - an existing ``LayoutSpec`` (returned as-is)
      - ``None`` (returns the legacy asymmetric T-shape, for back-compat:
        a config that names no layout predates the uniform grid)

    Raises ``KeyError`` for unknown names.
    """
    if name_or_layout is None:
        return LAYOUTS[TSHAPE_ROBOTWIN_384X320_NAME]
    if isinstance(name_or_layout, LayoutSpec):
        return name_or_layout
    if isinstance(name_or_layout, str):
        resolved = canonical_layout_name(name_or_layout)
        if resolved not in LAYOUTS:
            raise KeyError(
                f"Unknown composite layout {name_or_layout!r}. "
                f"Registered: {sorted(LAYOUTS)!r}"
            )
        return LAYOUTS[resolved]
    raise TypeError(
        f"Expected str | LayoutSpec | None; got {type(name_or_layout).__name__}"
    )


def is_layout_name(name) -> bool:
    """True iff ``name`` is a registered layout name (or a legacy alias). Used
    by datasets that also accept legacy ``"horizontal"`` / ``"vertical"`` modes."""
    return isinstance(name, str) and canonical_layout_name(name) in LAYOUTS

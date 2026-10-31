"""The FlexPi training dataset: per-camera RGB, optional per-camera depth.

One loader serves both regimes. Cameras, their order, and the RGB path come
from ``shape_meta.images``; declaring ``shape_meta.depth`` additionally turns
on the depth stream and sets its per-camera target grid. See
``RobotVideoDataset`` for the full contract.

Sample dict:
  - ``per_cam``:           dict[cam -> Tensor [C, T_video, H, W]] float32 in [-1, 1]
  - ``per_cam_depth``:     dict[cam -> Tensor [T_video, H, W]] uint16 mm
                           (present iff ``shape_meta.depth`` is declared)
  - ``camera_intrinsics``: Tensor [num_cams, 3, 3]
  - plus action, proprio, context, context_mask, prompt, and the *_is_pad flags.

RGB is served per-camera; the model composites it (``per_cam_compose``). The
pre-depth composite-``video`` loader and the RGB-only per-cam loader were folded
into this class — no config referenced them.

Depth/RGB frame alignment: verified bit-exact at encode time. Both streams are
encoded via the same ffmpeg path (``-framerate 50``, ``-f rawvideo``) so they
carry identical pts per frame across codecs/containers. The decoder uses the
same query timestamps and tolerance as LeRobot's RGB decode; query timestamps
past the episode end are clamped to the last valid frame, matching LeRobot's
repeat-last-frame padding convention (``image_is_pad`` flags which outputs were
padded).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import time as _dl_time   # alias: DLTIMING harness; no-op when env disabled
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Literal, Mapping, Optional

import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

from .base_lerobot_dataset import BaseLerobotDataset
from ...composite_layouts import (
    LayoutSpec, get_layout, is_layout_name,
    TSHAPE_ROBOTWIN_384X320_NAME,
)
from .lerobot.datasets.video_utils import (
    decode_depth_frames,
    _dl_timing_enabled,
    _dl_worker_id,
    _dl_emit,
)
from .utils.normalizer import save_dataset_stats_to_json, load_dataset_stats_from_json
from ..dataset_utils import ResizeSmallestSideAspectPreserving, CenterCrop, Normalize
from flexpi.utils.logging_config import get_logger
from flexpi.utils import misc, pytorch_utils

logger = get_logger(__name__)


# Legacy per-cam list used by some external callers (e.g. precompute scripts).
# The slot order matches `LAYOUTS["tshape_robotwin_384x320_uniform"].cam_slots()` exactly. Preserved
# for back-compat; new code should pull slot HW from the LayoutSpec directly.
_ROBOTWIN_COMPOSITE_CAM_HW = [
    ("cam_high",        (256, 320)),
    ("cam_left_wrist",  (128, 160)),
    ("cam_right_wrist", (128, 160)),
]

# Legacy module-level constant. Preserved as the RoboTwin per-cam HW table
# (head 256×320, wrists 224×224) for back-compat with the deploy_policy
# importers. New code should pull from the layout via
# ``LAYOUTS[name].cam_slots()[i].src_hw``.
_PER_CAM_HW = [
    ("cam_high",        (256, 320)),
    ("cam_left_wrist",  (224, 224)),
    ("cam_right_wrist", (224, 224)),
]


DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"


_DEPTH_VCODEC = {"ffv1": "ffv1", "x264rgb": "libx264rgb"}
_DEPTH_EXT = {"ffv1": ".mkv", "x264rgb": ".mp4"}

# Depth video PTS are encoded with ~1ms precision (3 decimals). At fps that
# don't divide cleanly into 1ms (e.g. AgiBot 30 fps, 1/30 = 0.0333...), the
# stored PTS rounds alternately to 0.033 / 0.034, producing up to ~0.5ms
# offset from the analytical k/fps query timestamp. lerobot's default
# `tolerance_s = 1e-4` (0.1ms) is too tight for this case and triggers
# AssertionError in `decode_depth_frames`. Bump to 1ms here to cover all
# fps with 1ms-precision PTS without picking the wrong frame (frame spacing
# is >= 1/60 s = 16.7ms even at 60 fps, far above this slack).
# LIBERO/RoboTwin at 50 fps store PTS exactly (1/50 = 0.02), so this bump
# is a no-op there.
_DEPTH_PTS_TOLERANCE_S = 1e-3


class RobotVideoDataset(torch.utils.data.Dataset):
    """Per-camera RGB + depth at native resolution. See module docstring.

    Declaring ``shape_meta.depth`` turns the depth stream ON and sets its
    per-camera target (H, W):

        shape_meta:
          images: [...]
          depth:
            - {key: cam_high,        raw_shape: [1, 240, 320], shape: [1, 240, 320]}
            - {key: cam_left_wrist,  raw_shape: [1, 240, 320], shape: [1, 240, 320]}
            - {key: cam_right_wrist, raw_shape: [1, 240, 320], shape: [1, 240, 320]}

    The depth resize (nearest, preserves integer mm) targets these sizes and K
    is rescaled to match. Omit the block and no depth is decoded or returned;
    K then tracks the ``shape_meta.images`` grid, and the cameras, their order,
    and the RGB path are unaffected either way — those come from
    ``shape_meta.images``, which every config declares.
    """

    def __init__(
        self,
        dataset_dirs,
        shape_meta,
        num_frames=33,
        video_size=[384, 640],
        camera_key=None,
        processor=None,
        text_embedding_cache_dir=None,
        context_len=128,
        pretrained_norm_stats=None,
        val_set_proportion=0.05,
        is_training_set=False,
        global_sample_stride=1,
        action_video_freq_ratio: int = 1,
        skip_padding_as_possible: bool = False,
        max_padding_retry: int = 3,
        concat_multi_camera: Optional[str] = None, # a registered LayoutSpec name ("tshape_robotwin_384x320_uniform", "tshape_libero_2cam_448x512"), or None
        composite_layout_slot_key_map: Optional[Mapping[str, str]] = None, # for layout-driven modes: override default_slot_key_map. Keys are layout slot placeholders (e.g. "slot_tl"); values are dataset cam keys (e.g. "cam_high"). None → use the layout's default.
        composite_layout_text_prefix: bool = False, # if True, prepend the layout's text_prefix to the T5 prompt (off by default)
        override_instruction: Optional[str] = None, # whether to hardcode a specific instruction for all samples, for debugging
        num_episodes: Optional[int] = None, # limit episodes per dataset dir (for quick experiments)
        dataset_weights: Optional[list] = None, # per-dataset *frame* sampling probabilities, parallel to dataset_dirs (size-agnostic mixing, e.g. [0.2,0.8]); requires samples_per_epoch
        samples_per_epoch: Optional[int] = None, # number of weighted draws per epoch (defines epoch length when dataset_weights is set)
        episode_ids: Optional[list] = None, # explicit episode indices (overrides train/val split)
        task_names: Optional[list] = None, # robotwin task names; resolved to episode_ids via task_episode_map.json
        num_episodes_per_task: Optional[int] = None, # random-sample N episodes per task (requires task_names)
        sample_language_annotations: bool = False, # For datasets carrying several language paraphrases per episode: uniformly sample one per item (matches DreamZero); off → always paraphrase #1 (task_index). Requires the text-embeds cache to cover every paraphrase.
        depth_codec: Literal["ffv1", "x264rgb"] = "ffv1",
        synthetic_zero_cams: Optional[list] = None,
        exterior_view_aug_prob: float = 0.0,
        exterior_view_aug_keys: Optional[list] = None,
    ):
        # ── shape_meta: cameras, and whether this run carries depth ──────
        # Read shape_meta BEFORE calling super, because super's __init__
        # triggers _load_layout_intrinsics_for_dir which reads self._per_cam_hw.
        # Accept either an OmegaConf config or a plain dict; we only inspect it,
        # we don't mutate. The original shape_meta (with its OmegaConf type) is
        # forwarded to super intact.
        if isinstance(shape_meta, (DictConfig, ListConfig)):
            sm_dict = OmegaConf.to_container(shape_meta, resolve=True)
        else:
            sm_dict = shape_meta
        images_meta = sm_dict["images"]
        depth_meta = sm_dict.get("depth")
        # THE depth switch: declaring the block is what turns the stream on.
        # No separate boolean, so a config cannot contradict itself.
        self.has_depth: bool = depth_meta is not None

        # Cameras and their order always come from shape_meta.images — every
        # config declares it, and the RGB path needs it whether or not depth
        # exists. shape_meta.depth contributes ONLY the depth target grid, so
        # the two must agree on the camera list; otherwise `camera_intrinsics`
        # (indexed positionally by the pointmap encoder) would silently pair a
        # camera's K with another camera's depth.
        if self.has_depth:
            img_cams = [m["key"] for m in images_meta]
            depth_cams = [m["key"] for m in depth_meta]
            if img_cams != depth_cams:
                raise ValueError(
                    "shape_meta.images and shape_meta.depth must list the same "
                    f"cameras in the same order; got images={img_cams} "
                    f"depth={depth_cams}."
                )
        # `shape` is a [C, H, W] list. Without a depth block K tracks the RGB
        # grid — nothing consumes it then (the pointmap stream is off), but it
        # keeps `camera_intrinsics` meaningful rather than arbitrary.
        hw_meta = depth_meta if self.has_depth else images_meta
        self._per_cam_hw: list[tuple[str, tuple[int, int]]] = [
            (m["key"], (int(m["shape"][-2]), int(m["shape"][-1])))
            for m in hw_meta
        ]
        self._cams: tuple[str, ...] = tuple(name for name, _ in self._per_cam_hw)

        # Parent expects concat_multi_camera to name a registered layout for
        # its per-dataset intrinsics bookkeeping; a config that names no layout
        # predates the uniform DINO grid, so it gets the legacy asymmetric T.
        if concat_multi_camera is None:
            concat_multi_camera = TSHAPE_ROBOTWIN_384X320_NAME
        # ── shared LeRobot / processor / layout setup ────────────────────
        # FLEXPI_RGB_DECODE_SUBSAMPLE (default "1" = enabled) tells lerobot to
        # decode only the `action_video_freq_ratio`-strided frames the model
        # actually consumes (9 of 33 for the standard 33/4 setup). Pre-fix
        # behavior: lerobot decoded all 33 frames per camera, then the
        # downstream `video[:, video_sample_indices]` discarded 24 of them —
        # 73% of RGB decode work was wasted. The decoded pixels for surviving
        # timestamps are byte-identical either way; only `image_is_pad`
        # changes length, and the consumer code is idempotent w.r.t. that.
        # Set to "0" for emergency rollback to legacy behavior.
        _rgb_subsample_enabled = (
            os.environ.get("FLEXPI_RGB_DECODE_SUBSAMPLE", "1") == "1"
            and int(action_video_freq_ratio) > 1
        )
        _image_sample_stride = int(action_video_freq_ratio) if _rgb_subsample_enabled else 1
        self.lerobot_dataset = BaseLerobotDataset(
            dataset_dirs=dataset_dirs,
            shape_meta=OmegaConf.to_container(shape_meta, resolve=True),
            obs_size=num_frames,
            action_size=num_frames - 1,
            val_set_proportion=val_set_proportion,
            is_training_set=is_training_set,
            global_sample_stride=global_sample_stride,
            num_episodes=num_episodes,
            dataset_weights=dataset_weights,
            samples_per_epoch=samples_per_epoch,
            episode_ids=episode_ids,
            task_names=task_names,
            num_episodes_per_task=num_episodes_per_task,
            image_sample_stride=_image_sample_stride,
            sample_language_annotations=sample_language_annotations,
        )

        # Weighted multi-dataset frame-sampling spec, surfaced for the trainer's
        # WeightedResumableEpochSampler. None when single-dataset / unweighted.
        self.dataset_weights = self.lerobot_dataset.dataset_weights
        self.samples_per_epoch = self.lerobot_dataset.samples_per_epoch
        self.per_dataset_num_frames = self.lerobot_dataset.per_dataset_num_frames

        self.num_frames = num_frames
        self.action_video_freq_ratio = action_video_freq_ratio

        assert (num_frames - 1) % self.action_video_freq_ratio == 0, \
            f"num_frames-1 must be divisible by action_video_freq_ratio, got {num_frames - 1} and {self.action_video_freq_ratio}"
        assert ((num_frames - 1) // self.action_video_freq_ratio) % 4 == 0, \
            f"video frames must be divisible by 4 for tokenization, got {(num_frames - 1) // self.action_video_freq_ratio}"
        self.video_sample_indices = list(range(0, num_frames, self.action_video_freq_ratio))

        self.camera_key = camera_key
        self.lerobot_dataset._set_return_images(True)

        self.video_size = video_size
        self.text_embedding_cache_dir = text_embedding_cache_dir
        self.context_len = context_len
        self.skip_padding_as_possible = skip_padding_as_possible
        self.max_padding_retry = max_padding_retry
        self.concat_multi_camera = concat_multi_camera
        # Resolve layout for layout-driven modes (i.e. registered names like
        # "tshape_robotwin_384x320_uniform", "tshape_libero_2cam_448x512"). Legacy modes
        # ("horizontal" / "vertical" / None) leave self._layout = None.
        if is_layout_name(concat_multi_camera):
            self._layout: Optional[LayoutSpec] = get_layout(concat_multi_camera)
            # Convert OmegaConf DictConfig → plain dict so frozen-dataclass equality
            # checks downstream don't choke on omegaconf wrappers.
            override = composite_layout_slot_key_map
            if isinstance(override, DictConfig):
                override = OmegaConf.to_container(override, resolve=True)
            self._slot_key_map = self._layout.resolve_slot_key_map(override)
        else:
            self._layout = None
            self._slot_key_map = None
        self._composite_layout_text_prefix = bool(composite_layout_text_prefix)
        self.override_instruction = override_instruction

        self.resize_transform = ResizeSmallestSideAspectPreserving(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.crop_transform = CenterCrop(
            args={"img_w": self.video_size[1], "img_h": self.video_size[0]},
        )
        self.normalize_transform = Normalize(
            args={"mean": 0.5, "std": 0.5},
        )
        if processor is not None:
            if isinstance(processor, DictConfig):
                processor = instantiate(processor)
            if not pretrained_norm_stats:
                if not is_training_set:
                    raise ValueError("pretrained_norm_stats must be provided for validation/test sets since we don't want to calculate stats on them.")
                if PartialState().is_main_process:
                    logger.info("Calculating dataset stats for normalization...")
                    dataset_stats = self.lerobot_dataset.get_dataset_stats(processor)
                    work_dir = misc.get_work_dir()
                    save_dataset_stats_to_json(dataset_stats, os.path.join(work_dir, "dataset_stats.json"))
                else:
                    dataset_stats = None
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    obj_list = [dataset_stats]
                    torch.distributed.broadcast_object_list(obj_list, src=0)
                    dataset_stats = obj_list[0]
            else:
                dataset_stats = load_dataset_stats_from_json(pretrained_norm_stats)
                logger.info(f"Using dataset stats: {pretrained_norm_stats}")
                if PartialState().is_main_process:
                    work_dir = misc.get_work_dir()
                    dst = os.path.join(work_dir, "dataset_stats.json")
                    # Skip self-overwrite: build_datasets passes
                    # work_dir/dataset_stats.json as pretrained_norm_stats for
                    # val, so src == dst and rewriting is a no-op. Worse, on
                    # GPFS the rewrite races with non-main ranks' reads (open
                    # 'w' truncates first → racing readers see empty file →
                    # JSONDecodeError).
                    if os.path.realpath(pretrained_norm_stats) != os.path.realpath(dst):
                        save_dataset_stats_to_json(dataset_stats, dst)

            processor.set_normalizer_from_stats(dataset_stats)
            self.lerobot_dataset.set_processor(processor)

        # Per-dataset camera intrinsics. Each underlying dataset_dir may ship its
        # own meta/camera_intrinsics.json, so we keep a list indexed by dataset
        # idx and look up per-sample at __getitem__ time. Available for any
        # layout-driven mode (the LayoutSpec defines per-cam K targets via slot
        # HW); legacy "horizontal"/"vertical" modes don't load K.
        self._per_dataset_intrinsics: list[Optional[torch.Tensor]] = []
        if self._layout is not None:
            dirs = list(dataset_dirs) if not isinstance(dataset_dirs, (str, os.PathLike)) else [dataset_dirs]
            for d in dirs:
                self._per_dataset_intrinsics.append(self._load_layout_intrinsics_for_dir(str(d)))
        # Cumulative frame bounds so we can map a flat sample index → dataset idx.
        self._dataset_frame_bounds: list[int] = []
        cumul = 0
        for d in getattr(self.lerobot_dataset.multi_dataset, "_datasets", []):
            cumul += d.num_frames
            self._dataset_frame_bounds.append(cumul)

        # ── depth stream + per-cam RGB geometry ──────────────────────────
        # Validate layout selection — depth dataset is layout-aware just like
        # RobotVideoDataset (legacy 'horizontal'/'vertical' modes lack
        # the per-cam slot geometry needed to build the RGB stream).
        if not is_layout_name(self.concat_multi_camera):
            raise ValueError(
                f"RobotVideoDataset requires `concat_multi_camera` to "
                f"name a registered layout (e.g. 'robotwin'); got "
                f"{self.concat_multi_camera!r}."
            )
        # RGB per-cam HW table from the layout's slot.src_hw, in cam_slots()
        # order. This is what _get uses for RGB resize. Note: distinct from
        # `self._per_cam_hw` which holds the DEPTH grid (from shape_meta.depth).
        self._rgb_per_cam_hw: list[tuple[str, tuple[int, int]]] = [
            (self._slot_key_map[s.key], s.src_hw)
            for s in self._layout.cam_slots()
        ]

        # ── Exterior-view augmentation ────────────────────────────────────────
        # Training-time view aug for multi-exterior rigs (two or more
        # exterior cams): with prob `exterior_view_aug_prob`, pick ONE of the
        # named exterior slots at random and repeat its view across all of them
        # — RGB (`per_cam`), depth (`per_cam_depth`), and intrinsics
        # (`camera_intrinsics`) duplicated in lock-step so the composite / DINO /
        # pointmap streams stay mutually consistent. The wrist (and any other
        # unnamed) slot is never touched. Default prob 0.0 = disabled =
        # byte-identical to before. Exterior slots are resolved once here as
        # positions in `layout.cam_slots()` order — the SAME order the pointmap
        # encoder uses to index `camera_intrinsics` (pointmap_encoder.py).
        self.exterior_view_aug_prob = float(exterior_view_aug_prob)
        self._ext_aug_slot_positions: list[int] = []
        self._ext_aug_slot_names: list[str] = []
        if self.exterior_view_aug_prob > 0.0:
            if isinstance(exterior_view_aug_keys, ListConfig):
                exterior_view_aug_keys = OmegaConf.to_container(
                    exterior_view_aug_keys, resolve=True
                )
            if not exterior_view_aug_keys or len(exterior_view_aug_keys) < 2:
                raise ValueError(
                    "exterior_view_aug_prob > 0 requires exterior_view_aug_keys "
                    "with >= 2 camera keys (the exterior slots to shuffle); got "
                    f"{exterior_view_aug_keys!r}"
                )
            slot_names = [self._slot_key_map[s.key] for s in self._layout.cam_slots()]
            for key in exterior_view_aug_keys:
                if key not in slot_names:
                    raise ValueError(
                        f"exterior_view_aug_keys entry {key!r} is not a camera in "
                        f"the layout slot map {slot_names}."
                    )
                self._ext_aug_slot_positions.append(slot_names.index(key))
                self._ext_aug_slot_names.append(key)
            logger.info(
                "RobotVideoDataset: exterior-view aug ON "
                f"(p={self.exterior_view_aug_prob}, exteriors={self._ext_aug_slot_names}, "
                f"slot_idxs={self._ext_aug_slot_positions})"
            )

        if depth_codec not in _DEPTH_VCODEC:
            raise ValueError(f"depth_codec must be one of {list(_DEPTH_VCODEC)}; got {depth_codec!r}")
        self.depth_codec = depth_codec
        self._depth_vcodec = _DEPTH_VCODEC[depth_codec]
        self._depth_ext = _DEPTH_EXT[depth_codec]
        self._depth_feat_prefix = f"observation.depth_{depth_codec}"

        if not self.has_depth:
            logger.info(
                "RobotVideoDataset: no shape_meta.depth block — depth "
                "decode skipped, `per_cam_depth` omitted. Pair with "
                "model.enable_pointmap=false."
            )

        # Synthetic zero cams: canonical cam names not present on disk. When
        # set (e.g., LIBERO has 2 physical cams but the model expects 3),
        # the named cams are fabricated with zero RGB / zero depth /
        # identity K so per_cam, per_cam_depth, and camera_intrinsics
        # always carry the canonical 3 cams. Disk cams (from
        # shape_meta.images) ∪ synthetic_zero_cams must equal the canonical
        # set defined by _PER_CAM_HW.
        self.synthetic_zero_cams: tuple[str, ...] = tuple(synthetic_zero_cams or [])
        if self.synthetic_zero_cams:
            canonical = tuple(name for name, _ in _PER_CAM_HW)
            bad = [c for c in self.synthetic_zero_cams if c not in canonical]
            if bad:
                raise ValueError(
                    f"synthetic_zero_cams contains non-canonical names {bad}; "
                    f"must be a subset of {list(canonical)}"
                )
            overlap = [c for c in self.synthetic_zero_cams if c in self._cams]
            if overlap:
                raise ValueError(
                    f"synthetic_zero_cams overlaps disk cams from shape_meta.images: {overlap}"
                )
            union = set(self._cams) | set(self.synthetic_zero_cams)
            if union != set(canonical):
                raise ValueError(
                    f"shape_meta.images disk cams {list(self._cams)} ∪ synthetic_zero_cams "
                    f"{list(self.synthetic_zero_cams)} must equal canonical {list(canonical)}; "
                    f"got {sorted(union)}"
                )

        # Depth is always decoded at the 9 RGB-anchor frames
        # (`video_sample_indices`), the same temporal density RGB uses:
        # 9 depth frames → WAN VAE → 3 pointmap latents, aligned 1:1 with the
        # 3 RGB video latents.

    # ---------------------------------------------------------------------
    # Override: load K and rescale to the per-cam target size declared in
    # self._per_cam_hw (the depth grid, or the RGB grid with no depth block).
    # When target equals native
    # this is an identity rescale (sx = sy = 1). When target differs, K
    # scales in lock-step with the resize in _get so RGB and depth
    # unprojection stay consistent.
    # ---------------------------------------------------------------------
    def _load_layout_intrinsics_for_dir(self, dataset_dir: str) -> Optional[torch.Tensor]:
        json_path = os.path.join(dataset_dir, "meta", "camera_intrinsics.json")
        if not os.path.exists(json_path):
            logger.info(f"No camera_intrinsics.json at {json_path}")
            return None
        with open(json_path, "r") as f:
            raw = json.load(f)
        Ks = []
        for cam_name, (h_target, w_target) in self._per_cam_hw:
            if cam_name not in raw:
                raise KeyError(f"{json_path} missing '{cam_name}' (found: {list(raw)})")
            e = raw[cam_name]
            raw_fx, raw_fy = float(e["fx"]), float(e["fy"])
            raw_cx, raw_cy = float(e["cx"]), float(e["cy"])
            raw_W, raw_H   = float(e["width"]), float(e["height"])
            # Endpoint convention: matches openpi's
            # ``scale_intrinsics(..., convention="endpoint")`` and pairs with
            # the linspace-endpoint nearest depth resize at decode time, so
            # the pixel grid and K stay aligned end-to-end. (The builder
            # writes K rescaled with the same convention.)
            sx = (w_target - 1) / (raw_W - 1) if raw_W > 1 else 1.0
            sy = (h_target - 1) / (raw_H - 1) if raw_H > 1 else 1.0
            Ks.append(torch.tensor(
                [[raw_fx * sx, 0.0,         raw_cx * sx],
                 [0.0,         raw_fy * sy, raw_cy * sy],
                 [0.0,         0.0,         1.0       ]], dtype=torch.float32,
            ))
        K_stack = torch.stack(Ks, dim=0)  # [num_cams, 3, 3]
        logger.info(
            f"Loaded camera intrinsics from {json_path} (target per-cam sizes "
            f"{self._per_cam_hw})"
        )
        return K_stack

    # Back-compat alias: old call sites (e.g. RobotVideoDataset.
    # _load_robotwin_intrinsics_for_dir override) keep working.
    def _load_robotwin_intrinsics_for_dir(self, dataset_dir: str) -> Optional[torch.Tensor]:
        return self._load_layout_intrinsics_for_dir(dataset_dir)

    def _dataset_idx_for(self, flat_idx: int) -> int:
        """Return which underlying dataset owns the given flat frame index."""
        for i, bound in enumerate(self._dataset_frame_bounds):
            if flat_idx < bound:
                return i
        raise IndexError(
            f"flat_idx={flat_idx} outside dataset bounds {self._dataset_frame_bounds}"
        )

    def __len__(self):
        return len(self.lerobot_dataset)

    # ---------------------------------------------------------------------
    # Override _get: output per-cam at native resolution + add depth stream.
    # Structure mirrors RobotVideoDataset._get; RGB path simplified (no
    # resize / composite assembly), depth path added.
    # ---------------------------------------------------------------------
    def _get(self, idx):
        # DLTIMING harness — env-gated, no-op when FLEXPI_DL_TIMING != "1".
        _dl_t_start = _dl_time.perf_counter() if _dl_timing_enabled() else None
        _dl_timings = {} if _dl_t_start is not None else None
        _dl_mark = (lambda: _dl_time.perf_counter()) if _dl_t_start is not None else (lambda: 0.0)

        sample_idx = idx
        sample = None
        _t = _dl_mark()
        for attempt in range(self.max_padding_retry + 1):
            sample = self.lerobot_dataset[sample_idx]
            if not self.skip_padding_as_possible:
                break
            has_pad = (
                bool(sample["action_is_pad"].any().item())
                or bool(sample["image_is_pad"].any().item())
                or bool(sample["proprio_is_pad"].any().item())
            )
            if not has_pad or attempt >= self.max_padding_retry:
                break
            sample_idx = np.random.randint(len(self.lerobot_dataset))
        if _dl_timings is not None:
            _dl_timings["lerobot_fetch_ms"] = int((_dl_mark() - _t) * 1000)
            _t = _dl_mark()

        # --- RGB: take pixel_values, subsample to video frames, normalize --------
        # Mirrors RobotVideoDataset._get (L140-172), with ONE difference:
        # we skip the per-cam slot-size resize (256×320 / 224×224 / 224×224),
        # since our input is already at the desired native resolution.
        image_is_pad = sample["image_is_pad"]
        _expected_T = len(self.video_sample_indices)
        _rgb_slot_hw = {name: hw for name, hw in self._rgb_per_cam_hw}

        # Heterogeneous per-cam path (audit-fix #5): the processor emitted
        # `per_cam_rgb` as a dict because the per-cam transforms produced
        # different (H, W) per cam. Use it directly — its sizes already match
        # `_rgb_per_cam_hw` so no second resize is needed.
        per_cam_rgb_in = sample.get("per_cam_rgb", None)
        if isinstance(per_cam_rgb_in, dict):
            T_video = None
            per_cam = {}
            for cam_name, _depth_hw in self._per_cam_hw:
                if cam_name not in per_cam_rgb_in:
                    raise KeyError(
                        f"per_cam_rgb missing '{cam_name}' (got: "
                        f"{sorted(per_cam_rgb_in)})"
                    )
                t = per_cam_rgb_in[cam_name]  # [T, C, H, W]
                if t.ndim != 4:
                    raise ValueError(
                        f"per_cam_rgb['{cam_name}'] expected 4D [T,C,H,W], "
                        f"got shape {tuple(t.shape)}"
                    )
                if t.shape[0] != _expected_T:
                    t = t[self.video_sample_indices, :, :, :]
                if T_video is None:
                    T_video = t.shape[0]
                elif t.shape[0] != T_video:
                    raise ValueError(
                        f"per_cam_rgb cams disagree on T_video: got "
                        f"{T_video} and {t.shape[0]} for '{cam_name}'"
                    )
                h_cam, w_cam = _rgb_slot_hw[cam_name]
                if (t.shape[-2], t.shape[-1]) != (h_cam, w_cam):
                    raise ValueError(
                        f"per_cam_rgb['{cam_name}'] is {(t.shape[-2], t.shape[-1])} "
                        f"but layout source HW is {(h_cam, w_cam)}. The per-cam "
                        f"transforms in the data config must target the layout's "
                        f"Slot.src_hw exactly."
                    )
                t = self.normalize_transform(t)  # [-1, 1]
                per_cam[cam_name] = t.permute(1, 0, 2, 3).contiguous()
            num_cameras = len(self._per_cam_hw)
        else:
            video = sample["pixel_values"]  # [num_cameras, T, C, H, W] or [T, C, H, W]
            # Idempotent w.r.t. BaseLerobotDataset.image_sample_stride: when the
            # base subsamples RGB timestamps at decode time, `video` arrives at
            # the expected T_video and we skip the slice. Legacy path (full
            # obs_size in) still subsamples here.
            if video.ndim == 5:
                if video.shape[1] != _expected_T:
                    video = video[:, self.video_sample_indices, :, :, :]
                num_cameras, T_video, C, H, W = video.shape
            else:
                assert video.ndim == 4, f"Expected [T, C, H, W], got {video.shape}"
                if video.shape[0] != _expected_T:
                    video = video[self.video_sample_indices, :, :, :]
                T_video, C, H, W = video.shape
                num_cameras = 1

            video = video.view(num_cameras, T_video, C, H, W)
            if num_cameras != len(self._per_cam_hw):
                raise ValueError(
                    f"Camera count mismatch: pixel_values has {num_cameras} cams but "
                    f"shape_meta.images declares {len(self._per_cam_hw)} "
                    f"({[n for n, _ in self._per_cam_hw]})"
                )

            # Per-camera RGB resize to the layout's per-cam SOURCE sizes (Slot.src_hw)
            # via self._rgb_per_cam_hw. Iterate self._per_cam_hw (disk cams, in
            # shape_meta.images order) rather than self._rgb_per_cam_hw so the
            # loop matches actual pixel_values cam count when synthetic_zero_cams
            # is non-empty — synthetic cams are fabricated separately below.
            # Independent of shape_meta.depth: depth can live on its own per-cam
            # grid for the pointmap encoder, while RGB uses the layout-prescribed
            # source sizes for the composite path.
            per_cam = {}
            for cam_idx, (cam_name, _depth_hw) in enumerate(self._per_cam_hw):
                h_cam, w_cam = _rgb_slot_hw[cam_name]
                resized = transforms_F.resize(
                    video[cam_idx],
                    size=[h_cam, w_cam],
                    interpolation=transforms_F.InterpolationMode.BILINEAR,
                    antialias=True,
                )  # [T_video, C, h_cam, w_cam] in [0, 1]
                resized = self.normalize_transform(resized)  # [-1, 1]
                per_cam[cam_name] = resized.permute(1, 0, 2, 3).contiguous()

        if image_is_pad.shape[0] != _expected_T:
            image_is_pad = image_is_pad[self.video_sample_indices]

        # Fabricate zero RGB tensors for synthetic cams (LIBERO etc. with
        # fewer physical cams than the canonical 3-cam model expects).
        # Fill with -1.0 (NOT 0.0): per_cam tensors live in normalize_transform's
        # [-1, 1] range, so the "no-signal" pixel is -1 (raw black before norm),
        # not 0 (which un-normalizes to 0.5 = mid-gray and shows up as a gray
        # tile in eval-video visualizations).
        for cam_name in self.synthetic_zero_cams:
            h_cam, w_cam = _rgb_slot_hw[cam_name]
            ref = next(iter(per_cam.values()))
            per_cam[cam_name] = torch.full(
                (3, T_video, h_cam, w_cam), fill_value=-1.0, dtype=ref.dtype
            )
        if _dl_timings is not None:
            _dl_timings["rgb_resize_ms"] = int((_dl_mark() - _t) * 1000)
            _t = _dl_mark()

        # --- Depth: decode only the video-frame timestamps we need ---------------
        # No shape_meta.depth block → RGB-only sample, matching what the
        # pre-depth RobotVideoDataset used to produce (plus our
        # camera_intrinsics). Skips episode-lookup + timestamp reconstruction
        # + pyav decode for all three cameras.
        per_cam_depth = None
        ds_idx = self._dataset_idx_for(int(sample_idx))

        if not self.has_depth:
            pass   # depth undeclared; fall through to the data dict below
        else:
            # The FlexPiProcessor strips episode_index / timestamp out of the
            # sample dict, but retains the global frame index as sample["idx"].
            # We recover the episode + in-episode frame index from the
            # multi-dataset's episode_data_index boundaries (constructed in
            # BaseLerobotDataset.__init__).
            per_cam_depth = self._decode_per_cam_depth(sample, ds_idx)
            # Fabricate zero depth for synthetic cams. Mirror the first real
            # depth cam's shape — pointmap encoder tolerates per-cam HW differences.
            for cam_name in self.synthetic_zero_cams:
                ref = next(iter(per_cam_depth.values()))
                per_cam_depth[cam_name] = torch.zeros_like(ref)
        if _dl_timings is not None:
            _dl_timings["depth_total_ms"] = int((_dl_mark() - _t) * 1000)
            _t = _dl_mark()

        # --- Action / proprio / text — identical to parent ----------------------
        action = sample["action"]
        proprio = sample["proprio"][:-1, :]
        # Use the first cam's tensor as the reference for T_video. (Was
        # hard-coded to "cam_high" — generalized to whichever key the layout
        # binds to its first cam slot.)
        first_cam_key = self._rgb_per_cam_hw[0][0]
        head_T = per_cam[first_cam_key].shape[1]   # [C, T_video, H, W] → dim 1
        if head_T <= 1:
            raise ValueError(f"video must have >=2 frames, got T_video={head_T}")
        if action.shape[0] % (head_T - 1) != 0:
            raise ValueError(
                f"action horizon {action.shape[0]} not divisible by video transitions {head_T - 1}"
            )

        task = sample["instruction"]
        if self.override_instruction is not None:
            task = self.override_instruction
        instruction = DEFAULT_PROMPT.format(task=task)
        # Optional layout-describing prefix for T5. Off by default.
        if (
            self._composite_layout_text_prefix
            and self._layout is not None
            and self._layout.text_prefix
        ):
            instruction = f"{self._layout.text_prefix} {instruction}"
        context, context_mask = self._get_cached_text_context(instruction)
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        if _dl_timings is not None:
            _dl_timings["action_text_ms"] = int((_dl_mark() - _t) * 1000)

        data = {
            "per_cam": per_cam,                  # dict[cam -> [C, T_video, H, W] float32 in [-1, 1]]
            "action": action,
            "proprio": proprio,
            "prompt": instruction,
            "context": context,
            "context_mask": context_mask,
            "image_is_pad": image_is_pad,
            "action_is_pad": sample["action_is_pad"],
            "proprio_is_pad": sample["proprio_is_pad"],
        }
        if per_cam_depth is not None:
            # dict[cam -> [T_video, H, W] uint16 mm]
            data["per_cam_depth"] = per_cam_depth

        if self._per_dataset_intrinsics:
            K = self._per_dataset_intrinsics[ds_idx]
            if K is not None:
                if self.synthetic_zero_cams:
                    # K is [n_disk, 3, 3] in disk order. Extend to the full cam
                    # set with identity K for synthetic cams (their depth is
                    # all-zero, so K is numerically irrelevant). The
                    # PointmapEncoder indexes camera_intrinsics BY SLOT (cam_idx
                    # over layout.cam_slots()), so stack in SLOT order under the
                    # resolved slot_key_map. For the default map this is the
                    # canonical (cam_high, cam_left_wrist, cam_right_wrist) order
                    # — byte-identical to before; a non-default map (e.g. the
                    # wrist-top swap) reorders K to stay aligned with the slots.
                    slot_order = [
                        self._slot_key_map[s.key] for s in self._layout.cam_slots()
                    ]
                    disk_to_idx = {n: i for i, n in enumerate(self._cams)}
                    stack = []
                    for name in slot_order:
                        if name in disk_to_idx:
                            stack.append(K[disk_to_idx[name]])
                        else:
                            stack.append(torch.eye(3, dtype=K.dtype))
                    K = torch.stack(stack, dim=0)
                data["camera_intrinsics"] = K   # [num_cams, 3, 3] float32 at native resolution

        # Exterior-view augmentation (in place on `data`; no-op when disabled).
        self._maybe_augment_exterior_views(data)

        if _dl_t_start is not None:
            # Compute episode/frame for context — only used in the log line.
            ep_idx_str, frame_in_ep_str = "?", "?"
            try:
                if per_cam_depth is not None and hasattr(self, "_last_episode_info"):
                    ep_idx_str, frame_in_ep_str = self._last_episode_info
            except Exception:
                pass
            total_ms = int((_dl_time.perf_counter() - _dl_t_start) * 1000)
            _dl_emit({
                "stage": "ds_get",
                "worker": _dl_worker_id(), "pid": os.getpid(),
                "idx": int(sample_idx),
                "ep": ep_idx_str, "frame": frame_in_ep_str,
                "total_ms": total_ms,
                **{k: v for k, v in _dl_timings.items()},
            })
        return data

    # ---------------------------------------------------------------------
    # Exterior-view augmentation. With prob `exterior_view_aug_prob`, pick ONE
    # exterior slot at random and copy its view into every other exterior slot —
    # RGB (`per_cam`), depth (`per_cam_depth`), and intrinsics
    # (`camera_intrinsics`) in lock-step, so the composite / DINO / pointmap
    # streams stay mutually consistent (the pointmap encoder pairs
    # per_cam_depth[slot_name] with camera_intrinsics[slot_idx], so both must
    # move together). The wrist slot is never touched. No-op when disabled or
    # when the Bernoulli draw misses. See __init__ for the resolved slot table.
    # ---------------------------------------------------------------------
    def _maybe_augment_exterior_views(self, data: dict) -> None:
        if self.exterior_view_aug_prob <= 0.0:
            return
        if np.random.random() >= self.exterior_view_aug_prob:
            return
        src = int(np.random.randint(len(self._ext_aug_slot_positions)))
        src_name = self._ext_aug_slot_names[src]
        src_slot = self._ext_aug_slot_positions[src]

        per_cam = data["per_cam"]
        per_cam_depth = data.get("per_cam_depth")
        K = data.get("camera_intrinsics")
        # `camera_intrinsics` may alias the cached per-dataset K tensor
        # (self._per_dataset_intrinsics[ds_idx]); clone before the in-place slot
        # copy so we never corrupt the shared cache for later samples.
        if K is not None:
            K = K.clone()
            data["camera_intrinsics"] = K

        src_rgb = per_cam[src_name]
        src_depth = per_cam_depth[src_name] if per_cam_depth is not None else None
        for pos, tgt_name in enumerate(self._ext_aug_slot_names):
            if pos == src:
                continue
            per_cam[tgt_name] = src_rgb.clone()
            if per_cam_depth is not None:
                per_cam_depth[tgt_name] = src_depth.clone()
            if K is not None:
                K[self._ext_aug_slot_positions[pos]] = K[src_slot]

    # ---------------------------------------------------------------------
    # Depth-decoding helper. Factored out of _get so the no-depth-block
    # ablation doesn't pay any of its cost (episode lookup, path build,
    # 3× pyav decode).
    # ---------------------------------------------------------------------
    def _decode_per_cam_depth(self, sample, ds_idx: int) -> dict[str, torch.Tensor]:
        underlying_lerobot_ds = self.lerobot_dataset.multi_dataset._datasets[ds_idx]
        dataset_root = Path(underlying_lerobot_ds.root)
        meta_info = underlying_lerobot_ds.meta.info
        chunks_size = meta_info["chunks_size"]
        fps = int(meta_info["fps"])
        # Superseded by the round-to-ms fix at query_ts below — kept commented
        # as an alternative if a future dataset breaks the ms-quantization
        # assumption.
        # tolerance_s = max(underlying_lerobot_ds.tolerance_s, 1.5e-3)
        tolerance_s = underlying_lerobot_ds.tolerance_s

        global_idx = int(
            sample["idx"].item() if isinstance(sample.get("idx"), torch.Tensor)
            else sample["idx"]
        )
        # Translate combined frame index → per-underlying-dataset frame index.
        # `BaseLerobotDataset.episode_data_index` is concatenated across all
        # dataset_dirs with cumulative offsets, but `underlying_lerobot_ds`
        # (and its `.episodes` list, used for ep_list[ep_idx_local] below) is
        # per-dataset. Mixing the two indexes the wrong list when a sample
        # comes from a non-first dataset_dir (IndexError on multi-dataset).
        prev_cum = 0 if ds_idx == 0 else self._dataset_frame_bounds[ds_idx - 1]
        local_idx = global_idx - prev_cum
        ep_data_index = underlying_lerobot_ds.episode_data_index
        from_arr = ep_data_index["from"]
        to_arr = ep_data_index["to"]
        ep_idx_local = None
        for e in range(len(from_arr)):
            if int(from_arr[e].item()) <= local_idx < int(to_arr[e].item()):
                ep_idx_local = e
                break
        if ep_idx_local is None:
            raise RuntimeError(
                f"Couldn't locate episode for ds_idx={ds_idx} local_idx={local_idx} "
                f"(global_idx={global_idx}, prev_cum={prev_cum}, range boundaries: "
                f"{from_arr.tolist()} → {to_arr.tolist()})"
            )
        ep_list = list(underlying_lerobot_ds.episodes) if underlying_lerobot_ds.episodes is not None \
                  else list(range(len(from_arr)))
        ep_idx = int(ep_list[ep_idx_local])
        frame_in_ep = local_idx - int(from_arr[ep_idx_local].item())
        anchor_ts = frame_in_ep / float(fps)
        # Stash for the parent _get's DLTIMING emit. No-op when timing disabled.
        if _dl_timing_enabled():
            self._last_episode_info = (ep_idx, frame_in_ep)
        # Clamp query ts to episode bounds so end-of-episode windows pad-by-repeat
        # the same way LeRobot does for RGB. `image_is_pad` flags the padded frames.
        ep_length = int(to_arr[ep_idx_local].item()) - int(from_arr[ep_idx_local].item())
        max_valid_ts = (ep_length - 1) / float(fps)

        # Reuse LeRobot's already-computed delta list for RGB — same deltas for
        # every cam, shared timeline between RGB and depth streams. Depth is
        # decoded at the 9 RGB-anchor positions (`video_sample_indices`) so the
        # pointmap stream stays 1:1 with the RGB stream the model trains on.
        rgb_key = f"observation.images.{self._cams[0]}"
        all_deltas = underlying_lerobot_ds.delta_timestamps[rgb_key]
        # If BaseLerobotDataset.image_sample_stride was > 1, `all_deltas` is
        # already the subsampled length (one entry per `video_sample_indices`
        # position). Iterate by position in that case; otherwise iterate by
        # `video_sample_indices` (legacy 33-entry full list).
        if len(all_deltas) == len(self.video_sample_indices):
            _iter_indices = list(range(len(all_deltas)))
        else:
            _iter_indices = self.video_sample_indices
        # Round to ms: depth MKVs use the container's 1ms time_base, so PTS for
        # any fps that doesn't divide 1000 (e.g. 30) drifts ~0.3-0.5ms from the
        # true 1/fps grid and exceeds the RGB tolerance_s (0.1ms).
        query_ts = [round(min(anchor_ts + all_deltas[i], max_valid_ts), 3)
                    for i in _iter_indices]

        # info.json.video_path is hardcoded to '.mp4'; swap in the depth ext
        # recorded per-feature by the converter.
        video_path_template = meta_info["video_path"]
        if not video_path_template.endswith(".mp4"):
            raise ValueError(
                f"Unexpected video_path template {video_path_template!r}; expected '.mp4' suffix."
            )
        ep_chunk = ep_idx // chunks_size

        # DLTIMING per-cam breakdown — env-gated; no-op when disabled.
        _dl_t_d0 = _dl_time.perf_counter() if _dl_timing_enabled() else None
        _dl_per_cam_ms: dict[str, int] = {}

        per_cam_depth: dict[str, torch.Tensor] = {}
        for cam_name, (h_cam, w_cam) in self._per_cam_hw:
            _dl_t_cam = _dl_time.perf_counter() if _dl_t_d0 is not None else None
            feat_key = f"{self._depth_feat_prefix}.{cam_name}"
            if feat_key not in meta_info["features"]:
                raise KeyError(
                    f"Missing depth feature '{feat_key}' in info.json — was the dataset "
                    f"converted with --depth-codec {self.depth_codec}?"
                )
            feat_info = meta_info["features"][feat_key].get("info", {})
            expected_ext = feat_info.get("depth_ext", self._depth_ext)
            relative = video_path_template.format(
                episode_chunk=ep_chunk, video_key=feat_key, episode_index=ep_idx,
            )
            video_path = dataset_root / (relative[:-4] + expected_ext)
            depth = decode_depth_frames(
                video_path=video_path, timestamps=query_ts,
                tolerance_s=max(tolerance_s, _DEPTH_PTS_TOLERANCE_S),
                vcodec=self._depth_vcodec,
            )  # [T_video, H_native, W_native] uint16 mm

            # Linspace-endpoint nearest sampler (matches
            # ``openpi.policies.depth_codec.resize_depth_nearest``). Pairs with
            # the endpoint K rescale above so pixel grid + K stay aligned, AND
            # with the builder's _resize_depth_nearest_endpoint at write time.
            # No-op when native equals target.
            if depth.shape[-2:] != (h_cam, w_cam):
                src_h, src_w = depth.shape[-2:]
                ys = torch.linspace(0, src_h - 1, h_cam, device=depth.device).round().long()
                xs = torch.linspace(0, src_w - 1, w_cam, device=depth.device).round().long()
                # PyTorch CPU has no index_cpu kernel for uint16, so advanced
                # indexing on a uint16 tensor raises at runtime. Round-trip
                # through int32 (lossless: uint16 fits in int32) for the gather,
                # then cast back to uint16 mm.
                if depth.dtype == torch.uint16:
                    depth = depth.to(torch.int32)[..., ys[:, None], xs[None, :]].to(torch.uint16)
                else:
                    depth = depth[..., ys[:, None], xs[None, :]]
            per_cam_depth[cam_name] = depth
            if _dl_t_cam is not None:
                _dl_per_cam_ms[cam_name] = int((_dl_time.perf_counter() - _dl_t_cam) * 1000)

        if _dl_t_d0 is not None:
            total_ms = int((_dl_time.perf_counter() - _dl_t_d0) * 1000)
            _dl_emit({
                "stage": "depth_per_cam",
                "worker": _dl_worker_id(), "pid": os.getpid(),
                "ep": ep_idx, "frame": frame_in_ep,
                "total_ms": total_ms,
                **{f"{n}_ms": v for n, v in _dl_per_cam_ms.items()},
            })
        return per_cam_depth

    def _get_cached_text_context(self, prompt: str):
        if self.text_embedding_cache_dir is None:
            raise ValueError("text_embedding_cache_dir is not set.")
        cache_dir = self.text_embedding_cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        hashed = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = os.path.join(cache_dir, f"{hashed}.t5_len{self.context_len}.wan22ti2v5b.pt")
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing text embedding cache: {cache_path}. "
                "Run scripts/precompute_text_embeds.py first."
            )
        payload = torch.load(cache_path, map_location="cpu")
        context = payload["context"]
        context_mask = payload["mask"].bool()
        if context.ndim != 2:
            raise ValueError(
                f"Cached `context` must be 2D [L, D], got shape {tuple(context.shape)} in {cache_path}"
            )
        if context_mask.ndim != 1:
            raise ValueError(
                f"Cached `mask` must be 1D [L], got shape {tuple(context_mask.shape)} in {cache_path}"
            )
        if context.shape[0] != self.context_len:
            raise ValueError(
                f"Cached context_len mismatch: expected {self.context_len}, got {context.shape[0]} in {cache_path}"
            )
        if context_mask.shape[0] != self.context_len:
            raise ValueError(
                f"Cached mask_len mismatch: expected {self.context_len}, got {context_mask.shape[0]} in {cache_path}"
            )

        return context, context_mask

    def __getitem__(self, idx):
        try:
            return self._get(idx)
        except Exception as e:
            print(f"Error processing sample idx {idx}: {e}. Returning a random sample instead.")
            print(traceback.format_exc())
            return self._get(np.random.randint(len(self)))



# =============================================================================
# Standalone smoke test
#
# From the FlexPi repo root (so hydra finds configs/):
#
#   # pulls one sample + 10 random probes; depth on/off follows the data config
#   python -m flexpi.datasets.lerobot.robot_video_dataset
#
#   # depth off = a data config with no shape_meta.depth block (sample dict
#   # becomes a RobotVideoDataset superset)
#   python -m flexpi.datasets.lerobot.robot_video_dataset \
#       --data-cfg <a config without shape_meta.depth>
#
#   # different codec / config / more probes
#   python -m flexpi.datasets.lerobot.robot_video_dataset \
#       --data-cfg robotwin --depth-codec x264rgb --n-samples 50
#
# Prints sample-dict shape anatomy, verifies invariants, runs a stability loop
# and a pointmap-unprojection sanity check. No training side effects.
# =============================================================================
if __name__ == "__main__":
    import argparse
    import os
    import time as _time

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-cfg",  default="robotwin",
                        help="hydra data config name under configs/data/")
    parser.add_argument("--model-cfg", default="flexpi",
                        help="hydra model config (needed for T5 encoder plumbing)")
    parser.add_argument("--task-cfg",  default="robotwin_uncond_3cam_384_1e-4",
                        help="hydra task config (sets batch_size etc.; any valid task works)")
    parser.add_argument("--depth-codec", default=None, choices=[None, "ffv1", "x264rgb"],
                        help="override data config's depth_codec")
    parser.add_argument("--n-samples", type=int, default=10,
                        help="random samples for stability/timing probe")
    parser.add_argument("--config-dir", default=None,
                        help="override configs/ dir (default: ./configs relative to CWD)")
    parser.add_argument("--override", action="append", default=[],
                        help="Hydra override forwarded to compose() (repeatable). "
                             "Example: --override 'data.train.dataset_dirs=[./data/agibot_beta/task_327]'")
    args = parser.parse_args()

    cfg_dir = args.config_dir or os.path.abspath("configs")
    if not os.path.isdir(cfg_dir):
        raise SystemExit(f"configs dir not found: {cfg_dir}. Run from FlexPi repo root.")

    overrides = [
        f"data={args.data_cfg}",
        f"model={args.model_cfg}",
        f"task={args.task_cfg}",
    ]
    if args.depth_codec is not None:
        overrides.append(f"++data.train.depth_codec={args.depth_codec}")
    overrides.extend(args.override)

    with initialize_config_dir(config_dir=cfg_dir, version_base="1.3"):
        cfg = compose("train", overrides=overrides)

    ds = instantiate(cfg.data.train)
    print(f"=== {type(ds).__name__} ===")
    print(f"  len(ds)        = {len(ds):,}")
    print(f"  has_depth      = {ds.has_depth}  (from shape_meta.depth presence)")
    print(f"  depth_codec    = {ds.depth_codec}")
    print(f"  per_cam_hw     = {ds._per_cam_hw}")
    print(f"  dataset_dirs   = {list(cfg.data.train.dataset_dirs)}")

    print("\n--- first sample anatomy ---")
    s = ds[0]
    for k in sorted(s.keys()):
        v = s[k]
        if isinstance(v, dict):
            print(f"  {k:22s} dict[{len(v)}]:")
            for sk in sorted(v):
                sv = v[sk]
                dt = sv.dtype
                rng = (f"  min={float(sv.min()):.3f} max={float(sv.max()):.3f}"
                       if dt.is_floating_point
                       else f"  min={int(sv.int().min())} max={int(sv.int().max())}")
                print(f"    {sk:22s} {str(dt):14s} shape={tuple(sv.shape)}{rng}")
        elif hasattr(v, "shape"):
            dt = v.dtype
            if dt.is_floating_point:
                rng = f"  min={float(v.min()):.3f} max={float(v.max()):.3f}"
            elif dt == torch.bool:
                rng = f"  any={bool(v.any())}"
            else:
                rng = f"  min={int(v.min())} max={int(v.max())}"
            print(f"  {k:22s} {str(dt):14s} shape={str(tuple(v.shape)):30s}{rng}")
        elif isinstance(v, str):
            print(f"  {k:22s} str  {v[:60]!r}")

    # Invariants
    assert isinstance(s["per_cam"], dict)
    if ds.has_depth:
        assert "per_cam_depth" in s and isinstance(s["per_cam_depth"], dict)
    else:
        assert "per_cam_depth" not in s, "per_cam_depth should be absent with no shape_meta.depth"
    print("\n  ✓ sample-dict invariants match the shape_meta.depth declaration")

    # Pointmap unprojection check (only if depth is on)
    if ds.has_depth:
        print("\n--- pointmap unprojection sanity ---")
        K_all = s["camera_intrinsics"]
        for cam_idx, (cam_name, (h_cam, w_cam)) in enumerate(ds._per_cam_hw):
            depth_m = s["per_cam_depth"][cam_name].float() / 1000.0
            z = depth_m[0]
            valid = z > 0.05
            if not bool(valid.any()):
                print(f"  {cam_name}: no valid depth in first frame (all < 0.05 m)")
                continue
            # Build per-cam (u, v) — sizes can differ across cams (e.g. YAM head 256x320,
            # wrists 224x224), so the meshgrid must follow the camera's own HW.
            v_grid, u_grid = torch.meshgrid(
                torch.arange(h_cam).float(),
                torch.arange(w_cam).float(),
                indexing="ij",
            )
            K = K_all[cam_idx]
            x = (u_grid - K[0, 2]) / K[0, 0] * z
            y = (v_grid - K[1, 2]) / K[1, 1] * z
            print(f"  {cam_name}: valid={int(valid.sum()):>6d}/{valid.numel()}  "
                  f"x∈[{x[valid].min():.2f}, {x[valid].max():.2f}] m  "
                  f"z∈[{z[valid].min():.2f}, {z[valid].max():.2f}] m")

    # Stability / timing probe
    print(f"\n--- {args.n_samples} random samples ---")
    rng = np.random.default_rng(0)
    t0 = _time.time()
    failed = 0
    for _ in range(args.n_samples):
        try:
            _ = ds[int(rng.integers(len(ds)))]
        except Exception as e:
            failed += 1
            print(f"  FAIL: {type(e).__name__}: {str(e)[:120]}")
    dt = _time.time() - t0
    print(f"  total {dt:.1f}s  ({dt * 1000 / args.n_samples:.0f} ms/sample, "
          f"{failed}/{args.n_samples} failed)")

    print("\n=== SMOKE TEST DONE ===")

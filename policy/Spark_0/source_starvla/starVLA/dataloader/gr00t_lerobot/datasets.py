# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
In this file, we define 3 types of datasets:
1. LeRobotSingleDataset: a single dataset for a given embodiment tag
2. LeRobotMixtureDataset: a mixture of datasets for a given list of embodiment tags
3. CachedLeRobotSingleDataset: a single dataset for a given embodiment tag,
                                with caching for the video frames

See `scripts/load_dataset.py` for examples on how to use these datasets.
"""
import os
import hashlib
import math
import time
import uuid
import io
import re
import json, torch
import copy
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Sequence
import os, random
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torch.distributed as dist

from starVLA.dataloader.gr00t_lerobot.video import get_all_frames, get_frames_by_timestamps

from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from starVLA.dataloader.gr00t_lerobot.schema import (
    DatasetMetadata,
    DatasetStatisticalValues,
    LeRobotModalityMetadata,
    LeRobotStateActionMetadata,
)
from starVLA.dataloader.gr00t_lerobot.streaming_stats import (
    _CollectBudget,
    _resolve_stats_mode,
    _StreamingStatsBuilder,
)
from starVLA.dataloader.gr00t_lerobot.tactile_pressure_stats import (
    calculate_tactile_pressure_statistics,
    tactile_pressure_stream_plan,
)
from starVLA.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionTransform

from functools import partial
from typing import Tuple, List
import pickle
import gc

# LeRobot v2.0 dataset file names 
LE_ROBOT_MODALITY_FILENAME = "meta/modality.json"
LE_ROBOT_EPISODE_FILENAME = "meta/episodes.jsonl"
LE_ROBOT_TASKS_FILENAME = "meta/tasks.jsonl"
LE_ROBOT_INFO_FILENAME = "meta/info.json"
LE_ROBOT_STATS_FILENAME = "meta/stats_xpolicy.json"
LE_ROBOT_TACTILE_PRESSURE_STATS_FILENAME = "meta/stats_xpolicy_tactile_pressure.json"
LE_ROBOT_DATA_FILENAME = "data/*/*.parquet"
LE_ROBOT_STEPS_FILENAME = "meta/steps.pkl"
LE_ROBOT_STATS_FORMAT_VERSION = 2
EPSILON = 5e-4

#  LeRobot v3.0 dataset file names 
LE_ROBOT3_TASKS_FILENAME = "meta/tasks.parquet"
LE_ROBOT3_EPISODE_FILENAME = "meta/episodes/*/*.parquet"

# Tactile streams (Spark-0 local addition). Packing is opt-in
# (`datasets.vla_data.pack_tactile`). Unset / false → sample dict has no
# tactile keys (the V-only / pre-tactile path) even if modality.json declares
# a `tactile` block; parquet columns are still skipped in pandas. The model
# resolves packed keys by default (starVLA/model/modules/tactile/batch_packing.py).
TACTILE_REGION_PRESSURE_KEY = "region_pressure"
TACTILE_TAXEL_PRESSURE_KEY = "taxel_pressure_3d"
TACTILE_TAXEL_VALID_KEY = "taxel_valid_mask"
# Head-camera HxW contact heatmap (tactile_align primary target). Not a
# per-hand stream: modality.json ``tactile.2d`` is a single parquet column.
TACTILE_HEATMAP_KEY = "tactile_2d"
# Next H glove frames of region force (variant A future-contact target).
# Exported next to the history window as its own column, so packing costs
# ONE row read: modality.json ``tactile.region_pressure_vector_future``,
# cell ``[future_frames, regions]`` with slot 0 nearest to the observation
# (EgoPro writes 50 frames at 60 Hz). Sliced to
# ``datasets.vla_data.tactile_future_horizon`` (unset / 0 = not packed).
TACTILE_REGION_FUTURE_STREAM = "region_pressure_vector_future"
TACTILE_REGION_FUTURE_KEY = "region_pressure_future"
TACTILE_REGION_FUTURE_VALID_KEY = "region_pressure_future_valid"
# Nested per-hand streams (left_ee / right_ee). Converter-style flat keys
# (3d_left / 3d_right) are wired in _get_tactile_spec, not this map.
TACTILE_STREAM_TO_BATCH_KEY = {
    "region_pressure_vector": TACTILE_REGION_PRESSURE_KEY,
    TACTILE_REGION_FUTURE_STREAM: TACTILE_REGION_FUTURE_KEY,
    "taxel_pressure_3D": TACTILE_TAXEL_PRESSURE_KEY,
    "2d": TACTILE_HEATMAP_KEY,
}
TACTILE_SAMPLE_KEYS = (
    TACTILE_REGION_PRESSURE_KEY,
    TACTILE_TAXEL_PRESSURE_KEY,
    TACTILE_TAXEL_VALID_KEY,
    TACTILE_HEATMAP_KEY,
    TACTILE_REGION_FUTURE_KEY,
    TACTILE_REGION_FUTURE_VALID_KEY,
)
# Hand slot order of the encoder's ``num_hands`` axis.
TACTILE_HAND_ORDER = ("left_ee", "right_ee")
# Legacy v2.1 Tianji converter: one column per hand, wrist-local xyz+pressure.
TACTILE_CONVERTER_TAXEL = (
    ("3d_left", "observation.tactile.3d_left"),
    ("3d_right", "observation.tactile.3d_right"),
)
TACTILE_CONVERTER_VALID = (
    ("validity_left", "observation.tactile.validity_left"),
    ("validity_right", "observation.tactile.validity_right"),
)


def _tactile_units_from_hand_shape(batch_key: str, shape) -> int | None:
    """Regions or taxels from modality.json ``shape``, if declared."""
    if not shape:
        return None
    dims = [int(x) for x in shape]
    if batch_key == TACTILE_REGION_PRESSURE_KEY:
        return dims[1] if len(dims) >= 2 else None
    if len(dims) == 2:
        return dims[0]
    if len(dims) >= 3:
        return dims[1]
    return None


def tactile_packed_layout_from_spec(spec: dict) -> dict:
    """Comparable packed-shape contract for one dataset's tactile spec."""
    out: dict[str, tuple] = {}
    for key, entry in (spec or {}).items():
        if key == TACTILE_TAXEL_VALID_KEY:
            continue
        if entry.get("kind") == "heatmap" or key == TACTILE_HEATMAP_KEY:
            out[key] = ("heatmap", entry.get("hw"), None, None)
        elif key == TACTILE_REGION_PRESSURE_KEY:
            channels = 2 if entry.get("append_delta") else 1
            out[key] = (
                "region",
                entry.get("history_len"),
                entry.get("num_units"),
                channels,
            )
        elif key == TACTILE_REGION_FUTURE_KEY:
            # Sliced future window; Δp is never appended here.
            out[key] = (
                "region_future",
                entry.get("future_len"),
                entry.get("num_units"),
                1,
            )
        elif key == TACTILE_TAXEL_PRESSURE_KEY:
            # Episode-stacked window (EgoPro parquet is per-row ``[N, 4]``).
            # Unset / None → T=1 (current frame). A bare ``tactile_history: N``
            # does not apply here — only an explicit mapping key does.
            out[key] = (
                "taxel",
                int(entry.get("history_len") or 1),
                entry.get("num_units"),
                4,
            )
        else:
            out[key] = (
                entry.get("kind"),
                entry.get("history_len"),
                entry.get("num_units"),
                entry.get("hw"),
            )
    return out


def assert_mixture_tactile_layouts(datasets: Sequence) -> None:
    """Fail at mixture init if tactile-packing members cannot stack.

    Non-tactile members are legal (those rows are zero-filled at collate).
    Two present members that disagree on history / taxels / channels would
    otherwise raise on the first mixed batch, often mid-run on multi-node.

    Fusion streams (``region_pressure`` / ``taxel_pressure_3d``) also fail
    when two or more packing members share the stream and any of them is
    missing ``num_units`` (modality.json ``shape``). Matching ``None``
    signatures would otherwise look compatible and hide an 80-vs-320 clash
    until the first mixed batch. A single tactile member plus V-only
    members stays legal; heatmap signatures already use ``None`` for unused
    fields and are not gated here.
    """
    by_key: dict[str, list[tuple[str, tuple]]] = {}
    fusion_keys = (TACTILE_REGION_PRESSURE_KEY, TACTILE_TAXEL_PRESSURE_KEY)
    for dataset in datasets:
        spec = getattr(dataset, "_tactile_spec", None) or {}
        layout = tactile_packed_layout_from_spec(spec)
        name = str(
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "_dataset_name", "?")
        )
        for key, signature in layout.items():
            by_key.setdefault(key, []).append((name, signature))
    for key, entries in by_key.items():
        signatures = {signature for _, signature in entries}
        if len(signatures) > 1:
            raise ValueError(
                f"mixture tactile stream {key!r} has conflicting packed layouts "
                f"{entries}. Every tactile-packing member must share "
                "history_len / num_units / channels so a mixed batch can stack; "
                "non-tactile members are fine (those rows are zero-filled)."
            )
        if key in fusion_keys and len(entries) >= 2:
            missing = [name for name, signature in entries if signature[2] is None]
            if missing:
                raise ValueError(
                    f"mixture tactile stream {key!r} is missing num_units "
                    f"(modality.json shape) on {missing}. Two or more "
                    "tactile-packing members share this fusion stream; "
                    "unspecified taxel/region counts cannot be asserted at "
                    "launch and an 80-vs-320 clash would wait for the first "
                    "mixed batch. Declare shape on every packing member, or "
                    "keep only one tactile member in the mixture."
                )

    # Validity packing may differ across taxel members: batch_packing defaults
    # the rows that carry no `taxel_valid_mask` to ALL VALID, so this is
    # correct rather than fatal. Report it so a run that expected per-taxel
    # validity everywhere does not discover the asymmetry from a loss curve.
    taxel_members = [
        (
            str(
                getattr(dataset, "dataset_name", None)
                or getattr(dataset, "_dataset_name", "?")
            ),
            TACTILE_TAXEL_VALID_KEY in (getattr(dataset, "_tactile_spec", None) or {}),
        )
        for dataset in datasets
        if TACTILE_TAXEL_PRESSURE_KEY
        in (getattr(dataset, "_tactile_spec", None) or {})
    ]
    if len({has_valid for _, has_valid in taxel_members}) > 1:
        with_valid = [name for name, has_valid in taxel_members if has_valid]
        without = [name for name, has_valid in taxel_members if not has_valid]
        print(
            f"[LeRobotMixtureDataset] tactile stream {TACTILE_TAXEL_PRESSURE_KEY!r} "
            f"is packed with {TACTILE_TAXEL_VALID_KEY!r} on {with_valid} but not on "
            f"{without}; rows from the latter are treated as ALL taxels valid. "
            "Declare the validity column on every taxel member if that is not "
            "what you want."
        )


_ENCODER_STEM_HINT = (
    "Pair tactile_fusion: fingertip_only with EgoPro 80/5; "
    "default / robust with the 320-point full-hand cloud "
    "(legacy Tianji 460: set num_taxels: 460 in the recipe)."
)


def encoder_tactile_layout(tactile_cfg) -> dict:
    """Packed-shape contract the encoder will require (active streams only).

    Same tuple layout as ``tactile_packed_layout_from_spec``:
    ``(kind, history_len, num_units, channels)``. Empty when fusion is off.
    Defaults match ``TactileEncoder`` (region 5/5/1, taxel 1/320/4).
    """
    tac = _as_mapping(tactile_cfg)
    out: dict[str, tuple] = {}
    streams = tac.get("streams") or {}
    if _tactile_stream_active(tac, TACTILE_REGION_PRESSURE_KEY):
        region = streams.get(TACTILE_REGION_PRESSURE_KEY) or {}
        if not isinstance(region, dict):
            region = {}
        out[TACTILE_REGION_PRESSURE_KEY] = (
            "region",
            int(region.get("history_len") or 5),
            int(region.get("num_regions") or 5),
            int(region.get("channel_dim") or 1),
        )
    if _tactile_stream_active(tac, TACTILE_TAXEL_PRESSURE_KEY):
        taxel = streams.get(TACTILE_TAXEL_PRESSURE_KEY) or {}
        if not isinstance(taxel, dict):
            taxel = {}
        out[TACTILE_TAXEL_PRESSURE_KEY] = (
            "taxel",
            int(taxel.get("history_len") or 1),
            int(taxel.get("num_taxels") or 320),
            4,
        )
    return out


def assert_encoder_tactile_layouts(datasets: Sequence, tactile_cfg=None) -> None:
    """Fail at launch if a packing member's modality.json ≠ the encoder.

    ``assert_mixture_tactile_layouts`` only compares members to each other.
    A single EgoPro 80-taxel member plus ``tactile_fusion: default`` (320)
    used to look fine until the first ``encode`` raised. V-only members and
    heatmap-only packs are skipped (those rows stay ``no_tactile`` / align).
    A packing member that is missing ``num_units`` also fails — matching
    ``None`` cannot prove 80 vs 320 vs 460. An omitted encoder stream is
    legal (those rows stay ``present=False`` / ``no_tactile``); only a
    packed stream with a conflicting signature is an error.
    """
    expected = encoder_tactile_layout(tactile_cfg)
    if not expected:
        return
    fusion_keys = (TACTILE_REGION_PRESSURE_KEY, TACTILE_TAXEL_PRESSURE_KEY)
    for dataset in datasets:
        spec = getattr(dataset, "_tactile_spec", None) or {}
        layout = tactile_packed_layout_from_spec(spec)
        if not any(key in layout for key in fusion_keys):
            continue
        name = str(
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "_dataset_name", "?")
        )
        for key, enc_sig in expected.items():
            if key not in layout:
                # Mixed batches zero-fill a missing stream and mark it
                # absent; that is the gloved + non-gloved contract.
                continue
            data_sig = layout[key]
            if data_sig[2] is None:
                raise ValueError(
                    f"[{name}] tactile stream {key!r} is missing num_units "
                    f"(modality.json shape); encoder expects {enc_sig}. "
                    "Cannot assert taxel/region count at launch."
                )
            if data_sig != enc_sig:
                raise ValueError(
                    f"[{name}] packed {key} {data_sig} != encoder {enc_sig}. "
                    f"{_ENCODER_STEM_HINT} A mismatch used to wait for the "
                    "first encode."
                )


def assert_packed_tactile_array(
    array,
    *,
    batch_key: str,
    spec: dict,
    dataset_name: str,
    n_hands: int | None = None,
) -> None:
    """Fail if a packed sample disagrees with its own modality.json spec.

    Launch-time ``assert_encoder_tactile_layouts`` only compares
    modality.json to the encoder. This catches parquet cells that do not
    match the declared shape (an 80-point cloud labeled 320, a truncated
    region window, a 3-channel taxel cell) at the first ``__getitem__``.
    """
    if array is None:
        return
    arr = np.asarray(array)
    hands = n_hands if n_hands is not None else len(spec.get("columns") or ()) or 2
    units = spec.get("num_units")
    name = f"[{dataset_name}] packed {batch_key}"

    if batch_key == TACTILE_HEATMAP_KEY:
        if arr.ndim != 2:
            raise ValueError(f"{name} must be HxW, got {tuple(arr.shape)}")
        hw = spec.get("hw")
        if hw is not None and tuple(arr.shape) != tuple(hw):
            raise ValueError(
                f"[{dataset_name}] {batch_key!r} has shape {tuple(arr.shape)}, "
                f"expected {tuple(hw)} from modality.json tactile.2d.shape"
            )
        return

    if batch_key == TACTILE_TAXEL_VALID_KEY:
        if arr.ndim != 2:
            raise ValueError(
                f"{name} must be [hands, taxels], got {tuple(arr.shape)}"
            )
        if int(arr.shape[0]) != int(hands):
            raise ValueError(
                f"{name} hands={arr.shape[0]} != {hands}"
            )
        if units is not None and int(arr.shape[1]) != int(units):
            raise ValueError(
                f"{name} taxels={arr.shape[1]} != modality.json "
                f"num_units={units}. {_ENCODER_STEM_HINT}"
            )
        return

    if batch_key == TACTILE_TAXEL_PRESSURE_KEY:
        history = int(spec.get("history_len") or 1)
        if arr.ndim != 4:
            raise ValueError(
                f"{name} must be [hands, taxels, T, 4], got {tuple(arr.shape)}"
            )
        got = tuple(int(x) for x in arr.shape)
        exp_hands_t_c = (int(hands), history, 4)
        if (got[0], got[2], got[3]) != exp_hands_t_c:
            raise ValueError(
                f"{name} shape {got} != [hands={hands}, taxels, T={history}, 4]. "
                f"{_ENCODER_STEM_HINT}"
            )
        if units is not None and got[1] != int(units):
            raise ValueError(
                f"{name} taxels={got[1]} != modality.json num_units={units} "
                f"(expected [{hands}, {int(units)}, {history}, 4]). "
                f"{_ENCODER_STEM_HINT}"
            )
        return

    if batch_key == TACTILE_REGION_PRESSURE_KEY:
        history = spec.get("history_len")
        channels = 2 if spec.get("append_delta") else 1
        if arr.ndim != 4:
            raise ValueError(
                f"{name} must be [hands, regions, history, C], got {tuple(arr.shape)}"
            )
        if int(arr.shape[0]) != int(hands):
            raise ValueError(f"{name} hands={arr.shape[0]} != {hands}")
        if units is not None and int(arr.shape[1]) != int(units):
            raise ValueError(
                f"{name} regions={arr.shape[1]} != modality.json "
                f"num_units={units}. {_ENCODER_STEM_HINT}"
            )
        if history is not None and int(arr.shape[2]) != int(history):
            raise ValueError(
                f"{name} history={arr.shape[2]} != spec history_len={history}"
            )
        if int(arr.shape[3]) != int(channels):
            raise ValueError(
                f"{name} channels={arr.shape[3]} != {channels} "
                f"(append_delta={bool(spec.get('append_delta'))})"
            )
        return

    if batch_key == TACTILE_REGION_FUTURE_KEY:
        horizon = spec.get("future_len")
        if arr.ndim != 4:
            raise ValueError(
                f"{name} must be [hands, regions, H, 1], got {tuple(arr.shape)}"
            )
        if int(arr.shape[0]) != int(hands):
            raise ValueError(f"{name} hands={arr.shape[0]} != {hands}")
        if units is not None and int(arr.shape[1]) != int(units):
            raise ValueError(
                f"{name} regions={arr.shape[1]} != modality.json "
                f"num_units={units}"
            )
        if horizon is not None and int(arr.shape[2]) != int(horizon):
            raise ValueError(
                f"{name} horizon={arr.shape[2]} != future_len={horizon}"
            )
        if int(arr.shape[3]) != 1:
            raise ValueError(f"{name} last dim must be 1, got {arr.shape[3]}")
        return


def assert_mixture_camera_slots(datasets: Sequence) -> None:
    """Fail at mixture init if members disagree on the packed camera order.

    ``_run_video_keys`` is the run-level camera slot list, and it comes from
    the member's OWN ``robot_type`` DataConfig — two members with different
    ``robot_type`` can therefore carry different slot lists. Packed image
    position *i* is "the i-th present run slot" and the VLM sees no slot-id
    tokens, so a mixture whose members order (or name) their slots
    differently trains position 0 as the head camera on one dataset and as a
    wrist on another. ``view_mask`` is per-member too, so nothing downstream
    can recover the intent.

    Only the slot LIST must match. Which of those slots a given member
    actually provides is exactly what ``flexible_views`` is for, and the
    per-member ``_view_mask_template`` keeps those aligned.
    """
    by_slots: dict[tuple, list[str]] = {}
    for dataset in datasets:
        name = str(
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "_dataset_name", "?")
        )
        slots = tuple(
            str(key)
            for key in (
                getattr(dataset, "_run_video_keys", None)
                or (getattr(dataset, "modality_keys", {}) or {}).get("video")
                or []
            )
        )
        by_slots.setdefault(slots, []).append(name)
    if len(by_slots) > 1:
        detail = "; ".join(
            f"{list(slots)} <- {names}" for slots, names in by_slots.items()
        )
        raise ValueError(
            "mixture members declare different camera slot orders: "
            f"{detail}. Packed image position i means 'the i-th present run "
            "slot' and the VLM gets no slot-id tokens, so mismatched slot "
            "lists make position i a different physical camera per dataset. "
            "Give every member the same robot_type / DataConfig video_keys "
            "(flexible_views then masks the cameras a dataset does not have)."
        )


# Deploy aliases cam_high ↔ cam_head (and a few other head names). Training
# matches modality.json keys exactly — a required slot that only exists under
# an alias is a hard error, with a hint so the miss is not silent.
VIDEO_NAME_ALIASES = {
    "cam_high": ("cam_head", "head_camera", "top_camera"),
    "cam_head": ("cam_high", "head_camera", "top_camera"),
    "head_camera": ("cam_high", "cam_head", "top_camera"),
    "top_camera": ("cam_high", "cam_head", "head_camera"),
}


def _collect_declared_tactile_parquet_keys(declared: dict) -> set[str]:
    """original_key values in a modality.json tactile block (no schema checks)."""
    keys: set[str] = set()
    if not isinstance(declared, dict):
        return keys
    converter_defaults = dict(TACTILE_CONVERTER_TAXEL + TACTILE_CONVERTER_VALID)
    for stream_name, entry in declared.items():
        if not isinstance(entry, dict):
            continue
        if "original_key" in entry:
            keys.add(str(entry["original_key"]))
            continue
        nested = False
        for child in entry.values():
            if isinstance(child, dict) and child.get("original_key"):
                keys.add(str(child["original_key"]))
                nested = True
        if nested:
            continue
        if stream_name in converter_defaults:
            keys.add(converter_defaults[stream_name])
        elif stream_name == "2d":
            keys.add("observation.tactile.2d")
        else:
            for hand in TACTILE_HAND_ORDER:
                keys.add(f"observation.tactile.{stream_name}.{hand}")
    return keys


def _flag_unset(value) -> bool:
    return value in (None, "", "null", "None")


def _as_mapping(value) -> dict:
    """Dict or OmegaConf → plain dict. Anything else → ``{}``."""
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            value = OmegaConf.to_container(value, resolve=True)
    except Exception:
        pass
    return dict(value) if isinstance(value, dict) else {}


def _role_flag_on(tac: dict, role: str) -> bool:
    """``fast_enable`` / ``slow_enable``; missing / auto → on (no-op)."""
    raw = tac.get("fast_enable" if role == "fast" else "slow_enable", True)
    if raw is None:
        return True
    if isinstance(raw, str) and raw.strip().lower() in ("", "auto", "none", "null"):
        return True
    if isinstance(raw, str) and raw.strip().lower() in ("false", "0", "no", "off"):
        return False
    return bool(raw)


def _tactile_stream_active(tac: dict, name: str) -> bool:
    """Stream is on iff top-level enable × ``streams.<name>.enable`` × role flag."""
    if not tac or not bool(tac.get("enable", False)):
        return False
    stream = (tac.get("streams") or {}).get(name) or {}
    if not isinstance(stream, dict) or not bool(stream.get("enable", False)):
        return False
    default_role = "fast" if name == TACTILE_REGION_PRESSURE_KEY else "slow"
    role = str(stream.get("role") or default_role).lower()
    return _role_flag_on(tac, role)


def _tactile_region_active(tac: dict) -> bool:
    """Region expert will actually be built (stem enable × role flag)."""
    return _tactile_stream_active(tac, TACTILE_REGION_PRESSURE_KEY)


def _tactile_taxel_active(tac: dict) -> bool:
    """3D taxel expert will actually be built (stem enable × role flag)."""
    return _tactile_stream_active(tac, TACTILE_TAXEL_PRESSURE_KEY)


def _sync_tactile_history_with_encoder(vla_data, tac: dict) -> None:
    """Keep packed windows equal to the encoder's per-stream ``history_len``.

    Deploy rebuilds those windows from ``framework.tactile.streams.*.history_len``.
    If ``tactile_history`` stays ``null``, the dataloader would emit the full
    stored region column (egopro: 50) and a single taxel frame while the
    encoder / server expect the stem windows — a shape error at train, or a
    silent train/deploy mismatch. Unset → copy every enabled encoder stream;
    an explicit value that disagrees is a hard error.

    A bare int applies to the **region** stream only (in-file history). Taxel
    files have no history axis; that window is stacked from episode policy
    frames and is only set from an explicit mapping key or this copy.

    ``tactile_append_delta`` adds a region channel; the encoder is
    shape-strict, so ``channel_dim`` must already be 2 (train.sh bumps it
    when writing the frozen config).

    Skipped per stream when that expert is off (``streams.*.enable`` or the
    matching ``fast_enable`` / ``slow_enable``) so an ablation does not
    force history / Δp constraints on a stream the encoder dropped.
    """
    region_on = _tactile_region_active(tac)
    taxel_on = _tactile_taxel_active(tac)
    if not region_on and not taxel_on:
        return
    streams = tac.get("streams") or {}
    region = streams.get("region_pressure") or {}
    taxel = streams.get("taxel_pressure_3d") or {}
    if not isinstance(region, dict):
        region = {}
    if not isinstance(taxel, dict):
        taxel = {}
    region_hist = int(region.get("history_len") or 5) if region_on else None
    taxel_hist = int(taxel.get("history_len") or 1) if taxel_on else None
    current = vla_data.get("tactile_history") if hasattr(vla_data, "get") else None
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(current):
            current = OmegaConf.to_container(current, resolve=True)
    except Exception:
        pass
    mapping: dict = dict(current) if isinstance(current, dict) else {}
    if region_on:
        got = current.get("region_pressure") if isinstance(current, dict) else current
        if got is not None and int(got) != region_hist:
            raise ValueError(
                "datasets.vla_data.tactile_history "
                f"({got}) != framework.tactile.streams.region_pressure.history_len "
                f"({region_hist}). Deploy uses the encoder length; they must match."
            )
        mapping["region_pressure"] = region_hist
    if taxel_on:
        # Bare int never applies to taxel (in-file region window only).
        got_taxel = current.get("taxel_pressure_3d") if isinstance(current, dict) else None
        if got_taxel is not None and int(got_taxel) != taxel_hist:
            raise ValueError(
                "datasets.vla_data.tactile_history.taxel_pressure_3d "
                f"({got_taxel}) != framework.tactile.streams."
                f"taxel_pressure_3d.history_len ({taxel_hist}). "
                "Deploy uses the encoder length; they must match."
            )
        mapping["taxel_pressure_3d"] = taxel_hist
    vla_data["tactile_history"] = mapping
    if not region_on:
        return
    delta_cfg = vla_data.get("tactile_append_delta") if hasattr(vla_data, "get") else None
    delta_on = (
        bool(delta_cfg.get("region_pressure"))
        if isinstance(delta_cfg, dict)
        else bool(delta_cfg)
    )
    if delta_on and int(region.get("channel_dim") or 1) != 2:
        raise ValueError(
            "datasets.vla_data.tactile_append_delta=true requires "
            "framework.tactile.streams.region_pressure.channel_dim: 2 "
            f"(got {region.get('channel_dim', 1)})."
        )


_PRESSURE_VLA_KEYS = (
    ("log1p", "tactile_pressure_log1p"),
    ("scale", "tactile_pressure_scale"),
    ("center", "tactile_pressure_center"),
    ("append_delta", "tactile_append_delta"),
)


def _parse_tactile_pressure_scale(value, batch_key: str) -> tuple[float | None, bool]:
    """``(scale, auto)``. ``auto`` is resolved from the |p| q99 cache later."""
    from starVLA.model.modules.tactile.pressure import is_auto_pressure_scale

    if is_auto_pressure_scale(value):
        return None, True
    if value is None:
        return None, False
    if isinstance(value, bool):
        # float(True) == 1.0 would silently disable scaling.
        raise ValueError(
            "datasets.vla_data.tactile_pressure_scale must be a positive "
            f"number or 'auto', got boolean {value!r} for {batch_key}"
        )
    scale = float(value)
    if scale <= 0.0:
        raise ValueError(
            "datasets.vla_data.tactile_pressure_scale must be > 0, "
            f"got {scale} for {batch_key}"
        )
    return scale, False


def _sync_tactile_pressure_from_stem(vla_data, tac: dict) -> None:
    """Copy ``framework.tactile.pressure`` into ``vla_data`` when unset.

    Same contract as ``tools.runtime_config.apply_tactile_stem_pressure`` so a
    dataset built from an already-merged framework (frozen yaml) still
    sees the stem's sensor scale without a second recipe key.
    """
    if vla_data is None or not isinstance(tac, dict):
        return
    pressure = tac.get("pressure")
    if not isinstance(pressure, dict):
        return
    for src, dest in _PRESSURE_VLA_KEYS:
        if src not in pressure:
            continue
        current = vla_data.get(dest) if hasattr(vla_data, "get") else None
        if _flag_unset(current):
            vla_data[dest] = pressure[src]


def _future_contact_cfg(framework) -> dict:
    tac = framework.get("tactile") if isinstance(framework, dict) else {}
    if not isinstance(tac, dict):
        tac = {}
    expert = tac.get("tactile_expert") if isinstance(tac.get("tactile_expert"), dict) else {}
    fc = expert.get("future_contact") if isinstance(expert.get("future_contact"), dict) else {}
    return fc


def _sync_tactile_future_horizon(vla_data, framework) -> None:
    """Copy variant-A horizon onto ``datasets.vla_data.tactile_future_horizon``.

    Unset follows ``tactile_expert.future_contact.horizon``, then the head's
    own default. The horizon counts GLOVE frames of the exported future
    column, not policy steps, so it is deliberately NOT tied to
    ``action_horizon``. Only written when future_contact is enabled, so a
    fusion-only run never reads the future column. Explicit
    ``tactile_future_horizon`` still wins.
    """
    if vla_data is None:
        return
    fc = _future_contact_cfg(framework)
    if not bool(fc.get("enable", False)):
        return
    current = vla_data.get("tactile_future_horizon") if hasattr(vla_data, "get") else None
    if not _flag_unset(current):
        return
    horizon = fc.get("horizon")
    if _flag_unset(horizon):
        from starVLA.model.modules.tactile.future_contact import (
            DEFAULT_FUTURE_HORIZON,
        )

        horizon = DEFAULT_FUTURE_HORIZON
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError(
            f"tactile_future_horizon must be >= 1, got {horizon}"
        )
    vla_data["tactile_future_horizon"] = horizon


def sync_tactile_pack_flags(vla_data, framework=None) -> None:
    """Fill pack / heatmap / pressure knobs from the merged tactile stems.

    Unset flags follow the enable bits so a V-only cfg (both off) never packs,
    and a fusion/align run packs without a second recipe key. Unset
    ``tactile_pressure_*`` copy ``framework.tactile.pressure``. Explicit
    values in ``vla_data`` win. Safe on dict and OmegaConf.
    """
    if vla_data is None:
        return
    fw = framework or {}
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(fw):
            fw = OmegaConf.to_container(fw, resolve=True) or {}
    except Exception:
        pass
    if not isinstance(fw, dict):
        fw = {}
    tac = fw.get("tactile") if isinstance(fw.get("tactile"), dict) else {}
    align = fw.get("tactile_align") if isinstance(fw.get("tactile_align"), dict) else {}
    tac_on = bool(tac.get("enable", False))
    align_on = bool(align.get("enable", False))
    if _flag_unset(vla_data.get("pack_tactile") if hasattr(vla_data, "get") else None):
        vla_data["pack_tactile"] = bool(tac_on or align_on)
    if _flag_unset(vla_data.get("tactile_pack_heatmap") if hasattr(vla_data, "get") else None):
        vla_data["tactile_pack_heatmap"] = bool(align_on)
    _sync_tactile_pressure_from_stem(vla_data, tac)
    _sync_tactile_history_with_encoder(vla_data, tac)
    _sync_tactile_future_horizon(vla_data, fw)


# Row groups above this are not worth caching (the egopro export writes ONE)
# group of ~110k rows, ~1.6 GB of tactile); the reader warns and streams
# instead. See LeRobotSingleDataset._read_tactile_row.
_TACTILE_MAX_CACHED_ROW_GROUP_ROWS = 20000
# Numpy-cache a column only when one cell is this small. Fusion arrays
# (region 50×6, taxel 460×4) stay numpy; a 480×640 heatmap would be
# 1000×1.2 MB ≈ 1.2 GB/worker as float32, so it stays an arrow column and
# is decoded one frame at a time.
_TACTILE_MAX_NUMPY_CACHE_CELL_ELEMS = 16 * 1024


def _arrow_nested_floats_to_ndarray(column):
    """Decode ``list<list<number>>`` / ``list<number>`` → float32 without ``as_py``.

    Nested Python lists of a 480×640 heatmap are tens of milliseconds per
    cell; flattening the arrow offsets is a single reshape.
    """
    import pyarrow as pa

    arr = column.combine_chunks() if hasattr(column, "combine_chunks") else column
    if arr.null_count:
        raise ValueError("null tactile cells")
    n = len(arr)
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if not (pa.types.is_list(arr.type) or pa.types.is_large_list(arr.type)):
        raise TypeError(f"expected list column, got {arr.type}")
    level1 = arr.flatten()
    outer = np.diff(arr.offsets.to_numpy())
    if pa.types.is_list(level1.type) or pa.types.is_large_list(level1.type):
        values = np.asarray(level1.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
        inner = np.diff(level1.offsets.to_numpy())
        if outer.size and not bool(np.all(outer == outer[0])):
            raise ValueError("ragged outer list lengths in tactile column")
        if inner.size and not bool(np.all(inner == inner[0])):
            raise ValueError("ragged inner list lengths in tactile column")
        height = int(outer[0]) if outer.size else 0
        width = int(inner[0]) if inner.size else 0
        return values.reshape(n, height, width)
    values = np.asarray(level1.to_numpy(zero_copy_only=False), dtype=np.float32)
    if outer.size and not bool(np.all(outer == outer[0])):
        raise ValueError("ragged list lengths in tactile column")
    width = int(outer[0]) if outer.size else 0
    return values.reshape(n, width)


def _arrow_cell_to_ndarray(scalar):
    """One list/list-list cell → float32 ndarray, preferring arrow flatten."""
    import pyarrow as pa

    values = getattr(scalar, "values", None)
    if values is None:
        return np.asarray(scalar.as_py(), dtype=np.float32)
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type):
        flat = np.asarray(values.flatten().to_numpy(zero_copy_only=False), dtype=np.float32)
        height = len(values)
        width = 0 if height == 0 else len(values[0])
        return flat.reshape(height, width)
    return np.asarray(values.to_numpy(zero_copy_only=False), dtype=np.float32)


def _tactile_cell_flag(value) -> bool:
    """Scalar parquet validity cell → bool (``[True]`` or a 0-d / 1-d array)."""
    if value is None:
        return False
    array = np.asarray(value).reshape(-1)
    return bool(array[0]) if array.size else False


def _tactile_column_cell_elems(column) -> int:
    if len(column) == 0:
        return 0
    return int(_arrow_cell_to_ndarray(column[0]).size)


def _cache_tactile_row_group(table, columns, group_rows: int) -> dict:
    """Cache small fusion columns as numpy; keep HxW heatmaps as arrow.

    Flattening a 1000-row 480×640 heatmap to float32 is ~1.2 GB *and* a
    second copy of the arrow buffers. Neighbouring frames still share the
    decoded row group; each ``__getitem__`` only materialises one cell.
    """
    numpy_cols: dict[str, np.ndarray] = {}
    arrow_cols: dict = {}
    for column in columns:
        col = table.column(column)
        if _tactile_column_cell_elems(col) > _TACTILE_MAX_NUMPY_CACHE_CELL_ELEMS:
            arrow_cols[column] = col.combine_chunks() if hasattr(col, "combine_chunks") else col
            continue
        try:
            numpy_cols[column] = _arrow_nested_floats_to_ndarray(col)
        except (TypeError, ValueError):
            numpy_cols[column] = np.stack(
                [_arrow_cell_to_ndarray(col[i]) for i in range(group_rows)]
            )
    return {"numpy": numpy_cols, "arrow": arrow_cols}


def _tactile_row_from_cache(payload: dict, columns, offset: int) -> dict:
    """One row from a hybrid cache; always a fresh ndarray (not a cache view)."""
    numpy_cols = payload["numpy"]
    arrow_cols = payload["arrow"]
    out = {}
    for column in columns:
        if column in numpy_cols:
            out[column] = np.array(numpy_cols[column][offset], dtype=np.float32, copy=True)
        else:
            out[column] = _arrow_cell_to_ndarray(arrow_cols[column][offset])
    return out


def _is_tactile_column_name(name: str) -> bool:
    """Parquet / pandas column that belongs to a tactile stream."""
    lowered = name.lower()
    return (
        lowered.startswith("observation.tactile.")
        or lowered.startswith("tactile.")
        or ".tactile." in lowered
    )


def _exclude_from_dataset_statistics(name: str) -> bool:
    """Tactile / task columns never enter q01/q99 action-state stats.

    A 480×640 heatmap is ``list<list<float>>``. The streaming fallback
    ``column.to_pandas()`` on a 65536-row batch would materialise tens of GB
    before the column is skipped as non-numeric. Name-skip them at read time.
    """
    if "task_info" in name:
        return True
    return _is_tactile_column_name(name)


def _arrow_column_is_nested_list(column) -> bool:
    """True for ``list<list<…>>`` / ``list<fixed_size_list>`` (heatmap, taxel)."""
    import pyarrow as pa

    column_type = column.type
    if not (pa.types.is_list(column_type) or pa.types.is_large_list(column_type)):
        return False
    inner = column_type.value_type
    return (
        pa.types.is_list(inner)
        or pa.types.is_large_list(inner)
        or pa.types.is_fixed_size_list(inner)
    )


def _parquet_columns_for_dataset_statistics(path: Path) -> list[str] | None:
    """Columns the raw-stats paths may load; ``None`` = every column."""
    import pyarrow.parquet as pq

    names = list(pq.ParquetFile(path).schema_arrow.names)
    keep = [name for name in names if not _exclude_from_dataset_statistics(name)]
    if len(keep) == len(names):
        return None
    return keep


def _low_dim_column_matrix(column) -> np.ndarray:
    """``(N, D)`` float32 matrix for one parquet column.

    Exactly the conversion the pooled path applies to the concatenated
    frame, so both modes see bit-identical inputs (scalar columns become
    ``(N, 1)``). Raises for anything not convertible to float32 (string /
    ragged columns); both callers treat that as "skip this column".
    """
    return np.vstack([np.asarray(x, dtype=np.float32) for x in column])


def calculate_dataset_statistics(
    parquet_paths: list[Path], *, mode: str = "memory"
) -> dict:
    """Calculate the dataset statistics of all columns for a list of parquet files.

    ``mode="streaming"`` keeps the peak RSS at ~1 file + histograms instead
    of materialising every parquet at once (see
    :func:`_calculate_dataset_statistics_streaming`); min/max/quantiles come
    out bit-identical, mean/std float64-accurate.
    """
    if mode == "streaming":
        return _calculate_dataset_statistics_streaming(parquet_paths)
    # Dataset statistics
    all_low_dim_data_list = []
    # Collect all the data
    # parquet_paths = parquet_paths[:3]
    for parquet_path in tqdm(
        sorted(list(parquet_paths)),
        desc="Collecting all parquet files...",
    ):
        # Load the parquet file
        parquet_data = pd.read_parquet(
            parquet_path, columns=_parquet_columns_for_dataset_statistics(parquet_path)
        )
        all_low_dim_data_list.append(parquet_data)

    if not all_low_dim_data_list:
        raise FileNotFoundError(
            f"No parquet files found under the provided paths: {[str(p) for p in parquet_paths[:3]]}..."
            f" — make sure the dataset has been downloaded/converted before training."
        )

    all_low_dim_data = pd.concat(all_low_dim_data_list, axis=0)
    # Compute dataset statistics
    dataset_statistics = {}
    for le_modality in tqdm(all_low_dim_data.columns, desc="Processing modalities"):
        print(le_modality)
        if "task_info" in le_modality:
            continue
        print(f"Computing statistics for {le_modality}...")
        try:
            np_data = np.vstack(
                [np.asarray(x, dtype=np.float32) for x in all_low_dim_data[le_modality]]
            )
        except Exception as e:
            print(f"Warning: Failed to process modality {le_modality} due to error: {e}")
            continue  

        dataset_statistics[le_modality] = {
            "mean": np.mean(np_data, axis=0).tolist(),
            "std": np.std(np_data, axis=0).tolist(),
            "min": np.min(np_data, axis=0).tolist(),
            "max": np.max(np_data, axis=0).tolist(),
            "q01": np.quantile(np_data, 0.01, axis=0).tolist(),
            "q99": np.quantile(np_data, 0.99, axis=0).tolist(),
        }
    return dataset_statistics


def _arrow_column_matrix(column) -> np.ndarray:
    """``(N, D)`` float32 view of one arrow column, without Python objects.

    ``pandas`` materialises a ``fixed_size_list`` column as one small ndarray
    OBJECT per row: 2.2 kB/row on open_aoe_100 against 216 B/row of actual
    values, and the freed objects fragment the allocator instead of
    returning to the OS. Reading through arrow keeps a batch at its true
    columnar size. Anything this cannot express (strings, ragged lists,
    nulls) raises and the caller falls back to the pandas conversion.
    """
    import pyarrow as pa

    if column.null_count:
        raise ValueError("null values require the pandas conversion")
    column_type = column.type
    if pa.types.is_fixed_size_list(column_type):
        width = column_type.list_size
        flat = column.flatten().to_numpy(zero_copy_only=False)
        return np.asarray(flat, dtype=np.float32).reshape(-1, width)
    if (
        pa.types.is_floating(column_type)
        or pa.types.is_integer(column_type)
        or pa.types.is_boolean(column_type)
    ):
        values = column.to_numpy(zero_copy_only=False)
        return np.asarray(values, dtype=np.float32).reshape(-1, 1)
    raise TypeError(f"unsupported arrow type {column_type}")


def _iter_parquet_column_matrices(path: Path, batch_size: int, skip: set[str]):
    """Yield ``(column_names, {column: (N, D) float32 | Exception})`` per batch.

    A column that cannot be expressed as a float32 matrix (task strings,
    ragged annotations) yields its exception instead; the caller drops it
    exactly like the pooled path does.
    """
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(path)
    schema_names = list(parquet_file.schema_arrow.names)
    read_names = [
        name
        for name in schema_names
        if name not in skip and not _exclude_from_dataset_statistics(name)
    ]
    if not read_names:
        return
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=read_names):
        names = [name for name in batch.schema.names if "task_info" not in name]
        matrices: dict[str, np.ndarray | Exception] = {}
        for name in names:
            if name in skip:
                continue
            column = batch.column(name)
            try:
                matrices[name] = _arrow_column_matrix(column)
            except Exception as first:
                if _arrow_column_is_nested_list(column):
                    matrices[name] = first
                    continue
                try:
                    # Same tolerance (and the same values) as the pooled path.
                    matrices[name] = _low_dim_column_matrix(column.to_pandas())
                except Exception as exc:
                    matrices[name] = exc
        yield names, matrices


# Rows per parquet batch in streaming mode. 65536 x 54 float32 is ~14 MB, so
# the resident data is a rounding error next to the histograms.
_STREAMING_PARQUET_BATCH_ROWS = 1 << 16


def _calculate_dataset_statistics_streaming(parquet_paths: list[Path]) -> dict:
    """Same statistics, one parquet BATCH resident at a time.

    The pooled path keeps every parquet in a list, concatenates them, and
    then materialises one float32 copy plus ``np.quantile``'s partition copy
    per column: measured 1.53 kB/row on open_aoe_100, i.e. ~22 GB peak for
    its 10.97M rows and growing linearly with the dataset — the same cliff
    the converted-pose statistics already avoid. Here pass 1 histograms each
    batch and pass 2 re-reads them to select the quantile ranks exactly, so
    the peak is one batch plus ~0.5 MB of histogram per column DIMENSION,
    independent of both the dataset size and the parquet file size.

    Files are read twice (I/O, not RAM), and the whole run shares one pass-2
    candidate budget.
    """
    paths = sorted(list(parquet_paths))
    if not paths:
        raise FileNotFoundError(
            f"No parquet files found under the provided paths: {[str(p) for p in parquet_paths[:3]]}..."
            f" — make sure the dataset has been downloaded/converted before training."
        )

    budget = _CollectBudget()
    builders: dict[str, _StreamingStatsBuilder] = {}
    columns: list[str] | None = None
    skipped: set[str] = set()

    def check_columns(names, path):
        nonlocal columns
        if columns is None:
            columns = list(names)
            return
        # `skipped` grows while we read: a column that fails to convert to a
        # numeric matrix (a string `task`, say) is dropped, and every later
        # batch stops yielding it. Compare against the FIRST file's list with
        # those same drops applied — comparing against the raw first list made
        # the first non-convertible column look like a schema change on file
        # two, so any dataset with a text column failed here.
        expected = [name for name in columns if name not in skipped]
        if list(names) != expected:
            # The pooled path pads a missing column with NaN through
            # pd.concat and then dies on the ragged vstack (or silently
            # drops the column); neither is a useful answer, and a
            # half-covered statistic would poison normalization.
            raise ValueError(
                f"{path} declares columns {list(names)} but the first parquet "
                f"file declared {expected}; statistics require one schema "
                "across the dataset."
            )

    for path in tqdm(paths, desc="Dataset statistics (pass 1/2: histograms)"):
        for names, matrices in _iter_parquet_column_matrices(
            path, _STREAMING_PARQUET_BATCH_ROWS, skipped
        ):
            check_columns(names, path)
            for column, matrix in matrices.items():
                if isinstance(matrix, Exception):
                    print(
                        f"Warning: Failed to process modality {column} due to "
                        f"error: {matrix}"
                    )
                    skipped.add(column)
                    builders.pop(column, None)
                    continue
                if column not in builders:
                    builders[column] = _StreamingStatsBuilder(budget=budget)
                builders[column].update(matrix)

    for builder in builders.values():
        builder.finalize_pass1()

    for path in tqdm(paths, desc="Dataset statistics (pass 2/2: quantiles)"):
        for _, matrices in _iter_parquet_column_matrices(
            path, _STREAMING_PARQUET_BATCH_ROWS, skipped
        ):
            for column, builder in builders.items():
                builder.collect_update(matrices[column])

    return {column: builder.result() for column, builder in builders.items()}


def _normalize_action_mode(mode: str) -> str:
    """Normalize action mode names to {abs, delta, rel}.""" 
    # @gaoning plz move this, we want dataloader to be independent of the action mode logic, we can move this to transform or a separate utils tool to handle lerobot dataset
    mode = str(mode).lower()
    if mode in {"absolute", "raw"}:
        mode = "abs"
    if mode not in {"abs", "delta", "rel"}:
        mode = "abs"
    return mode


def _normalize_action_mode_apply_keys(
    action_mode_apply_keys: Sequence[str] | None,
    fallback_keys: Sequence[str] | None = None,
) -> list[str]:
    source_keys = action_mode_apply_keys if action_mode_apply_keys else (fallback_keys or [])
    normalized = []
    for key in source_keys:
        key = str(key)
        if not key.startswith("action."):
            key = f"action.{key}"
        normalized.append(key)
    return normalized


def _normalize_action_mode_state_map(action_mode_state_map: dict[str, str] | None) -> dict[str, str]:
    normalized = {}
    for action_key, state_key in (action_mode_state_map or {}).items():
        action_key = str(action_key)
        state_key = str(state_key)
        if not action_key.startswith("action."):
            action_key = f"action.{action_key}"
        if not state_key.startswith("state."):
            state_key = f"state.{state_key}"
        normalized[action_key] = state_key
    return normalized


_EPISODE_PARQUET_RE = re.compile(r"episode_(\d+)\.parquet$")


def _episode_index_from_parquet_name(name: str) -> int | None:
    """Episode index encoded in a per-episode parquet filename, else None."""
    match = _EPISODE_PARQUET_RE.search(name)
    return int(match.group(1)) if match else None


def resolve_excluded_episodes(data_cfg) -> set[int]:
    """Episode indices excluded from BOTH sampling and statistics.

    Read from ``datasets.vla_data.exclude_episodes`` (list of episode indices,
    default empty). Applying the same set to trajectory enumeration and to the
    statistics passes keeps the stats and the training data over the exact
    same episode population.
    """
    raw = data_cfg.get("exclude_episodes", None) if data_cfg else None
    if not raw:
        return set()
    return {int(v) for v in raw}


def _path_under_dataset(path: Path | str, dataset_path: Path | str) -> str:
    """Dataset-relative posix path, stable across symlink vs realpath access.

    Training often reaches a shared dataset through a per-user symlink
    (``…/tianxing/…/data/X -> …/zijian/…/data/X``). Fingerprints must NOT
    embed either spelling — otherwise two arms on the same physical data
    get different cache keys and recompute past each other. Resolving both
    sides before relativizing makes the recorded path identical either way.
    """
    path = Path(path)
    dataset_path = Path(dataset_path)
    try:
        return path.resolve().relative_to(dataset_path.resolve()).as_posix()
    except ValueError:
        # Last resort when the two paths do not share a resolved root
        # (should not happen for files we glob under dataset_path).
        return Path(
            os.path.relpath(os.path.abspath(path), os.path.abspath(dataset_path))
        ).as_posix()


def _fingerprint_paths(dataset_path: Path | str, paths) -> list:
    """``[relpath, size, mtime_ns]`` for each existing file, sorted by relpath.

    Relpaths are dataset-relative (see :func:`_path_under_dataset`), so the
    fingerprint never carries an absolute or cwd-dependent path. Bare
    filenames are NOT enough: LeRobot v3 stores the same ``file-000.parquet``
    under every ``data/chunk-NNN/``, and a basename-only key cannot tell those
    files apart.
    """
    fingerprint = []
    for path in paths:
        path = Path(path)
        if not path.is_file():
            continue
        stat = os.stat(path)
        fingerprint.append(
            [
                _path_under_dataset(path, dataset_path),
                int(stat.st_size),
                int(stat.st_mtime_ns),
            ]
        )
    fingerprint.sort(key=lambda entry: entry[0])
    return fingerprint


def _dataset_data_fingerprint(dataset_path: Path | str, parquet_paths) -> list:
    """Fingerprint of the data parquet files a statistics/step cache covers.

    Embedded in cache configs so that regenerating a dataset IN PLACE (same
    directory, e.g. rewriting the action columns) automatically invalidates
    the derived caches. Without it, a cache is only keyed on configuration
    and silently survives data changes.
    """
    return _fingerprint_paths(dataset_path, parquet_paths)


def _dataset_meta_fingerprint(dataset_path: Path, extra_globs: Sequence[str] = ()) -> list:
    """``[relpath, size, mtime_ns]`` for the METADATA the statistics read.

    The data fingerprint covers only the parquet files, so editing
    ``meta/modality.json`` in place — renaming an ``original_key``, moving a
    key's ``start``/``end`` — kept every derived cache valid while changing
    what the statistics are computed over. Camera-extrinsics ``.npy`` sidecars
    fall under the same rule for datasets that store them outside the parquet
    (``extra_globs``); a dataset with the extrinsics in a parquet column
    matches nothing and is covered by the data fingerprint instead.
    """
    dataset_path = Path(dataset_path)
    candidates = [dataset_path / LE_ROBOT_MODALITY_FILENAME]
    for pattern in extra_globs:
        candidates.extend(dataset_path.glob(pattern))
    return _fingerprint_paths(dataset_path, candidates)


def _env_int(name: str, default: int) -> int:
    """``int`` of an env var, falling back on unset/empty/garbage."""
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _stats_process_rank() -> int:
    """This process's rank, WITHOUT requiring an initialized process group.

    Every cache in this module elects a single writer. Keying that election
    purely on ``dist.is_initialized()`` means that any launcher which builds
    the datasets before the process group exists turns all N ranks into
    "rank 0", and they scan the dataset concurrently (N x the peak RSS, N
    writers racing on the same file). The launcher-provided ``RANK`` is set
    long before ``init_process_group`` runs, so it is the safer default; a
    single-process run has neither and stays rank 0.
    """
    if dist.is_initialized():
        return dist.get_rank()
    return _env_int("RANK", 0)


def _stats_world_size() -> int:
    """World size, with the same pre-process-group fallback as _stats_process_rank."""
    if dist.is_initialized():
        return dist.get_world_size()
    return _env_int("WORLD_SIZE", 1)


def _is_stats_main_process() -> bool:
    """Whether this process computes and publishes the shared caches."""
    return _stats_process_rank() == 0


def _build_stats_cache_config(
    action_mode: str,
    modality_key_ranges: dict | None = None,
    exclude_episodes: set[int] | None = None,
    data_fingerprint: list | None = None,
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    action_indices: list[int] | None = None,
    state_indices: list[int] | None = None,
    meta_fingerprint: list | None = None,
    stats_mode: str | None = None,
) -> dict:
    config = {
        "mode": action_mode,
    }
    # Streaming statistics are float64-accurate in mean/std (the pooled path
    # accumulates float32 pairwise and drifts with the row count), so the two
    # compute modes must not share a cache. Recorded ONLY when streaming, so
    # every existing memory-mode cache keeps its hash.
    if stats_mode == "streaming":
        config["stats_mode"] = "streaming"
    # delta/rel statistics are computed over CONVERTED chunks
    # (calculate_delta/rel_action_statistics), so every input that changes the
    # conversion must also change the cache key — two delta-mode runs with
    # different chunk lengths or state mappings would otherwise share one
    # statistics file and normalize against the other run's conversion.
    # Added only for non-abs modes so every existing abs cache stays valid.
    if action_mode != "abs":
        config["action_mode_apply_keys"] = list(action_mode_apply_keys or [])
        config["action_mode_state_map"] = dict(action_mode_state_map or {})
        config["action_indices"] = [int(i) for i in (action_indices or [])]
        config["state_indices"] = [int(i) for i in (state_indices or [])]
    if modality_key_ranges:
        config["modality_key_ranges"] = {
            str(modality): {
                str(key): [int(key_range[0]), int(key_range[1])]
                for key, key_range in key_ranges.items()
            }
            for modality, key_ranges in modality_key_ranges.items()
        }
    # Key added only when non-empty so caches from runs without exclusions
    # stay valid.
    if exclude_episodes:
        config["exclude_episodes"] = sorted(int(v) for v in exclude_episodes)
    if data_fingerprint is not None:
        config["data_fingerprint"] = data_fingerprint
    if meta_fingerprint is not None:
        config["meta_fingerprint"] = meta_fingerprint
    return config


def _stats_cache_path(base_path: Path, cache_config: dict) -> Path:
    """Content-addressed sibling of ``base_path`` for one cache config.

    The cache lives inside the DATASET directory, so every run that trains on
    that dataset shares it. A single fixed filename made runs with different
    configs (a different pose format, chunk length, camera frame, episode
    exclusion, ...) evict each other's statistics: each launch found a config
    mismatch, deleted the file and recomputed, which both wastes the (up to
    hour-long) scan and lets one run overwrite the cache another run is about
    to read. Hashing the config into the filename lets those caches coexist.

    Stale files are never pruned automatically: they are cheap, and deleting
    them is exactly the eviction this addresses.
    """
    digest = hashlib.sha1(
        json.dumps(cache_config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:12]
    return base_path.with_name(f"{base_path.stem}__{digest}{base_path.suffix}")


def _load_stats_cache_for_config(
    base_path: Path, cache_config: dict
) -> tuple[dict | None, bool]:
    """``(statistics, already_saved_under_the_hashed_name)``.

    Pre-``_stats_cache_path`` runs wrote a single ``base_path``. When that file
    still matches the requested config, reuse it instead of paying for a full
    recompute — reported as not-yet-saved so the caller migrates it into this
    config's own file. A hit on the hashed path is reported as saved, so a
    launch that only reads the cache never rewrites a file other runs share.
    """
    stats_path = _stats_cache_path(base_path, cache_config)
    # Never unlink a shared hashed file merely because one read hit an NFS/I/O
    # or JSON parse failure. Rank 0 treats the miss as a recompute and replaces
    # it atomically; other ranks keep polling. The unhashed legacy path is also
    # read-only here because it may belong to another config.
    statistics = _load_stats_cache(stats_path, cache_config, invalidate_legacy=False)
    if statistics is not None:
        return statistics, True
    # invalidate_legacy=False: the shared legacy path may belong to a run with
    # a different config, which is not ours to delete.
    return _load_stats_cache(base_path, cache_config, invalidate_legacy=False), False


def _invalidate_legacy_stats_cache(stats_path: Path, reason: str) -> None:
    if not stats_path.exists():
        return
    print(f"Removing stale dataset statistics cache at {stats_path}: {reason}")
    stats_path.unlink()


def _load_stats_cache(
    stats_path: Path,
    expected_config: dict,
    *,
    invalidate_legacy: bool,
) -> dict | None:
    if not stats_path.exists():
        return None

    try:
        with open(stats_path, "r") as f:
            payload = json.load(f)
    except Exception as exc:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, f"failed to load JSON ({exc})")
        return None

    if not isinstance(payload, dict):
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "unexpected top-level format")
        return None

    format_version = payload.get("__format_version")
    cache_config = payload.get("__cache_config")
    statistics = payload.get("statistics")
    if format_version != LE_ROBOT_STATS_FORMAT_VERSION or cache_config is None or statistics is None:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "legacy statistics format detected")
        return None

    if cache_config != expected_config:
        if invalidate_legacy:
            _invalidate_legacy_stats_cache(stats_path, "statistics config mismatch, rebuilding cache")
        return None

    return statistics


def _save_stats_cache(stats_path: Path, cache_config: dict, statistics: dict) -> None:
    payload = {
        "__format_version": LE_ROBOT_STATS_FORMAT_VERSION,
        "__cache_config": cache_config,
        "statistics": statistics,
    }
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # A config-addressed cache can still be produced concurrently by two
    # independent training jobs. A fixed sibling ".tmp" lets one job rename
    # the file while the other still has it open, after which the second
    # os.replace fails (or publishes an incomplete shared temporary inode).
    # Give every writer its own temp file; os.replace then atomically publishes
    # one complete, equivalent payload and last-writer-wins safely.
    tmp_path = None
    try:
        tmp_path = stats_path.with_name(
            f".{stats_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        # "x" both guarantees this UUID path is ours and preserves ordinary
        # open()/umask permissions. NamedTemporaryFile would force 0600 and
        # prevent other Unix users from reusing a cache in a shared dataset.
        with open(tmp_path, "x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=4)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, stats_path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def _wait_for_stats_cache(
    stats_path: Path,
    cache_config: dict,
    *,
    description: str,
) -> dict:
    """Poll a rank-0-produced cache without holding a process-group barrier.

    Converted-pose statistics can take much longer than the common 600-second
    NCCL timeout. Non-zero ranks used to enter ``dist.barrier()`` immediately
    and time out while rank 0 was still scanning the shared dataset. The cache
    file itself is the synchronization primitive: it is published atomically,
    so readers either see no file or one complete payload.
    """
    timeout_s = float(os.environ.get("SPARK_0_STATS_CACHE_TIMEOUT_S", 6 * 60 * 60))
    poll_s = float(os.environ.get("SPARK_0_STATS_CACHE_POLL_S", 5.0))
    if (
        not math.isfinite(timeout_s)
        or not math.isfinite(poll_s)
        or timeout_s <= 0
        or poll_s <= 0
    ):
        raise ValueError(
            "SPARK_0_STATS_CACHE_TIMEOUT_S and SPARK_0_STATS_CACHE_POLL_S must be > 0"
        )
    deadline = time.monotonic() + timeout_s
    while True:
        statistics = _load_stats_cache(
            stats_path, cache_config, invalidate_legacy=False
        )
        if statistics is not None:
            return statistics
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out after {timeout_s:g}s waiting for {description}: {stats_path}. "
                "Rank 0 may have failed while computing statistics; inspect its logs. "
                "Increase SPARK_0_STATS_CACHE_TIMEOUT_S for an intentionally longer scan."
            )
        time.sleep(poll_s)


def _precomputed_stats_required(data_cfg=None) -> bool:
    """Whether statistics must come from a precomputed cache (never computed
    at training launch).

    Enabled by ``SPARK_0_REQUIRE_PRECOMPUTED_STATS=1`` (exported by default by
    the Spark_0 adapter's train.sh) or ``datasets.vla_data.require_precomputed_stats:
    true``. The env var is tri-state and wins over the config key: ``1``
    forces on, ``0`` forces off (the offline precompute relies on this),
    unset falls back to the config. Caches are produced offline by
    ``starVLA/training/precompute_stats.py`` (Spark_0
    ``scripts/precompute_stats.sh``), which runs the same code path with the
    same resolved config, so the content-addressed file names match.
    """
    env = os.environ.get("SPARK_0_REQUIRE_PRECOMPUTED_STATS")
    if env is not None:
        return env == "1"
    if data_cfg is not None and bool(data_cfg.get("require_precomputed_stats", False)):
        return True
    return False


def _force_stats_recompute() -> bool:
    """``SPARK_0_STATS_FORCE_RECOMPUTE=1``: ignore existing caches and rebuild.

    Escape hatch for the offline precompute (implementation A/Bs, stale-cache
    suspicion). The rebuilt payload is published atomically under the same
    content-addressed name as usual. Single-process use only: with multiple
    ranks the non-zero ranks would read the pre-rebuild file while rank 0 is
    still computing.
    """
    return os.environ.get("SPARK_0_STATS_FORCE_RECOMPUTE", "0") == "1"


def _check_force_stats_recompute_allowed(data_cfg=None) -> None:
    """Refuse the two combinations where force-recompute silently misbehaves.

    Strict precomputed-stats mode never computes, so the flag would be a
    silent no-op; multi-rank non-strict mode would let non-zero ranks read
    the pre-rebuild cache while rank 0 is still computing.
    """
    if _precomputed_stats_required(data_cfg):
        raise ValueError(
            "SPARK_0_STATS_FORCE_RECOMPUTE=1 is incompatible with "
            "precomputed-statistics mode (SPARK_0_REQUIRE_PRECOMPUTED_STATS=1): "
            "strict mode never computes, so the force flag would be silently "
            "ignored. Rebuild the caches offline with "
            "`bash scripts/precompute_stats.sh <config> --force-recompute`."
        )
    if _stats_world_size() > 1:
        raise RuntimeError(
            "SPARK_0_STATS_FORCE_RECOMPUTE=1 is single-process only: non-zero "
            "ranks would read the pre-rebuild statistics cache while rank 0 is "
            "still computing. Run the offline precompute instead "
            "(`bash scripts/precompute_stats.sh <config> --force-recompute`)."
        )


def _load_precomputed_stats_or_raise(
    base_path: Path, cache_config: dict, *, description: str
) -> dict:
    """Strict-mode cache read: the content-addressed file only, on every rank,
    with no compute and no polling.

    Only the hashed path counts: adopting the legacy unhashed file here would
    leave non-zero ranks waiting on a hashed file nobody writes. The offline
    precompute runs the normal path, which publishes the hashed name.
    """
    stats_path = _stats_cache_path(base_path, cache_config)
    statistics = _load_stats_cache(stats_path, cache_config, invalidate_legacy=False)
    if statistics is None:
        raise FileNotFoundError(
            f"Precomputed {description} not found (or cache config mismatch): "
            f"{stats_path}. Precomputed-statistics mode is on "
            "(SPARK_0_REQUIRE_PRECOMPUTED_STATS=1 or "
            "datasets.vla_data.require_precomputed_stats: true), so training "
            "refuses to compute statistics at launch. Run the offline "
            "precompute first with the SAME training config, e.g. "
            "`bash scripts/precompute_stats.sh <train_config>` in the Spark_0 "
            "adapter, or set SPARK_0_REQUIRE_PRECOMPUTED_STATS=0 to let "
            "rank 0 compute at launch."
        )
    return statistics


def _build_tactile_pressure_stats_cache_config(
    plan: dict[str, dict],
    *,
    exclude_episodes: set[int] | None = None,
    data_fingerprint: list | None = None,
    stats_mode: str | None = None,
) -> dict:
    config = {
        "kind": "tactile_pressure",
        "streams": {
            key: {"kind": spec["kind"], "columns": list(spec["columns"])}
            for key, spec in sorted(plan.items())
        },
    }
    if stats_mode == "streaming":
        config["stats_mode"] = "streaming"
    if exclude_episodes:
        config["exclude_episodes"] = sorted(int(v) for v in exclude_episodes)
    if data_fingerprint is not None:
        config["data_fingerprint"] = data_fingerprint
    return config


def _load_or_compute_tactile_pressure_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    plan: dict[str, dict],
    *,
    dataset_name: str,
    stats_mode: str,
    data_cfg=None,
) -> dict:
    """Same writer-election / precomputed-strict contract as action-state stats."""
    if _force_stats_recompute():
        _check_force_stats_recompute_allowed(data_cfg)
        statistics, cached = None, False
    elif _precomputed_stats_required(data_cfg):
        return _load_precomputed_stats_or_raise(
            stats_path,
            stats_cache_config,
            description="tactile pressure statistics cache",
        )
    else:
        statistics, cached = _load_stats_cache_for_config(
            stats_path, stats_cache_config
        )
    if statistics is None:
        if not _is_stats_main_process():
            return _wait_for_stats_cache(
                _stats_cache_path(stats_path, stats_cache_config),
                stats_cache_config,
                description="tactile pressure statistics cache",
            )
        if _stats_process_rank() == 0:
            print(
                f"[RANK 0] Calculating tactile pressure statistics for "
                f"{dataset_name} (stats_mode={stats_mode})"
            )
        statistics = calculate_tactile_pressure_statistics(
            parquet_paths, plan, mode=stats_mode
        )
        cached = False
    if not cached and _is_stats_main_process():
        _save_stats_cache(
            _stats_cache_path(stats_path, stats_cache_config),
            stats_cache_config,
            statistics,
        )
    return statistics


def apply_merged_tactile_pressure_scales(datasets, data_cfg=None) -> dict:
    """Widest-q99 merge across mixture members; write numbers onto ``data_cfg``.

    Same envelope rule as action ``percentile_mixing_method: min_max``: the
    run uses ``max(q99)`` so no member's typical contact exceeds ~1. Members
    with an explicit numeric scale are left alone. The resolved mapping is
    written back so ``config.full.yaml`` freezes numbers for deploy.
    """
    from starVLA.model.modules.tactile.pressure import (
        is_auto_pressure_scale,
        scale_from_pressure_q99,
    )

    q99_by_key: dict[str, list[float]] = {}
    log1p_by_key: dict[str, bool] = {}
    auto_keys: set[str] = set()
    for dataset in datasets:
        spec = getattr(dataset, "_tactile_spec", None) or {}
        stats = getattr(dataset, "_tactile_pressure_stats", None) or {}
        for key, entry in spec.items():
            if not isinstance(entry, dict) or entry.get("kind") == "heatmap":
                continue
            if not entry.get("pressure_scale_auto"):
                continue
            auto_keys.add(key)
            log1p = bool(entry.get("pressure_log1p"))
            if key in log1p_by_key and log1p_by_key[key] != log1p:
                raise ValueError(
                    f"mixture members disagree on tactile_pressure_log1p "
                    f"for {key} ({log1p_by_key[key]} vs {log1p}); the "
                    "merged auto scale cannot pick both"
                )
            log1p_by_key[key] = log1p
            payload = stats.get(key) or {}
            if "q99" in payload:
                q99_by_key.setdefault(key, []).append(float(payload["q99"]))
    if not auto_keys:
        return {}
    resolved: dict[str, float] = {}
    for key in sorted(auto_keys):
        values = q99_by_key.get(key) or []
        if not values:
            raise ValueError(
                f"tactile_pressure_scale for {key} is auto but no |p| q99 "
                "was computed on any mixture member"
            )
        q99 = max(values)
        scale = scale_from_pressure_q99(q99, log1p_by_key.get(key, True))
        resolved[key] = scale
        for dataset in datasets:
            spec = getattr(dataset, "_tactile_spec", None) or {}
            entry = spec.get(key)
            if isinstance(entry, dict) and entry.get("pressure_scale_auto"):
                entry["pressure_scale"] = scale
    if data_cfg is not None and hasattr(data_cfg, "__setitem__"):
        current = data_cfg.get("tactile_pressure_scale") if hasattr(data_cfg, "get") else None
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(current):
                current = OmegaConf.to_container(current, resolve=True)
        except Exception:
            pass
        if current is None or is_auto_pressure_scale(current):
            data_cfg["tactile_pressure_scale"] = dict(resolved)
        elif isinstance(current, dict):
            for key, scale in resolved.items():
                if key not in current or is_auto_pressure_scale(current.get(key)):
                    current[key] = scale
            data_cfg["tactile_pressure_scale"] = current
    if _is_stats_main_process() and resolved:
        parts = ", ".join(f"{key}={scale:.4g}" for key, scale in resolved.items())
        print(f"[LeRobotMixtureDataset] tactile pressure scale from stats: {parts}")
    return resolved


def _compute_statistics_for_mode(
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
    stats_mode: str = "memory",
) -> dict:
    if _stats_process_rank() == 0:
        print(
            f"[RANK 0] Calculating dataset statistics for {dataset_name} "
            f"(mode={action_mode}, stats_mode={stats_mode})"
        )

    # delta/rel still pool their CONVERTED chunks below; only the base
    # per-column statistics stream today.
    base_stats = calculate_dataset_statistics(parquet_paths, mode=stats_mode)
    
    if action_mode == "abs":
        return base_stats

    if action_indices is None or state_indices is None:
        raise ValueError(
            "Both action and state modalities are required to compute "
            f"{action_mode} action mode statistics."
        )

    if action_mode == "delta":
        return calculate_delta_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    if action_mode == "rel":
        return calculate_rel_action_statistics(
            parquet_paths=parquet_paths,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            base_stats=base_stats,
        )
    raise ValueError(f"Unsupported action mode for statistics: {action_mode}")


def _load_or_compute_statistics(
    stats_path: Path,
    stats_cache_config: dict,
    parquet_paths: list[Path],
    dataset_name: str,
    action_mode: str,
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int] | None,
    state_indices: list[int] | None,
    action_mode_apply_keys: list[str] | None,
    action_mode_state_map: dict[str, str] | None,
    stats_mode: str = "memory",
) -> dict:
    if _force_stats_recompute():
        le_statistics, cached = None, False
    else:
        le_statistics, cached = _load_stats_cache_for_config(stats_path, stats_cache_config)
    if le_statistics is None:
        le_statistics = _compute_statistics_for_mode(
            parquet_paths=parquet_paths,
            dataset_name=dataset_name,
            action_mode=action_mode,
            lerobot_modality_meta=lerobot_modality_meta,
            action_keys_full=action_keys_full,
            state_keys_full=state_keys_full,
            action_indices=action_indices,
            state_indices=state_indices,
            action_mode_apply_keys=action_mode_apply_keys,
            action_mode_state_map=action_mode_state_map,
            stats_mode=stats_mode,
        )
    if not cached:
        # Freshly computed, or adopted from the legacy shared path: give this
        # config its own file so later runs read it from there.
        _save_stats_cache(
            _stats_cache_path(stats_path, stats_cache_config),
            stats_cache_config,
            le_statistics,
        )
    return le_statistics


def _get_action_col_slices(
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
) -> dict[str, list[tuple[tuple[int, int], str, tuple[int, int], str, str]]]:
    apply_keys = _normalize_action_mode_apply_keys(action_mode_apply_keys, action_keys_full)
    action_mode_state_map = _normalize_action_mode_state_map(action_mode_state_map)

    action_meta = lerobot_modality_meta.action
    state_meta = lerobot_modality_meta.state

    # Build per-column mapping: action column -> list of (action_slice, state_column, state_slice)
    action_col_slices: dict[str, list[tuple[tuple[int, int], str, tuple[int, int]]]] = {}
    for action_key in apply_keys:
        if not action_key.startswith("action."):
            raise ValueError(f"Invalid action key {action_key}. Expected prefix 'action.'.")
        state_key = action_mode_state_map.get(action_key, action_key.replace("action.", "state.", 1))
        if state_key not in state_keys_full:
            raise ValueError(
                f"State key {state_key} not found for action key {action_key}. "
                f"Add it to action_mode_state_map or remove {action_key} from action_mode_apply_keys."
            )

        action_subkey = action_key.replace("action.", "", 1)
        state_subkey = state_key.replace("state.", "", 1)
        if action_subkey not in action_meta or state_subkey not in state_meta:
            raise ValueError(f"Action/state key missing in metadata: {action_key} -> {state_key}")

        action_cfg = action_meta[action_subkey]
        state_cfg = state_meta[state_subkey]
        action_col = action_cfg.original_key or action_subkey
        state_col = state_cfg.original_key or state_subkey
        action_slice = (action_cfg.start, action_cfg.end)
        state_slice = (state_cfg.start, state_cfg.end)
        action_padding = "first_last" if action_cfg.absolute else "zero"
        state_padding = "first_last" if state_cfg.absolute else "zero"
        action_col_slices.setdefault(action_col, []).append(
            (action_slice, state_col, state_slice, action_padding, state_padding)
        )

    return action_col_slices


def calculate_delta_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using delta mode.

    Rule:
      - For t>0: a_t - a_{t-1}
      - For t=0: a_0 - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting delta action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk.copy()
                    if len(out) > 1:
                        out[1:] = action_part_chunk[1:] - action_part_chunk[:-1]
                    out[0] = action_part_chunk[0] - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    delta_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        delta_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return delta_stats


def calculate_rel_action_statistics(
    parquet_paths: list[Path],
    lerobot_modality_meta: "LeRobotModalityMetadata",
    action_keys_full: list[str],
    state_keys_full: list[str],
    action_indices: list[int],
    state_indices: list[int],
    action_mode_apply_keys: list[str] | None = None,
    action_mode_state_map: dict[str, str] | None = None,
    base_stats: dict | None = None,
) -> dict:
    """
    Calculate action statistics using rel mode.

    Rule:
      - For all t: a_t - s_0

    Mapping rule (only two cases):
      1) Use explicit action_mode_state_map if provided.
      2) Otherwise, replace 'action.' with 'state.' directly.
    """
    if base_stats is None:
        base_stats = calculate_dataset_statistics(parquet_paths)

    action_col_slices = _get_action_col_slices(
        lerobot_modality_meta, action_keys_full, state_keys_full, action_mode_apply_keys, action_mode_state_map
    )
    if not action_col_slices:
        raise ValueError("No action columns found in the dataset.")

    def _get_chunk(array: np.ndarray, step_indices: np.ndarray, padding_strategy: str) -> np.ndarray:
        max_length = array.shape[0]
        front_padding = step_indices < 0
        end_padding = step_indices >= max_length
        padding_positions = np.logical_or(front_padding, end_padding)
        output = np.zeros((len(step_indices), array.shape[1]), dtype=array.dtype)
        if (~padding_positions).any():
            output[~padding_positions] = array[step_indices[~padding_positions]]
        if padding_positions.any():
            if padding_strategy == "first_last":
                output[front_padding] = array[0]
                output[end_padding] = array[-1]
            elif padding_strategy == "zero":
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    accum: dict[str, list[np.ndarray]] = {col: [] for col in action_col_slices.keys()}
    for parquet_path in tqdm(sorted(list(parquet_paths)), desc="Collecting rel action stats"):
        data = pd.read_parquet(parquet_path)
        trajectory_length = len(data)
        for action_col, slice_list in action_col_slices.items():
            if action_col not in data.columns:
                raise ValueError(f"{action_col} not found in parquet columns.")
            action_matrix = np.stack(data[action_col])
            action_padding_ref = slice_list[0][3]
            prepared_slices = []
            for a_slice, state_col, s_slice, action_padding, state_padding in slice_list:
                if state_col not in data.columns:
                    raise ValueError(f"{state_col} not found in parquet columns.")
                state_matrix = np.stack(data[state_col])
                state_part_full = state_matrix[:, s_slice[0] : s_slice[1]]
                prepared_slices.append((a_slice, state_part_full, state_padding))
            for base_index in range(trajectory_length):
                action_steps = np.array(action_indices) + base_index
                action_chunk_full = _get_chunk(action_matrix, action_steps, action_padding_ref)

                for a_slice, state_part_full, state_padding in prepared_slices:
                    action_part_chunk = action_chunk_full[:, a_slice[0] : a_slice[1]]
                    state_chunk = _get_chunk(state_part_full, np.array(state_indices) + base_index, state_padding)
                    if action_part_chunk.shape[1] != state_chunk.shape[1]:
                        raise ValueError(f"Action/state dim mismatch for {action_col}:{a_slice}")

                    out = action_part_chunk - state_chunk[0]
                    action_chunk_full[:, a_slice[0] : a_slice[1]] = out

                accum[action_col].append(action_chunk_full)

    rel_stats = copy.deepcopy(base_stats)
    for action_col, series_list in accum.items():
        if not series_list:
            continue
        all_values = np.concatenate(series_list, axis=0).astype(np.float32)
        rel_stats[action_col] = {
            "mean": np.mean(all_values, axis=0).tolist(),
            "std": np.std(all_values, axis=0).tolist(),
            "min": np.min(all_values, axis=0).tolist(),
            "max": np.max(all_values, axis=0).tolist(),
            "q01": np.quantile(all_values, 0.01, axis=0).tolist(),
            "q99": np.quantile(all_values, 0.99, axis=0).tolist(),
        }
    return rel_stats

class ModalityConfig(BaseModel):
    """Configuration for a modality."""

    delta_indices: list[int]
    """Delta indices to sample relative to the current index. The returned data will correspond to the original data at a sampled base index + delta indices."""
    modality_keys: list[str]
    """The keys to load for the modality in the dataset."""


class LeRobotSingleDataset(Dataset):
    """
    Base dataset class for LeRobot that supports sharding.
    """

    # Process-wide, so a mixture of datasets warns once in total.
    _warned_tactile_row_groups = False
    _warned_taxel_delta = False
    # Class-level defaults for the tactile accessors: statistics helpers and
    # subclasses reach them through partially-constructed instances, and "no
    # tactile" is the correct answer for every dataset that declares none.
    _tactile_spec: dict = {}
    _tactile_pressure_stats: dict = {}
    # Parquet columns named in modality.json's tactile block — including a
    # heatmap that this run chose not to pack. Always excluded from pandas.
    _tactile_parquet_columns: set = set()
    _curr_traj_source: tuple = (None, 0)
    _parquet_handle_pid: int | None = None
    _parquet_handles: dict = {}
    _tactile_row_group_cache: tuple = (None, None)

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg = None,
        **kwargs,
    ):
        """
        Initialize the dataset.

        Args:
            dataset_path (Path | str): The path to the dataset.
            modality_configs (dict[str, ModalityConfig]): The configuration for each modality. The keys are the modality names, and the values are the modality configurations.
                See `ModalityConfig` for more details.
            video_backend (str): Backend for video reading.
            video_backend_kwargs (dict): Keyword arguments for the video backend when initializing the video reader.
            transforms (ComposedModalityTransform): The transforms to apply to the dataset.
            embodiment_tag (EmbodimentTag): Overload the embodiment tag for the dataset. e.g. define it as "new_embodiment"
        """
        # first check if the path directory exists
        self.data_cfg = data_cfg
        if not Path(dataset_path).exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        # indict letobot version
        self._lerobot_version =  self.data_cfg.get("lerobot_version", "v2.0") #self._indict_lerobot_version(**kwargs)

        # Episodes dropped from BOTH sampling and statistics (see
        # resolve_excluded_episodes).
        self._excluded_episodes = resolve_excluded_episodes(self.data_cfg)

        self._action_mode = None
        self._action_mode_state_map = {}
        self._action_mode_apply_keys = None

        self.delete_pause_frame = delete_pause_frame

        self.modality_configs = modality_configs
        self.video_backend = video_backend
        self.video_backend_kwargs = video_backend_kwargs if video_backend_kwargs is not None else {}
        self.transforms = (
            transforms if transforms is not None else ComposedModalityTransform(transforms=[])
        )

        self._dataset_path = Path(dataset_path)
        self._dataset_name = self._dataset_path.name
        if isinstance(embodiment_tag, EmbodimentTag):
            self.tag = embodiment_tag.value
        else:
            self.tag = embodiment_tag

        self._init_action_mode()
        # Resolved before the metadata/statistics pass: that pass reads the
        # parquet files too, and it must skip the heavy tactile columns.
        self._tactile_spec = self._get_tactile_spec()
        self._tactile_pressure_stats: dict = {}
        self._metadata = self._get_metadata(EmbodimentTag(self.tag))
        self._resolve_tactile_pressure_scales()
        self._report_tactile_conditioning()

        # LeRobot-specific config
        self._lerobot_modality_meta = self._get_lerobot_modality_meta()
        self._lerobot_info_meta = self._get_lerobot_info_meta()
        self._data_path_pattern = self._get_data_path_pattern()
        self._video_path_pattern = self._get_video_path_pattern()
        self._chunk_size = self._get_chunk_size()
        self._tasks = self._get_tasks()
        # self._episodes = self._get_episode_info() # TODO why we need this func
        self.curr_traj_data = None
        self.curr_traj_id = None

        self._trajectory_ids, self._trajectory_lengths = self._get_trajectories()
        self._modality_keys = self._get_modality_keys()
        # Run-level camera slots (frozen enable set) before any per-dataset
        # intersection. flexible_views may shrink modality_keys["video"] to the
        # cameras this dataset's modality.json actually provides; packing still
        # reports a view_mask over these run slots so mixed batches stay aligned.
        self._run_video_keys = list(self._modality_keys.get("video") or [])
        self._view_mask_template = [1] * len(self._run_video_keys)
        self._apply_flexible_views()
        self._delta_indices = self._get_delta_indices()
        self._all_steps = self._get_all_steps()
        self.set_transforms_metadata(self.metadata)
        self.set_epoch(0)

        if int(os.environ.get("RANK", "0")) == 0:
            print(f"Initialized dataset {self.dataset_name} with {embodiment_tag}")

        # Check if the dataset is valid
        self._check_integrity()

    @property
    def dataset_path(self) -> Path:
        """The path to the dataset that contains the METADATA_FILENAME file."""
        return self._dataset_path

    @property
    def metadata(self) -> DatasetMetadata:
        """The metadata for the dataset, loaded from metadata.json in the dataset directory"""
        return self._metadata

    @property
    def trajectory_ids(self) -> np.ndarray:
        """The trajectory IDs in the dataset, stored as a 1D numpy array of strings."""
        return self._trajectory_ids

    @property
    def trajectory_lengths(self) -> np.ndarray:
        """The trajectory lengths in the dataset, stored as a 1D numpy array of integers.
        The order of the lengths is the same as the order of the trajectory IDs.
        """
        return self._trajectory_lengths

    @property
    def all_steps(self) -> list[tuple[int, int]]:
        """The trajectory IDs and base indices for all steps in the dataset.
        Example:
            self.trajectory_ids: [0, 1, 2]
            self.trajectory_lengths: [3, 2, 4]
            return: [
                ("traj_0", 0), ("traj_0", 1), ("traj_0", 2),
                ("traj_1", 0), ("traj_1", 1),
                ("traj_2", 0), ("traj_2", 1), ("traj_2", 2), ("traj_2", 3)
            ]
        """
        return self._all_steps

    @property
    def modality_keys(self) -> dict:
        """The modality keys for the dataset. The keys are the modality names, and the values are the keys for each modality.

        Example: {
            "video": ["video.image_side_0", "video.image_side_1"],
            "state": ["state.eef_position", "state.eef_rotation"],
            "action": ["action.eef_position", "action.eef_rotation"],
            "language": ["language.human.task"],
            "timestamp": ["timestamp"],
            "reward": ["reward"],
        }
        """
        return self._modality_keys

    @property
    def delta_indices(self) -> dict[str, np.ndarray]:
        """The delta indices for the dataset. The keys are the modality.key, and the values are the delta indices for each modality.key."""
        return self._delta_indices

    @property
    def dataset_name(self) -> str:
        """The name of the dataset."""
        return self._dataset_name

    @property
    def lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_modality_meta

    @property
    def lerobot_info_meta(self) -> dict:
        """The metadata for the LeRobot dataset."""
        return self._lerobot_info_meta

    @property
    def data_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._data_path_pattern

    @property
    def video_path_pattern(self) -> str:
        """The path pattern for the LeRobot dataset."""
        return self._video_path_pattern

    @property
    def chunk_size(self) -> int:
        """The chunk size for the LeRobot dataset."""
        return self._chunk_size

    @property
    def tasks(self) -> pd.DataFrame:
        """The tasks for the dataset."""
        return self._tasks

    def _get_metadata(self, embodiment_tag: EmbodimentTag) -> DatasetMetadata:
        """Get the metadata for the dataset.

        Returns:
            dict: The metadata for the dataset.
        """

        # 1. Modality metadata
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        # 1.1. State and action modalities
        simplified_modality_meta: dict[str, dict] = {}
        with open(modality_meta_path, "r") as f:
            le_modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        for modality in ["state", "action"]:
            simplified_modality_meta[modality] = {}
            le_state_action_meta: dict[str, LeRobotStateActionMetadata] = getattr(
                le_modality_meta, modality
            )
            for subkey in le_state_action_meta:
                state_action_dtype = np.dtype(le_state_action_meta[subkey].dtype)
                if np.issubdtype(state_action_dtype, np.floating):
                    continuous = True
                else:
                    continuous = False
                simplified_modality_meta[modality][subkey] = {
                    "absolute": le_state_action_meta[subkey].absolute,
                    "rotation_type": le_state_action_meta[subkey].rotation_type,
                    "shape": [
                        le_state_action_meta[subkey].end - le_state_action_meta[subkey].start
                    ],
                    "continuous": continuous,
                }

        # 1.2. Video modalities
        le_info_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        assert (
            le_info_path.exists()
        ), f"Please provide a {LE_ROBOT_INFO_FILENAME} file in {self.dataset_path}"
        with open(le_info_path, "r") as f:
            le_info = json.load(f)
        simplified_modality_meta["video"] = {}
        for new_key in le_modality_meta.video:
            original_key = le_modality_meta.video[new_key].original_key
            if original_key is None:
                original_key = new_key
            le_video_meta = le_info["features"][original_key]
            height = le_video_meta["shape"][le_video_meta["names"].index("height")]
            width = le_video_meta["shape"][le_video_meta["names"].index("width")]
            # NOTE(FH): different lerobot dataset versions have different keys for the number of channels and fps
            try:
                channels = le_video_meta["shape"][le_video_meta["names"].index("channel")]
                fps = le_video_meta["video_info"]["video.fps"]
            except (ValueError, KeyError):
                try:
                    channels = le_video_meta["info"]["video.channels"]
                    fps = le_video_meta["info"]["video.fps"]
                except (ValueError, KeyError):
                    # Fallback for image-only datasets (e.g. VLA-Arena) that lack video_info
                    channels = 3
                    fps = le_info.get("fps", 30)
            simplified_modality_meta["video"][new_key] = {
                "resolution": [width, height],
                "channels": channels,
                "fps": fps,
            }


        # 2. Dataset statistics
        is_main = _is_stats_main_process

        action_mode = _normalize_action_mode(self.data_cfg.get("action_mode", "abs") if self.data_cfg else "abs")

        stats_path = self.dataset_path / LE_ROBOT_STATS_FILENAME
        action_cfg = self.modality_configs.get("action")
        state_cfg = self.modality_configs.get("state")
        action_keys_full = list(action_cfg.modality_keys) if action_cfg else []
        state_keys_full = list(state_cfg.modality_keys) if state_cfg else []
        action_indices = list(action_cfg.delta_indices) if action_cfg else None
        state_indices = list(state_cfg.delta_indices) if state_cfg else None

        apply_keys = _normalize_action_mode_apply_keys(
            self.data_cfg.get("action_mode_apply_keys", None) if self.data_cfg else None,
            action_keys_full,
        )
        normalized_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) if self.data_cfg else {}
        )
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        # Same episode exclusion as _get_trajectories, so stats and sampling
        # cover the same episodes (v3 multi-episode files carry no episode
        # index in the name and are kept).
        parquet_files_filtered = [
            pf
            for pf in parquet_files
            if _episode_index_from_parquet_name(pf.name) not in self._excluded_episodes
        ]
        data_fingerprint = _dataset_data_fingerprint(
            self.dataset_path, parquet_files_filtered
        )
        # Resolved from the fingerprint's parquet bytes, so the offline
        # precompute and the training launch always agree on the mode (and
        # therefore on the cache hash) for the same data.
        raw_stats_mode = _resolve_stats_mode(
            self.data_cfg, sum(int(entry[1]) for entry in data_fingerprint)
        )
        stats_cache_config = _build_stats_cache_config(
            action_mode=action_mode,
            modality_key_ranges=self.data_cfg.get("modality_key_ranges") if self.data_cfg else None,
            exclude_episodes=self._excluded_episodes,
            data_fingerprint=data_fingerprint,
            action_mode_apply_keys=apply_keys,
            action_mode_state_map=normalized_state_map,
            action_indices=action_indices,
            state_indices=state_indices,
            meta_fingerprint=_dataset_meta_fingerprint(self.dataset_path),
            stats_mode=raw_stats_mode,
        )

        if _force_stats_recompute():
            # Refuse silently-misbehaving combinations up front: strict mode
            # (never computes) and multi-rank (stale reads on other ranks).
            _check_force_stats_recompute_allowed(self.data_cfg)

        if _precomputed_stats_required(self.data_cfg):
            # Strict mode: every rank reads the content-addressed cache
            # directly; nobody computes and nobody polls (there is no rank-0
            # writer to wait for). A miss fails the launch immediately.
            le_statistics = _load_precomputed_stats_or_raise(
                stats_path,
                stats_cache_config,
                description="dataset statistics cache",
            )
        else:
            if is_main():
                le_statistics = _load_or_compute_statistics(
                    stats_path,
                    stats_cache_config=stats_cache_config,
                    parquet_paths=parquet_files_filtered,
                    dataset_name=self.dataset_name,
                    action_mode=action_mode,
                    lerobot_modality_meta=le_modality_meta,
                    action_keys_full=action_keys_full,
                    state_keys_full=state_keys_full,
                    action_indices=action_indices,
                    state_indices=state_indices,
                    action_mode_apply_keys=apply_keys,
                    action_mode_state_map=normalized_state_map,
                    stats_mode=raw_stats_mode,
                )
            else:
                le_statistics = None

            if le_statistics is None:
                le_statistics = _wait_for_stats_cache(
                    _stats_cache_path(stats_path, stats_cache_config),
                    stats_cache_config,
                    description="dataset statistics cache",
                )

        for stat in le_statistics.values():
            DatasetStatisticalValues.model_validate(stat)


        dataset_statistics = {}
        for our_modality in ["state", "action"]:
            dataset_statistics[our_modality] = {}
            for subkey in simplified_modality_meta[our_modality]:
                dataset_statistics[our_modality][subkey] = {}
                state_action_meta = le_modality_meta.get_key_meta(f"{our_modality}.{subkey}")
                assert isinstance(state_action_meta, LeRobotStateActionMetadata)
                le_modality = state_action_meta.original_key
                for stat_name in le_statistics[le_modality]:
                    indices = np.arange(
                        state_action_meta.start,
                        state_action_meta.end,
                    )
                    stat = np.array(le_statistics[le_modality][stat_name])
                    dataset_statistics[our_modality][subkey][stat_name] = stat[indices].tolist()

        # 3. Full dataset metadata
        metadata = DatasetMetadata(
            statistics=dataset_statistics,  # type: ignore
            modalities=simplified_modality_meta,  # type: ignore
            embodiment_tag=embodiment_tag,
        )

        return metadata

    def _get_trajectories(self) -> tuple[np.ndarray, np.ndarray]:
        """Get the trajectories in the dataset."""
        # Get trajectory lengths, IDs, and whitelist from dataset metadata
        # v2.0
        if self._lerobot_version == "v2.0":
            file_path = self.dataset_path / LE_ROBOT_EPISODE_FILENAME
            with open(file_path, "r") as f:
                episode_metadata = [json.loads(line) for line in f]
            trajectory_ids = []
            trajectory_lengths = []
            for episode in episode_metadata:
                if int(episode["episode_index"]) in self._excluded_episodes:
                    continue
                trajectory_ids.append(episode["episode_index"])
                trajectory_lengths.append(episode["length"])
            return np.array(trajectory_ids), np.array(trajectory_lengths)
        # v3.0
        elif self._lerobot_version == "v3.0":
            file_paths = sorted(list((self.dataset_path).glob(LE_ROBOT3_EPISODE_FILENAME)))
            trajectory_ids = []
            trajectory_lengths = []
            # data_chunck_index = []
            # data_file_index = []
            # vido_from_index = []
            self.trajectory_ids_to_metadata = {}
            for file_path in file_paths:
                episodes_data = pd.read_parquet(file_path)
                timestamp_cols = [
                    c
                    for c in episodes_data.columns
                    if str(c).startswith("videos/") and str(c).endswith("/from_timestamp")
                ]
                for index, episode in episodes_data.iterrows():
                    if int(episode["episode_index"]) in self._excluded_episodes:
                        continue
                    trajectory_ids.append(episode["episode_index"])
                    trajectory_lengths.append(episode["length"])

                    from_timestamps = {}
                    for col in timestamp_cols:
                        value = episode[col]
                        if pd.isna(value):
                            continue
                        # videos/{video_key}/from_timestamp -> {video_key}
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        from_timestamps[video_key] = float(value)

                    # TODO auto map key 
                    # Collect video file indices for each video key
                    # Modified lerobot v3.0 video index (extract video and file indices)
                    video_file_indices = {}
                    for col in timestamp_cols:
                        video_key = str(col)[len("videos/") : -len("/from_timestamp")]
                        chunk_col = f"videos/{video_key}/chunk_index"
                        file_col = f"videos/{video_key}/file_index"
                        if chunk_col in episode and file_col in episode:
                            video_file_indices[video_key] = {
                                "chunk_index": int(episode[chunk_col]),
                                "file_index": int(episode[file_col]),
                            }
                    print(video_file_indices)
                    episode_meta = {
                        "data/chunk_index": episode["data/chunk_index"],
                        "data/file_index": episode["data/file_index"],
                        "data/file_from_index": index,
                        "videos/from_timestamps": from_timestamps,
                        "videos/file_indices": video_file_indices,
                    }
                    # episode_meta = {
                    #     "data/chunk_index": episode["data/chunk_index"],
                    #     "data/file_index": episode["data/file_index"],
                    #     "data/file_from_index": index,
                    #     "videos/from_timestamps": from_timestamps,
                    # }
                    self.trajectory_ids_to_metadata[trajectory_ids[-1]] = episode_meta

            # Should be able to directly read the saved index info here
            return np.array(trajectory_ids), np.array(trajectory_lengths)

    def _get_all_steps(self) -> list[tuple[int, int]]:
        """Get the trajectory IDs and base indices for all steps in the dataset.

        Returns:
            list[tuple[str, int]]: A list of (trajectory_id, base_index) tuples.
        """
        is_main = _is_stats_main_process

        config_key = self._get_steps_config_key()
        base_path = self.dataset_path / "meta" / "steps_data_index.pkl"
        # Content-addressed like the statistics caches. The file lives in the
        # shared DATASET directory, so with one fixed name two runs whose step
        # index differs (exclude_episodes, delete_pause_frame) each found a
        # mismatch, rebuilt and overwrote the other's file — and the read after
        # the barrier below took whatever was on disk, which could by then be
        # the OTHER run's index. Hashing the config into the name lets ablation
        # arms coexist.
        steps_path = base_path.with_name(f"{base_path.stem}__{config_key}{base_path.suffix}")

        def load(path: Path):
            """Steps from ``path``, or None when absent / another config's."""
            if not path.exists():
                return None
            try:
                with open(path, "rb") as f:
                    cached_data = pickle.load(f)
                # config_key embeds the data fingerprint, so a dataset
                # regenerated in place is rejected here instead of silently
                # sampling stale trajectory lengths. Kept inside the `try` with
                # the unpickling: a payload from an older format may not be a
                # dict at all, and that is a miss, not a crash.
                if cached_data.get("config_key") != config_key:
                    return None
                return cached_data["steps"]
            except Exception as e:
                # include EOFError / PickleError / AttributeError / KeyError; a
                # partially written or transiently unreadable file is a miss,
                # never a deletion.
                print(
                    f"[RANK {os.environ.get('RANK', 'NA')}] "
                    f"Failed to load cached steps from {path.name} ({e}), will rebuild."
                )
                return None

        steps = load(steps_path)
        if steps is not None:
            return steps
        # Pre-content-addressing runs wrote one shared filename; adopt it when
        # it belongs to THIS config rather than paying for a rebuild. A file
        # belonging to another config is left alone, never deleted.
        legacy_steps = load(base_path)
        if legacy_steps is not None:
            if is_main():
                self._save_steps_cache(steps_path, config_key, legacy_steps)
            return legacy_steps

        if is_main():
            all_steps = self._get_all_steps_single_process()
            self._save_steps_cache(steps_path, config_key, all_steps)
            # Never poll for our OWN output: on a single-process run nobody
            # else will ever publish it, so a failed write would hang here for
            # the full six-hour poll timeout instead of reporting the problem.
            return all_steps

        # The atomically published file is the synchronization primitive, as
        # for the statistics caches: a `dist.barrier()` here both risked the
        # NCCL timeout on a long scan and deadlocked whenever the ranks
        # disagreed about the cache hit (a hit returns early, without it).
        steps = load(steps_path)
        if steps is None:
            steps = self._wait_for_steps_cache(steps_path, load)
        return steps

    def _save_steps_cache(self, steps_path: Path, config_key: str, all_steps: list) -> None:
        cache_data = {
            "config_key": config_key,
            "steps": all_steps,
            "num_trajectories": len(self.trajectory_ids),
            "total_steps": len(all_steps),
            "computed_timestamp": pd.Timestamp.now().isoformat(),
            "delete_pause_frame": self.delete_pause_frame,
        }
        steps_path.parent.mkdir(parents=True, exist_ok=True)
        # Unique temp per writer + atomic replace (same reasoning as
        # _save_stats_cache): a fixed sibling ".tmp" lets one job rename the
        # file while another still has it open.
        tmp_path = None
        try:
            tmp_path = steps_path.with_name(
                f".{steps_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            with open(tmp_path, "xb") as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, steps_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)
        print(f"[RANK {os.environ.get('RANK', 'NA')}] Cached steps saved to {steps_path}")

    @staticmethod
    def _wait_for_steps_cache(steps_path: Path, load) -> list:
        """Poll for the rank-0-published step index (see _wait_for_stats_cache)."""
        timeout_s = float(os.environ.get("SPARK_0_STATS_CACHE_TIMEOUT_S", 6 * 60 * 60))
        poll_s = float(os.environ.get("SPARK_0_STATS_CACHE_POLL_S", 5.0))
        if (
            not math.isfinite(timeout_s)
            or not math.isfinite(poll_s)
            or timeout_s <= 0
            or poll_s <= 0
        ):
            raise ValueError(
                "SPARK_0_STATS_CACHE_TIMEOUT_S and SPARK_0_STATS_CACHE_POLL_S must be > 0"
            )
        deadline = time.monotonic() + timeout_s
        while True:
            steps = load(steps_path)
            if steps is not None:
                return steps
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_s}s waiting for the step-index cache "
                    f"at {steps_path}. Raise SPARK_0_STATS_CACHE_TIMEOUT_S if rank 0 "
                    "is still scanning the dataset."
                )
            time.sleep(poll_s)

    def _all_parquet_fingerprint(self) -> list:
        """Memoized ``[relpath, size, mtime_ns]`` for every data parquet.

        Three cache keys per dataset need it (step index, converted-pose
        statistics, and the streaming-mode threshold), and a fourth call site
        fingerprints the episode-filtered subset. Recomputing means a fresh
        glob + one ``stat`` per file EACH time, on every rank — thousands of
        NFS round trips per launch on a large tree. Everything that needs it
        runs during ``__init__``, so one snapshot is also more consistent
        than four.
        """
        # getattr: tests drive these helpers on partially-initialized fakes.
        cached = getattr(self, "_data_fingerprint_cache", None)
        if cached is None:
            cached = _dataset_data_fingerprint(
                self.dataset_path,
                self.dataset_path.glob(LE_ROBOT_DATA_FILENAME),
            )
            self._data_fingerprint_cache = cached
        return cached

    def _get_steps_config_key(self) -> str:
        """Generate a configuration key for steps caching."""
        # Intentionally omit ``dataset_name`` (``Path.name`` of the access
        # path): a shared tree reached as ``…/ego_pro_lerobot`` vs a
        # differently named symlink would otherwise mint two keys for one
        # physical ``meta/`` directory. The data fingerprint already scopes
        # the cache to this dataset's parquet files.
        config_dict = {
            "delete_pause_frame": self.delete_pause_frame,
            # Data fingerprint: regenerating the dataset in place changes the
            # key and forces a rebuild of the cached step index.
            "data_fingerprint": self._all_parquet_fingerprint(),
        }
        # The step index is built from the FILTERED trajectory list
        # (_get_trajectories drops datasets.vla_data.exclude_episodes), so a
        # changed exclusion list must invalidate the cache. Added only when
        # non-empty so caches from runs without exclusions stay valid (same
        # convention as the stats caches).
        if self._excluded_episodes:
            config_dict["exclude_episodes"] = sorted(int(v) for v in self._excluded_episodes)
        # Create a hash of the configuration
        config_str = str(sorted(config_dict.items()))
        return hashlib.md5(config_str.encode()).hexdigest()[:12]  #


    def _get_all_steps_single_process(self) -> list[tuple[int, int]]:
        """Original single-process implementation as fallback."""
        all_steps: list[tuple[int, int]] = []
        skipped_trajectories = 0
        processed_trajectories = 0
        
        # Check if language modality is configured
        has_language_modality = 'language' in self.modality_keys and len(self.modality_keys['language']) > 0
        # TODO why trajectory_length here, why not use data length?
        for trajectory_id, trajectory_length in tqdm(zip(self.trajectory_ids, self.trajectory_lengths), total=len(self.trajectory_ids), desc="Getting All Step"):
            try:
                if self._lerobot_version == "v2.0":
                    data = self.get_trajectory_data(trajectory_id)
                elif self._lerobot_version == "v3.0":
                    data = self.get_trajectory_data_lerobot_v3(trajectory_id)
                
                trajectory_skipped = False
            
                # Check if trajectory has valid language instruction (if language modality is configured)
                if has_language_modality:
                    self.curr_traj_data = data  # Set current trajectory data for get_language to work

                    language_instruction = self.get_language(trajectory_id, self.modality_keys['language'][0], 0)
                    if not language_instruction or language_instruction[0] == "":
                        print(f"Skipping trajectory {trajectory_id} due to empty language instruction")
                        skipped_trajectories += 1
                        trajectory_skipped = True
                        continue

            except Exception as e:
                print(f"Skipping trajectory {trajectory_id} due to read error: {e}")
                skipped_trajectories += 1
                trajectory_skipped = True
                continue
        
            if not trajectory_skipped:
                processed_trajectories += 1
        
            for base_index in range(trajectory_length):
                all_steps.append((trajectory_id, base_index))
                
        # Print summary statistics
        print(f"Single-process summary: Processed {processed_trajectories} trajectories, skipped {skipped_trajectories} empty trajectories")
        print(f"Total steps: {len(all_steps)} from {len(self.trajectory_ids)} trajectories")
                   
        return all_steps

    def _get_position_and_gripper_values(self, data: pd.DataFrame) -> tuple[list, list]:
        """Get position and gripper values based on available columns in the dataset."""
        # Get action keys from modality_keys
        action_keys = self.modality_keys.get('action', [])
        
        # Extract position data
        delta_position_values = None
        position_candidates = ['delta_eef_position']
        coordinate_candidates = ['x', 'y', 'z']
        
        # First try combined position fields
        for pos_key in position_candidates:
            full_key = f"action.{pos_key}"
            if full_key in action_keys:
                try:
                    # Get the lerobot key for this modality
                    le_action_cfg = self.lerobot_modality_meta.action
                    subkey = pos_key
                    if subkey in le_action_cfg:
                        le_key = le_action_cfg[subkey].original_key or subkey
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[subkey].start, le_action_cfg[subkey].end)
                            filtered_data = data_array[:, le_indices]
                            delta_position_values = filtered_data.tolist()
                            break
                except Exception:
                    continue
        
        # If combined fields not found, try individual x,y,z coordinates
        if delta_position_values is None:
            x_data, y_data, z_data = None, None, None
            for coord in coordinate_candidates:
                full_key = f"action.{coord}"
                if full_key in action_keys:
                    try:
                        le_action_cfg = self.lerobot_modality_meta.action
                        if coord in le_action_cfg:
                            le_key = le_action_cfg[coord].original_key or coord
                            if le_key in data.columns:
                                data_array = np.stack(data[le_key])
                                le_indices = np.arange(le_action_cfg[coord].start, le_action_cfg[coord].end)
                                coord_data = data_array[:, le_indices].flatten()
                                if coord == 'x':
                                    x_data = coord_data
                                elif coord == 'y':
                                    y_data = coord_data
                                elif coord == 'z':
                                    z_data = coord_data
                    except Exception:
                        continue
            
            if x_data is not None and y_data is not None and z_data is not None:
                delta_position_values = np.column_stack((x_data, y_data, z_data)).tolist()
        
        if delta_position_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.delta_eef_position' in data.columns:
                delta_position_values = data['action.delta_eef_position'].to_numpy().tolist()
            elif all(col in data.columns for col in ['action.x', 'action.y', 'action.z']):
                x_vals = data['action.x'].to_numpy()
                y_vals = data['action.y'].to_numpy() 
                z_vals = data['action.z'].to_numpy()
                delta_position_values = np.column_stack((x_vals, y_vals, z_vals)).tolist()
            else:
                raise ValueError(f"No suitable position columns found. Available columns: {data.columns.tolist()}")
        
        # Extract gripper data
        gripper_values = None
        gripper_candidates = ['gripper_close', 'gripper']
        
        for grip_key in gripper_candidates:
            full_key = f"action.{grip_key}"
            if full_key in action_keys:
                try:
                    le_action_cfg = self.lerobot_modality_meta.action
                    if grip_key in le_action_cfg:
                        le_key = le_action_cfg[grip_key].original_key or grip_key
                        if le_key in data.columns:
                            data_array = np.stack(data[le_key])
                            le_indices = np.arange(le_action_cfg[grip_key].start, le_action_cfg[grip_key].end)
                            gripper_data = data_array[:, le_indices].flatten()
                            gripper_values = gripper_data.tolist()
                            break
                except Exception:
                    continue
        
        if gripper_values is None:
            # Fallback to the old hardcoded approach if metadata approach fails
            if 'action.gripper_close' in data.columns:
                gripper_values = data['action.gripper_close'].to_numpy().tolist()
            elif 'action.gripper' in data.columns:
                gripper_values = data['action.gripper'].to_numpy().tolist()
            else:
                raise ValueError(f"No suitable gripper columns found. Available columns: {data.columns.tolist()}")
        
        return delta_position_values, gripper_values

    def _get_modality_keys(self) -> dict:
        """Get the modality keys for the dataset.
        The keys are the modality names, and the values are the keys for each modality.
        See property `modality_keys` for the expected format.
        """
        modality_keys = defaultdict(list)
        for modality, config in self.modality_configs.items():
            modality_keys[modality] = config.modality_keys
        return modality_keys

    @staticmethod
    def _short_camera_name(key: str) -> str:
        text = str(key)
        return text.split(".", 1)[-1] if text.startswith("video.") else text

    @staticmethod
    def _parse_flexible_views_flag(value) -> bool:
        """Parse ``flexible_views`` without treating the string ``\"false\"`` as True."""
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off", ""}:
                return False
        raise ValueError(
            f"datasets.vla_data.flexible_views must be a boolean, got {value!r}."
        )

    def _flexible_views_enabled(self) -> bool:
        if self.data_cfg is None:
            return False
        return self._parse_flexible_views_flag(
            self.data_cfg.get("flexible_views", False)
        )

    def _required_view_shorts(self) -> list[str]:
        """Cameras that must exist in modality.json when flexible_views is on.

        Default: the first run-slot camera (typically the head). An empty list
        disables the required-view check (still need at least one present view).
        """
        if self.data_cfg is None:
            return []
        raw = self.data_cfg.get("required_views", None)
        if raw is None:
            if self._run_video_keys:
                return [self._short_camera_name(self._run_video_keys[0])]
            return []
        if isinstance(raw, str):
            raw = [raw]
        return [self._short_camera_name(item) for item in raw]

    @staticmethod
    def _assert_trailing_absent_mask(view_mask: list[int], run_keys: list[str]) -> None:
        """Reject non-prefix masks so packed image positions stay unambiguous.

        The VLM only sees the present subsequence (no slot-id tokens). A mask
        like ``[1,0,1]`` (head+right, no left) would pack ``[head, right]`` and
        make image[1] mean a different camera than ``[1,1,0]`` → ``[head, left]``.
        Requiring absences to be a trailing suffix of the run slots keeps
        packed position i = the i-th present catalog camera (head-only /
        head+left / head+both wrists are all valid under the default catalog).
        """
        seen_absent = False
        for present in view_mask:
            if not present:
                seen_absent = True
            elif seen_absent:
                raise ValueError(
                    "flexible_views requires absent cameras to be a trailing "
                    f"suffix of the run slots {run_keys} (got view_mask="
                    f"{view_mask}). Non-prefix masks make packed image "
                    "positions ambiguous to the VLM (no slot-id tokens). "
                    "Reorder the catalog or provide the missing middle camera."
                )

    def _apply_flexible_views(self) -> None:
        """Intersect run camera slots with this dataset's modality.json keys.

        When ``datasets.vla_data.flexible_views`` is false (default), leave
        modality_keys and the all-ones mask untouched so ``_check_integrity``
        still requires every active camera — today's behaviour.

        When true, drop cameras missing from ``meta/modality.json`` from the
        load set, keep ``_run_video_keys`` as the fixed slot order, and build
        ``_view_mask_template`` so packing can emit a stable mask for mixture
        batches that mix head-only and multi-view datasets.
        """
        run_keys = list(self._run_video_keys)
        if not run_keys or not self._flexible_views_enabled():
            return

        available = set(self.lerobot_modality_meta.video.keys())
        present = [key for key in run_keys if self._short_camera_name(key) in available]
        missing = [key for key in run_keys if self._short_camera_name(key) not in available]
        required = self._required_view_shorts()
        missing_required = [name for name in required if name not in available]
        if missing_required:
            alias_hints = []
            for name in missing_required:
                found = [alias for alias in VIDEO_NAME_ALIASES.get(name, ()) if alias in available]
                if found:
                    alias_hints.append(
                        f"{name!r} is missing but modality.json has {found} — "
                        "training matches camera names exactly (deploy aliases "
                        "cam_high↔cam_head). Rename the modality.json video key "
                        "or the run slot."
                    )
            extra = (" " + " ".join(alias_hints)) if alias_hints else ""
            raise ValueError(
                f"Dataset {self.dataset_name}: flexible_views requires "
                f"{required} in modality.json, but missing {missing_required}. "
                f"Available video keys: {sorted(available)}.{extra}"
            )
        if not present:
            raise ValueError(
                f"Dataset {self.dataset_name}: flexible_views selected no cameras "
                f"from run slots {run_keys}; modality.json has {sorted(available)}."
            )

        view_mask = [
            1 if self._short_camera_name(key) in available else 0 for key in run_keys
        ]
        self._assert_trailing_absent_mask(view_mask, run_keys)
        self._view_mask_template = view_mask
        self._modality_keys["video"] = present
        if missing and int(os.environ.get("RANK", "0")) == 0:
            print(
                f"[flexible_views] {self.dataset_name}: using {present}, "
                f"masking absent {missing} (view_mask={self._view_mask_template})."
            )

    def _get_delta_indices(self) -> dict[str, np.ndarray]:
        """Restructure the delta indices to use modality.key as keys instead of just the modalities."""
        # Prefer the (possibly flexible_views-shrunk) modality_keys so we do not
        # build delta indices for cameras this dataset will never load.
        delta_indices: dict[str, np.ndarray] = {}
        for modality, keys in self.modality_keys.items():
            config = self.modality_configs.get(modality)
            if config is None:
                continue
            for key in keys:
                delta_indices[key] = np.array(config.delta_indices)
        return delta_indices

    def _init_action_mode(self) -> None:
        if self.data_cfg is None:
            self._action_mode = "abs"
            return

        action_mode = self.data_cfg.get("action_mode", "abs")
        if action_mode is None:
            action_mode = "abs"
        action_mode = _normalize_action_mode(action_mode)
        if action_mode not in {"abs", "delta", "rel"}:
            raise ValueError(f"Invalid action_mode: {action_mode}. Expected one of: abs, delta, rel.")
        self._action_mode = action_mode

        apply_keys = _normalize_action_mode_apply_keys(self.data_cfg.get("action_mode_apply_keys", None))
        if apply_keys:
            self._action_mode_apply_keys = apply_keys

        self._action_mode_state_map = _normalize_action_mode_state_map(
            self.data_cfg.get("action_mode_state_map", {}) or {}
        )

    def _infer_state_key_for_action(self, action_key: str) -> str | None:
        if action_key in self._action_mode_state_map:
            return self._action_mode_state_map[action_key]

        if not action_key.startswith("action."):
            return None
        base = action_key.replace("action.", "", 1)
        if f"state.{base}" in self.modality_keys.get("state", []):
            return f"state.{base}"
        return None

    def _apply_action_mode(self, data: dict) -> dict:
        if self._action_mode in (None, "abs"):
            return data

        action_keys = self._action_mode_apply_keys or self.modality_keys.get("action", [])
        for action_key in action_keys:
            if action_key not in data:
                print(f"[WARNING] Action key {action_key} not found in data")
                continue
            state_key = self._infer_state_key_for_action(action_key)

            # for safety, check if the state key is valid
            if state_key is None or state_key not in data:
                continue

            action_values = np.asarray(data[action_key])
            state_values = np.asarray(data[state_key])
            if action_values.ndim != 2 or state_values.ndim != 2:
                raise ValueError(
                    f"Expected 2D arrays for action/state, got {action_key}: {action_values.shape}, {state_key}: {state_values.shape}"
                )
            if action_values.shape[1] != state_values.shape[1]:
                raise ValueError(
                    f"Action/state dim mismatch for {action_key} vs {state_key}: {action_values.shape} vs {state_values.shape}"
                )

            state0 = state_values[0]
            if self._action_mode == "delta":
                out = action_values.copy()
                if len(out) > 1:
                    out[1:] = action_values[1:] - action_values[:-1]
                out[0] = action_values[0] - state0
            elif self._action_mode == "rel":
                out = action_values - state0
            else:
                out = action_values

            data[action_key] = out

        return data

    def _get_lerobot_modality_meta(self) -> LeRobotModalityMetadata:
        """Get the metadata for the LeRobot dataset."""
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        assert (
            modality_meta_path.exists()
        ), f"Please provide a {LE_ROBOT_MODALITY_FILENAME} file in {self.dataset_path}"
        with open(modality_meta_path, "r") as f:
            modality_meta = LeRobotModalityMetadata.model_validate(json.load(f))
        modality_key_ranges = self.data_cfg.get("modality_key_ranges") if self.data_cfg else None
        if modality_key_ranges:
            for modality, key_ranges in modality_key_ranges.items():
                meta_for_modality = getattr(modality_meta, modality, None)
                if meta_for_modality is None:
                    continue
                for key, key_range in key_ranges.items():
                    if key not in meta_for_modality:
                        continue
                    start, end = key_range
                    meta_for_modality[key].start = int(start)
                    meta_for_modality[key].end = int(end)
        return modality_meta

    def _get_tactile_spec(self) -> dict:
        """Resolve the optional ``tactile`` block of ``meta/modality.json``.

        Returns ``{batch_key: spec}`` — empty for every dataset that declares
        no tactile block, which is the no-op path.

        Two LeRobot layouts are accepted:

        * Nested (optional region history): ``region_pressure_vector`` /
          ``taxel_pressure_3D`` with ``left_ee, right_ee`` hands.
        * Legacy LeRobot v2.1 Tianji converter:
          ``3d_left`` / ``3d_right`` (wrist-local xyz+pressure; shipped
          ``default`` stem expects 320, legacy converter files are 460)
          plus optional ``validity_left`` / ``validity_right``. Packed as
          ``taxel_pressure_3d`` / ``taxel_valid_mask``.

        ``datasets.vla_data.pack_tactile`` is opt-in. Unset / false returns
        an empty spec (no tactile keys on the sample — V-only / baseline)
        and does not validate the tactile schema. Parquet column names are
        still recorded so pandas never materialises them. ``true`` (written
        by ``sync_tactile_pack_flags`` / ``train.sh`` when a tactile stem is
        enabled) packs the declared streams.

        ``tactile.2d`` is a single HxW column (aligned with ``cam_high``).
        Set ``datasets.vla_data.tactile_pack_heatmap: true`` to pack it
        (``sync_tactile_pack_flags`` does this when ``tactile_align.enable``).

        ``datasets.vla_data.tactile_history`` (int, per-stream mapping keyed by
        batch key, or ``null``) sets the packed window. A bare int / the
        ``region_pressure`` mapping key truncates the **region** in-file
        history. An explicit ``taxel_pressure_3d`` mapping key stacks the last
        N episode policy frames (files have no taxel history axis). A bare
        int is not applied to taxel.

        ``datasets.vla_data.tactile_future_horizon`` (int, unset / 0 = off)
        slices the first H frames of the exported
        ``tactile.region_pressure_vector_future`` column into
        ``region_pressure_future`` ``[hands, regions, H, 1]`` plus
        ``region_pressure_future_valid`` ``[H]``. H counts GLOVE frames of
        that column (slot 0 nearest to now), not policy steps. Used by
        variant A future-contact; ``sync_tactile_pack_flags`` writes it from
        the stem when ``tactile_expert.future_contact.enable``. Off → the
        column is skipped like any other unpacked stream.

        ``datasets.vla_data.tactile_pressure_log1p`` / ``tactile_pressure_scale``
        (bool / float / ``auto``, or per-stream mappings) condition the
        PRESSURE channel. ``auto`` reads ``log1p(q99)`` of packed ``|p|``
        from ``meta/stats_xpolicy_tactile_pressure*``;
        optional ``tactile_pressure_center`` (``none`` / ``median``) and
        ``tactile_append_delta`` (region stream only) run in that order —
        see :meth:`_get_tactile_step_data`.
        """
        modality_meta_path = self.dataset_path / LE_ROBOT_MODALITY_FILENAME
        with open(modality_meta_path, "r") as f:
            declared = (json.load(f) or {}).get("tactile") or {}
        if not isinstance(declared, dict):
            self._tactile_parquet_columns = set()
            self._tactile_future_horizon = 0
            return {}

        def _stream_cfg(name):
            value = (self.data_cfg or {}).get(name)
            try:
                from omegaconf import OmegaConf

                if OmegaConf.is_config(value):
                    value = OmegaConf.to_container(value, resolve=True)
            except Exception:
                pass
            return value

        def _for_stream(value, batch_key):
            return value.get(batch_key) if isinstance(value, dict) else value

        def _requested_history(batch_key):
            # Bare int = region in-file window only. Taxel stacking needs
            # an explicit mapping key (or the encoder-sync copy).
            if batch_key == TACTILE_REGION_PRESSURE_KEY:
                return _for_stream(history_cfg, batch_key)
            if isinstance(history_cfg, dict):
                return history_cfg.get(batch_key)
            return None

        def _pressure_opts(batch_key):
            center = _for_stream(center_cfg, batch_key)
            center = str(center or "none").lower()
            if center in ("", "off", "false"):
                center = "none"
            if center not in ("none", "median"):
                raise ValueError(
                    "datasets.vla_data.tactile_pressure_center must be "
                    f"none|median, got {center!r} for {batch_key}"
                )
            append_delta = bool(_for_stream(delta_cfg, batch_key))
            if append_delta and batch_key != TACTILE_REGION_PRESSURE_KEY:
                # Δp is a region-stream-only channel. Taxel history is an
                # episode-stacked xyz+p window, not a Δp channel.
                if not LeRobotSingleDataset._warned_taxel_delta:
                    LeRobotSingleDataset._warned_taxel_delta = True
                    warnings.warn(
                        "datasets.vla_data.tactile_append_delta applies to the "
                        f"region stream only; ignoring it for {batch_key}.",
                        stacklevel=2,
                    )
                append_delta = False
            return {
                "pressure_center": center,
                "append_delta": append_delta,
            }

        history_cfg = _stream_cfg("tactile_history")
        log1p_cfg = _stream_cfg("tactile_pressure_log1p")
        scale_cfg = _stream_cfg("tactile_pressure_scale")
        center_cfg = _stream_cfg("tactile_pressure_center")
        delta_cfg = _stream_cfg("tactile_append_delta")
        pack_heatmap_cfg = _stream_cfg("tactile_pack_heatmap")
        pack_heatmap = bool(pack_heatmap_cfg)
        pack_tactile_cfg = _stream_cfg("pack_tactile")
        pack_tactile = bool(pack_tactile_cfg)
        future_horizon_cfg = _stream_cfg("tactile_future_horizon")
        future_horizon = 0
        if not _flag_unset(future_horizon_cfg) and future_horizon_cfg not in (False, 0, "0"):
            future_horizon = int(future_horizon_cfg)
            if future_horizon < 1:
                raise ValueError(
                    "datasets.vla_data.tactile_future_horizon must be >= 1, "
                    f"got {future_horizon}"
                )
        self._tactile_future_horizon = future_horizon

        if not pack_tactile:
            self._tactile_parquet_columns = _collect_declared_tactile_parquet_keys(
                declared
            )
            self._tactile_future_horizon = 0
            return {}

        spec: dict[str, dict] = {}
        parquet_columns: set[str] = set()
        for stream_name, entry in declared.items():
            batch_key = TACTILE_STREAM_TO_BATCH_KEY.get(stream_name)
            if batch_key is None or not isinstance(entry, dict):
                continue
            if batch_key == TACTILE_HEATMAP_KEY:
                original = str(entry.get("original_key") or "observation.tactile.2d")
                parquet_columns.add(original)
                if not pack_heatmap:
                    continue
                shape = entry.get("shape")
                hw = None
                if shape is not None and len(shape) >= 2:
                    hw = (int(shape[0]), int(shape[1]))
                    if min(hw) < 1:
                        raise ValueError(
                            f"[{self.dataset_name}] tactile.2d shape must be "
                            f"positive [H, W], got {list(shape)}"
                        )
                spec[batch_key] = {
                    "kind": "heatmap",
                    "columns": [original],
                    "hw": hw,
                    "history_len": None,
                    "history_order": "",
                    "pressure_log1p": False,
                    "pressure_scale": None,
                }
                continue
            hands = entry
            columns = []
            for hand in TACTILE_HAND_ORDER:
                hand_entry = hands.get(hand)
                if not isinstance(hand_entry, dict):
                    raise ValueError(
                        f"[{self.dataset_name}] modality.json tactile.{stream_name} "
                        f"is missing hand {hand!r}; expected {list(TACTILE_HAND_ORDER)}"
                    )
                columns.append(
                    str(
                        hand_entry.get("original_key")
                        or f"observation.tactile.{stream_name}.{hand}"
                    )
                )
            parquet_columns.update(columns)
            first_hand = hands[TACTILE_HAND_ORDER[0]]
            history_len = None
            hand_shape = first_hand.get("shape")
            if batch_key == TACTILE_REGION_FUTURE_KEY:
                # Pre-exported window: the run slices a prefix of the stored
                # frames, so nothing here depends on the episode. Left out of
                # the spec entirely when variant A is off, which is what keeps
                # a fusion-only run from decoding the column.
                if future_horizon < 1:
                    continue
                stored = int(hand_shape[0]) if hand_shape else None
                if stored is not None and stored < future_horizon:
                    raise ValueError(
                        f"[{self.dataset_name}] "
                        f"datasets.vla_data.tactile_future_horizon="
                        f"{future_horizon} exceeds the exported "
                        f"tactile.{TACTILE_REGION_FUTURE_STREAM} window "
                        f"({stored} frames)"
                    )
                spec[batch_key] = {
                    "kind": "region_future",
                    "columns": columns,
                    "future_len": future_horizon,
                    "stored_len": stored,
                    "num_units": _tactile_units_from_hand_shape(
                        TACTILE_REGION_PRESSURE_KEY, hand_shape
                    ),
                    "future_order": str(
                        first_hand.get("future_order") or "nearest_first"
                    ),
                }
                continue
            if batch_key == TACTILE_REGION_PRESSURE_KEY:
                if hand_shape:
                    history_len = int(hand_shape[0])
            num_units = _tactile_units_from_hand_shape(batch_key, hand_shape)
            requested = _requested_history(batch_key)
            if requested is not None:
                requested = int(requested)
                if requested < 1:
                    raise ValueError(
                        "datasets.vla_data.tactile_history must be >= 1, "
                        f"got {requested} for {batch_key}"
                    )
                if batch_key == TACTILE_REGION_PRESSURE_KEY:
                    history_len = (
                        requested if history_len is None else min(history_len, requested)
                    )
                else:
                    # Episode-stack length; the file cell is one frame.
                    history_len = requested
            scale, scale_auto = _parse_tactile_pressure_scale(
                _for_stream(scale_cfg, batch_key), batch_key
            )
            spec[batch_key] = {
                "kind": "hands",
                "columns": columns,
                "history_len": history_len,
                "num_units": num_units,
                "history_order": str(first_hand.get("history_order") or "most_recent_first"),
                "pressure_log1p": bool(_for_stream(log1p_cfg, batch_key)),
                "pressure_scale": scale,
                "pressure_scale_auto": scale_auto,
                **_pressure_opts(batch_key),
            }

        # Converter schema: tactile.3d_left / 3d_right (+ optional validity_*).
        if TACTILE_TAXEL_PRESSURE_KEY in spec and (
            isinstance(declared.get("3d_left"), dict)
            or isinstance(declared.get("3d_right"), dict)
        ):
            raise ValueError(
                f"[{self.dataset_name}] modality.json tactile block has both "
                "taxel_pressure_3D and 3d_left/3d_right; pick one layout"
            )
        if TACTILE_TAXEL_PRESSURE_KEY not in spec:
            left_entry = declared.get("3d_left")
            right_entry = declared.get("3d_right")
            if isinstance(left_entry, dict) or isinstance(right_entry, dict):
                if not isinstance(left_entry, dict) or not isinstance(right_entry, dict):
                    raise ValueError(
                        f"[{self.dataset_name}] modality.json tactile.3d_left and "
                        "tactile.3d_right must both be present"
                    )
                columns = []
                for (_, default_key), entry in zip(
                    TACTILE_CONVERTER_TAXEL, (left_entry, right_entry)
                ):
                    columns.append(str(entry.get("original_key") or default_key))
                parquet_columns.update(columns)
                valid_columns = []
                for stream_name, default_key in TACTILE_CONVERTER_VALID:
                    entry = declared.get(stream_name)
                    if isinstance(entry, dict):
                        valid_columns.append(
                            str(entry.get("original_key") or default_key)
                        )
                parquet_columns.update(valid_columns)
                if valid_columns and len(valid_columns) != 2:
                    raise ValueError(
                        f"[{self.dataset_name}] tactile validity_left and "
                        "validity_right must both be declared when either is present"
                    )
                scale, scale_auto = _parse_tactile_pressure_scale(
                    _for_stream(scale_cfg, TACTILE_TAXEL_PRESSURE_KEY),
                    TACTILE_TAXEL_PRESSURE_KEY,
                )
                requested = _requested_history(TACTILE_TAXEL_PRESSURE_KEY)
                taxel_history = 1
                if requested is not None:
                    taxel_history = int(requested)
                    if taxel_history < 1:
                        raise ValueError(
                            "datasets.vla_data.tactile_history must be >= 1, "
                            f"got {taxel_history} for {TACTILE_TAXEL_PRESSURE_KEY}"
                        )
                spec[TACTILE_TAXEL_PRESSURE_KEY] = {
                    "kind": "hands",
                    "columns": columns,
                    "history_len": taxel_history,
                    "num_units": _tactile_units_from_hand_shape(
                        TACTILE_TAXEL_PRESSURE_KEY, left_entry.get("shape")
                    ),
                    "history_order": "",
                    "pressure_log1p": bool(
                        _for_stream(log1p_cfg, TACTILE_TAXEL_PRESSURE_KEY)
                    ),
                    "pressure_scale": scale,
                    "pressure_scale_auto": scale_auto,
                    "valid_columns": valid_columns or None,
                    **_pressure_opts(TACTILE_TAXEL_PRESSURE_KEY),
                }
            else:
                for stream_name, default_key in TACTILE_CONVERTER_VALID:
                    entry = declared.get(stream_name)
                    if isinstance(entry, dict):
                        parquet_columns.add(
                            str(entry.get("original_key") or default_key)
                        )

        if (
            future_horizon >= 1
            and TACTILE_REGION_PRESSURE_KEY in spec
            and TACTILE_REGION_FUTURE_KEY not in spec
        ):
            raise ValueError(
                f"[{self.dataset_name}] future_contact is on "
                f"(tactile_future_horizon={future_horizon}) but "
                f"meta/modality.json declares no tactile."
                f"{TACTILE_REGION_FUTURE_STREAM}. Export the future window "
                f"next to the history one (columns "
                f"observation.tactile.{TACTILE_REGION_FUTURE_STREAM}."
                "{left_ee,right_ee}, cell [future_frames, regions], slot 0 "
                "nearest to the observation, frames past the episode end "
                "written as NaN), or turn the head off."
            )
        if TACTILE_REGION_FUTURE_KEY in spec:
            region = spec.get(TACTILE_REGION_PRESSURE_KEY)
            if region is None:
                # The head predicts FROM the region tokens, and the future
                # window borrows that stream's conditioning; a future column
                # without its history column is not a usable pack.
                raise ValueError(
                    f"[{self.dataset_name}] modality.json declares tactile."
                    f"{TACTILE_REGION_FUTURE_STREAM} without "
                    "tactile.region_pressure_vector; future-contact needs both."
                )
            future_units = spec[TACTILE_REGION_FUTURE_KEY].get("num_units")
            region_units = region.get("num_units")
            if (
                future_units is not None
                and region_units is not None
                and int(future_units) != int(region_units)
            ):
                # Targets are per region token; a mismatch would otherwise
                # surface as a shape error on the first batch.
                raise ValueError(
                    f"[{self.dataset_name}] tactile."
                    f"{TACTILE_REGION_FUTURE_STREAM} has {future_units} regions "
                    f"!= region_pressure_vector's {region_units}"
                )
            if region.get("pressure_center") == "median":
                # Both windows would be centered by their OWN median, so the
                # mean-Δp target (future minus current) would compare two
                # different offsets.
                raise ValueError(
                    f"[{self.dataset_name}] tactile_pressure_center=median "
                    "cannot be combined with future_contact: the history and "
                    "future windows would each subtract their own median and "
                    "the Δp target would be meaningless. Use center: none."
                )

        self._tactile_parquet_columns = parquet_columns
        return spec

    def _parquet_files_for_statistics(self) -> list[Path]:
        parquet_files = list(self.dataset_path.glob(LE_ROBOT_DATA_FILENAME))
        return [
            path
            for path in parquet_files
            if _episode_index_from_parquet_name(path.name) not in self._excluded_episodes
        ]

    def _report_tactile_conditioning(self) -> None:
        """One launch-time line per packing member: what the encoder is fed.

        The shipped fusion stems turned pressure conditioning on by default
        (log1p + ``scale: auto``); printing the resolved knobs keeps a rerun
        of an older recipe from silently training on a different distribution.
        """
        spec = self._tactile_spec or {}
        keys = sorted(
            key
            for key, entry in spec.items()
            if isinstance(entry, dict) and entry.get("kind") == "hands"
        )
        if not keys or not _is_stats_main_process():
            return
        parts = []
        for key in keys:
            entry = spec[key]
            scale = entry.get("pressure_scale")
            scale_s = (
                f"{scale:.4g}" if isinstance(scale, (int, float)) else str(scale)
            )
            extras = []
            if entry.get("pressure_center", "none") != "none":
                extras.append(f"center={entry['pressure_center']}")
            if entry.get("append_delta"):
                extras.append("Δp")
            extras_s = f", {', '.join(extras)}" if extras else ""
            history = entry.get("history_len")
            if history is None:
                history = 1
            future_s = ""
            if key == TACTILE_REGION_PRESSURE_KEY:
                future_h = int(getattr(self, "_tactile_future_horizon", 0) or 0)
                if future_h > 0:
                    future_s = f", future={future_h}"
            parts.append(
                f"{key}[T={history}, "
                f"log1p={entry.get('pressure_log1p')}, scale={scale_s}{extras_s}{future_s}]"
            )
        print(f"[{self.dataset_name}] tactile conditioning: " + " ".join(parts))

    def _resolve_tactile_pressure_scales(self) -> None:
        """Fill ``pressure_scale_auto`` streams from this dataset's |p| q99."""
        from starVLA.model.modules.tactile.pressure import scale_from_pressure_q99

        spec = self._tactile_spec or {}
        auto_keys = [
            key
            for key, entry in spec.items()
            if isinstance(entry, dict) and entry.get("pressure_scale_auto")
        ]
        if not auto_keys:
            self._tactile_pressure_stats = {}
            return
        plan = {
            key: entry
            for key, entry in tactile_pressure_stream_plan(spec).items()
            if key in auto_keys
        }
        if not plan:
            raise ValueError(
                f"[{self.dataset_name}] tactile_pressure_scale is auto but no "
                "region/taxel pressure columns are packed"
            )
        parquet_files = self._parquet_files_for_statistics()
        data_fingerprint = _dataset_data_fingerprint(self.dataset_path, parquet_files)
        stats_mode = _resolve_stats_mode(
            self.data_cfg, sum(int(entry[1]) for entry in data_fingerprint)
        )
        cache_config = _build_tactile_pressure_stats_cache_config(
            plan,
            exclude_episodes=self._excluded_episodes,
            data_fingerprint=data_fingerprint,
            stats_mode=stats_mode,
        )
        stats = _load_or_compute_tactile_pressure_statistics(
            self.dataset_path / LE_ROBOT_TACTILE_PRESSURE_STATS_FILENAME,
            cache_config,
            parquet_files,
            plan,
            dataset_name=self.dataset_name,
            stats_mode=stats_mode,
            data_cfg=self.data_cfg,
        )
        self._tactile_pressure_stats = stats
        for key in auto_keys:
            payload = stats.get(key)
            if not payload or "q99" not in payload:
                raise ValueError(
                    f"[{self.dataset_name}] tactile pressure stats missing q99 "
                    f"for {key}"
                )
            spec[key]["pressure_scale"] = scale_from_pressure_q99(
                payload["q99"], spec[key]["pressure_log1p"]
            )
        if _is_stats_main_process():
            parts = ", ".join(
                f"{key}={spec[key]['pressure_scale']:.4g} "
                f"(q99={stats[key]['q99']:.4g})"
                for key in auto_keys
            )
            print(
                f"[{self.dataset_name}] tactile pressure scale from stats: {parts}"
            )

    def tactile_packed_layout(self) -> dict:
        """Comparable packed-shape contract. Empty if this member packs no tactile."""
        return tactile_packed_layout_from_spec(self._tactile_spec)

    def _parquet_file(self, parquet_path: Path):
        """Process-local ``pq.ParquetFile`` handle cache (handles are not fork-safe)."""
        import pyarrow.parquet as pq

        pid = os.getpid()
        if getattr(self, "_parquet_handle_pid", None) != pid:
            self._parquet_handle_pid = pid
            self._parquet_handles = {}
            self._tactile_row_group_cache = (None, None)
        key = str(parquet_path)
        handle = self._parquet_handles.get(key)
        if handle is None:
            handle = pq.ParquetFile(parquet_path)
            self._parquet_handles[key] = handle
        return handle

    def _read_tactile_row(self, parquet_path: Path, row: int) -> dict:
        """Tactile columns of ONE row, decoding only that row's row group.

        Parquet's decode unit is the row group, so the cost of a random frame
        is set by how the file was written. LeRobot's writer produces one row
        group per file, which makes a single frame cost a full-file decode
        (measured 3.7 s on the 1.1 GB egopro files); repacking them into ~1k-row
        groups (``tools/repack_lerobot_row_groups.py``) takes that to ~26 ms,
        and the one-entry cache below then serves neighbouring frames for free.
        Fusion columns are cached as numpy; HxW heatmap columns stay arrow
        and are decoded per frame (see ``_TACTILE_MAX_NUMPY_CACHE_CELL_ELEMS``).
        A file that still has oversized row groups warns once rather than
        silently running at seconds per sample.
        """
        columns = []
        for spec in self._tactile_spec.values():
            columns.extend(spec["columns"])
            columns.extend(spec.get("valid_columns") or ())
        columns = list(dict.fromkeys(columns))
        handle = self._parquet_file(parquet_path)
        metadata = handle.metadata
        group_index, offset = 0, row
        for candidate in range(metadata.num_row_groups):
            group_rows = metadata.row_group(candidate).num_rows
            if offset < group_rows:
                group_index = candidate
                break
            offset -= group_rows
        else:  # row past the end; clamp to the last row of the last group
            group_index = metadata.num_row_groups - 1
            offset = metadata.row_group(group_index).num_rows - 1

        cache_key = (str(parquet_path), group_index)
        cached_key, cached_payload = self._tactile_row_group_cache
        if cached_key != cache_key:
            group_rows = metadata.row_group(group_index).num_rows
            if group_rows > _TACTILE_MAX_CACHED_ROW_GROUP_ROWS:
                if not LeRobotSingleDataset._warned_tactile_row_groups:
                    LeRobotSingleDataset._warned_tactile_row_groups = True
                    warnings.warn(
                        f"{parquet_path.name} stores {group_rows} rows per parquet row "
                        "group, so reading one frame's tactile decodes the whole group "
                        "(seconds per sample). Repack the dataset with "
                        "`python tools/repack_lerobot_row_groups.py <dataset>` to get "
                        "random-access reads.",
                        stacklevel=2,
                    )
                table = handle.read_row_group(group_index, columns=columns)
                return {
                    column: _arrow_cell_to_ndarray(table.column(column)[offset])
                    for column in columns
                }
            table = handle.read_row_group(group_index, columns=columns)
            cached_payload = _cache_tactile_row_group(table, columns, group_rows)
            self._tactile_row_group_cache = (cache_key, cached_payload)
        return _tactile_row_from_cache(cached_payload, columns, offset)

    @staticmethod
    def _condition_tactile_pressure(pressure: np.ndarray, spec: dict) -> np.ndarray:
        """Median-center (optional) then ``sign(p) * log1p(|p|) / scale``.

        Shared with deploy via ``modules/tactile/pressure.py``. Contact forces
        are heavy-tailed and mostly zero; raw newtons alias the taxel Fourier
        features. Center removes donning / thermal offset before the log.
        """
        from starVLA.model.modules.tactile.pressure import (
            center_pressure,
            condition_tactile_pressure,
        )

        pressure = center_pressure(pressure, spec.get("pressure_center"))
        return condition_tactile_pressure(
            pressure, spec["pressure_log1p"], spec["pressure_scale"]
        )

    def _taxel_hands_from_columns(self, columns: dict, spec: dict) -> np.ndarray:
        """One episode frame → ``[hands, taxels, 4]`` (raw, unconditioned)."""
        hands = []
        for column in spec["columns"]:
            value = columns.get(column)
            if value is None:
                raise KeyError(
                    f"[{self.dataset_name}] tactile column {column!r} declared in "
                    "modality.json is missing from the parquet data"
                )
            hands.append(np.array(value, dtype=np.float32, copy=True))
        return np.stack(hands, axis=0)

    def _stack_taxel_episode_frames(
        self,
        spec: dict,
        *,
        parquet_path: Path,
        row_offset: int,
        row: int,
        current_columns: dict,
    ) -> np.ndarray:
        """Last N episode policy frames → ``[hands, taxels, T, 4]``.

        EgoPro / converter cells are one frame (``[taxels, 4]``). Oldest
        first. A window that would leave the episode is left-padded by
        repeating the earliest available frame — zeros would look like a
        fake origin in xyz.
        """
        history_len = int(spec.get("history_len") or 1)
        if history_len < 1:
            raise ValueError(
                f"[{self.dataset_name}] taxel history_len must be >= 1, "
                f"got {history_len}"
            )
        if history_len == 1:
            return self._taxel_hands_from_columns(current_columns, spec)[:, :, None, :]
        n_traj = len(self.curr_traj_data)
        row_end = int(np.clip(row, 0, n_traj - 1))
        start = row_end - history_len + 1
        first = max(0, start)
        frames = []
        for t in range(first, row_end + 1):
            cols_t = (
                current_columns
                if t == row_end
                else self._read_tactile_row(parquet_path, row_offset + t)
            )
            frames.append(self._taxel_hands_from_columns(cols_t, spec))
        if first > start:
            frames = [frames[0]] * (first - start) + frames
        return np.stack(frames, axis=2)

    def _slice_region_future(
        self, spec: dict, columns: dict
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exported future window → ``[hands, regions, H, 1]`` plus ``valid [H]``.

        The cell is ``[future_frames, regions]`` with slot 0 the frame
        NEAREST the observation, so the run's horizon is a prefix slice of
        the same row the history came from — one row read, no episode walk.
        Frames past the episode end must be exported as **NaN** (0.0 is a
        legitimate no-contact reading and would look like a real release);
        they come back zeroed with ``valid=False``. Conditioning is the
        region stream's own (same log1p / scale), so ``current`` and the
        future window are in one unit system; no Δp channel is appended.
        """
        horizon = int(spec["future_len"])
        order = spec.get("future_order") or "nearest_first"
        if order not in ("nearest_first", "farthest_first"):
            raise ValueError(
                f"[{self.dataset_name}] tactile."
                f"{TACTILE_REGION_FUTURE_STREAM} future_order must be "
                f"nearest_first|farthest_first, got {order!r}"
            )
        hands = []
        for column in spec["columns"]:
            value = columns.get(column)
            if value is None:
                raise KeyError(
                    f"[{self.dataset_name}] tactile column {column!r} declared in "
                    "modality.json is missing from the parquet data"
                )
            array = np.array(value, dtype=np.float32, copy=True)
            if array.ndim != 2:
                raise ValueError(
                    f"[{self.dataset_name}] {column!r} must be "
                    f"[future_frames, regions], got {tuple(array.shape)}"
                )
            if order == "farthest_first":
                array = array[::-1]
            if array.shape[0] < horizon:
                raise ValueError(
                    f"[{self.dataset_name}] {column!r} stores "
                    f"{array.shape[0]} future frames < "
                    f"tactile_future_horizon={horizon}"
                )
            hands.append(array[:horizon])
        # [hands, H, regions] -> [hands, regions, H, 1]
        stacked = np.transpose(np.stack(hands, axis=0), (0, 2, 1))[..., None]
        finite = np.isfinite(stacked)
        valid = finite.all(axis=(0, 1, 3))
        stacked = np.where(finite, stacked, 0.0)
        stacked = self._condition_tactile_pressure(
            stacked, self._tactile_spec[TACTILE_REGION_PRESSURE_KEY]
        )
        return (
            np.ascontiguousarray(stacked, dtype=np.float32),
            np.ascontiguousarray(valid),
        )

    def _get_tactile_step_data(self, base_index: int) -> dict:
        """Current-step tactile tensors, keyed by the model-side batch keys.

        * ``region_pressure`` -> ``[hands, regions, history, 1]``, history in
          chronological order (oldest first), truncated to the run's window
          from the in-file history axis.
        * ``taxel_pressure_3d`` -> ``[hands, taxels, T, 4]`` (xyz + pressure).
          ``T`` is the encoder window: 1 = current row; ``T>1`` stacks the
          last T episode policy frames (oldest first; left-pad by repeating
          the earliest available frame). Accepts nested
          ``taxel_pressure_3D.{left,right}_ee`` or converter ``3d_left`` /
          ``3d_right``. Converter ``validity_*`` becomes ``taxel_valid_mask``
          ``[hands, taxels]`` from the **current** row; both hands invalid
          omits the taxel key (``present_slow=False``).
        * ``tactile_2d`` -> ``[H, W]`` head-camera contact heatmap (no log1p).
        * ``region_pressure_future`` -> ``[hands, regions, H, 1]``, the first
          ``tactile_future_horizon`` frames of the exported future window
          (slot 0 nearest to now), same conditioning as ``region_pressure``.
          NaN frames (episode end) are 0 with
          ``region_pressure_future_valid[i]=False``.

        Region / heatmap history lives inside the stored arrays, so
        ``observation_indices`` does not apply to those streams. Taxel
        history is stacked across episode rows of this trajectory.

        The pressure channel (all of ``region_pressure``; field 3 of
        ``taxel_pressure_3d``) passes through
        :meth:`_condition_tactile_pressure`; the taxel xyz stays in metres.
        The heatmap is left in the stored units.
        """
        if not self._tactile_spec:
            return {}
        assert self.curr_traj_data is not None
        parquet_path, row_offset = self._curr_traj_source
        if parquet_path is None:
            raise RuntimeError(
                f"[{self.dataset_name}] tactile requested before the trajectory's "
                "parquet source was resolved; get_trajectory_data must run first."
            )
        row = int(np.clip(base_index, 0, len(self.curr_traj_data) - 1))
        columns = self._read_tactile_row(parquet_path, row_offset + row)
        data: dict[str, np.ndarray] = {}
        for batch_key, spec in self._tactile_spec.items():
            if spec.get("kind") == "region_future":
                # Needs the region spec's resolved conditioning; packed below.
                continue
            if spec.get("kind") == "heatmap":
                column = spec["columns"][0]
                value = columns.get(column)
                if value is None:
                    raise KeyError(
                        f"[{self.dataset_name}] tactile column {column!r} declared in "
                        "modality.json is missing from the parquet data"
                    )
                array = np.array(value, dtype=np.float32, copy=True)
                data[batch_key] = np.ascontiguousarray(array, dtype=np.float32)
                assert_packed_tactile_array(
                    data[batch_key],
                    batch_key=batch_key,
                    spec=spec,
                    dataset_name=self.dataset_name,
                )
                continue
            hands = []
            for column in spec["columns"]:
                value = columns.get(column)
                if value is None:
                    raise KeyError(
                        f"[{self.dataset_name}] tactile column {column!r} declared in "
                        "modality.json is missing from the parquet data"
                    )
                hands.append(np.array(value, dtype=np.float32, copy=True))
            array = np.stack(hands, axis=0)
            if batch_key == TACTILE_REGION_PRESSURE_KEY:
                # [hands, history, regions] -> [hands, regions, history, 1]
                history_len = spec["history_len"] or array.shape[1]
                # Always the most recent N, then chronological (oldest first).
                if spec["history_order"] == "most_recent_first":
                    array = array[:, :history_len]
                    array = array[:, ::-1]
                else:
                    array = array[:, -history_len:]
                array = np.transpose(array, (0, 2, 1))[..., None]
                # Every channel of this stream IS a force.
                array = self._condition_tactile_pressure(array, spec)
                if spec.get("append_delta"):
                    from starVLA.model.modules.tactile.pressure import (
                        append_temporal_delta,
                    )

                    array = append_temporal_delta(array, history_axis=2)
            else:
                # [hands, taxels, T, fields] — T=1 is the current row;
                # T>1 stacks episode policy frames (oldest first).
                array = self._stack_taxel_episode_frames(
                    spec,
                    parquet_path=parquet_path,
                    row_offset=row_offset,
                    row=row,
                    current_columns=columns,
                )
                # Only the pressure field; xyz stays in metres.
                array = np.concatenate(
                    [
                        array[..., :3],
                        self._condition_tactile_pressure(array[..., 3:], spec),
                    ],
                    axis=-1,
                )
                valid_columns = spec.get("valid_columns")
                if valid_columns:
                    flags = [
                        _tactile_cell_flag(columns.get(column))
                        for column in valid_columns
                    ]
                    if not any(flags):
                        continue
                    mask = np.zeros(array.shape[:2], dtype=np.bool_)
                    for hand_index, flag in enumerate(flags):
                        if flag:
                            mask[hand_index] = True
                    data[TACTILE_TAXEL_VALID_KEY] = np.ascontiguousarray(mask)
            data[batch_key] = np.ascontiguousarray(array, dtype=np.float32)
            assert_packed_tactile_array(
                data[batch_key],
                batch_key=batch_key,
                spec=spec,
                dataset_name=self.dataset_name,
            )
            if (
                batch_key == TACTILE_TAXEL_PRESSURE_KEY
                and TACTILE_TAXEL_VALID_KEY in data
            ):
                assert_packed_tactile_array(
                    data[TACTILE_TAXEL_VALID_KEY],
                    batch_key=TACTILE_TAXEL_VALID_KEY,
                    spec=spec,
                    dataset_name=self.dataset_name,
                )
        future_spec = self._tactile_spec.get(TACTILE_REGION_FUTURE_KEY)
        if future_spec is not None and TACTILE_REGION_PRESSURE_KEY in data:
            future, valid = self._slice_region_future(future_spec, columns)
            data[TACTILE_REGION_FUTURE_KEY] = future
            data[TACTILE_REGION_FUTURE_VALID_KEY] = valid
            assert_packed_tactile_array(
                future,
                batch_key=TACTILE_REGION_FUTURE_KEY,
                spec=future_spec,
                dataset_name=self.dataset_name,
            )
            expected_h = int(future_spec["future_len"])
            if tuple(valid.shape) != (expected_h,):
                raise ValueError(
                    f"[{self.dataset_name}] packed "
                    f"{TACTILE_REGION_FUTURE_VALID_KEY} shape {tuple(valid.shape)} "
                    f"!= ({expected_h},)"
                )
        return data

    def _get_lerobot_info_meta(self) -> dict:
        """Get the metadata for the LeRobot dataset."""
        info_meta_path = self.dataset_path / LE_ROBOT_INFO_FILENAME
        with open(info_meta_path, "r") as f:
            info_meta = json.load(f)
        return info_meta

    def _get_data_path_pattern(self) -> str:
        """Get the data path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["data_path"]

    def _get_video_path_pattern(self) -> str:
        """Get the video path pattern for the LeRobot dataset."""
        return self.lerobot_info_meta["video_path"]

    def _get_chunk_size(self) -> int:
        """Get the chunk size for the LeRobot dataset."""
        return self.lerobot_info_meta["chunks_size"]

    def _get_tasks(self) -> pd.DataFrame:
        """Get the tasks for the dataset."""
        if self._lerobot_version == "v2.0":
            tasks_path = self.dataset_path / LE_ROBOT_TASKS_FILENAME
            with open(tasks_path, "r") as f:
                tasks = [json.loads(line) for line in f]
            df = pd.DataFrame(tasks)
            return df.set_index("task_index")
        
        elif self._lerobot_version == "v3.0":
            tasks_path = self.dataset_path / LE_ROBOT3_TASKS_FILENAME
            df = pd.read_parquet(tasks_path)
            df = df.reset_index()  # convert index to a column, typically named 'index'
            df = df.rename(columns={'index': 'task'})  # rename 'index' column to 'task'
            df = df[['task_index', 'task']]  # reorder columns
            # get_language looks tasks up with .loc[task_indices], so the index
            # must BE task_index (same contract as the v2.0 branch above) —
            # with a RangeIndex the lookup only works while task_index happens
            # to equal the row position and silently returns the wrong task
            # text otherwise.
            return df.set_index("task_index")
    def _check_integrity(self):
        """Use the config to check if the keys are valid and detect silent data corruption.

        Iterates the (possibly flexible_views-shrunk) ``modality_keys`` rather
        than the raw modality_configs, so cameras masked out as absent from
        this dataset's modality.json are not required on disk.
        """
        ERROR_MSG_HEADER = f"Error occurred in initializing dataset {self.dataset_name}:\n"

        for keys in self.modality_keys.values():
            for key in keys:
                if key == "lapa_action" or key == "dream_actions":
                    continue  # no need for any metadata for lapa actions because it comes normalized
                # Check if the key is valid
                try:
                    self.lerobot_modality_meta.get_key_meta(key)
                except Exception as e:
                    raise ValueError(
                        ERROR_MSG_HEADER + f"Unable to find key {key} in modality metadata:\n{e}"
                    )

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        self.transforms.set_metadata(metadata)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Args:
            epoch (int): The epoch to set.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Get the total number of data points in the dataset.

        Returns:
            int: the total number of data points in the dataset.
        """
        return len(self.all_steps)

    def __str__(self) -> str:
        """Get the description of the dataset."""
        return f"{self.dataset_name} ({len(self)} steps)"


    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step in a trajectory.

        Args:
            index (int): The index of the step to get.

        Returns:
            dict: The data for the step.
        """
        trajectory_id, base_index = self.all_steps[index]
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        return self._pack_sample(data)

    def _resolve_obs_image_size(self) -> tuple[int, int]:
        """Uniform observation resize from datasets.vla_data.obs_image_size.

        Prefer :meth:`_resolve_obs_image_sizes` when cameras may differ. A
        per-camera mapping collapses only when every resolved size matches;
        otherwise raises (callers that need mixed sizes must use the list API).
        """
        sizes = self._resolve_obs_image_sizes(self.modality_keys.get("video") or ["video.cam_high"])
        unique = set(sizes)
        if len(unique) != 1:
            raise ValueError(
                "obs_image_size resolves to mixed per-camera sizes "
                f"{sizes}; use _resolve_obs_image_sizes / pack per video key."
            )
        return sizes[0]

    def _resolve_obs_image_sizes(self, video_keys: list[str] | None = None) -> list[tuple[int, int]]:
        """One ``(W, H)`` PIL resize target per video key.

        ``datasets.vla_data.obs_image_size`` (alias ``image_size``) accepts:
          - int / ``[W, H]`` — same size for every camera
          - mapping keyed by short name (``cam_high``) or full key
            (``video.cam_high``), with optional ``default`` for the rest
        """
        default_size = (384, 384)
        keys = list(video_keys if video_keys is not None else self.modality_keys.get("video") or [])
        if not keys:
            return [default_size]

        if self.data_cfg is None:
            return [default_size for _ in keys]
        spec = self.data_cfg.get("obs_image_size")
        if spec is None:
            spec = self.data_cfg.get("image_size")
        if spec is None:
            return [default_size for _ in keys]

        # OmegaConf DictConfig / ListConfig -> plain containers.
        try:
            from omegaconf import OmegaConf

            if OmegaConf.is_config(spec):
                spec = OmegaConf.to_container(spec, resolve=True)
        except Exception:
            pass

        def _parse_hw(value) -> tuple[int, int]:
            if isinstance(value, bool):
                raise ValueError(f"obs_image_size entry must be [W, H], got {value!r}.")
            if isinstance(value, int):
                return (value, value)
            if isinstance(value, (list, tuple)) and len(value) == 2:
                return (int(value[0]), int(value[1]))
            raise ValueError(f"obs_image_size entry must be [W, H], got {value!r}.")

        def _short(name: str) -> str:
            text = str(name)
            return text.split(".", 1)[-1] if "." in text else text

        if isinstance(spec, bool):
            raise ValueError(f"obs_image_size must be int / [W, H] / mapping, got {spec!r}.")
        if isinstance(spec, int) or (
            isinstance(spec, (list, tuple))
            and len(spec) == 2
            and not isinstance(spec[0], (list, tuple, dict))
        ):
            size = _parse_hw(spec)
            return [size for _ in keys]
        if not isinstance(spec, dict):
            raise ValueError(
                "obs_image_size must be an int, [W, H], or a per-camera mapping "
                f"(e.g. {{cam_high: [384, 384], cam_left_wrist: [224, 224]}}), got {spec!r}."
            )

        by_short: dict[str, tuple[int, int]] = {}
        mapping_default: tuple[int, int] | None = None
        for raw_key, raw_size in spec.items():
            key = str(raw_key)
            if key in {"default", "_default"}:
                mapping_default = _parse_hw(raw_size)
                continue
            by_short[_short(key)] = _parse_hw(raw_size)

        sizes: list[tuple[int, int]] = []
        missing: list[str] = []
        for video_key in keys:
            short = _short(video_key)
            if short in by_short:
                sizes.append(by_short[short])
            elif mapping_default is not None:
                sizes.append(mapping_default)
            else:
                missing.append(str(video_key))
        if missing:
            raise ValueError(
                "obs_image_size mapping is missing camera(s) "
                f"{missing}; known keys={sorted(by_short)}. "
                "Add each camera or a 'default' entry."
            )
        return sizes

    def _resolve_sample_dtype(self) -> np.dtype:
        """dtype of the packed state/action arrays (datasets.vla_data.sample_dtype).

        The legacy upstream behaviour casts normalized state/action to float16,
        which quantizes the training targets (and the state fed to the 256-bin
        discretizer) for no benefit. Set ``sample_dtype: float32`` in the
        training YAML to keep full precision. The key is recorded in the run's
        config.full.yaml, and deployment replays the same choice for the state
        input (see deployment/model_server/policy_wrapper.py) — so it must NOT
        be changed for an already-trained checkpoint.
        """
        default = "float16"
        if self.data_cfg is None:
            return np.dtype(default)
        return np.dtype(str(self.data_cfg.get("sample_dtype", default)))

    def _current_frame_position(self, key: str) -> int:
        """Position of the CURRENT frame within a key's delta_indices.

        The model contract is single-frame video/language, so when
        ``observation_indices`` carries extra history offsets (e.g.
        [-2, 0]) only the frame at delta 0 — the timestep the action chunk
        starts from — may be packed. Taking index 0 instead would silently
        pick the OLDEST frame. Extra frames are dropped with a one-time
        warning: image history needs model- and deploy-side support, state
        history already has its own path (state_history_frames).
        """
        deltas = [int(d) for d in self.delta_indices[key]]
        position = deltas.index(0) if 0 in deltas else int(np.argmax(deltas))
        if len(deltas) > 1 and not getattr(self, "_warned_obs_history_drop", False):
            self._warned_obs_history_drop = True
            import warnings

            warnings.warn(
                f"observation_indices={deltas} packs multiple observation frames, but the "
                "model contract is single-frame video/language: only the delta==0 (current) "
                "frame is packed and the rest are DROPPED. State history is unaffected "
                "(state_history_frames). Remove the extra offsets or add multi-frame image "
                "support to the model and deploy client first.",
                stacklevel=2,
            )
        return position

    def _pack_sample(self, data: dict) -> dict:
        """Pack transformed modality data into training sample format.

        Camera packing uses the run-level slot order (``_run_video_keys``).
        With ``flexible_views``, cameras absent from this dataset's
        ``modality.json`` are omitted from ``image`` (so the VLM never sees
        black placeholders) and recorded as 0 in ``view_mask``; the mask
        length always equals the run slot count so mixed head-only /
        multi-view batches stay index-aligned.
        """
        run_video_keys = list(getattr(self, "_run_video_keys", None) or self.modality_keys["video"])
        view_mask = list(
            getattr(self, "_view_mask_template", None) or [1] * len(run_video_keys)
        )
        if len(view_mask) != len(run_video_keys):
            raise ValueError(
                f"view_mask length {len(view_mask)} != run video slots "
                f"{len(run_video_keys)} for dataset {self.dataset_name}."
            )
        # Resize targets follow run slots; only present slots are packed.
        obs_image_sizes = self._resolve_obs_image_sizes(run_video_keys)
        sample_dtype = self._resolve_sample_dtype()
        step_images = []
        for video_key, obs_image_size, present in zip(
            run_video_keys, obs_image_sizes, view_mask
        ):
            if not present:
                continue
            if video_key not in data:
                raise KeyError(
                    f"view_mask marks {video_key} present but it is missing from "
                    f"loaded data for dataset {self.dataset_name}."
                )
            image = data[video_key][self._current_frame_position(video_key)]
            # PIL resize takes (width, height); obs_image_size is [W, H].
            image = Image.fromarray(image).resize(obs_image_size)
            step_images.append(image)

        language_key = self.modality_keys["language"][0]
        language = data[language_key][self._current_frame_position(language_key)]
        action = []
        for action_key in self.modality_keys["action"]:
            action.append(data[action_key])
        action = np.concatenate(action, axis=1).astype(sample_dtype)

        sample = {
            "action": action,
            "image": step_images,
            "lang": language,
            "robot_tag": self.tag,
        }
        if self._flexible_views_enabled() or any(m == 0 for m in view_mask):
            sample["view_mask"] = view_mask

        if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
            state = []
            for state_key in self.modality_keys.get("state", []):
                state.append(data[state_key])
            if not state:
                import warnings
                warnings.warn(
                    "include_state=True but no state modality keys found in modality_configs "
                    "(state modality may be disabled in the dataset's DataConfig). "
                    "Skipping state packing.",
                    stacklevel=2,
                )
            else:
                state = np.concatenate(state, axis=1).astype(sample_dtype)
                sample["state"] = state

        # Tactile streams (optional). Only datasets whose meta/modality.json
        # declares a `tactile` block produce these keys (see get_step_data);
        # the names match the built-in aliases that the model-side tactile
        # batch packing (modules/tactile/batch_packing.py) looks up.
        for tactile_key in TACTILE_SAMPLE_KEYS:
            if tactile_key in data:
                sample[tactile_key] = data[tactile_key]

        return sample

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step in a trajectory. No transforms are applied.

        Args:
            trajectory_id (int): The name of the trajectory.
            base_index (int): The base step index in the trajectory.

        Returns:
            dict: The RAW data for the step.

        Example return:
            {
                "video": {
                    "video.image_side_0": [B, T, H, W, C],
                    "video.image_side_1": [B, T, H, W, C],
                },
                "state": {
                    "state.eef_position": [B, T, state_dim],
                    "state.eef_rotation": [B, T, state_dim],
                },
                "action": {
                    "action.eef_position": [B, T, action_dim],
                    "action.eef_rotation": [B, T, action_dim],
                },
            }
        """
        data = {}
        # Get the data for all modalities # just for action base data
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # TODO @JinhuiYE The logic below is poorly implemented. Data reading should be directly based on curr_traj_data.
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data = self._apply_action_mode(data)
        data.update(self._get_tactile_step_data(base_index))
        return data

    def _pandas_columns(self, parquet_path: Path) -> list[str] | None:
        """Columns to materialise as pandas; ``None`` = every column.

        The tactile columns (fusion arrays plus the optional HxW heatmap) are
        2-D per frame. pandas turns each into a per-row object ndarray:
        `pd.read_parquet` on a 1.1 GB file measures 17.7 s against 0.16 s
        without them. They are excluded here — including streams this run
        chose not to pack (`pack_tactile: false` / `tactile_pack_heatmap:
        false`) — and served per frame by :meth:`_read_tactile_row` when the
        spec is non-empty. Name-matching also drops undeclared `tactile.*`
        columns in the same file. Datasets with no tactile parquet columns
        take the original path (``None`` = every column) without opening
        the schema.
        """
        if not self._tactile_parquet_columns:
            return None
        schema_names = self._parquet_file(parquet_path).schema_arrow.names
        skip = self._tactile_parquet_columns
        keep = [
            name
            for name in schema_names
            if name not in skip and not _is_tactile_column_name(name)
        ]
        if len(keep) == len(schema_names):
            return None
        return keep

    def get_trajectory_data(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory."""
        if self._lerobot_version == "v2.0":
        
            if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
                return self.curr_traj_data
            else:
                chunk_index = self.get_episode_chunk(trajectory_id)
                parquet_path = self.dataset_path / self.data_path_pattern.format(
                    episode_chunk=chunk_index, episode_index=trajectory_id
                )
                assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
                # One episode per file, so the episode starts at row 0.
                self._curr_traj_source = (parquet_path, 0)
                return pd.read_parquet(
                    parquet_path, columns=self._pandas_columns(parquet_path)
                )
        elif self._lerobot_version == "v3.0":
            return self.get_trajectory_data_lerobot_v3(trajectory_id)
    
    def get_trajectory_data_lerobot_v3(self, trajectory_id: int) -> pd.DataFrame:
        """Get the data for a trajectory from lerobot v3.

        v3 packs ~100 episodes into one parquet file (single row group), and
        the previous implementation ran ``pd.read_parquet`` over the WHOLE
        file just to keep one episode — the pandas materialisation of the
        fixed-size-list state/action columns builds a per-row object array
        for every episode in the file (measured 300-1150 ms per sample on
        this repo's pretrain mixture, ~38% of ``__getitem__``). Read the file
        as an arrow table instead, locate the episode's contiguous row span
        from the ``episode_index`` column, and convert ONLY that slice to
        pandas (1.9-25x per read, byte-identical values). The returned frame
        keeps the old ``.loc[mask]`` row labels (file row positions) so any
        label-based consumer sees the exact same object. Non-contiguous
        episodes (never produced by the LeRobot writer) fall back to the
        original whole-file mask path.
        """
        if self.curr_traj_id == trajectory_id and self.curr_traj_data is not None:
            return self.curr_traj_data
        else: #TODO check detail later
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]
            chunk_index = episode_meta["data/chunk_index"]
            file_index = self.get_episode_file_index(trajectory_id)
            # file_from_index = self.get_episode_file_from_index(trajectory_id)
            
            
            parquet_path = self.dataset_path / self.data_path_pattern.format(
                chunk_index=chunk_index, file_index=file_index
            )
            assert parquet_path.exists(), f"Parquet file not found at {parquet_path}"
            # A fresh open per call, NOT self._parquet_file: that handle cache
            # never evicts, and a v3 dataset can have tens of thousands of
            # data files (humantouch: 67970) — persistent workers would leak
            # one fd per touched file until the run dies on EMFILE. Opening
            # costs a few ms; the arrow decode itself is unavoidable because
            # LeRobot's writer emits one row group per file. Only the pandas
            # conversion below is restricted to this episode's rows.
            import pyarrow.parquet as pq

            table = pq.read_table(
                parquet_path, columns=self._pandas_columns(parquet_path)
            )

            episode_column = table.column("episode_index").to_numpy()
            row_positions = np.flatnonzero(episode_column == trajectory_id)
            # Episodes are contiguous in the file; remember where this one starts
            # so the tactile reader can address its rows without the mask.
            self._curr_traj_source = (
                parquet_path,
                int(row_positions[0]) if len(row_positions) else 0,
            )
            if len(row_positions) == 0:
                return table.slice(0, 0).to_pandas()
            start = int(row_positions[0])
            stop = int(row_positions[-1]) + 1
            if stop - start != len(row_positions):
                # Non-contiguous episode rows: reproduce the original path.
                file_data = table.to_pandas()
                episode_mask = episode_column == trajectory_id
                return file_data.loc[episode_mask].copy()
            episode_data = table.slice(start, stop - start).to_pandas()
            # Same row labels as file_data.loc[mask] used to produce.
            episode_data.index = pd.RangeIndex(start, stop)
            return episode_data


    def get_trajectory_index(self, trajectory_id: int) -> int:
        """Get the index of the trajectory in the dataset by the trajectory ID.
        This is useful when you need to get the trajectory length or sampling weight corresponding to the trajectory ID.

        Args:
            trajectory_id (str): The ID of the trajectory.

        Returns:
            int: The index of the trajectory in the dataset.
        """
        trajectory_indices = np.where(self.trajectory_ids == trajectory_id)[0]
        if len(trajectory_indices) != 1:
            raise ValueError(
                f"Error finding trajectory index for {trajectory_id}, found {trajectory_indices=}"
            )
        return trajectory_indices[0]

    def get_episode_chunk(self, ep_index: int) -> int:
        """Get the chunk index for an episode index."""
        return ep_index // self.chunk_size
    def get_episode_file_index(self, ep_index: int) -> int:
        """Get the file index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_index"]
    
    def get_episode_file_from_index(self, ep_index: int) -> int:
        """Get the file from index for an episode index."""
        episode_meta = self.trajectory_ids_to_metadata[ep_index]
        return episode_meta["data/file_from_index"]


    def retrieve_data_and_pad(
        self,
        array: np.ndarray,
        step_indices: np.ndarray,
        max_length: int,
        padding_strategy: str = "first_last",
    ) -> np.ndarray:
        """Retrieve the data from the dataset and pad it if necessary.
        Args:
            array (np.ndarray): The array to retrieve the data from.
            step_indices (np.ndarray): The step indices to retrieve the data for.
            max_length (int): The maximum length of the data.
            padding_strategy (str): The padding strategy, either "first" or "last".
        """
        # Get the padding indices
        front_padding_indices = step_indices < 0
        end_padding_indices = step_indices >= max_length
        padding_positions = np.logical_or(front_padding_indices, end_padding_indices)
        # Retrieve the data with the non-padding indices
        # If there exists some padding, Given T step_indices, the shape of the retrieved data will be (T', ...) where T' < T
        raw_data = array[step_indices[~padding_positions]]
        assert isinstance(raw_data, np.ndarray), f"{type(raw_data)=}"
        # This is the shape of the output, (T, ...)
        if raw_data.ndim == 1:
            expected_shape = (len(step_indices),)
        else:
            expected_shape = (len(step_indices), *array.shape[1:])

        # Pad the data
        output = np.zeros(expected_shape)
        # Assign the non-padded data
        output[~padding_positions] = raw_data
        # If there exists some padding, pad the data
        if padding_positions.any():
            if padding_strategy == "first_last":
                # Use first / last step data to pad
                front_padding_data = array[0]
                end_padding_data = array[-1]
                output[front_padding_indices] = front_padding_data
                output[end_padding_indices] = end_padding_data
            elif padding_strategy == "zero":
                # Use zero padding
                output[padding_positions] = 0
            else:
                raise ValueError(f"Invalid padding strategy: {padding_strategy}")
        return output

    def get_video_path(self, trajectory_id: int, key: str) -> Path:
        chunk_index = self.get_episode_chunk(trajectory_id)
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self._lerobot_version == "v2.0":
            video_filename = self.video_path_pattern.format(
                episode_chunk=chunk_index, episode_index=trajectory_id, video_key=original_key
            )
        elif self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata[trajectory_id]

            video_file_indices = episode_meta.get("videos/file_indices", {})
            # print(f"{video_file_indices=}")
            # Modified lerobot v3.0 video index
            if original_key in video_file_indices:
                video_chunk_index = video_file_indices[original_key]["chunk_index"]
                video_file_index = video_file_indices[original_key]["file_index"]
            else:
                video_chunk_index = episode_meta["data/chunk_index"]
                video_file_index = episode_meta["data/file_index"]
            video_filename = self.video_path_pattern.format(
                video_key=original_key,
                chunk_index=video_chunk_index,
                file_index=video_file_index,
            )
        return self.dataset_path / video_filename

    def get_video(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the video frames for a trajectory by a base index.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (str): The ID of the trajectory.
            key (str): The key of the video.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The video frames for the trajectory and frame indices. Shape: (T, H, W, C)
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # print(f"{step_indices=}")
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")

        # Image-only LeRobot datasets (e.g. VLA-Arena) may store frames directly
        # in parquet columns and have total_videos == 0 (no mp4 files). In this
        # case, load frames from the original image column and pad by step indices.
        original_key = self.lerobot_modality_meta.video[key].original_key
        if original_key is None:
            original_key = key
        if self.curr_traj_data is not None and original_key in self.curr_traj_data.columns:
            image_entries = self.curr_traj_data[original_key].tolist()

            def _decode_image_entry(entry):
                if isinstance(entry, np.ndarray):
                    return entry
                if isinstance(entry, Image.Image):
                    return np.array(entry)
                if isinstance(entry, dict):
                    img_bytes = entry.get("bytes", None)
                    img_path = entry.get("path", None)

                    if img_bytes is not None:
                        return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

                    if img_path is not None:
                        path_obj = Path(img_path)
                        if not path_obj.is_absolute():
                            path_obj = self.dataset_path / path_obj
                        return np.array(Image.open(path_obj).convert("RGB"))

                raise TypeError(f"Unsupported image entry type: {type(entry)}")

            frames = []
            for idx in step_indices:
                safe_idx = int(min(max(idx, 0), len(image_entries) - 1))
                frames.append(_decode_image_entry(image_entries[safe_idx]))

            return np.stack(frames)

        video_path = self.get_video_path(trajectory_id, key)
        # Get the action/state timestamps for each frame in the video
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert "timestamp" in self.curr_traj_data.columns, f"No timestamp found in {trajectory_id=}"
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        # Get the corresponding video timestamps from the step indices
        video_timestamp = timestamp[step_indices]
        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key].original_key
            if original_video_key is None:
                original_video_key = key
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        return get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend, # TODO
            video_backend_kwargs=self.video_backend_kwargs,
        )

    def get_state_or_action(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ) -> np.ndarray:
        """Get the state or action data for a trajectory by a base index.
        If the step indices are out of range, pad with the data:
            if the data is stored in absolute format, pad with the first or last step data;
            otherwise, pad with zero.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.

        Returns:
            np.ndarray: The data for the trajectory and step indices.
        """
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        assert key.startswith(modality + "."), f"{key} must start with {modality + '.'}, got {key}"
        # Get the sub-key, e.g. state.joint_angles -> joint_angles
        key = key.replace(modality + ".", "")
        # Get the lerobot key
        le_state_or_action_cfg = getattr(self.lerobot_modality_meta, modality)
        le_key = le_state_or_action_cfg[key].original_key
        if le_key is None:
            le_key = key
        # Get the data array, shape: (T, D)
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        assert le_key in self.curr_traj_data.columns, f"No {le_key} found in {trajectory_id=}"
        data_array: np.ndarray = np.stack(self.curr_traj_data[le_key])  # type: ignore
        assert data_array.ndim == 2, f"Expected 2D array, got key {le_key} is{data_array.shape} array"
        le_indices = np.arange(
            le_state_or_action_cfg[key].start,
            le_state_or_action_cfg[key].end,
        )
        data_array = data_array[:, le_indices]
        # Get the state or action configuration
        state_or_action_cfg = getattr(self.metadata.modalities, modality)[key]

        # Pad the data
        return self.retrieve_data_and_pad(
            array=data_array,
            step_indices=step_indices,
            max_length=max_length,
            padding_strategy="first_last" if state_or_action_cfg.absolute else "zero",
            # padding_strategy="zero",           # HACK for realdata
        )

    def get_language(
        self,
        trajectory_id: int,
        key: str,
        base_index: int,
    ) -> list[str]:
        """Get the language annotation data for a trajectory by step indices.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            key (str): The key of the annotation.
            base_index (int): The base index of the trajectory.

        Returns:
            list[str]: The annotation data for the trajectory and step indices. If no matching data is found, return empty strings.
        """
        assert self.curr_traj_data is not None, f"No data found for {trajectory_id=}"
        # Get the step indices
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Get the maximum length of the trajectory
        max_length = self.trajectory_lengths[trajectory_index]
        # Get the end times corresponding to the closest indices
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, max_length - 1)
        # Get the annotations
        task_indices: list[int] = []
        assert key.startswith(
            "annotation."
        ), f"Language key must start with 'annotation.', got {key}"
        subkey = key.replace("annotation.", "")
        annotation_meta = self.lerobot_modality_meta.annotation
        assert annotation_meta is not None, f"Annotation metadata is None for {subkey}"
        assert (
            subkey in annotation_meta
        ), f"Annotation key {subkey} not found in metadata, available annotation keys: {annotation_meta.keys()}"
        subkey_meta = annotation_meta[subkey]
        original_key = subkey_meta.original_key
        if original_key is None:
            original_key = key
        for i in range(len(step_indices)): # 
            # task_indices.append(self.curr_traj_data[original_key][step_indices[i]].item())
            value = self.curr_traj_data[original_key].iloc[step_indices[i]] # TODO check v2.0 
            task_indices.append(value if isinstance(value, (int, float)) else value.item())

        return self.tasks.loc[task_indices]["task"].tolist()

    def get_data_by_modality(
        self,
        trajectory_id: int,
        modality: str,
        key: str,
        base_index: int,
    ):
        """Get the data corresponding to the modality for a trajectory by a base index.
        This method will call the corresponding helper method based on the modality.
        See the helper methods for more details.
        NOTE: For the language modality, the data is padded with empty strings if no matching data is found.

        Args:
            dataset (BaseSingleDataset): The dataset to retrieve the data from.
            trajectory_id (int): The ID of the trajectory.
            modality (str): The modality of the data.
            key (str): The key of the data.
            base_index (int): The base index of the trajectory.
        """
        if modality == "video":
            return self.get_video(trajectory_id, key, base_index)
        elif modality == "state" or modality == "action":
            return self.get_state_or_action(trajectory_id, modality, key, base_index)
        elif modality == "language":
            return self.get_language(trajectory_id, key, base_index)
        else:
            raise ValueError(f"Invalid modality: {modality}")

    def _save_dataset_statistics_(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the dataset.
        Keys are stored in modality_keys (embodiment) order — the same order the
        model's flat state/action vectors use.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Get used modality keys
        used_action_keys, used_state_keys = get_used_modality_keys(self.modality_keys)
        
        # Organize statistics by tag
        tag = self.tag
        tag_stats = {}
        
        # Process action statistics (only for used keys).
        if hasattr(self.metadata.statistics, 'action') and self.metadata.statistics.action:
            action_stats = self.metadata.statistics.action
            
            # Keep the modality_keys (embodiment) order: the model's flat action
            # vector, the mixture save path, and the eval-side stats split
            # (PolicyNormProcessor) all use this order. The old "gripper keys
            # last" reorder mis-assigned per-dim stats at eval for embodiments
            # whose gripper keys are interleaved between the arms.
            filtered_action_stats = {}
            for key in used_action_keys:
                if key in action_stats:
                    filtered_action_stats[key] = action_stats[key]
            
            if filtered_action_stats:
                # Combine statistics from filtered action sub-keys
                combined_action_stats = combine_modality_stats(filtered_action_stats)
                
                # mask=False for dimensions whose normalization mode is "binary"
                _action_norm_modes = _extract_action_normalization_modes(self.transforms)
                mask = generate_action_mask_for_used_keys(
                    self.metadata.modalities.action, filtered_action_stats.keys(),
                    normalization_modes=_action_norm_modes,
                )
                combined_action_stats["mask"] = mask
                
                tag_stats["action"] = combined_action_stats
                per_row_action_stats = collect_per_row_modality_stats(filtered_action_stats)
                if per_row_action_stats:
                    tag_stats["action_per_row"] = per_row_action_stats
        
        # Process state statistics (only for used keys), same key order as above.
        if hasattr(self.metadata.statistics, 'state') and self.metadata.statistics.state:
            state_stats = self.metadata.statistics.state
            
            filtered_state_stats = {}
            for key in used_state_keys:
                if key in state_stats:
                    filtered_state_stats[key] = state_stats[key]
            
            if filtered_state_stats:
                combined_state_stats = combine_modality_stats(filtered_state_stats)
                tag_stats["state"] = combined_state_stats
        
        # Add dataset counts
        tag_stats["num_transitions"] = len(self)
        tag_stats["num_trajectories"] = len(self.trajectory_ids)
        
        statistics_data[tag] = tag_stats
        
        # Save as JSON file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Single dataset statistics saved to: {save_path}")
        print(f"Used action keys (modality order): {list(used_action_keys)}")
        print(f"Used state keys (modality order): {list(used_state_keys)}")


class CachedLeRobotSingleDataset(LeRobotSingleDataset):
    def __init__(self, img_resize: tuple[int, int] | None = None, *args, **kwargs):
        """
        This class caches the video frames for each trajectory and key.
        It is recommended to use this class if the video frames need to be accessed multiple times.

        Args:
            resize_img (tuple[int, int], optional): The size to resize the video frames to reduce memory usage.
        """
        # Convert img_resize to tuple if it is not already
        if img_resize is not None and not isinstance(img_resize, tuple):
            img_resize = tuple(img_resize)
            assert len(img_resize) == 2, f"Expected tuple of length 2, got {img_resize}"
        self.img_resize = img_resize

        # Initialize img_resize attribute first to ensure it exists
        super().__init__(*args, **kwargs)
        cached_frames: dict[str, np.ndarray] = {}

        for key in self.modality_keys["video"]:
            all_frames = []
            original_key = key
            key = key.replace("video.", "")
            for trajectory_id, trajectory_length in tqdm(
                zip(self.trajectory_ids, self.trajectory_lengths),
                total=len(self.trajectory_ids),
                desc=f"Caching {key} frames",
            ):
                video_path = self.get_video_path(trajectory_id, key)
                frames = get_all_frames(
                    video_path.as_posix(),
                    video_backend=self.video_backend,
                    video_backend_kwargs=self.video_backend_kwargs,
                    resize_size=img_resize,
                )
                assert frames.ndim == 4, f"Expected 4D array, got {frames.shape} array"
                assert frames.shape[3] == 3, f"Expected 3 channels, got {frames.shape[3]} channels"
                
                # Apply image cropping if enabled and the video key is base_view
                # Note: crop_obs_camera functionality has been removed
                
                # assert (
                #     frames.shape[0] == trajectory_length
                # ), f"Expected {trajectory_length} frames, got {frames.shape[0]} frames"
                all_frames.append(frames)
            cached_frames[key] = np.concatenate(all_frames, axis=0)
            print(f"{key}: {cached_frames[key].shape}")
        self.cached_frames = cached_frames
        self.start_indices = np.cumsum(self.trajectory_lengths) - self.trajectory_lengths

    def get_video(self, trajectory_id: int, key: str, base_index: int) -> np.ndarray:
        step_indices = self.delta_indices[key] + base_index
        # Get the trajectory index
        trajectory_index = self.get_trajectory_index(trajectory_id)
        # Ensure the indices are within the valid range
        # This is equivalent to padding the video with extra frames at the beginning and end
        step_indices = np.maximum(step_indices, 0)
        step_indices = np.minimum(step_indices, self.trajectory_lengths[trajectory_index] - 1)
        assert key.startswith("video."), f"Video key must start with 'video.', got {key}"
        # Get the sub-key
        key = key.replace("video.", "")
        # Calculate the absolute indices
        absolute_indices = self.start_indices[trajectory_index] + step_indices
        return self.cached_frames[key][absolute_indices]

    def get_step_data(self, trajectory_id: int, base_index: int) -> dict:
        """Get the RAW data for a single step. No transforms are applied.

        Args:
            trajectory_id (str): The ID of the trajectory.
            base_index (int): The base index of the step.

        Returns:
            dict: The data for the step.
        """
        data = {}
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        # Get the data for all modalities
        for modality in self.modality_keys:
            # Get the data corresponding to each key in the modality
            for key in self.modality_keys[modality]:
                data[key] = self.get_data_by_modality(trajectory_id, modality, key, base_index)
        data.update(self._get_tactile_step_data(base_index))
        return data

    def set_transforms_metadata(self, metadata: DatasetMetadata):
        """Set the metadata for the transforms. This is useful for transforms that need to know the metadata, such as the normalization values."""
        if self.img_resize is not None:
            all_video_keys = [key for key in self.modality_keys["video"]]
            for key in metadata.modalities.video:
                if key in all_video_keys:
                    metadata.modalities.video[key].resolution = self.img_resize
        super().set_transforms_metadata(metadata)


def safe_hash(input_tuple):
    # keep 128 bits of the hash
    tuple_string = repr(input_tuple).encode("utf-8")
    sha256 = hashlib.sha256()
    sha256.update(tuple_string)

    seed = int(sha256.hexdigest(), 16)

    return seed & 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF


class MixtureSpecElement(BaseModel):
    dataset_path: list[Path] | Path = Field(..., description="The path to the dataset.")
    dataset_weight: float = Field(..., description="The weight of the dataset in the mixture.")
    distribute_weights: bool = Field(
        default=False,
        description="Whether to distribute the weights of the dataset across all the paths. If True, the weights will be evenly distributed across all the paths.",
    )


# Helper functions for dataset statistics

PER_ROW_STAT_NAMES = ("q01_per_row", "q99_per_row")


def collect_per_row_modality_stats(modality_stats: dict) -> dict:
    """Per-subkey ``(H, D)`` statistics, keyed by sub-key.

    These cannot go into the flat combined arrays of
    :func:`combine_modality_stats` (they have a horizon axis the pooled keys do
    not), and only the keys that actually use them carry them, so they are
    saved as their own ``<modality>_per_row`` block. Empty dict when no sub-key
    has per-row statistics, which is the case for every pooled-only run.
    """
    per_row: dict[str, dict] = {}
    for subkey, subkey_stats in modality_stats.items():
        entry = {}
        for stat_name in PER_ROW_STAT_NAMES:
            value = getattr(subkey_stats, stat_name, None)
            if value is None:
                continue
            entry[stat_name] = value.tolist() if hasattr(value, "tolist") else value
        if entry:
            per_row[subkey] = entry
    return per_row


def combine_modality_stats(modality_stats: dict) -> dict:
    """
    Combine statistics from all sub-keys under a modality.
    
    Args:
        modality_stats (dict): Statistics for a modality, containing multiple sub-keys.
                               Each sub-key contains DatasetStatisticalValues object.
        
    Returns:
        dict: Combined statistics (the pooled, per-dimension fields only; see
              collect_per_row_modality_stats for the optional per-row block)
    """
    combined_stats = {
        "mean": [],
        "std": [],
        "max": [],
        "min": [],
        "q01": [],
        "q99": []
    }
    
    # Combine statistics in sub-key order
    for subkey in modality_stats.keys():
        subkey_stats = modality_stats[subkey]  # This is a DatasetStatisticalValues object
        
        # Convert DatasetStatisticalValues to dict-like access
        for stat_name in ["mean", "std", "max", "min", "q01", "q99"]:
            stat_value = getattr(subkey_stats, stat_name)
            if isinstance(stat_value, (list, tuple)):
                combined_stats[stat_name].extend(stat_value)
            else:
                # Handle NDArray case - convert to list
                if hasattr(stat_value, 'tolist'):
                    combined_stats[stat_name].extend(stat_value.tolist())
                else:
                    combined_stats[stat_name].append(float(stat_value))
    
    return combined_stats

def _extract_action_normalization_modes(transforms) -> dict:
    """Extract normalization modes for action keys from a ComposedModalityTransform.

    Returns:
        dict: {subkey_without_action_prefix -> normalization_mode_str}
    """
    modes = {}
    for t in transforms.transforms:
        if isinstance(t, StateActionTransform):
            for key, mode in t.normalization_modes.items():
                if key.startswith("action."):
                    subkey = key[len("action."):]
                    modes[subkey] = mode
    return modes


def generate_action_mask_for_used_keys(
    action_modalities: dict,
    used_action_keys_ordered,
    normalization_modes: dict | None = None,
) -> list[bool]:
    """Generate per-dimension mask for action statistics.

    A dimension gets ``mask=False`` only when its normalization mode is ``"binary"``.
    This tells the inference code to skip continuous de-normalization for that dimension.
    All other modes (q99, mean_std, min_max ...) produce ``mask=True``.

    Args:
        action_modalities (dict): Configuration information for action modalities.
        used_action_keys_ordered: Iterable of actually used action keys (no "action." prefix).
        normalization_modes (dict | None): Mapping {subkey -> mode} (no "action." prefix).
            If ``None``, all dimensions default to ``mask=True``.

    Returns:
        list[bool]: Per-dimension mask values.
    """
    mask = []

    for subkey in used_action_keys_ordered:
        if subkey in action_modalities:
            subkey_config = action_modalities[subkey]

            # Get dimension count from shape
            if hasattr(subkey_config, 'shape') and len(subkey_config.shape) > 0:
                dim_count = subkey_config.shape[0]
            else:
                dim_count = 1

            # mask=False only when the normalization mode is explicitly "binary"
            is_binary = (
                normalization_modes.get(subkey) == "binary"
                if normalization_modes is not None
                else False
            )

            for _ in range(dim_count):
                mask.append(not is_binary)

    return mask

def get_used_modality_keys(modality_keys: dict) -> tuple[list, list]:
    """Extract used action and state keys from modality configuration."""
    used_action_keys = []
    used_state_keys = []
    
    # Extract action keys (remove "action." prefix)
    for action_key in modality_keys.get("action", []):
        if action_key.startswith("action."):
            clean_key = action_key.replace("action.", "")
            used_action_keys.append(clean_key)
    
    # Extract state keys (remove "state." prefix)  
    for state_key in modality_keys.get("state", []):
        if state_key.startswith("state."):
            clean_key = state_key.replace("state.", "")
            used_state_keys.append(clean_key)
    
    return used_action_keys, used_state_keys


def coerce_mixture_members(
    data_mixture: Sequence,
) -> tuple[list, list[float]]:
    """Keep every declared member; fail on empty datasets or non-positive weights.

    ``weight: 0`` used to clamp to ``1e-8`` (still sampled). An empty member
    used to be dropped with a warning. Both look like "exclude this dataset"
    in a YAML and silently change the mixture.
    """
    datasets: list = []
    weights: list[float] = []
    for dataset, weight in data_mixture:
        name = str(
            getattr(dataset, "dataset_name", None)
            or getattr(dataset, "_dataset_name", "?")
        )
        n = len(dataset)
        if n == 0:
            raise ValueError(
                f"mixture member {name!r} is empty. Fix the dataset path / "
                "filters, or remove it from data_mixture — empty members are "
                "not skipped."
            )
        try:
            w = float(weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mixture member {name!r} has non-numeric weight {weight!r}."
            ) from exc
        if not math.isfinite(w) or w <= 0.0:
            raise ValueError(
                f"mixture member {name!r} has weight {weight!r}; weights must "
                "be finite and > 0. Omit the entry to exclude a dataset "
                "(weight 0 is not a drop)."
            )
        datasets.append(dataset)
        weights.append(w)
    if not datasets:
        raise ValueError("No valid datasets found in the mixture.")
    return datasets, weights


def _sampleable_trajectory_positions(dataset) -> np.ndarray:
    """Positions in ``dataset.trajectory_ids`` that survived ``all_steps``.

    ``LeRobotSingleDataset.__getitem__`` indexes ``all_steps``, which is the
    step list built by ``_get_all_steps_single_process`` AFTER dropping every
    trajectory with an empty language instruction or an unreadable parquet.
    ``LeRobotMixtureDataset`` does not go through that index — it draws its own
    ``(trajectory, base_index)`` pair — so without this filter it re-admits
    exactly the trajectories the index excluded, and every training run goes
    through the mixture (``get_vla_dataset`` builds one even for a single
    dataset). Empty instructions would then reach the model silently.

    ``all_steps`` is already materialized at dataset construction, so this is a
    single pass over an in-memory list. An empty index (every trajectory
    dropped) returns no positions so the mixture raises; a missing attribute
    (custom member) still treats every trajectory as sampleable.
    """
    steps = getattr(dataset, "all_steps", None)
    if steps is None:
        # Custom member without a step index: every trajectory is sampleable.
        return np.arange(len(dataset.trajectory_lengths), dtype=np.int64)
    if not steps:
        # The index exists and kept nothing (empty language / unreadable
        # parquet). Re-admitting the raw table would undo that filter.
        return np.array([], dtype=np.int64)
    kept = {int(trajectory_id) for trajectory_id, _ in steps}
    return np.array(
        [
            position
            for position, trajectory_id in enumerate(dataset.trajectory_ids)
            if int(trajectory_id) in kept
        ],
        dtype=np.int64,
    )


class LeRobotMixtureDataset(Dataset):
    """
    A mixture of multiple datasets. This class samples a single dataset based on the dataset weights and then calls the `__getitem__` method of the sampled dataset.
    It is recommended to modify the single dataset class instead of this class.
    """

    def __init__(
        self,
        data_mixture: Sequence[tuple[LeRobotSingleDataset, float]],
        mode: str,
        balance_dataset_weights: bool = True,
        balance_trajectory_weights: bool = True,
        seed: int = 42,
        metadata_config: dict = {
            "percentile_mixing_method": "min_max",
        },
        **kwargs,
    ):
        """
        Initialize the mixture dataset.

        Args:
            data_mixture (list[tuple[LeRobotSingleDataset, float]]): Datasets and their corresponding weights.
            mode (str): If "train", __getitem__ will return different samples every epoch; if "val" or "test", __getitem__ will return the same sample every epoch.
            balance_dataset_weights (bool): If True, the weight of dataset will be multiplied by the total trajectory length of each dataset.
            balance_trajectory_weights (bool): If True, sample trajectories within a dataset weighted by their length; otherwise, use equal weighting.
            seed (int): Random seed for sampling.
            sample_without_replacement (kwargs / data_cfg): If True, a SINGLE
                member indexes ``all_steps`` (one epoch = each step once;
                DataLoader shuffle permutes). Default False keeps the original
                with-replacement draw. Illegal on a multi-member mixture.
        """
        datasets, dataset_sampling_weights = coerce_mixture_members(data_mixture)
        self.datasets = datasets
        assert_mixture_tactile_layouts(self.datasets)
        assert_mixture_camera_slots(self.datasets)
        self.data_cfg = kwargs["data_cfg"] if "data_cfg" in kwargs else None
        apply_merged_tactile_pressure_scales(self.datasets, self.data_cfg)
        self.balance_dataset_weights = balance_dataset_weights
        self.balance_trajectory_weights = balance_trajectory_weights
        self.seed = seed
        self.mode = mode
        without_replacement = kwargs.pop("sample_without_replacement", None)
        if without_replacement is None and self.data_cfg is not None:
            without_replacement = self.data_cfg.get("sample_without_replacement", False)
        self.sample_without_replacement = False
        self._without_replacement_requested = bool(without_replacement)

        # Set properties for sampling

        # 1. Dataset lengths
        self._dataset_lengths = np.array([len(dataset) for dataset in self.datasets])
        print(f"Dataset lengths: {self._dataset_lengths}")
        self._getitem_count = 0
        # 2. Dataset sampling weights
        self._dataset_sampling_weights = np.array(dataset_sampling_weights, dtype=np.float64)
        
        if self.balance_dataset_weights:
            self._dataset_sampling_weights *= self._dataset_lengths
        if np.any(self._dataset_sampling_weights <= 0) or not np.all(
            np.isfinite(self._dataset_sampling_weights)
        ):
            raise ValueError(
                f"mixture sampling weights must be finite and > 0 after "
                f"balancing, got {self._dataset_sampling_weights}."
            )
        self._dataset_sampling_weights /= self._dataset_sampling_weights.sum()

        # 3. Trajectory sampling weights, over the SAMPLEABLE trajectories only
        # (see _sampleable_trajectory_positions: all_steps is the filtered step
        # index, and this class does its own (trajectory, base_index) draw).
        self._trajectory_sampling_weights: list[np.ndarray] = []
        self._sampleable_trajectory_positions: list[np.ndarray] = []
        for i, dataset in enumerate(self.datasets):
            name = str(getattr(dataset, "dataset_name", i))
            n_traj = len(dataset.trajectory_lengths)
            if n_traj == 0:
                raise ValueError(f"mixture member {name!r} has no trajectories.")
            positions = _sampleable_trajectory_positions(dataset)
            if positions.size == 0:
                raise ValueError(
                    f"mixture member {name!r} has no sampleable trajectories: "
                    "every one was dropped from all_steps (empty language "
                    "instruction or unreadable data). Fix or remove the member."
                )
            if positions.size < n_traj:
                print(
                    f"[LeRobotMixtureDataset] member {name!r}: sampling "
                    f"{positions.size}/{n_traj} trajectories; the rest were "
                    "dropped from all_steps (empty language instruction or "
                    "read error) and are excluded from the mixture draw too."
                )
            self._sampleable_trajectory_positions.append(positions)
            trajectory_sampling_weights = np.ones(positions.size, dtype=np.float64)
            if self.balance_trajectory_weights:
                trajectory_sampling_weights *= np.asarray(
                    dataset.trajectory_lengths, dtype=np.float64
                )[positions]
            if np.any(trajectory_sampling_weights <= 0) or not np.all(
                np.isfinite(trajectory_sampling_weights)
            ):
                raise ValueError(
                    f"mixture member {name!r} has a zero-length or non-finite "
                    "trajectory; cannot sample it."
                )
            trajectory_sampling_weights /= trajectory_sampling_weights.sum()
            self._trajectory_sampling_weights.append(trajectory_sampling_weights)

        self._configure_without_replacement(self._without_replacement_requested)

        # 4. Primary dataset indices
        self._primary_dataset_indices = np.array(dataset_sampling_weights) == 1.0
        if not np.any(self._primary_dataset_indices):
            # Epoch length is max(len_i / weight_i) over "primary" members.
            # Recipes that use probabilities (all weights < 1, e.g. 0814
            # pretrain) still sample by the normalized weights; only
            # __len__ / set_epoch cadence uses the max-weight member(s).
            max_weight = max(dataset_sampling_weights)
            self._primary_dataset_indices = np.array(dataset_sampling_weights) == max_weight
            primary_names = [
                getattr(dataset, "dataset_name", f"member[{index}]")
                for index, dataset in enumerate(self.datasets)
                if self._primary_dataset_indices[index]
            ]
            print(
                "[LeRobotMixtureDataset] no member has weight==1.0 "
                f"(weights={list(map(float, dataset_sampling_weights))}); "
                f"using max-weight member(s) {primary_names} "
                f"(weight={float(max_weight)}) as the epoch-length primary. "
                "Sampling ratios still follow the normalized weights — this "
                "only sets __len__ / set_epoch cadence, not which dataset "
                "is drawn."
            )

        if not np.any(self._primary_dataset_indices):
            # This should never happen, but just in case
            print("Error: Still no primary dataset found. Using first dataset as primary.")
            self._primary_dataset_indices = np.zeros(len(self.datasets), dtype=bool)
            self._primary_dataset_indices[0] = True

        # Shared so persistent DataLoader workers see set_epoch (a plain
        # ``self.epoch = n`` stays at 0 in those processes).
        self._epoch_shared = torch.zeros((), dtype=torch.int64)
        self._epoch_shared_ok = False
        try:
            self._epoch_shared.share_memory_()
            self._epoch_shared_ok = True
        except RuntimeError as exc:
            print(
                "[LeRobotMixtureDataset] _epoch_shared.share_memory_() failed "
                f"({exc}); persistent DataLoader workers will not see set_epoch "
                "and every epoch will redraw the same (dataset, trajectory, "
                "frame) sequence. Use num_workers=0 or a start method that "
                "supports shared tensors."
            )
        self.set_epoch(0)

        self.update_metadata(metadata_config)

    @property
    def dataset_lengths(self) -> np.ndarray:
        """The lengths of each dataset."""
        return self._dataset_lengths

    @property
    def dataset_sampling_weights(self) -> np.ndarray:
        """The sampling weights for each dataset."""
        return self._dataset_sampling_weights

    @property
    def trajectory_sampling_weights(self) -> list[np.ndarray]:
        """The sampling weights for each trajectory in each dataset."""
        return self._trajectory_sampling_weights

    @property
    def primary_dataset_indices(self) -> np.ndarray:
        """The indices of the primary datasets."""
        return self._primary_dataset_indices

    def __str__(self) -> str:
        dataset_descriptions = []
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            dataset_description = {
                "Dataset": str(dataset),
                "Sampling weight": float(weight),
            }
            dataset_descriptions.append(dataset_description)
        return json.dumps({"Mixture dataset": dataset_descriptions}, indent=2)

    def set_epoch(self, epoch: int):
        """Set the epoch for the dataset.

        Writes ``self.epoch`` and the shared buffer persistent workers read.
        """
        self.epoch = int(epoch)
        buf = getattr(self, "_epoch_shared", None)
        if buf is not None:
            buf.fill_(int(epoch))

    def _sampling_epoch(self) -> int:
        buf = getattr(self, "_epoch_shared", None)
        if buf is not None:
            return int(buf.item())
        return int(getattr(self, "epoch", 0))

    def _configure_without_replacement(self, enabled: bool) -> None:
        """Opt-in single-member ``all_steps`` indexing. Default stays off.

        Multi-member mixtures keep the original with-replacement draw — a
        without-replacement pass over the union would ignore mixture weights.
        """
        self.sample_without_replacement = False
        if not enabled:
            return
        if len(self.datasets) != 1:
            names = [
                str(getattr(dataset, "dataset_name", f"member[{index}]"))
                for index, dataset in enumerate(self.datasets)
            ]
            print(
                "[LeRobotMixtureDataset] sample_without_replacement requested "
                f"for {len(self.datasets)} members {names}; "
                "falling back to weighted sampling with replacement."
            )
            return
        dataset = self.datasets[0]
        name = str(getattr(dataset, "dataset_name", "member[0]"))
        steps = getattr(dataset, "all_steps", None)
        if steps is None:
            raise ValueError(
                f"sample_without_replacement requires member {name!r} to "
                "expose all_steps."
            )
        if len(steps) == 0:
            raise ValueError(
                f"sample_without_replacement: member {name!r} has an empty "
                "all_steps index."
            )
        self.sample_without_replacement = True
        print(
            "[LeRobotMixtureDataset] sample_without_replacement: indexing "
            f"all_steps of {name!r} ({len(steps)} steps). DataLoader shuffle "
            "/ DistributedSampler permute indices each epoch."
        )

    def _sample_step_without_replacement(
        self, index: int
    ) -> tuple[LeRobotSingleDataset, int, int]:
        dataset = self.datasets[0]
        steps = dataset.all_steps
        n = len(steps)
        idx = int(index)
        if idx < 0 or idx >= n:
            raise IndexError(
                f"sample_without_replacement index {idx} out of range "
                f"for all_steps length {n}."
            )
        trajectory_id, base_index = steps[idx]
        return dataset, int(trajectory_id), int(base_index)

    def _resample_index(self, index: int, attempt: int) -> int:
        """Deterministic replacement index for a missing-video / exception retry."""
        n = len(self)
        if n <= 0:
            raise RuntimeError("mixture dataset is empty")
        rng = np.random.default_rng(
            safe_hash((self._sampling_epoch(), int(index), self.seed, int(attempt)))
        )
        return int(rng.integers(0, n))

    def sample_step(self, index: int) -> tuple[LeRobotSingleDataset, int, int]:
        """Sample a single step from the dataset."""
        # return self.sampled_steps[index]
        if self.sample_without_replacement:
            return self._sample_step_without_replacement(index)

        # Set seed
        seed = (
            index
            if self.mode != "train"
            else safe_hash((self._sampling_epoch(), index, self.seed))
        )
        rng = np.random.default_rng(seed)

        # Sample dataset
        dataset_index = rng.choice(len(self.datasets), p=self.dataset_sampling_weights)
        dataset = self.datasets[dataset_index]

        # Sample trajectory. The draw is over the SAMPLEABLE positions only:
        # `all_steps` (what the single-dataset __getitem__ indexes) drops
        # trajectories with an empty language instruction or a read error, and
        # this class must not re-admit them.
        positions = self._sampleable_trajectory_positions[dataset_index]
        trajectory_index = int(
            positions[
                rng.choice(
                    positions.size, p=self.trajectory_sampling_weights[dataset_index]
                )
            ]
        )
        trajectory_id = dataset.trajectory_ids[trajectory_index]

        # Sample step
        base_index = rng.choice(dataset.trajectory_lengths[trajectory_index])
        return dataset, trajectory_id, base_index

    

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single trajectory and start index.

        Args:
            index (int): The index of the trajectory to get.

        Returns:
            dict: The data for the trajectory and start index.
        """
        self._getitem_count += 1
        if self._getitem_count % 1000 == 0:
            gc.collect()

        max_retries = 10
        last_exception = None
        requested = int(index)

        for attempt in range(max_retries):
            try:
                sample_tries = 0
                max_sample_tries = 200
                while True:
                    sample_tries += 1
                    if sample_tries > max_sample_tries:
                        raise RuntimeError(
                            f"Unable to sample a valid item after {max_sample_tries} attempts. "
                            f"dataset={self.datasets[0].dataset_name if len(self.datasets)>0 else 'unknown'}"
                        )

                    dataset, trajectory_id, step = self.sample_step(index)
                    # If dataset has no physical videos (e.g., image frames in parquet
                    # for VLA-Arena), do not gate sampling on mp4 existence.
                    total_videos = int(dataset.lerobot_info_meta.get("total_videos", 0))
                    if total_videos == 0:
                        break

                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = self._resample_index(requested, sample_tries)
                    
                raw_data = dataset.get_step_data(trajectory_id, step)    
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                
                return sample
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    # Log the error but continue trying
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    index = self._resample_index(requested, max_sample_tries + attempt)
                else:
                    # All retries exhausted
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    # Return a dummy sample or re-raise the exception
                    raise last_exception

    def __len__(self) -> int:
        """Get the length of a single epoch in the mixture.

        Returns:
            int: The length of a single epoch in the mixture.
        """
        if getattr(self, "sample_without_replacement", False):
            return len(self.datasets[0].all_steps)

        # Check for potential issues
        if len(self.datasets) == 0:
            return 0
            
        # Check if any dataset lengths are 0 or NaN
        if np.any(self.dataset_lengths == 0) or np.any(np.isnan(self.dataset_lengths)):
            print(f"Warning: Found zero or NaN dataset lengths: {self.dataset_lengths}")
            # Filter out zero/NaN length datasets
            valid_indices = (self.dataset_lengths > 0) & (~np.isnan(self.dataset_lengths))
            if not np.any(valid_indices):
                print("Error: All datasets have zero or NaN length")
                return 0
        else:
            valid_indices = np.ones(len(self.datasets), dtype=bool)
        
        # Check if any sampling weights are 0 or NaN
        if np.any(self.dataset_sampling_weights == 0) or np.any(np.isnan(self.dataset_sampling_weights)):
            print(f"Warning: Found zero or NaN sampling weights: {self.dataset_sampling_weights}")
            # Use only valid weights
            valid_weights = (self.dataset_sampling_weights > 0) & (~np.isnan(self.dataset_sampling_weights))
            valid_indices = valid_indices & valid_weights
            if not np.any(valid_indices):
                print("Error: All sampling weights are zero or NaN")
                return 0
        
        # Check primary dataset indices
        primary_and_valid = self.primary_dataset_indices & valid_indices
        if not np.any(primary_and_valid):
            print(f"Warning: No valid primary datasets found. Primary indices: {self.primary_dataset_indices}, Valid indices: {valid_indices}")
            # Fallback: use the largest valid dataset
            if np.any(valid_indices):
                max_length = self.dataset_lengths[valid_indices].max()
                print(f"Fallback: Using maximum dataset length: {max_length}")
                return int(max_length)
            else:
                return 0
        
        # Calculate the ratio and get max
        ratios = (self.dataset_lengths / self.dataset_sampling_weights)[primary_and_valid]
        
        # Check for NaN or inf in ratios
        if np.any(np.isnan(ratios)) or np.any(np.isinf(ratios)):
            print(f"Warning: Found NaN or inf in ratios: {ratios}")
            print(f"Dataset lengths: {self.dataset_lengths[primary_and_valid]}")
            print(f"Sampling weights: {self.dataset_sampling_weights[primary_and_valid]}")
            # Filter out invalid ratios
            valid_ratios = ratios[~np.isnan(ratios) & ~np.isinf(ratios)]
            if len(valid_ratios) == 0:
                print("Error: All ratios are NaN or inf")
                return 0
            max_ratio = valid_ratios.max()
        else:
            max_ratio = ratios.max()
        
        result = int(max_ratio)
        if result == 0:
            print(f"Warning: Dataset mixture length is 0")
        return result

    @staticmethod
    def compute_overall_statistics(
        per_task_stats: list[dict[str, dict[str, list[float] | np.ndarray]]],
        dataset_sampling_weights: list[float] | np.ndarray,
        percentile_mixing_method: str = "weighted_average",
    ) -> dict[str, dict[str, list[float]]]:
        """
        Computes overall statistics from per-task statistics using dataset sample weights.

        Args:
            per_task_stats: List of per-task statistics.
            Example format of one element in the per-task statistics list:
                {
                    "state.gripper": {
                        "min": [...],
                        "max": [...],
                        "mean": [...],
                        "std": [...],
                        "q01": [...],
                        "q99": [...],
                    },
                    ...
                }
            dataset_sampling_weights: List of sample weights for each task.
            percentile_mixing_method: The method to mix the percentiles, either "weighted_average" or "weighted_std".

        Returns:
            A dict of overall statistics per modality.
        """
        # Normalize the sample weights to sum to 1
        dataset_sampling_weights = np.array(dataset_sampling_weights)
        normalized_weights = dataset_sampling_weights / dataset_sampling_weights.sum()

        # Initialize overall statistics dict
        overall_stats: dict[str, dict[str, list[float]]] = {}

        # Get the list of modality keys
        modality_keys = per_task_stats[0].keys()

        for modality in modality_keys:
            # Number of dimensions (assuming consistent across tasks)
            num_dims = len(per_task_stats[0][modality]["mean"])

            # Initialize accumulators for means and variances
            weighted_means = np.zeros(num_dims)
            weighted_squares = np.zeros(num_dims)

            # Collect min, max, q01, q99 from all tasks
            min_list = []
            max_list = []
            q01_list = []
            q99_list = []

            for task_idx, task_stats in enumerate(per_task_stats):
                w_i = normalized_weights[task_idx]
                stats = task_stats[modality]
                means = np.array(stats["mean"])
                stds = np.array(stats["std"])

                # Update weighted sums for mean and variance
                weighted_means += w_i * means
                weighted_squares += w_i * (stds**2 + means**2)

                # Collect min, max, q01, q99
                min_list.append(stats["min"])
                max_list.append(stats["max"])
                q01_list.append(stats["q01"])
                q99_list.append(stats["q99"])

            # Compute overall mean
            overall_mean = weighted_means.tolist()

            # Compute overall variance and std deviation
            overall_variance = weighted_squares - weighted_means**2
            overall_std = np.sqrt(overall_variance).tolist()

            # Compute overall min and max per dimension
            overall_min = np.min(np.array(min_list), axis=0).tolist()
            overall_max = np.max(np.array(max_list), axis=0).tolist()

            # Compute overall q01 and q99 per dimension
            # Use weighted average of per-task quantiles
            q01_array = np.array(q01_list)
            q99_array = np.array(q99_list)
            if percentile_mixing_method == "weighted_average":
                weighted_q01 = np.average(q01_array, axis=0, weights=normalized_weights).tolist()
                weighted_q99 = np.average(q99_array, axis=0, weights=normalized_weights).tolist()
                # std_q01 = np.std(q01_array, axis=0).tolist()
                # std_q99 = np.std(q99_array, axis=0).tolist()
                # print(modality)
                # print(f"{std_q01=}, {std_q99=}")
                # print(f"{weighted_q01=}, {weighted_q99=}")
            elif percentile_mixing_method == "min_max":
                weighted_q01 = np.min(q01_array, axis=0).tolist()
                weighted_q99 = np.max(q99_array, axis=0).tolist()
            else:
                raise ValueError(f"Invalid percentile mixing method: {percentile_mixing_method}")

            # Store the overall statistics for the modality
            overall_stats[modality] = {
                "min": overall_min,
                "max": overall_max,
                "mean": overall_mean,
                "std": overall_std,
                "q01": weighted_q01,
                "q99": weighted_q99,
            }

            # Optional per-horizon-row quantiles: mixed with the same rule as
            # the pooled ones, elementwise over the (H, D) grid. All or none of
            # the datasets must carry them, otherwise the merged stats would
            # silently describe a subset of the mixture.
            for stat_name in PER_ROW_STAT_NAMES:
                present = [
                    task_stats[modality].get(stat_name) is not None
                    for task_stats in per_task_stats
                ]
                if not any(present):
                    continue
                if not all(present):
                    raise ValueError(
                        f"{stat_name} for {modality} is present in only "
                        f"{sum(present)}/{len(present)} datasets of the mixture. "
                        "Recompute the converted-pose statistics for every "
                        "dataset (they must share one action horizon)."
                    )
                per_task_per_row = [
                    np.asarray(task_stats[modality][stat_name], dtype=np.float64)
                    for task_stats in per_task_stats
                ]
                shapes = [tuple(values.shape) for values in per_task_per_row]
                if any(values.ndim != 2 for values in per_task_per_row) or len(
                    set(shapes)
                ) != 1:
                    raise ValueError(
                        f"{stat_name} for {modality} must have one shared (H, D) "
                        f"shape across the mixture, got {shapes}. All datasets "
                        "must use the same action_horizon and action layout."
                    )
                per_row = np.stack(per_task_per_row, axis=0)
                if percentile_mixing_method == "weighted_average":
                    merged = np.average(per_row, axis=0, weights=normalized_weights)
                else:
                    merged = (
                        np.min(per_row, axis=0)
                        if stat_name == "q01_per_row"
                        else np.max(per_row, axis=0)
                    )
                overall_stats[modality][stat_name] = merged.tolist()

        return overall_stats

    @staticmethod
    def _metadata_for_merge(dataset: "LeRobotSingleDataset") -> DatasetMetadata:
        """One dataset's metadata, restricted to what the run actually consumes.

        The video section of ``DatasetMetadata`` mirrors the dataset's FULL
        ``modality.json`` camera catalog, so two same-embodiment datasets whose
        on-disk catalogs differ (e.g. one declares wrist cameras, the other
        does not) could never merge — even when the run only enables cameras
        both provide. Restrict the video section to the cameras the dataset
        actually loads (its configured modality keys) before comparing; the
        run's active camera list is shared across the mixture, so the
        restricted sections are directly comparable. State/action sections
        stay untouched: their key sets and layouts must match exactly.
        """
        used_video_keys = {
            str(key).split(".", 1)[-1]
            for key in dataset.modality_keys.get("video", [])
        }
        meta_dict = dataset.metadata.model_dump(mode="json")
        video = (meta_dict.get("modalities") or {}).get("video") or {}
        meta_dict["modalities"]["video"] = {
            key: value for key, value in video.items() if key in used_video_keys
        }
        return DatasetMetadata.model_validate(meta_dict)

    @staticmethod
    def merge_metadata(
        metadatas: list[DatasetMetadata],
        dataset_sampling_weights: list[float],
        percentile_mixing_method: str,
        *,
        flexible_views: bool = False,
    ) -> DatasetMetadata:
        """Merge multiple metadata into one.

        With ``flexible_views=True``, video modality key *sets* may differ
        across members (head-only ∩ three-camera). Shared cameras must still
        agree on contract fields (not ``fps`` / native ``resolution`` —
        sampling is frame-indexed and ``obs_image_size`` re-resizes every
        view); the merged video section is the union (first dataset wins
        per key). State/action layouts stay byte-strict.
        """
        # Convert to dicts
        metadata_dicts = [metadata.model_dump(mode="json") for metadata in metadatas]
        # Create a new metadata dict
        merged_metadata = {}

        # Check all metadata have the same embodiment tag
        assert all(
            metadata.embodiment_tag == metadatas[0].embodiment_tag for metadata in metadatas
        ), "All metadata must have the same embodiment tag"
        merged_metadata["embodiment_tag"] = metadatas[0].embodiment_tag

        # Merge the dataset statistics
        dataset_statistics = {}
        dataset_statistics["state"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["state"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        dataset_statistics["action"] = LeRobotMixtureDataset.compute_overall_statistics(
            per_task_stats=[m["statistics"]["action"] for m in metadata_dicts],
            dataset_sampling_weights=dataset_sampling_weights,
            percentile_mixing_method=percentile_mixing_method,
        )
        merged_metadata["statistics"] = dataset_statistics

        # Merge the modality configs. Camera fps and native resolution are
        # recording properties, not a tensor contract: sampling is
        # frame-indexed and obs_image_size re-resizes every view, so mixed-
        # rate / mixed-native-res datasets (e.g. 25 fps vs 30 fps, 640x480 vs
        # 1280x720) are allowed to co-train. Report mismatches once and
        # compare the video configs with those fields stripped; channels and
        # the rest stay byte-strict.
        _VIDEO_CONTRACT_EXEMPT_FIELDS = frozenset({"fps", "resolution"})

        def _shared_camera_field_conflicts(field: str) -> dict[str, list]:
            by_cam: dict[str, set] = defaultdict(set)
            for metadata in metadata_dicts:
                for key, entry in (metadata["modalities"].get("video") or {}).items():
                    by_cam[key].add(
                        json.dumps((entry or {}).get(field), sort_keys=True)
                    )
            return {
                key: sorted(values)
                for key, values in by_cam.items()
                if len(values) > 1
            }

        fps_conflicts = _shared_camera_field_conflicts("fps")
        if fps_conflicts:
            print(
                "[LeRobotMixtureDataset] mixture members record different camera "
                f"fps on shared views {fps_conflicts}; fps is exempt from the "
                "metadata merge (frame-indexed sampling is unaffected)."
            )
        res_conflicts = _shared_camera_field_conflicts("resolution")
        if res_conflicts:
            print(
                "[LeRobotMixtureDataset] mixture members record different native "
                f"camera resolution on shared views {res_conflicts}; resolution "
                "is exempt from the metadata merge (obs_image_size re-resizes "
                "every view)."
            )

        def _merge_key(modality: str, configs: dict) -> str:
            if modality == "video":
                configs = {
                    key: {
                        f: v
                        for f, v in (entry or {}).items()
                        if f not in _VIDEO_CONTRACT_EXEMPT_FIELDS
                    }
                    for key, entry in configs.items()
                }
            return json.dumps(configs, sort_keys=True)

        def _video_entry_for_contract(entry: dict | None) -> dict:
            return {
                f: v
                for f, v in (entry or {}).items()
                if f not in _VIDEO_CONTRACT_EXEMPT_FIELDS
            }

        # Equality is judged on the merge key (video: fps/resolution-stripped);
        # the merged output keeps the original configs (VideoMetadata requires
        # fps), first dataset wins so the kept fps/resolution are deterministic.
        modality_merge_keys = defaultdict(set)
        modality_outputs: dict[str, dict[str, dict]] = defaultdict(dict)
        for metadata in metadata_dicts:
            for modality, configs in metadata["modalities"].items():
                merge_key = _merge_key(modality, configs)
                modality_merge_keys[modality].add(merge_key)
                modality_outputs[modality].setdefault(merge_key, configs)
        merged_metadata["modalities"] = {}
        for modality, merge_keys in modality_merge_keys.items():
            if modality == "video" and flexible_views:
                merged_video: dict[str, dict] = {}
                for metadata in metadata_dicts:
                    for key, entry in (metadata["modalities"].get("video") or {}).items():
                        if key in merged_video:
                            if _video_entry_for_contract(merged_video[key]) != _video_entry_for_contract(
                                entry
                            ):
                                raise AssertionError(
                                    f"flexible_views video merge: camera {key!r} has "
                                    f"conflicting contract metadata across mixture members "
                                    f"(fps/resolution exempt): "
                                    f"{_video_entry_for_contract(merged_video[key])} vs "
                                    f"{_video_entry_for_contract(entry)}."
                                )
                        else:
                            merged_video[key] = entry
                merged_metadata["modalities"]["video"] = merged_video
                continue
            # Check that all modality configs correspond to the same tag matches
            assert (
                len(merge_keys) == 1
            ), f"Multiple modality configs for modality {modality}: {sorted(merge_keys)}"
            merged_metadata["modalities"][modality] = modality_outputs[modality][
                next(iter(merge_keys))
            ]

        return DatasetMetadata.model_validate(merged_metadata)

    def update_metadata(self, metadata_config: dict, cached_statistics_path: Path | str | None = None) -> None:
        """
        Merge multiple metadatas into one and set the transforms with the merged metadata.

        Args:
            metadata_config (dict): Configuration for the metadata.
                "percentile_mixing_method": The method to mix the percentiles, either "weighted_average" or "min_max".
                    weighted_average: Use the weighted average of the percentiles using the weight used in sampling the datasets.
                    min_max: Use the min of the 1st percentile and max of the 99th percentile.
        """
        # If cached path is provided, try to load and apply
        if cached_statistics_path is not None:
            try:
                cached_stats = self.load_merged_statistics(cached_statistics_path)
                self.apply_cached_statistics(cached_stats)
                return
            except (FileNotFoundError, KeyError, ValidationError) as e:
                print(f"Failed to load cached statistics: {e}")
                print("Falling back to computing statistics from scratch...")

        self.tag = EmbodimentTag.NEW_EMBODIMENT.value
        self.merged_metadata: dict[str, DatasetMetadata] = {}
        # Group metadata by tag
        all_metadatas: dict[str, list[DatasetMetadata]] = {}
        all_weights: dict[str, list[float]] = {}
        for dataset, weight in zip(self.datasets, self.dataset_sampling_weights):
            if dataset.tag not in all_metadatas:
                all_metadatas[dataset.tag] = []
                all_weights[dataset.tag] = []
            all_metadatas[dataset.tag].append(self._metadata_for_merge(dataset))
            all_weights[dataset.tag].append(float(weight))
        flexible_views = (
            LeRobotSingleDataset._parse_flexible_views_flag(
                self.data_cfg.get("flexible_views", False)
            )
            if self.data_cfg is not None
            else False
        )
        for tag, metadatas in all_metadatas.items():
            self.merged_metadata[tag] = self.merge_metadata(
                metadatas=metadatas,
                dataset_sampling_weights=all_weights[tag],
                percentile_mixing_method=metadata_config["percentile_mixing_method"],
                flexible_views=flexible_views,
            )
        for dataset in self.datasets:
            dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])

    def save_dataset_statistics(self, save_path: Path | str, format: str = "json") -> None:
        """
        Save merged dataset statistics to specified path in the required format.
        Only includes statistics for keys that are actually used in the datasets.
        Gripper-related keys will be placed at the end.
        
        Args:
            save_path (Path | str): Path to save the statistics file
            format (str): Save format, currently only supports "json"
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build the data structure to save
        statistics_data = {}
        
        # Collect actually used keys from all datasets
        all_used_action_keys = []
        all_used_state_keys = []
        
        for dataset in self.datasets:
            used_action_keys, used_state_keys = get_used_modality_keys(dataset.modality_keys)
            for used_action_key in used_action_keys:
                if used_action_key not in all_used_action_keys:
                    all_used_action_keys.append(used_action_key)
            for used_state_key in used_state_keys:
                if used_state_key not in all_used_state_keys:
                    all_used_state_keys.append(used_state_key)
        
        # Organize statistics by tag
        for tag, merged_metadata in self.merged_metadata.items():
            tag_stats = {}
            
            # Process action statistics
            if hasattr(merged_metadata.statistics, 'action') and merged_metadata.statistics.action:
                action_stats = merged_metadata.statistics.action
                
                # Filter and reorder keys - iterate in all_used_action_keys order
                non_gripper_keys = []
                gripper_keys = []
                
                for key in all_used_action_keys:
                    if key in action_stats:
                        non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_action_stats = {}
                for key in reordered_keys:
                    filtered_action_stats[key] = action_stats[key]
                
                if filtered_action_stats:
                    combined_action_stats = combine_modality_stats(filtered_action_stats)
                    
                    # Collect action normalization modes from datasets of this tag.
                    # "binary" takes precedence: if any dataset marks a key as binary, use binary.
                    _action_norm_modes: dict = {}
                    for _ds in self.datasets:
                        if _ds.tag == tag:
                            for _k, _m in _extract_action_normalization_modes(_ds.transforms).items():
                                if _k not in _action_norm_modes or _m == "binary":
                                    _action_norm_modes[_k] = _m
                    mask = generate_action_mask_for_used_keys(
                        merged_metadata.modalities.action, filtered_action_stats.keys(),
                        normalization_modes=_action_norm_modes,
                    )
                    combined_action_stats["mask"] = mask
                    
                    tag_stats["action"] = combined_action_stats
                    per_row_action_stats = collect_per_row_modality_stats(
                        filtered_action_stats
                    )
                    if per_row_action_stats:
                        tag_stats["action_per_row"] = per_row_action_stats
            
            # Process state statistics
            if hasattr(merged_metadata.statistics, 'state') and merged_metadata.statistics.state:
                state_stats = merged_metadata.statistics.state
                
                # Filter and reorder keys - iterate in all_used_state_keys order
                # Filter and reorder keys - iterate in all_used_state_keys order
                non_gripper_keys = []
                gripper_keys = []
                
                for key in all_used_state_keys:
                    if key in state_stats:
                        non_gripper_keys.append(key)
                
                reordered_keys = non_gripper_keys + gripper_keys
                
                filtered_state_stats = {}
                for key in reordered_keys:
                    filtered_state_stats[key] = state_stats[key]
                
                if filtered_state_stats:
                    combined_state_stats = combine_modality_stats(filtered_state_stats)
                    tag_stats["state"] = combined_state_stats
            
            # Add dataset counts
            tag_stats.update(self._get_dataset_counts(tag))
            
            statistics_data[tag] = tag_stats
        
        # Save file
        if format.lower() == "json":
            if not str(save_path).endswith('.json'):
                save_path = save_path.with_suffix('.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(statistics_data, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}. Currently only 'json' is supported.")
        
        print(f"Merged dataset statistics saved to: {save_path}")
        print(f"Used action keys (reordered): {list(all_used_action_keys)}")
        print(f"Used state keys (reordered): {list(all_used_state_keys)}")


    def _combine_modality_stats(self, modality_stats: dict) -> dict:
        """Backward compatibility wrapper."""
        return combine_modality_stats(modality_stats)

    def _generate_action_mask_for_used_keys(self, action_modalities: dict, used_action_keys_ordered) -> list[bool]:
        """Backward compatibility wrapper."""
        return generate_action_mask_for_used_keys(action_modalities, used_action_keys_ordered)

    def _get_dataset_counts(self, tag: str) -> dict:
        """
        Get dataset count information for specified tag.
        
        Args:
            tag (str): embodiment tag
            
        Returns:
            dict: Dictionary containing num_transitions and num_trajectories
        """
        num_transitions = 0
        num_trajectories = 0
        
        # Count dataset information belonging to this tag
        for dataset in self.datasets:
            if dataset.tag == tag:
                num_transitions += len(dataset)
                num_trajectories += len(dataset.trajectory_ids)
        
        return {
            "num_transitions": num_transitions,
            "num_trajectories": num_trajectories
        }

    @classmethod
    def load_merged_statistics(cls, load_path: Path | str) -> dict:
        """
        Load merged dataset statistics from file.
        
        Args:
            load_path (Path | str): Path to the statistics file
            
        Returns:
            dict: Dictionary containing merged statistics
        """
        load_path = Path(load_path)
        if not load_path.exists():
            raise FileNotFoundError(f"Statistics file not found: {load_path}")
        
        if load_path.suffix.lower() == '.json':
            with open(load_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif load_path.suffix.lower() == '.pkl':
            import pickle
            with open(load_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported file format: {load_path.suffix}")

    def apply_cached_statistics(self, cached_statistics: dict) -> None:
        """
        Apply cached statistics to avoid recomputation.
        
        Args:
            cached_statistics (dict): Statistics loaded from file
        """
        # Validate that cached statistics match current datasets
        if "metadata" in cached_statistics:
            cached_dataset_names = set(cached_statistics["metadata"]["dataset_names"])
            current_dataset_names = set(dataset.dataset_name for dataset in self.datasets)
            
            if cached_dataset_names != current_dataset_names:
                print("Warning: Cached statistics dataset names don't match current datasets.")
                print(f"Cached: {cached_dataset_names}")
                print(f"Current: {current_dataset_names}")
                return
        
        # Apply cached statistics
        self.merged_metadata = {}
        for tag, stats_data in cached_statistics.items():
            if tag == "metadata":  # Skip metadata field
                continue
                
            # Convert back to DatasetMetadata format
            metadata_dict = {
                "embodiment_tag": tag,
                "statistics": {
                    "action": {},
                    "state": {}
                },
                "modalities": {}
            }
            
            # Convert action statistics back
            if "action" in stats_data:
                action_data = stats_data["action"]
                # This is simplified - you may need to split back to sub-keys
                metadata_dict["statistics"]["action"] = action_data
            
            # Convert state statistics back
            if "state" in stats_data:
                state_data = stats_data["state"]
                metadata_dict["statistics"]["state"] = state_data
            
            self.merged_metadata[tag] = DatasetMetadata.model_validate(metadata_dict)
        
        # Update transforms metadata for each dataset
        for dataset in self.datasets:
            if dataset.tag in self.merged_metadata:
                dataset.set_transforms_metadata(self.merged_metadata[dataset.tag])
        
        print(f"Applied cached statistics for {len(self.merged_metadata)} embodiment tags.")

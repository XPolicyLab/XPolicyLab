from __future__ import annotations

import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image as PILImage

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils import process_data as _xpl_process_data
from XPolicyLab.utils.process_data import pack_robot_state, unpack_robot_state

from .tools.action_codec import CAMERA_FRAME_POSE_FORMATS, EmbodimentActionCodec
from collections import deque

from .tools.runtime_config import (
    _parse_bool,
    assert_trailing_view_mask,
    resolve_camera_extrinsics_convention,
    resolve_ee_in_camera_frame,
    resolve_embodiment_spec,
    resolve_flexible_views,
    resolve_obs_image_sizes,
    resolve_include_state,
    resolve_required_views,
    resolve_state_history_frames,
    resolve_tactile_refresh,
    resolve_tactile_runtime,
    tactile_slow_refresh_control_period,
    TACTILE_REGION_KEY,
    TACTILE_TAXEL_KEY,
    condition_tactile_pressure,
)
from .tools.relative_pose import check_pose7, normalize_camera_pose


_CUR_DIR = Path(__file__).resolve().parent

# Obs camera-name candidates per embodiment video key (short name, without the
# "video." prefix). The order of the embodiment YAML's `video_keys` defines
# the image count AND order sent to the model — it must match training.
_CAMERA_CANDIDATES = {
    "cam_high": ["cam_head", "head_camera", "cam_high", "top_camera"],
    "cam_head": ["cam_head", "head_camera", "cam_high", "top_camera"],
    "cam_left_wrist": ["cam_left_wrist", "left_camera", "left_wrist"],
    "cam_right_wrist": ["cam_right_wrist", "right_camera", "right_wrist"],
}
# Legacy default (checkpoints without an embodiment spec): head + both wrists.
_DEFAULT_VIDEO_KEYS = ["video.cam_high", "video.cam_left_wrist", "video.cam_right_wrist"]


def _optional_path(value: str | None, *base_dirs: Path) -> Path | None:
    if value in (None, "", "null", "None"):
        return None
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path
    for base_dir in base_dirs:
        candidate = base_dir / path
        if candidate.exists():
            return candidate
    return base_dirs[0] / path


def _resolve_robot_action_dim_info(env_cfg_type: str) -> dict:
    """Resolve {arm_dim, ee_dim} for the eval robot.

    Prefers the bench helper (env_cfg/<type>.yml -> robot -> _robot_info.json);
    falls back to indexing XPolicyLab/utils/robot/_robot_info.json directly by
    env_cfg_type (the convention used by utils/get_action_dim.sh), so the
    adapter works on benches without an env_cfg folder.
    """
    try:
        return _xpl_process_data.get_robot_action_dim_info(env_cfg_type)
    except (FileNotFoundError, KeyError, OSError):
        pass
    robot_info_path = (
        Path(_xpl_process_data.__file__).resolve().parent / "robot" / "_robot_info.json"
    )
    with robot_info_path.open("r", encoding="utf-8") as stream:
        robot_info = json.load(stream)
    if env_cfg_type not in robot_info:
        raise KeyError(
            f"env_cfg_type={env_cfg_type!r} not found in {robot_info_path} "
            f"(available: {sorted(robot_info)})"
        )
    return robot_info[env_cfg_type]


def _normalize_image(image: Any) -> np.ndarray:
    """One DECODED camera frame -> HWC uint8 RGB.

    The XPolicyLab policy server decodes every observation before the model
    sees it (``decode_obs_images`` on call and infer payloads, batched lists
    included), so encoded bits never arrive here — a payload that bypassed
    the server fails loudly on the rank check below rather than being decoded
    a second time. What remains is validation plus layout/dtype normalization,
    which the server deliberately does NOT do: already-decoded arrays pass
    through it untouched. Images are RGB from data conversion through
    evaluation — a channel swap here would train the model on one order and
    evaluate it on the other.
    """
    image = np.asarray(image)

    if image.ndim != 3:
        raise ValueError(f"Expected HWC/CHW image, got shape {image.shape}.")
    if image.shape[0] in (1, 3) and image.shape[-1] not in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] != 3:
        raise ValueError(f"Expected 3 image channels, got shape {image.shape}.")

    if np.issubdtype(image.dtype, np.floating):
        # Round like the server's to_pil_preserve, not truncate: the two would
        # otherwise disagree by one level on a bench that publishes floats.
        image = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)
    return image


def _normalize_and_resize(raw_image: Any, size: tuple[int, int]) -> np.ndarray:
    """One decoded camera frame -> the HWC uint8 frame the model was trained on.

    ``size`` is ``[W, H]`` (the training ``obs_image_size`` order, which is also
    PIL's). Resampling goes through PIL exactly like the training dataloader
    (``LeRobotSingleDataset._pack_sample``: ``Image.fromarray(...).resize(size)``)
    so eval frames match the training resize bit-for-bit rather than picking up
    a different cv2 interpolation kernel.

    A frame the bench already publishes at that resolution skips the
    numpy -> PIL -> numpy round trip: Pillow short-circuits an unchanged size to
    a plain copy, so the only difference is the copy itself, which
    ``ascontiguousarray`` reproduces (and elides when the source is already
    contiguous). Bit-exact, not an approximation.
    """
    image = _normalize_image(raw_image)
    target = (int(size[0]), int(size[1]))
    if (image.shape[1], image.shape[0]) == target:
        return np.ascontiguousarray(image)
    return np.asarray(PILImage.fromarray(image).resize(target))


def _locate_camera(observation: dict[str, Any], camera_names: list[str]) -> Any:
    """The raw payload of the first camera name the obs publishes.

    Already decoded by the policy server; split from the normalize+resize so
    ``update_obs`` can keep validating the camera contract on every control
    step while the expensive part is deferred to the steps that actually
    infer (see :class:`_LazyExample`).
    """
    vision = observation.get("vision", {})
    for camera_name in camera_names:
        if camera_name not in vision:
            continue
        camera_obs = vision[camera_name]
        if isinstance(camera_obs, dict):
            for image_key in ("color", "rgb", "colors"):
                if image_key in camera_obs:
                    return camera_obs[image_key]
        else:
            return camera_obs
    raise KeyError(f"Missing camera from candidates: {camera_names}")


class _LazyExample(dict):
    """A model example whose camera images are normalized/resized on first use.

    ``update_obs`` runs once per control step, but the images are read only
    when a chunk is inferred — once per ``execute_horizon`` rows on the sync
    path, once per prefetch under RTC. Normalizing and resizing them eagerly
    threw that work away on every step in between (49 of every 50 at the
    default sync cadence, 7 of every 8 under ``rtc/default.yaml``).

    Subclasses ``dict`` so ``lang`` / ``state`` stay readable exactly as
    before; only ``image`` is deferred. ``state`` is assigned by
    ``update_obs_batch`` after construction, which is safe because nothing
    materializes before the example is stored.
    """

    __slots__ = ("_raw_images", "_image_sizes", "_images")

    def __init__(
        self,
        lang: str,
        raw_images: list[Any],
        image_sizes: list[tuple[int, int]],
    ) -> None:
        super().__init__(lang=lang)
        self._raw_images = raw_images
        self._image_sizes = list(image_sizes)
        self._images: list[np.ndarray] | None = None

    def materialize(self) -> dict[str, Any]:
        """The plain example dict the server expects, images included.

        Cached, and safe to call from the RTC prefetch worker: the
        normalize+resize is a pure function of the stored payloads, so a
        main-thread call racing the worker on the same snapshot can only
        duplicate work, never disagree.
        """
        if self._images is None:
            # strict=True: a length mismatch here would silently drop a camera
            # and send the model fewer views than it was trained on.
            self._images = [
                _normalize_and_resize(raw, size)
                for raw, size in zip(self._raw_images, self._image_sizes, strict=True)
            ]
        return {**self, "image": self._images}


def _materialize_example(example: Any) -> dict[str, Any]:
    """``_LazyExample`` -> a plain request example; anything else passes through."""
    materialize = getattr(example, "materialize", None)
    return materialize() if callable(materialize) else example


def _parse_optional_pose7(value: Any) -> np.ndarray | None:
    """``auto``/empty -> None; anything else must be a 7-d ``[xyz, qwxyz]``.

    A pinned pose is authored against the documented contract (already
    ``world_from_camera`` in the training axes), so it is validated but never
    re-interpreted by the extrinsics-convention knobs.
    """
    if value is None:
        return None
    if isinstance(value, str):
        if value.strip().lower() in {"", "auto", "none", "null"}:
            return None
        value = yaml.safe_load(value)
    try:
        return check_pose7(value, label="ee_camera_pose")
    except ValueError as exc:
        raise ValueError(
            f"ee_camera_pose must be 'auto' or a 7-d world_from_camera "
            f"[xyz, qwxyz] pose: {exc}"
        ) from exc


def _coerce_published_tactile_window(
    raw: Any, spec: dict, batch_key: str
) -> np.ndarray:
    """Published ``obs['tactile']`` array → encoder layout ``[H, U, T, C]``.

    The bench ships the current history window (oldest → current). This
    adapter overwrites; it does not accumulate frames. A single frame is
    accepted only when ``history_len == 1`` (the taxel default).
    """
    hands = spec["num_hands"]
    units = spec["num_units"]
    fields = spec["num_fields"]
    history = spec["history_len"]
    arr = np.asarray(raw, dtype=np.float32)
    if arr.ndim == 2 and fields == 1:
        arr = arr[..., None]
    if arr.ndim == 3 and arr.shape == (hands, units, fields):
        if history != 1:
            raise ValueError(
                f"obs['tactile'][{batch_key!r}] is a single frame {arr.shape} but "
                f"this checkpoint's history_len={history}. Publish the full window "
                f"[{hands}, {units}, {history}, {fields}] (oldest -> current); "
                "the adapter overwrites, it does not accumulate frames."
            )
        arr = arr[:, :, None, :]
    if arr.ndim == 3 and fields == 1 and arr.shape == (hands, units, history):
        arr = arr[..., None]
    expected = (hands, units, history, fields)
    if arr.shape != expected:
        raise ValueError(
            f"obs['tactile'][{batch_key!r}] has shape {arr.shape}, expected "
            f"{expected} (hands, "
            f"{'regions' if batch_key == TACTILE_REGION_KEY else 'taxels'}, "
            "history, fields), oldest -> current, matching the training window."
        )
    return arr


def _as_pose7(value: Any) -> np.ndarray:
    """Accept a 7-d pose or a (T, 7) array (last / only row)."""
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (7,):
        return arr
    if arr.ndim == 2 and arr.shape[-1] == 7 and arr.shape[0] >= 1:
        return arr[-1]
    raise ValueError(f"Expected camera pose shape (7,) or (T, 7), got {arr.shape}")


def _resolve_rtc_fire_threshold(
    fire_at: Any,
    server_execution_horizon: Any,
    delay_init: int,
    action_chunk_size: int,
    execute_horizon: int | None = None,
) -> int:
    """Rows-remaining at which the client prefetches the next chunk.

    This single number sets two things at once: the replan period
    (``action_chunk_size - threshold``) and the leftover length, i.e. the
    guidance window the server sees. Defaulting it to the server's
    ``execution_horizon`` makes that server-side cap exact instead of a no-op.

    Note the naming trap: this threshold is the SAME quantity as the server's
    ``rtc.execution_horizon`` — and as lerobot's ``RTCConfig.execution_horizon``,
    which is the prefix-attention ``end`` and is likewise clamped down to the
    leftover length. It is the COMPLEMENT of Kinetix's ``execute_horizon`` and
    of this module's own ``execute_horizon`` knob, both of which count the rows
    EXECUTED per cycle.

    ``auto`` without a server value resolves to
    ``max(H - cadence, min(10, H-1))``: the replan cadence matches the sync
    path (the explicit ``execute_horizon`` when given, else 16 steps), while
    never shrinking the guidance window below the historical 10-row default.
    It is floored at ``delay_init + 1`` — firing with fewer rows left than
    the inference takes stalls the queue every cycle. The RUNTIME floor in
    ``Model._rtc_fire_threshold`` (observed-delay + 1) then keeps it above
    the true latency even when that grows past this static estimate.
    """
    if fire_at in (None, "", "null", "None", "auto"):
        if server_execution_horizon:
            fire_at = int(server_execution_horizon)
        else:
            cadence = (
                int(execute_horizon)
                if execute_horizon
                else min(16, action_chunk_size - 1)
            )
            window = min(10, action_chunk_size - 1)
            fire_at = max(action_chunk_size - cadence, window)
        fire_at = max(int(fire_at), int(delay_init) + 1)
    return max(1, min(int(fire_at), action_chunk_size - 1))


# Request payload entries that carry ONE value per batch sample and must
# therefore be sliced alongside the examples when a batch is split.
_PER_SAMPLE_PAYLOAD_KEYS = ("prev_action_chunk_model", "inference_delay")


def _slice_extra_payload(
    extra_payload: dict[str, Any] | None, lo: int, hi: int
) -> dict[str, Any] | None:
    """The request payload restricted to examples ``[lo, hi)``.

    Ragged per-env lists and uniform (B, T, D) arrays are sliced; a scalar
    (the single-env fire's broadcast delay) means the same thing for every
    sample, so it is passed through untouched.
    """
    if not extra_payload:
        return extra_payload
    sliced: dict[str, Any] = {}
    for key, value in extra_payload.items():
        if key not in _PER_SAMPLE_PAYLOAD_KEYS:
            sliced[key] = value
        elif isinstance(value, (list, tuple)):
            sliced[key] = list(value)[lo:hi]
        elif isinstance(value, np.ndarray) and value.ndim == 3:
            sliced[key] = value[lo:hi]
        else:
            sliced[key] = value
    return sliced


def _merge_rtc_meta(
    parts: list[tuple[int, dict[str, Any] | None]],
) -> dict[str, Any] | None:
    """Per-sub-batch ``rtc`` metadata -> one report for the whole batch.

    ``parts`` is ``[(sub_batch_size, meta_or_None)]`` in request order. The
    merged ``per_sample`` list is the concatenation, padded with unsaturated
    placeholders for a sub-batch that reported nothing, so row i of the
    merged response still maps back to env i of the original request.
    """
    if all(meta is None for _size, meta in parts):
        return None
    per_sample: list[dict[str, Any]] = []
    for size, meta in parts:
        entries = (meta or {}).get("per_sample")
        if entries is None:
            entries = [meta or {"leftover_saturated": False}] * size
        entries = list(entries)[:size]
        entries += [{"leftover_saturated": False}] * (size - len(entries))
        per_sample.extend(entries)
    saturated = [entry for entry in per_sample if entry.get("leftover_saturated")]
    merged: dict[str, Any] = {
        "leftover_saturated": bool(saturated),
        "saturated_cells": sum(int(e.get("saturated_cells") or 0) for e in saturated),
        "per_sample": per_sample,
    }
    first_rows = [
        int(e["first_saturated_row"])
        for e in saturated
        if e.get("first_saturated_row") is not None
    ]
    if first_rows:
        merged["first_saturated_row"] = min(first_rows)
    return merged


# Fallback head-camera names when the embodiment has no head video key at all
# (cannot happen for a camera-frame checkpoint). Ordered like the training
# pipeline's CAMERA_EXTRINSICS_NAMES so a bench publishing several aliases
# resolves the same camera training did.
_EXTRINSICS_CAMERA_FALLBACK = ("cam_high", "cam_head", "head_camera", "top_camera")


def _extract_head_camera_pose(
    observation: dict[str, Any], camera_names: list[str] | None = None
) -> np.ndarray:
    """Head-cam extrinsics from ``vision/*/extrinsics|extrinsic`` only.

    ``camera_names`` is the head video key's own candidate list (the same
    list the image lookup walks), so the extrinsics always come from the
    camera whose image the model actually consumes. Returned as published;
    the caller applies the convention knobs
    (``ee_camera_extrinsics_semantics`` / ``ee_camera_axes``).
    """
    vision = observation.get("vision") or {}
    if not isinstance(vision, dict):
        raise KeyError("observation['vision'] must be a dict when ee_in_camera_frame=true")
    names = camera_names or list(_EXTRINSICS_CAMERA_FALLBACK)
    for camera_name in names:
        camera_obs = vision.get(camera_name)
        if not isinstance(camera_obs, dict):
            continue
        for key in ("extrinsics", "extrinsic"):
            if key in camera_obs:
                return _as_pose7(camera_obs[key])
    raise KeyError(
        f"ee_in_camera_frame=true requires vision/<head_cam>/extrinsics "
        f"(or extrinsic) as a 7-d [xyz, qwxyz] world-from-camera pose; tried "
        f"{names}. Benches that do not publish extrinsics can instead pin the "
        "training camera via deploy.yml's ee_camera_pose."
    )


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = self.model_cfg.get("action_type", "joint")
        if self.action_type not in ("joint", "ee"):
            raise ValueError(
                f"Spark_0 supports action_type 'joint' or 'ee', got {self.action_type!r}."
            )

        self.env_cfg_type = self.model_cfg.get("env_cfg_type")
        if self.env_cfg_type is None:
            raise ValueError("Spark_0 requires env_cfg_type.")
        self.robot_action_dim_info = _resolve_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )

        # Training-time embodiment schema, read from the checkpoint's saved
        # config (train.sh embeds name + full spec). It defines the camera
        # set/order, the pose representation and the model-side dims — the
        # deployment path below replays exactly that contract.
        checkpoint_path = self.model_cfg.get("checkpoint_path")
        embodiment_spec = resolve_embodiment_spec(
            checkpoint_path, _CUR_DIR / "model_configs" / "embodiments"
        )
        if embodiment_spec is None and self.action_type == "ee":
            raise ValueError(
                "action_type='ee' requires the checkpoint's embodiment schema "
                "(embodiment_spec/embodiment in the run's config.yaml) to decode "
                f"pose actions; none found next to checkpoint_path={checkpoint_path!r}."
            )
        self.codec = EmbodimentActionCodec(
            embodiment_spec, self.robot_action_dim_info, self.action_type
        )

        self.video_keys = (
            list(embodiment_spec.get("video_keys", _DEFAULT_VIDEO_KEYS))
            if embodiment_spec
            else list(_DEFAULT_VIDEO_KEYS)
        )
        # Replay datasets.vla_data.flexible_views: when on, a missing wrist
        # (or other optional camera) is masked out instead of KeyError — same
        # contract as training packing against modality.json.
        self.flexible_views = resolve_flexible_views(
            self.model_cfg.get("flexible_views", "auto"),
            checkpoint_path,
        )
        self.required_views = resolve_required_views(
            self.model_cfg.get("required_views", "auto"),
            checkpoint_path,
            self.video_keys,
        )
        self.camera_candidates = []
        for video_key in self.video_keys:
            short = str(video_key).split(".")[-1]
            self.camera_candidates.append(_CAMERA_CANDIDATES.get(short, [short]))
        # Extrinsics must be read from the SAME camera the head image comes
        # from, so the extrinsics lookup reuses that video key's candidate
        # list (first match wins in both) instead of a separate name order.
        self._head_extrinsics_candidates: list[str] | None = None
        for video_key, candidates in zip(self.video_keys, self.camera_candidates):
            if str(video_key).split(".")[-1] in ("cam_high", "cam_head"):
                self._head_extrinsics_candidates = list(candidates)
                break

        starvla_root = _optional_path(
            self.model_cfg.get("starvla_root"),
            _CUR_DIR,
        ) or (_CUR_DIR / "source_starvla")
        if str(starvla_root) not in sys.path:
            sys.path.insert(0, str(starvla_root))

        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

        self.client = WebsocketClientPolicy(
            self.model_cfg.get("Spark_0_server_host", "127.0.0.1"),
            int(self.model_cfg.get("Spark_0_server_port", 5694)),
        )
        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = int(server_meta["action_chunk_size"])

        # delta_xyz_rot6d chunks are RT-relative to the current state pose, so
        # row 0 is already the first action to execute (no identity prefix to
        # skip); chunk_offset is 0 for both delta and joint checkpoints.
        self.chunk_offset = self.codec.identity_prefix_rows
        usable_rows = self.action_chunk_size - self.chunk_offset

        # "auto"/null (the default) derives the horizon from the TRAINING
        # config: action_chunk_size comes from the checkpoint's saved
        # framework.action_model.action_horizon (via the server handshake).
        # An explicit number is injected from the LIVE model config's
        # top-level execute_horizon (or SPARK_0_EXECUTE_HORIZON) by the
        # launcher — never from deploy.yml.
        execute_horizon = self.model_cfg.get("execute_horizon", "auto")
        self._execute_horizon_is_explicit = execute_horizon not in (
            None,
            "",
            "null",
            "None",
            "auto",
        )
        if not self._execute_horizon_is_explicit:
            execute_horizon = usable_rows
        self.execute_horizon = int(execute_horizon)
        if self.execute_horizon <= 0:
            raise ValueError(f"execute_horizon must be positive, got {self.execute_horizon}.")
        if self.execute_horizon > usable_rows:
            raise ValueError(
                f"execute_horizon={self.execute_horizon} exceeds the usable chunk rows "
                f"({usable_rows} = action_chunk_size {self.action_chunk_size} - "
                f"{self.chunk_offset} prefix row(s))."
            )

        # Examples per server round trip. Parallel-env rollouts send every
        # env that needs a chunk in ONE request, and the server-side cost of
        # that request grows with the env count (the VLM prefills B x cameras
        # images and keeps every layer's hidden states for the action DiT).
        # A cap splits an oversized batch into consecutive round trips —
        # slower than one forward, but bounded memory instead of an OOM
        # mid-episode. auto/null/0 -> no cap (send the whole batch).
        max_infer_batch = self.model_cfg.get("max_infer_batch", "auto")
        if max_infer_batch in (None, "", "null", "None", "auto", 0, "0"):
            self.max_infer_batch: int | None = None
        else:
            self.max_infer_batch = int(max_infer_batch)
            if self.max_infer_batch < 1:
                raise ValueError(
                    f"max_infer_batch must be a positive integer or 'auto', got "
                    f"{max_infer_batch!r}."
                )

        self.unnorm_key = self.model_cfg.get("unnorm_key", "auto")
        if self.unnorm_key in (None, "", "null", "None", "auto"):
            self.unnorm_key = None  # server auto-picks the checkpoint's only key
        # "auto" (the default) replays the checkpoint's saved
        # datasets.vla_data.obs_image_size (uniform or per-camera mapping), so
        # the eval-side resize always matches training. An explicit uniform
        # [W, H] or per-camera mapping still overrides.
        self.image_sizes = resolve_obs_image_sizes(
            self.model_cfg.get("image_size", "auto"),
            checkpoint_path,
            self.video_keys,
        )
        if len(self.image_sizes) != len(self.camera_candidates):
            raise ValueError(
                f"image_sizes length {len(self.image_sizes)} != camera count "
                f"{len(self.camera_candidates)} for video_keys={self.video_keys}."
            )
        # Cross-check the SERVER's exported training contract against the
        # client-side resolution above. Both read the same checkpoint, so a
        # mismatch means the server is serving a different/stale checkpoint
        # than deploy.yml's checkpoint_path points at (or an override ate the
        # camera config) — every observation would be misaligned silently.
        server_video_keys = server_meta.get("training_video_keys")
        if server_video_keys is not None and [str(k) for k in server_video_keys] != [
            str(k) for k in self.video_keys
        ]:
            print(
                f"[Spark_0][WARN] camera contract mismatch: server trained on "
                f"{server_video_keys} but this client resolved {self.video_keys} from "
                f"checkpoint_path={checkpoint_path!r}. The server is likely serving a "
                "different checkpoint; observations will be misaligned."
            )
        server_image_sizes = server_meta.get("training_obs_image_sizes")
        if server_image_sizes:
            server_sizes = [tuple(int(v) for v in size) for size in server_image_sizes]
            if server_sizes != list(self.image_sizes):
                print(
                    f"[Spark_0][WARN] image-size mismatch: server trained at "
                    f"{server_sizes} but this client resizes to {self.image_sizes}. "
                    "Eval frames will not match the training resolution."
                )
        if "flexible_views" in server_meta:
            server_flex = bool(server_meta.get("flexible_views"))
            if server_flex != self.flexible_views:
                print(
                    f"[Spark_0][WARN] flexible_views mismatch: server="
                    f"{server_flex} client={self.flexible_views}. "
                    "Keep deploy.yml at flexible_views: auto (checkpoint replay) "
                    "or the image-count / view_mask contract will disagree."
                )
        # Backward-compatible scalar when every camera shares one size; otherwise
        # the full per-camera list (used by logs / older callers).
        unique_sizes = set(self.image_sizes)
        self.image_size = (
            next(iter(unique_sizes)) if len(unique_sizes) == 1 else list(self.image_sizes)
        )
        self.include_state = resolve_include_state(
            self.model_cfg.get("include_state", "auto"),
            checkpoint_path,
        )
        # State history frames sent to the model (training key
        # datasets.vla_data.state_history_frames; legacy checkpoints -> 1).
        # The per-env buffer below replays the training convention: frames
        # [1-N .. 0] with episode starts first/last-padded (repeat frame 0).
        self.state_history_frames = resolve_state_history_frames(
            self.model_cfg.get("state_history_frames", "auto"),
            checkpoint_path,
        )
        # Tactile streams the checkpoint's encoder expects, with the pressure
        # conditioning the dataloader applied (None when the checkpoint has no
        # tactile encoder, which leaves this whole path inert).
        self.tactile_runtime = resolve_tactile_runtime(
            self.model_cfg.get("tactile", "auto"),
            checkpoint_path,
            fast=self.model_cfg.get("tactile_fast", "auto"),
            slow=self.model_cfg.get("tactile_slow", "auto"),
        )
        self.tactile_refresh = resolve_tactile_refresh(checkpoint_path)
        # Replay datasets.vla_data.ee_in_camera_frame from the train config
        # saved next to the checkpoint (deploy.yml default "auto").
        self.ee_in_camera_frame = resolve_ee_in_camera_frame(
            self.model_cfg.get("ee_in_camera_frame", "auto"),
            checkpoint_path,
        )
        # cam_abs_xyz_rot6d is camera-frame absolute pose by definition; force
        # the deploy path to match training even if a legacy snapshot omitted
        # the flag (train.sh already rejects ee_in_camera_frame=false).
        if (
            self.codec.state_pose_format in CAMERA_FRAME_POSE_FORMATS
            or self.codec.action_pose_format in CAMERA_FRAME_POSE_FORMATS
        ):
            self.ee_in_camera_frame = True
        if self.ee_in_camera_frame and self.codec.action_pose_format is None:
            # Joint-only checkpoints have no EE pose keys to rewrite.
            self.ee_in_camera_frame = False
        # Optional pinned world-from-camera pose for benches that do not
        # publish extrinsics in the obs (valid only for a fixed camera, which
        # is what the tianji datasets record).
        self.ee_camera_pose = _parse_optional_pose7(
            self.model_cfg.get("ee_camera_pose", "auto")
        )
        # How to read the extrinsics the BENCH publishes. Training normalizes
        # its stored column to world_from_camera and refuses an ambiguous
        # declaration, but applies no axis correction and saves neither value
        # in the checkpoint — so this contract has to be asserted here, or a
        # mirrored camera frame silently corrupts every EE pose.
        (
            self.ee_camera_extrinsics_semantics,
            self.ee_camera_axes,
        ) = resolve_camera_extrinsics_convention(
            self.model_cfg.get("ee_camera_extrinsics_semantics", "auto"),
            self.model_cfg.get("ee_camera_axes", "auto"),
        )
        # The raw packed state is needed for the model input (include_state)
        # and/or as the t=0 reference pose when decoding delta pose actions.
        self._needs_raw_state = (
            self.include_state
            or self.codec.action_pose_format == "delta_xyz_rot6d"
            or self.ee_in_camera_frame
        )

        # SINGLE-TENANT CONTRACT. Every per-episode buffer below — and the
        # framework's slow-3D tactile cache, which is keyed by the `env_idx`
        # stamped onto each example — is keyed by `env_idx` ALONE. There is no
        # client/session id anywhere in the protocol, so two eval clients
        # sharing one policy server and both defaulting to `env_idx=0` (the
        # default in `deploy._env_idx`) silently read and overwrite each
        # other's observations, action chunks, RTC bookkeeping and cached 3D
        # tactile tokens. `reset()` from either client wipes both.
        #
        # Run one server per client, or give each client a disjoint `env_idx`
        # range in its obs. Do not "share" a server across independent runs.
        self.obs_by_env: dict[int, dict[str, Any]] = {}
        self.raw_state_by_env: dict[int, np.ndarray] = {}
        # Per-env world-frame raw-state history (oldest -> current). Rows are
        # kept in the WORLD frame and re-expressed in the CURRENT obs camera
        # at each update, matching training where every history row uses the
        # chunk-t0 camera.
        self.state_history_by_env: dict[int, deque] = {}
        # Bumped on reset() and stamped onto each example so the server
        # drops its slow-3D token cache across episodes. The bench publishes
        # the current history window; this adapter overwrites, it does not
        # accumulate frames.
        self._tactile_cache_gen = 0
        self.camera_pose_by_env: dict[int, np.ndarray] = {}
        # (published extrinsics, normalized pose) of the last conversion. A
        # camera property, not episode state, so `reset` deliberately keeps it.
        self._camera_pose_memo: tuple[np.ndarray, np.ndarray] | None = None
        self.action_chunks_by_env: dict[int, np.ndarray] = {}
        self.step_by_env: dict[int, int] = {}
        # Control step each env's stored observation was captured on, so an
        # inference can notice it is running on a stale one under FAST_OBS_UPDATE.
        self._obs_step_by_env: dict[int, int] = {}
        self._stale_obs_warned_envs: set[int] = set()
        # Last raw action row actually emitted per env — the hemisphere anchor
        # for the next decode. Anchoring on the observed STATE quat instead
        # re-imports the env's own q/-q canonicalization into a stream that is
        # supposed to be sign-continuous.
        self.last_action_by_env: dict[int, np.ndarray] = {}
        self._latest_env_idx_list = [0]

        # ── RTC (real-time chunking, arXiv:2506.07339) async execution ──
        # RTC is training-free: the launcher (setup_eval_policy_server.sh)
        # resolves it from the model config's top-level `rtc` pointer ->
        # model_configs/rtc/<name>.yaml and injects the client keys below as
        # deploy overrides — never from the checkpoint's training config.
        # `rtc` (auto) follows the server's framework.action_model.rtc.enable,
        # forwarded at the handshake. When disabled, `_next_action_vector`
        # below is the exact pre-RTC synchronous path. When enabled, chunks
        # are prefetched asynchronously while the robot keeps executing the
        # current one, and the server guides each new chunk with the previous
        # chunk's un-executed tail so the splice is smooth.
        server_rtc = server_meta.get("rtc") or {}
        deploy_rtc = self.model_cfg.get("rtc", "auto")
        if deploy_rtc in (None, "", "null", "None", "auto"):
            self.rtc_enabled = bool(server_rtc.get("enable", False))
        else:
            self.rtc_enabled = _parse_bool(deploy_rtc) and bool(server_rtc.get("enable", False))
            if _parse_bool(deploy_rtc) and not self.rtc_enabled:
                print(
                    "[Spark_0][WARN] rtc=true but the checkpoint/server has "
                    "framework.action_model.rtc.enable=false; using the synchronous path."
                )
        # The server-side pass-through keys (rtc_enable etc.) are applied by
        # the LAUNCHER (setup_eval_policy_server.sh translates the model
        # config's rtc pointer into --config_override for the Spark_0 server);
        # nothing in this client consumes them. If the requested server state
        # disagrees with the handshake, the server was started another way
        # (manual server_policy.py, stale server) and the pass-through was
        # lost.
        deploy_rtc_enable = self.model_cfg.get("rtc_enable", "auto")
        if deploy_rtc_enable not in (None, "", "null", "None", "auto"):
            want_enable = _parse_bool(deploy_rtc_enable)
            got_enable = bool(server_rtc.get("enable", False))
            if want_enable != got_enable:
                print(
                    f"[Spark_0][WARN] rtc_enable={want_enable} but the server "
                    f"handshake reports rtc.enable={got_enable}: the pass-through only "
                    "applies when the server is launched via setup_eval_policy_server.sh. "
                    "Restart the server through the launcher (or set "
                    "SPARK_0_CONFIG_OVERRIDE) for the model config's rtc value "
                    "to take effect."
                )
        if self.rtc_enabled and self.chunk_offset != 0:
            # The async queue emits chunk[consumed] with no offset, because a
            # leading no-op row would also shift the leftover/new-chunk
            # timestep alignment the whole splice is built on.
            raise ValueError(
                f"RTC does not support a non-zero identity prefix "
                f"(chunk_offset={self.chunk_offset}); disable rtc or the prefix."
            )
        self.rtc_delay_init = max(1, int(server_rtc.get("inference_delay") or 1))
        # Sliding window (rounds) of measured inference delays; the estimate
        # sent with each fire is its MAX (paper Algorithm 1: d = max(Q)).
        self.rtc_delay_window = max(1, int(self.model_cfg.get("rtc_delay_window", 5)))
        self.rtc_debug = _parse_bool(self.model_cfg.get("rtc_debug", False))
        fire_at_cfg = self.model_cfg.get("rtc_fire_at_remaining", "auto")
        self.rtc_execution_horizon = _resolve_rtc_fire_threshold(
            fire_at_cfg,
            server_rtc.get("execution_horizon"),
            self.rtc_delay_init,
            self.action_chunk_size,
            self.execute_horizon if self._execute_horizon_is_explicit else None,
        )
        if self.rtc_enabled and self.rtc_execution_horizon <= self.rtc_delay_init:
            print(
                f"[Spark_0][WARN] RTC fire threshold ({self.rtc_execution_horizon} rows) "
                f"does not exceed the inference-delay estimate ({self.rtc_delay_init} rows); "
                "prefetches will overrun the queue and block like the synchronous path "
                "until the runtime floor (observed delay + 1) raises it. Raise "
                "rtc_fire_at_remaining or lower rtc.inference_delay."
            )
        # An explicit execute_horizon is only IGNORED when the user or the
        # server pinned the fire threshold independently; otherwise it sets
        # the async loop's replan cadence (threshold = H - execute_horizon).
        execute_drives_rtc = (
            self.rtc_enabled
            and self._execute_horizon_is_explicit
            and fire_at_cfg in (None, "", "null", "None", "auto")
            and not server_rtc.get("execution_horizon")
        )
        if self.rtc_enabled and self._execute_horizon_is_explicit:
            if execute_drives_rtc:
                print(
                    f"[Spark_0] execute_horizon={self.execute_horizon} sets the RTC "
                    f"replan cadence: prefetch fires at {self.rtc_execution_horizon} "
                    "rows remaining (H - execute_horizon, floored at a 10-row guidance "
                    "window). Pin rtc_fire_at_remaining to override."
                )
            else:
                print(
                    f"[Spark_0][WARN] execute_horizon={self.execute_horizon} is ignored "
                    "while RTC is on: the async loop re-plans whenever the queue drops "
                    f"to {self.rtc_execution_horizon} rows (replan period "
                    f"{self.action_chunk_size - self.rtc_execution_horizon} steps). "
                    "Use rtc_fire_at_remaining to tune it."
                )
        # One worker: at most one batched prefetch is ever in flight (see
        # _rtc_maybe_fire). Extra workers could only queue behind the shared
        # websocket lock anyway.
        self._rtc_pool = ThreadPoolExecutor(max_workers=1) if self.rtc_enabled else None
        self._rtc_ws_lock = threading.Lock()
        # Consecutive async failures tolerated per env before the prefetch is
        # given up on for the rest of the episode. A single websocket hiccup
        # should not cost the whole episode's latency hiding.
        self.rtc_max_failures = max(1, int(self.model_cfg.get("rtc_max_failures", 3)))
        # Per-env async state (main thread only; worker threads return values
        # through futures and never touch these dicts). The guidance leftover
        # is NOT cached here: it is derived from the decoded (absolute, env
        # frame) chunk at fire time, which is the only representation that
        # stays valid once the reference pose has moved.
        self.rtc_consumed_by_env: dict[int, int] = {}
        # ONE batched prefetch in flight at most: every due env shares a
        # single guided request (one server forward for the whole parallel-env
        # batch) instead of serializing per-env requests behind the shared
        # websocket. `_rtc_pending_envs[i]` is the env of response row i.
        self._rtc_pending_future: Any | None = None
        self._rtc_pending_envs: list[int] = []
        # Snapshots taken this control step but not yet submitted, plus the
        # round bookkeeping that decides when to submit them. A transport that
        # splits the env batch into one call per env would otherwise fire one
        # request per env and pay the fixed per-request server cost N times.
        self._rtc_pending_entries: list[dict[str, Any]] = []
        self._rtc_round_envs: list[int] = []
        self._rtc_round_size: int = 1
        self.rtc_fire_index_by_env: dict[int, int] = {}
        # Delay observations (steps) per env, deque(maxlen=rtc_delay_window).
        # Each entry is max(queue-derived d_real, wall-clock-derived steps):
        # the queue value saturates at the fire threshold on an overrun, the
        # wall clock does not.
        self.rtc_delay_window_by_env: dict[int, deque] = {}
        self.rtc_submit_wall_by_env: dict[int, float] = {}
        self._rtc_step_wall_by_env: dict[int, float] = {}
        # Control-step period EWMA (seconds). Robot property, kept across
        # episodes; only feeds the wall->steps conversion above.
        self.rtc_step_period_s: float | None = None
        self.rtc_failures_by_env: dict[int, int] = {}
        self.rtc_failed_envs: set[int] = set()
        self.rtc_saturated_warned_envs: set[int] = set()

        # Tactile is overwrite-from-obs (the bench ships the current history
        # window), so a client that skips intermediate update_obs does not
        # thin the region history — only state still needs the client ring buffer.
        self._log_tactile_rtc_cadence()

        if self.tactile_runtime:
            tactile_summary = ", ".join(
                "{}[{}x{}x{}]".format(
                    key, spec["num_hands"], spec["num_units"], spec["history_len"]
                )
                for key, spec in self.tactile_runtime.items()
            )
        else:
            tactile_summary = "off"

        print(
            f"[Spark_0] connected to Spark-0 server, action_type={self.action_type}, "
            f"env_action_dim={self.action_dim}, model_action_dim={self.codec.model_action_dim}, "
            f"chunk={self.action_chunk_size} (identity_offset={self.chunk_offset}), "
            f"execute_horizon={self.execute_horizon}, include_state={self.include_state}, "
            f"state_history_frames={self.state_history_frames}, "
            f"tactile={tactile_summary}, "
            f"image_size={self.image_size}, "
            f"state_pose_format={self.codec.state_pose_format}, "
            f"action_pose_format={self.codec.action_pose_format}, "
            f"ee_in_camera_frame={self.ee_in_camera_frame}"
            f"{'' if self.ee_camera_pose is None else ' (pinned camera)'}"
            f"{'' if not self.ee_in_camera_frame else f' [extrinsics={self.ee_camera_extrinsics_semantics}, axes={self.ee_camera_axes}]'}, "
            f"cameras={[c[0] for c in self.camera_candidates]}, "
            f"flexible_views={self.flexible_views}"
            f"{'' if not self.flexible_views else f' (required={self.required_views})'}, "
            f"max_infer_batch={self.max_infer_batch or 'uncapped'}, "
            f"rtc={'off' if not self.rtc_enabled else f'on (fire_at<=max({self.rtc_execution_horizon}, delay+1), delay0={self.rtc_delay_init}, window={self.rtc_delay_window})'}, "
            f"unnorm_key={self.unnorm_key or 'auto'}, metadata={server_meta}"
        )

    def _head_camera_pose(self, observation: dict[str, Any]) -> np.ndarray:
        """``T_world_from_camera`` in TRAINING axes for this obs.

        A pinned ``ee_camera_pose`` is authored against that contract already;
        obs-published extrinsics go through the convention knobs first, so a
        ``camera_from_world`` column or an OpenCV camera frame is corrected
        here rather than silently mirroring every EE pose downstream.
        """
        if self.ee_camera_pose is not None:
            return self.ee_camera_pose
        published = _extract_head_camera_pose(
            observation, self._head_extrinsics_candidates
        )
        # A fixed head camera (what the Tianji datasets record) republishes the
        # same extrinsics every control step, and normalizing them is a pure
        # function of those 7 numbers — so one memo slot turns a per-step
        # matrix round trip into an array compare. Keyed on a COPY: the
        # published array can be a view into the obs buffer.
        memo = self._camera_pose_memo
        if memo is not None and np.array_equal(memo[0], published):
            return memo[1]
        normalized = normalize_camera_pose(
            published,
            semantics=self.ee_camera_extrinsics_semantics,
            axes=self.ee_camera_axes,
        )
        self._camera_pose_memo = (published.copy(), normalized)
        return normalized

    def _convert_obs(
        self, observation: dict[str, Any]
    ) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray | None]:
        """RoboDojo obs -> (model example, raw packed state, camera pose).

        The example carries the images (embodiment camera order, normalized and
        resized lazily on the first inference that reads them), the language
        instruction and — when the checkpoint was trained with state input —
        the state converted to the training representation (the server applies
        the training-time q99 normalization on top). The raw packed state is
        kept for the delta-action reference frame.
        """
        # Located (contract checked) now, normalized and resized only when a
        # chunk is inferred — see _LazyExample. With flexible_views, optional
        # cameras may be absent: omit them from the image list and record
        # view_mask over the training slot order (same as dataloader packing).
        raw_images: list[Any] = []
        present_sizes: list[tuple[int, int]] = []
        view_mask: list[int] = []
        for video_key, candidates, size in zip(
            self.video_keys, self.camera_candidates, self.image_sizes
        ):
            short = str(video_key).split(".")[-1]
            try:
                raw_images.append(_locate_camera(observation, candidates))
                present_sizes.append(size)
                view_mask.append(1)
            except KeyError:
                if not self.flexible_views or short in self.required_views:
                    raise
                view_mask.append(0)
        if not any(view_mask):
            raise KeyError(
                f"flexible_views found no cameras among {self.video_keys}; "
                f"required_views={self.required_views}."
            )
        # Same trailing-suffix contract the training dataloader enforces:
        # without it a missing middle camera (e.g. left wrist absent, right
        # present) would silently shift later views one packed slot left.
        assert_trailing_view_mask(view_mask, self.video_keys)

        instruction = observation.get("instruction") or observation.get("instructions")
        if isinstance(instruction, (list, tuple)):
            instruction = instruction[0] if instruction else ""
        if instruction in (None, ""):
            instruction = self.model_cfg.get("task_name", "")

        converted_obs = _LazyExample(str(instruction), raw_images, present_sizes)
        if self.flexible_views or any(m == 0 for m in view_mask):
            converted_obs["view_mask"] = view_mask

        raw_state_world = None
        camera_pose = None
        if self._needs_raw_state:
            raw_state_world = pack_robot_state(
                observation,
                self.action_type,
                self.robot_action_dim_info,
                source_type="obs",
            ).astype(np.float64)
            if raw_state_world.ndim != 1 or raw_state_world.shape[-1] != self.action_dim:
                raise ValueError(
                    f"Expected packed state shape ({self.action_dim},), "
                    f"got {raw_state_world.shape}."
                )
            # Training expresses BOTH the state frames and action EE poses in
            # the chunk-t0 camera frame when ee_in_camera_frame is on — the
            # caller (update_obs_batch) replays that on the whole state
            # history when assembling the model state block.
            if self.ee_in_camera_frame:
                camera_pose = self._head_camera_pose(observation)

        return converted_obs, raw_state_world, camera_pose

    def update_obs(self, obs):
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list):
        # Route the whole batch BEFORE converting anything: a missing
        # obs["env_idx"] defaults to 0, so a bench that does not publish it
        # collapses every parallel env onto env 0 — one chunk cache, one
        # state history, one RTC queue for all of them. That is silently
        # wrong and only ever shows up as a bad success rate.
        env_idx_list = [int(obs.get("env_idx", 0)) for obs in obs_list]
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(
                f"update_obs_batch got {len(obs_list)} observations mapping to only "
                f"{len(set(env_idx_list))} env index/indices ({env_idx_list}). Every "
                "parallel env must publish its own obs['env_idx']; without it they "
                "share env 0's chunk cache, state history and RTC queue."
            )
        self._latest_env_idx_list = env_idx_list
        for env_idx, obs in zip(env_idx_list, obs_list):
            converted_obs, raw_state, camera_pose = self._convert_obs(obs)
            if self.include_state and raw_state is not None:
                history_payload = obs.get("state_history")
                if history_payload is None:
                    converted_obs["state"] = self._model_state_block(
                        env_idx, raw_state, camera_pose
                    )
                else:
                    converted_obs["state"] = self._model_state_block_from_payload(
                        env_idx, history_payload, raw_state, camera_pose
                    )
            converted_obs["env_idx"] = env_idx
            for key, block in self._tactile_blocks(env_idx, obs).items():
                converted_obs[key] = block
            if self.tactile_runtime:
                converted_obs["tactile_cache_gen"] = int(self._tactile_cache_gen)
            self.obs_by_env[env_idx] = converted_obs
            # Which control step this observation belongs to, so an inference
            # can tell whether it is running on a stale one (see _infer_chunks).
            self._obs_step_by_env[env_idx] = int(self.step_by_env.get(env_idx, 0))
            if raw_state is not None:
                self.raw_state_by_env[env_idx] = raw_state
            if camera_pose is not None:
                self.camera_pose_by_env[env_idx] = camera_pose
            else:
                self.camera_pose_by_env.pop(env_idx, None)

    def _log_tactile_rtc_cadence(self) -> None:
        """Log that ``slow_refresh_every`` is infer-count, not control steps."""
        refresh = getattr(self, "tactile_refresh", None)
        if not refresh or refresh.get("mode") != "fast_cached_slow":
            return
        every = int(refresh["every"])
        if not self.rtc_enabled:
            print(
                f"[Spark_0] tactile slow 3D refresh every {every} infer call(s) "
                "(slow_refresh_every is infer-count, not control steps)"
            )
            return
        rtc_period = self.action_chunk_size - self.rtc_execution_horizon
        control = tactile_slow_refresh_control_period(rtc_period, every)
        print(
            f"[Spark_0] tactile slow 3D refresh every {every} infer calls = "
            f"{control} control steps (RTC period {rtc_period} × {every}). "
            "slow_refresh_every is infer-count, not control steps."
        )
        if every > 1 and every == rtc_period:
            print(
                f"[Spark_0][WARN] slow_refresh_every={every} equals the RTC "
                f"replan period ({rtc_period} steps). Those are different units: "
                f"3D refreshes every {control} control steps. Use "
                "slow_refresh_every: 1 to refresh 3D on every RTC fire."
            )

    def _tactile_blocks(self, env_idx: int, observation: dict) -> dict[str, np.ndarray]:
        """Encoder-layout tactile tensors for one env, or ``{}``.

        The bench publishes the CURRENT history window under ``obs['tactile']``
        (oldest → current, matching the training packed layout):

        * ``region_pressure``: ``[hands, regions, history]`` or
          ``[hands, regions, history, 1]``
        * ``taxel_pressure_3d``: ``[hands, taxels, T, 4]`` or
          ``[hands, taxels, 4]`` when ``history_len=1`` — xyz in the
          checkpoint's ``xyz_frame`` (Tianji converter: wrist-local / EE),
          metres, plus pressure. ``T`` matches
          ``streams.taxel_pressure_3d.history_len`` (every shipped stem is
          1 — the bench publishes the current cloud, no window).

        A single frame is accepted only when ``history_len == 1``. The adapter
        overwrites; it does not accumulate. A stream the bench omits is left
        off the batch, which packing turns into the learned ``no_tactile``
        fallback.
        """
        del env_idx  # overwrite contract: no per-env ring buffer
        if not self.tactile_runtime:
            return {}
        published = observation.get("tactile") or {}
        if not isinstance(published, dict):
            raise TypeError(
                f"obs['tactile'] must be a dict of stream -> array, got "
                f"{type(published).__name__}."
            )

        blocks: dict[str, np.ndarray] = {}
        for batch_key, spec in self.tactile_runtime.items():
            raw = published.get(batch_key)
            if raw is None and batch_key == TACTILE_TAXEL_KEY:
                left = published.get("3d_left")
                right = published.get("3d_right")
                if left is not None or right is not None:
                    if left is None or right is None:
                        raise ValueError(
                            "obs['tactile'] needs both 3d_left and 3d_right "
                            "(or stacked taxel_pressure_3d)"
                        )
                    raw = np.stack(
                        [
                            np.asarray(left, dtype=np.float32),
                            np.asarray(right, dtype=np.float32),
                        ],
                        axis=0,
                    )
            if raw is None:
                continue
            window = _coerce_published_tactile_window(raw, spec, batch_key)
            if batch_key == TACTILE_TAXEL_KEY:
                window = np.concatenate(
                    [
                        window[..., :3],
                        condition_tactile_pressure(
                            window[..., 3:],
                            spec["pressure_log1p"],
                            spec["pressure_scale"],
                            center=spec.get("pressure_center"),
                        ),
                    ],
                    axis=-1,
                )
            else:
                window = condition_tactile_pressure(
                    window,
                    spec["pressure_log1p"],
                    spec["pressure_scale"],
                    center=spec.get("pressure_center"),
                )
                if spec.get("append_delta"):
                    from starVLA.model.modules.tactile.pressure import (
                        append_temporal_delta,
                    )

                    window = append_temporal_delta(window, history_axis=2)
            blocks[batch_key] = np.ascontiguousarray(window, dtype=np.float32)
        return blocks

    def _model_state_block(
        self,
        env_idx: int,
        raw_state_world: np.ndarray,
        camera_pose: np.ndarray | None,
    ) -> np.ndarray:
        """(state_history_frames, model_state_dim) block for one env.

        Maintains the per-env world-frame history (oldest -> current); an
        empty buffer (episode start) is seeded by repeating the first frame —
        the same first/last padding the training dataloader applies to
        history indices that fall before the episode start. All rows are
        expressed in the CURRENT obs camera (training: chunk-t0 camera) and
        converted to the training representation. With state_history_frames=1
        this reduces exactly to the legacy single-frame block.
        """
        history = self.state_history_by_env.get(env_idx)
        if history is None:
            history = deque(maxlen=self.state_history_frames)
            self.state_history_by_env[env_idx] = history
        if not history:
            history.extend([raw_state_world] * self.state_history_frames)
        else:
            history.append(raw_state_world)
        rows = np.stack(list(history), axis=0)  # (N, raw_state_dim), world
        if self.ee_in_camera_frame:
            rows = self.codec.poses_world_to_camera(
                rows, camera_pose, modality="state"
            )
        return self.codec.state_raw_to_model(rows)

    def _model_state_block_from_payload(
        self,
        env_idx: int,
        history_payload: Any,
        raw_state_world: np.ndarray,
        camera_pose: np.ndarray | None,
    ) -> np.ndarray:
        """Same block as :meth:`_model_state_block`, but from a client-supplied
        history instead of the per-env buffer this class maintains.

        ``deploy.py``'s fast obs-update path skips ``update_obs`` on the steps
        that only read a cached action row, which would leave the buffer with
        gaps; instead it keeps the ring buffer itself and ships the whole
        window (oldest -> current, already first-padded at the episode start)
        with the observation that infers. Each entry is a raw ``obs['state']``
        payload — packing stays here so the caller needs no embodiment
        knowledge.
        """
        frames = list(history_payload)
        if len(frames) != self.state_history_frames:
            raise ValueError(
                f"obs['state_history'] must carry exactly "
                f"{self.state_history_frames} frames (oldest -> current), got "
                f"{len(frames)}. The episode start is first-padded by repeating "
                "the first frame, the same way training pads history indices "
                "before the episode start."
            )
        rows = np.stack(
            [
                pack_robot_state(
                    {"state": frame},
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                ).astype(np.float64)
                for frame in frames
            ],
            axis=0,
        )
        # The last history row IS the current observation; a mismatch means the
        # client's buffer drifted out of step with the obs it just sent, which
        # would condition the model on a history that never happened.
        if not np.allclose(rows[-1], raw_state_world):
            raise ValueError(
                "obs['state_history'][-1] must be the current obs['state']; the "
                "client-side history buffer is out of sync with the observation."
            )
        # Own rows, not views into the block below: the buffer outlives this
        # call and must stay in the WORLD frame, like _model_state_block's.
        self.state_history_by_env[env_idx] = deque(
            (row.copy() for row in rows), maxlen=self.state_history_frames
        )
        if self.ee_in_camera_frame:
            rows = self.codec.poses_world_to_camera(
                rows, camera_pose, modality="state"
            )
        return self.codec.state_raw_to_model(rows)

    def _request_chunks_full(
        self,
        examples: list[dict[str, Any]],
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """A whole env batch -> the (B, T, D) un-normalized model action
        chunks plus the server-side RTC metadata (if any).

        One round trip, unless ``max_infer_batch`` caps the examples per
        request: the batch is then split into consecutive sub-batches whose
        responses are concatenated back into the caller's env order. Batch
        rows are independent (the VLM and the DiT both attend within a
        sample), so splitting changes no result — only the peak server-side
        activation memory.
        """
        examples = list(examples)
        if not examples:
            raise ValueError("_request_chunks_full requires at least one example.")
        cap = self.max_infer_batch or len(examples)
        if len(examples) <= cap:
            return self._request_chunks_once(examples, extra_payload)
        chunks: list[np.ndarray] = []
        parts: list[tuple[int, dict[str, Any] | None]] = []
        for lo in range(0, len(examples), cap):
            hi = min(lo + cap, len(examples))
            sub_chunks, sub_meta = self._request_chunks_once(
                examples[lo:hi], _slice_extra_payload(extra_payload, lo, hi)
            )
            chunks.append(sub_chunks)
            parts.append((hi - lo, sub_meta))
        return np.concatenate(chunks, axis=0), _merge_rtc_meta(parts)

    def _request_chunks_once(
        self,
        examples: list[dict[str, Any]],
        extra_payload: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any] | None]:
        """One server round trip for a whole env batch.

        The RTC metadata rides along in the response's ``data`` dict
        (additive key ``rtc`` — older servers simply do not send it); it
        carries the leftover-saturation report so the client can surface it
        on its own side instead of leaving it in a server log the operator
        may not see.
        """
        # Note: the sampler step count is fixed by the checkpoint's
        # framework.action_model.num_inference_timesteps; it is not a
        # client-side knob.
        vla_input = {
            "examples": list(examples),
            "do_sample": False,
        }
        if self.unnorm_key is not None:
            vla_input["unnorm_key"] = self.unnorm_key
        if extra_payload:
            vla_input.update(extra_payload)
        with self._rtc_ws_lock:
            response = self.client.predict_action(vla_input)
        if not response.get("ok", False):
            raise RuntimeError(f"Spark_0 inference failed: {response.get('error', response)}")
        data = response["data"]
        chunks_model = np.asarray(data["actions"], dtype=np.float64)
        if (
            chunks_model.ndim != 3
            or chunks_model.shape[0] != len(examples)
            or chunks_model.shape[-1] != self.codec.model_action_dim
        ):
            raise ValueError(
                f"Expected model chunks (B={len(examples)}, T, {self.codec.model_action_dim}), "
                f"got {chunks_model.shape}; the checkpoint does not match the "
                f"embodiment schema resolved for env_cfg_type={self.env_cfg_type!r}."
            )
        rtc_meta = data.get("rtc") if isinstance(data, dict) else None
        return chunks_model, rtc_meta

    def _decode_chunk(
        self,
        chunk_model: np.ndarray,
        reference_state_world: np.ndarray | None,
        camera_pose: np.ndarray | None,
        anchor_quats: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Model chunk -> a fully decoded, directly deployable raw chunk.

        The delta pose rows are composed onto the state pose from the SAME
        observation snapshot that produced the chunk, so the returned chunk
        already contains absolute [xyz, quat_wxyz] targets in the env (world)
        frame. ``anchor_quats`` is the hemisphere anchor for the output quats
        (see :meth:`_anchor_quats`).
        """
        if self.codec.action_pose_format == "delta_xyz_rot6d" and reference_state_world is None:
            raise AssertionError(
                "delta pose actions need the current robot state as reference; "
                "obs['state'] was not available in update_obs."
            )
        if reference_state_world is None:
            reference_state_world = np.zeros(self.codec.raw_state_dim, dtype=np.float64)

        if self.ee_in_camera_frame:
            if camera_pose is None:
                raise AssertionError(
                    "ee_in_camera_frame requires vision/cam_head/extrinsics in the "
                    "same observation used for inference."
                )
            # Decode delta in the camera frame (matches training), then lift
            # absolute EE targets back to the world/env frame for the robot.
            reference_state_cam = self.codec.poses_world_to_camera(
                reference_state_world, camera_pose, modality="state"
            )
            chunk_cam = self.codec.action_model_to_raw(chunk_model, reference_state_cam)
            chunk_world = self.codec.poses_camera_to_world(
                chunk_cam, camera_pose, modality="action"
            )
            # camera->world canonicalizes every row to w >= 0, so the camera-
            # frame alignment is gone by here and has to be redone in world.
            return self.codec.realign_action_pose_quats_to_state(
                chunk_world, reference_state_world, anchor_quats
            ).astype(np.float32)

        return self.codec.action_model_to_raw(
            chunk_model, reference_state_world, anchor_quats
        )

    def _anchor_quats(self, env_idx: int) -> dict[str, np.ndarray] | None:
        """Hemisphere anchor for the next decode: the last action emitted.

        ``None`` at episode start (nothing sent yet), where the codec falls
        back to the reference state's own quats.
        """
        last_action = self.last_action_by_env.get(env_idx)
        if last_action is None:
            return None
        return self.codec.action_pose_quats(last_action)

    def _emit_action(self, env_idx: int, action_raw: np.ndarray) -> np.ndarray:
        """Record a chunk row as emitted and return it as the env action."""
        action = np.asarray(action_raw, dtype=np.float32)
        if action.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got {action.shape[-1]}.")
        self.last_action_by_env[env_idx] = np.asarray(action_raw, dtype=np.float64)
        return action

    def _encode_chunk(
        self,
        chunk_world: np.ndarray,
        reference_state_world: np.ndarray | None,
        camera_pose: np.ndarray | None,
    ) -> np.ndarray:
        """Decoded world-frame chunk -> model space, against a NEW reference.

        Exact inverse of :meth:`_decode_chunk`, and the reason RTC caches
        decoded chunks rather than the model-space ones the server returned:
        a ``delta_xyz_rot6d`` row is meaningful only together with the state
        pose it was measured from, so the leftover of a chunk generated ~one
        execution horizon ago cannot guide a chunk generated from the current
        observation until it is re-expressed against the current reference.
        Absolute env poses are the shared frame that makes that re-expression
        possible.
        """
        if self.codec.action_pose_format == "delta_xyz_rot6d" and reference_state_world is None:
            raise AssertionError(
                "delta pose actions need the current robot state as reference; "
                "obs['state'] was not available in update_obs."
            )
        if reference_state_world is None:
            reference_state_world = np.zeros(self.codec.raw_state_dim, dtype=np.float64)

        if self.ee_in_camera_frame:
            if camera_pose is None:
                raise AssertionError(
                    "ee_in_camera_frame requires vision/cam_head/extrinsics in the "
                    "same observation used for inference."
                )
            reference_state_cam = self.codec.poses_world_to_camera(
                reference_state_world, camera_pose, modality="state"
            )
            chunk_cam = self.codec.poses_world_to_camera(
                chunk_world, camera_pose, modality="action"
            )
            return self.codec.action_raw_to_model(chunk_cam, reference_state_cam)

        return self.codec.action_raw_to_model(chunk_world, reference_state_world)

    def _infer_chunks(self, env_idx_list: list[int]) -> list[np.ndarray]:
        """ONE plain (unguided) request for a whole env batch -> a fully
        decoded, directly deployable raw chunk per env, in the same order.

        Every per-env quantity the decode needs — reference state, camera
        pose, hemisphere anchor — is read from that env's own snapshot, so a
        batched call is exactly the per-env calls it replaces.
        """
        for env_idx in env_idx_list:
            if env_idx not in self.obs_by_env:
                raise AssertionError("update_obs must be called before get_action.")
        self._warn_on_stale_obs(env_idx_list)
        chunks_model, _rtc_meta = self._request_chunks_full(
            [_materialize_example(self.obs_by_env[env_idx]) for env_idx in env_idx_list]
        )
        return [
            self._decode_chunk(
                chunks_model[i],
                self.raw_state_by_env.get(env_idx),
                self.camera_pose_by_env.get(env_idx),
                self._anchor_quats(env_idx),
            )
            for i, env_idx in enumerate(env_idx_list)
        ]

    def _warn_on_stale_obs(self, env_idx_list: list[int]) -> None:
        """Report once per env when an inference runs on an old observation.

        Only reachable under ``deploy.py``'s ``FAST_OBS_UPDATE``, which sends
        a full observation on a fixed client period and skips the steps in
        between. A fire that lands off that period — an RTC threshold raised
        by the observed delay, or a queue drain — conditions on the last
        uploaded snapshot instead of failing. Without this the only symptom
        is a worse success rate.
        """
        obs_steps = getattr(self, "_obs_step_by_env", None) or {}
        warned = getattr(self, "_stale_obs_warned_envs", None)
        if warned is None:
            warned = set()
            self._stale_obs_warned_envs = warned
        for env_idx in env_idx_list:
            obs_step = obs_steps.get(env_idx)
            if obs_step is None:
                continue
            age = int(self.step_by_env.get(env_idx, 0)) - int(obs_step)
            if age <= 0 or env_idx in warned:
                continue
            warned.add(env_idx)
            print(
                f"[Spark_0][WARN] env {env_idx} inferred on an observation "
                f"{age} control step(s) old. The bench uploads observations on "
                "a fixed period (deploy.py FAST_OBS_UPDATE) and this inference "
                "fell between uploads — an RTC fire pulled in by a rising "
                "delay estimate, or a drained queue. Set FAST_OBS_UPDATE = "
                "False if this repeats. Logged once per env."
            )

    def _infer_chunk(self, env_idx: int) -> np.ndarray:
        """Single-env wrapper around :meth:`_infer_chunks`."""
        return self._infer_chunks([env_idx])[0]

    def _next_action_vectors(self, env_idx_list: list[int]) -> list[np.ndarray]:
        """One control step for a parallel-env batch (RTC on or off)."""
        if self.rtc_enabled:
            return self._next_action_vectors_rtc(env_idx_list)
        return self._next_action_vectors_sync(env_idx_list)

    def _next_action_vector(self, env_idx: int) -> np.ndarray:
        """Single-env entry of the batched control step (same semantics)."""
        return self._next_action_vectors([env_idx])[0]

    def _next_action_vectors_sync(self, env_idx_list: list[int]) -> list[np.ndarray]:
        """One control step for a parallel-env batch, RTC off.

        The per-env schedule is the pre-batch one — re-infer every
        ``execute_horizon`` steps, emit ``chunk[offset + step % horizon]`` —
        but every env whose chunk expires on THIS step shares one server
        round trip instead of paying its own. Envs keep their own step
        counters, so a batch that desynchronized (staggered episode starts,
        envs that dropped out and left the list) simply sends the smaller due
        set; envs stepping in lockstep re-infer together and send one full
        batch every ``execute_horizon`` steps.
        """
        due = [
            env_idx
            for env_idx in dict.fromkeys(env_idx_list)
            if self.action_chunks_by_env.get(env_idx) is None
            or self.step_by_env.get(env_idx, 0) % self.execute_horizon == 0
        ]
        if due:
            for env_idx, chunk in zip(due, self._infer_chunks(due)):
                self.action_chunks_by_env[env_idx] = chunk

        out = []
        for env_idx in env_idx_list:
            step = self.step_by_env.get(env_idx, 0)
            chunk = self.action_chunks_by_env[env_idx]
            action_idx = min(
                self.chunk_offset + step % self.execute_horizon, len(chunk) - 1
            )
            self.step_by_env[env_idx] = step + 1
            out.append(self._emit_action(env_idx, chunk[action_idx]))
        return out

    # ─────────────────────────── RTC async loop ───────────────────────────
    # Mirrors lerobot's RTC ActionQueue semantics: the next chunk is generated
    # while the current one executes; on arrival the new chunk REPLACES the
    # queue and its first `d_real` rows (the steps already executed from the
    # old chunk during inference) are discarded. The server-side ΠGDM guidance
    # freezes/guides those rows toward the leftover sent with the request.
    #
    # Parallel envs (sim rollouts) step in lockstep, so every phase is
    # batched: ONE collect splices a finished prefetch into every env it
    # covers, ONE blocking request refills every drained queue, and ONE
    # guided prefetch (per-env ragged leftovers + per-env delays) covers
    # every env that hit the fire threshold — one server round trip each
    # instead of per-env requests serialized behind the shared websocket.

    def _next_action_vector_rtc(self, env_idx: int) -> np.ndarray:
        """Single-env entry of the batched RTC loop (identical semantics)."""
        return self._next_action_vectors_rtc([env_idx])[0]

    def _next_action_vectors_rtc(self, env_idx_list: list[int]) -> list[np.ndarray]:
        """One control step for a parallel-env batch (or one env of it, when
        the transport splits the batch into a call per env)."""
        self._rtc_note_round(env_idx_list)
        for env_idx in env_idx_list:
            self._rtc_note_step_wall(env_idx)
        self._rtc_collect(block=False)
        self._rtc_block_for_chunks(env_idx_list)

        # Fire BEFORE consuming. Row 0 of a chunk is the action for the
        # observation it was inferred from (the sync path emits it right after
        # inferring), so the new chunk starts at the row about to be emitted:
        # firing after the emit would generate a chunk whose first row is
        # already in the past, replaying one action and leaving every later
        # one a step late. Both `leftover[0]` and `new_chunk[0]` are then the
        # same timestep — the alignment the Kinetix loop and lerobot's
        # `get_left_over` both use.
        self._rtc_maybe_fire(env_idx_list)

        out = []
        for env_idx in env_idx_list:
            chunk = self.action_chunks_by_env[env_idx]
            consumed = self.rtc_consumed_by_env.get(env_idx, 0)
            self.rtc_consumed_by_env[env_idx] = consumed + 1
            self.step_by_env[env_idx] = self.step_by_env.get(env_idx, 0) + 1
            out.append(self._emit_action(env_idx, chunk[consumed]))
        return out

    # ── Delay estimation (paper Algorithm 1: d = max(Q), horizon = max(d, s)) ──

    def _rtc_note_step_wall(self, env_idx: int) -> None:
        """Track the control-step period (EWMA) for wall-clock delay estimates.

        The queue-derived ``d_real`` saturates at the rows left at fire time:
        once the queue has drained, a chronically slow inference looks exactly
        like a fast one. The wall clock is the only unsaturated signal. A 3x
        outlier rejection keeps blocking inferences and episode pauses out of
        the period estimate.
        """
        now = time.monotonic()
        last = self._rtc_step_wall_by_env.get(env_idx)
        self._rtc_step_wall_by_env[env_idx] = now
        if last is None:
            return
        dt = now - last
        if dt <= 0:
            return
        if self.rtc_step_period_s is None:
            # Seed only from a normal step pair — a dt that spans a blocking
            # inference (episode start, overrun) would poison the estimate.
            # Control loops slower than 5 Hz simply never enable the
            # wall-clock path; the queue-derived delay still works.
            if dt <= 0.2:
                self.rtc_step_period_s = max(dt, 1e-3)
        elif dt <= 3.0 * self.rtc_step_period_s:
            self.rtc_step_period_s = 0.9 * self.rtc_step_period_s + 0.1 * dt

    def _rtc_delay_estimate(self, env_idx: int) -> int:
        """Frozen-prefix length for the next fire: MAX of the recent delay
        window (paper Algorithm 1: ``d = max(Q)``). Conservative on purpose:
        an overestimate freezes rows that are skipped at the splice anyway,
        while an underestimate leaves the rows that DO execute right after
        the splice with decayed guidance weight — a visible seam."""
        window = self.rtc_delay_window_by_env.get(env_idx)
        if window:
            return max(1, int(max(window)))
        return self.rtc_delay_init

    def _rtc_fire_threshold(self, env_idx: int) -> int:
        """Effective fire threshold: the configured base raised above the
        observed delay (paper: the execution horizon is ``max(d, s_min)``).

        A static threshold below the true latency starves the queue every
        cycle, and the queue-derived delay saturates at that same threshold,
        so a static scheme can never discover it should have fired earlier —
        the wall-clock observations in the window are what let this floor
        track reality.
        """
        return min(
            self.action_chunk_size - 1,
            max(self.rtc_execution_horizon, self._rtc_delay_estimate(env_idx) + 1),
        )

    def _rtc_delay_observation(self, env_idx: int, queue_d_real: int) -> int:
        """One delay observation for the window: max(queue, wall), capped at H.

        Queue-derived is exact when the prefetch lands before the queue
        drains; wall-derived (submit -> splice over the period EWMA, ceiled —
        the safe direction for a frozen-prefix length) stays meaningful on an
        overrun. The cap keeps a mid-episode pause from poisoning the window:
        a frozen prefix never usefully exceeds the horizon (the server clamps
        it to the leftover length anyway).
        """
        observed = max(0, int(queue_d_real))
        submit_wall = self.rtc_submit_wall_by_env.get(env_idx)
        period = self.rtc_step_period_s
        if submit_wall is not None and period:
            wall_steps = math.ceil((time.monotonic() - submit_wall) / period)
            observed = max(observed, int(wall_steps))
        return min(observed, self.action_chunk_size)

    def _rtc_note_round(self, env_idx_list: list[int]) -> None:
        """Track how many envs make up ONE control step.

        A split transport delivers the parallel batch as one `Model` call per
        env (``protocol: ws`` turns ``get_action_batch`` into one INFER frame
        each), so firing on the spot would send one prefetch request per env
        and the server would pay the fixed per-request cost N times over —
        measured to saturate the GPU at 8 envs where a batched drive sits at
        19%. Coalescing needs to know when the round ends, which is what this
        learns: an env seen twice means a new round started.
        """
        if any(env_idx in self._rtc_round_envs for env_idx in env_idx_list):
            # The previous round ended without reaching the expected size (an
            # env finished and left the batch): re-learn it, and submit what
            # was already snapshotted rather than stranding it.
            self._rtc_round_size = max(1, len(self._rtc_round_envs))
            self._rtc_round_envs = []
            self._rtc_submit_pending()
        for env_idx in env_idx_list:
            if env_idx not in self._rtc_round_envs:
                self._rtc_round_envs.append(env_idx)

    def _rtc_submit_pending(self) -> None:
        """Send the snapshots accumulated this round as ONE guided request."""
        entries = self._rtc_pending_entries
        self._rtc_pending_entries = []
        if not entries:
            return
        if self._rtc_pending_future is not None:
            # Nothing else submits, so this cannot normally happen; if it
            # somehow does, the snapshots are stale by the time the slot
            # frees. Drop them (and their fire index) — the envs simply
            # re-snapshot next round.
            for entry in entries:
                self.rtc_fire_index_by_env.pop(entry["env_idx"], None)
            return
        submit_wall = time.monotonic()
        for entry in entries:
            self.rtc_submit_wall_by_env[entry["env_idx"]] = submit_wall
        self._rtc_pending_envs = [entry["env_idx"] for entry in entries]
        self._rtc_pending_future = self._rtc_pool.submit(
            self._rtc_infer_batch_async, entries
        )

    def _rtc_maybe_fire(self, env_idx_list: list[int]) -> None:
        """Prefetch the next chunk for every env whose queue runs low — ONE
        batched guided request covering the whole due set.

        Never raises: a prefetch is an optimization, so a bad snapshot
        degrades that env to the blocking path instead of killing the episode
        (the async half already has that contract in :meth:`_rtc_collect`).
        At most one batched prefetch is in flight: the shared websocket
        serializes requests anyway, so a second batch could start no earlier
        — an env that becomes due mid-flight joins the next fire (or blocks
        on its drained queue, exactly like the sync path).

        Each due env is SNAPSHOTTED here (obs, reference pose, leftover,
        consumed index) and the batch is submitted once the round is complete
        — immediately when the whole env list arrives in one call, at the
        round's last frame when the transport splits it per env. Snapshot and
        submission therefore always sit in the same control step, so the
        deferral costs no extra delay and `s_fire` stays the consumed index
        the splice measures against.
        """
        if self._rtc_pending_future is not None:
            return
        entries = []
        pending_envs = {entry["env_idx"] for entry in self._rtc_pending_entries}
        for env_idx in env_idx_list:
            if env_idx in self.rtc_failed_envs or env_idx in pending_envs:
                continue
            chunk = self.action_chunks_by_env.get(env_idx)
            if chunk is None:
                continue
            consumed = self.rtc_consumed_by_env.get(env_idx, 0)
            remaining = len(chunk) - consumed
            threshold = self._rtc_fire_threshold(env_idx)
            if remaining <= 0 or remaining > threshold:
                continue
            example = self.obs_by_env.get(env_idx)
            if example is None:
                continue
            # Prefetch is the common FAST miss: `_infer_chunks` only sees the
            # blocking refill, not this path.
            self._warn_on_stale_obs([env_idx])
            # Leftover aligned to the NEW chunk's time base: prev[consumed:],
            # exactly lerobot's ActionQueue.get_left_over / the Kinetix shift.
            # These rows are ABSOLUTE env poses, so the shift alone keeps them
            # valid; `_encode_chunk` below is what puts them in the frame of
            # the inference they are about to guide.
            leftover_world = chunk[consumed:]
            delay = self._rtc_delay_estimate(env_idx)
            # Snapshot the per-env references NOW: update_obs replaces (never
            # mutates) these entries, so the async worker sees this exact
            # obs — and the leftover is encoded against the SAME reference
            # the server will condition on.
            reference_state_world = self.raw_state_by_env.get(env_idx)
            camera_pose = self.camera_pose_by_env.get(env_idx)
            try:
                leftover_model = self._encode_chunk(
                    leftover_world, reference_state_world, camera_pose
                )
                # leftover[0] is the row this call is about to emit, i.e. the
                # same timestep as the new chunk's row 0 — the tightest
                # possible anchor for the decode once the prefetch lands.
                anchor_quats = self.codec.action_pose_quats(leftover_world[0])
            except Exception as exc:
                self._rtc_note_failure(env_idx, exc, "leftover re-encode failed")
                continue
            entries.append(
                {
                    "env_idx": env_idx,
                    "example": example,
                    "reference_state_world": reference_state_world,
                    "camera_pose": camera_pose,
                    "anchor_quats": anchor_quats,
                    "leftover_model": np.asarray(leftover_model, dtype=np.float32),
                    "delay": int(delay),
                    "consumed": consumed,
                    "remaining": remaining,
                    "threshold": threshold,
                }
            )
        for entry in entries:
            env_idx = entry["env_idx"]
            self.rtc_fire_index_by_env[env_idx] = entry["consumed"]
            if self.rtc_debug:
                print(
                    f"[Spark_0][RTC] env{env_idx} fire: remaining={entry['remaining']} "
                    f"threshold={entry['threshold']} leftover={len(entry['leftover_model'])} "
                    f"delay_sent={entry['delay']}"
                )
        self._rtc_pending_entries.extend(entries)
        # Round complete -> one request for every env snapshotted this step.
        # `>=` keeps the un-learned case (a stale or first round) on the old
        # fire-immediately behaviour instead of stranding the snapshots.
        if len(self._rtc_round_envs) >= self._rtc_round_size:
            self._rtc_submit_pending()

    def _rtc_infer_batch_async(self, entries: list[dict[str, Any]]):
        """Worker: ONE batched guided request, then per-env decode.

        Wire format: a single-env fire keeps the legacy uniform payload (one
        (1, T, D) array + scalar delay); a multi-env fire sends a ragged list
        of per-env leftovers plus a per-env delay list, which the server
        normalizes per sample and guides with per-sample weight schedules.
        """
        # Materialized HERE rather than at fire time: the image
        # normalize+resize is the only per-observation cost left on the fire
        # path, and doing it in the worker keeps the control step that fires
        # as cheap as any other.
        examples = [_materialize_example(entry["example"]) for entry in entries]
        if len(entries) == 1:
            extra_payload = {
                "prev_action_chunk_model": entries[0]["leftover_model"][None],
                "inference_delay": entries[0]["delay"],
            }
        else:
            extra_payload = {
                "prev_action_chunk_model": [entry["leftover_model"] for entry in entries],
                "inference_delay": [entry["delay"] for entry in entries],
            }
        chunks_model, rtc_meta = self._request_chunks_full(examples, extra_payload)
        decoded = [
            self._decode_chunk(
                chunks_model[i],
                entry["reference_state_world"],
                entry["camera_pose"],
                entry["anchor_quats"],
            )
            for i, entry in enumerate(entries)
        ]
        return decoded, rtc_meta

    def _rtc_note_failure(self, env_idx: int, exc: BaseException, what: str) -> None:
        """Count one prefetch failure; give up on this env past the budget."""
        failures = self.rtc_failures_by_env.get(env_idx, 0) + 1
        self.rtc_failures_by_env[env_idx] = failures
        if failures >= self.rtc_max_failures:
            self.rtc_failed_envs.add(env_idx)
            tail = (
                f"disabling RTC prefetch for env {env_idx} after {failures} "
                "consecutive failures (synchronous path from here)"
            )
        else:
            tail = (
                f"retrying on the next fire ({failures}/{self.rtc_max_failures} "
                "consecutive failures)"
            )
        print(f"[Spark_0][WARN] RTC {what} ({exc}); {tail}.")

    def _rtc_splice(self, env_idx: int, decoded: np.ndarray) -> None:
        """Replace the queue with the new chunk, skipping the rows that were
        already executed from the old chunk while inference ran."""
        s_fire = self.rtc_fire_index_by_env.get(env_idx, 0)
        consumed_now = self.rtc_consumed_by_env.get(env_idx, 0)
        d_real = max(0, consumed_now - s_fire)
        if d_real > 0:
            observed = self._rtc_delay_observation(env_idx, d_real)
            window = self.rtc_delay_window_by_env.setdefault(
                env_idx, deque(maxlen=self.rtc_delay_window)
            )
            window.append(observed)
            if self.rtc_debug:
                print(
                    f"[Spark_0][RTC] env{env_idx} splice: d_real={d_real} "
                    f"observed={observed} window_max={max(window)} "
                    f"new_queue={len(decoded) - min(d_real, len(decoded))}"
                )
        self.rtc_submit_wall_by_env.pop(env_idx, None)
        skip = min(d_real, len(decoded))
        self.action_chunks_by_env[env_idx] = decoded[skip:]
        self.rtc_consumed_by_env[env_idx] = 0

    def _rtc_note_server_meta(self, env_idx: int, rtc_meta: dict[str, Any] | None) -> None:
        """Surface server-reported guidance problems on the client side.

        The leftover-saturation report otherwise lives only in the server
        log, which runs on a different process (and often a different
        machine) from the eval driver that the operator watches.
        """
        if not rtc_meta or not rtc_meta.get("leftover_saturated"):
            return
        if env_idx in self.rtc_saturated_warned_envs:
            return
        self.rtc_saturated_warned_envs.add(env_idx)
        print(
            f"[Spark_0][WARN] RTC guidance leftover saturated the normalized action "
            f"space (env {env_idx}, first row {rtc_meta.get('first_saturated_row')}, "
            f"{rtc_meta.get('saturated_cells')} cells): the frozen prefix is guided "
            "toward a clamped, SHORTENED target and the splice keeps part of the "
            "discontinuity RTC removes. Cause: controller tracking lag — leftover "
            "row j sits j+lag steps from the measured state while row j's "
            "normalization span covers ~j steps, so the fire threshold cannot fix "
            "this; reduce the lag. Logged once per env."
        )

    def _rtc_collect(self, block: bool = False) -> None:
        """Splice a finished batched prefetch into every env it covers.

        Non-blocking by default (a still-running prefetch changes nothing);
        ``block=True`` waits — used when a drained queue can only be refilled
        by the in-flight request. An env that left the batch mid-flight (a
        sim episode that succeeded and dropped out of ``env_idx_list``) is
        spliced like the others; its state simply dies at the next reset().
        """
        future = self._rtc_pending_future
        if future is None or (not block and not future.done()):
            return
        envs = self._rtc_pending_envs
        self._rtc_pending_future = None
        self._rtc_pending_envs = []
        try:
            decoded_list, rtc_meta = future.result()
        except Exception as exc:
            for env_idx in envs:
                self._rtc_note_failure(
                    env_idx, exc, "async inference failed, keeping the current chunk"
                )
                self.rtc_fire_index_by_env.pop(env_idx, None)
                self.rtc_submit_wall_by_env.pop(env_idx, None)
            return
        # The server's per-sample saturation report is aligned with the
        # request's example order; older servers send only the aggregate.
        per_sample = rtc_meta.get("per_sample") if isinstance(rtc_meta, dict) else None
        for i, env_idx in enumerate(envs):
            self.rtc_failures_by_env.pop(env_idx, None)
            meta_i = per_sample[i] if per_sample is not None else rtc_meta
            self._rtc_note_server_meta(env_idx, meta_i)
            self._rtc_splice(env_idx, decoded_list[i])
            self.rtc_fire_index_by_env.pop(env_idx, None)

    def _rtc_block_for_chunks(self, env_idx_list: list[int]) -> None:
        """Refill drained queues — episode start, inference overrun, or after
        an async failure (the same stall the synchronous path has per chunk).

        ONE batched plain request covers every drained env. A drained env
        covered by the in-flight prefetch waits for it instead of
        re-requesting; a FAILED prefetch leaves the queue drained, so the env
        falls through to the plain request instead of replaying the
        exhausted chunk.
        """

        def _drained(env_idx: int) -> bool:
            chunk = self.action_chunks_by_env.get(env_idx)
            return chunk is None or self.rtc_consumed_by_env.get(env_idx, 0) >= len(chunk)

        if not any(_drained(e) for e in env_idx_list):
            return
        if self._rtc_pending_future is not None and any(
            e in self._rtc_pending_envs for e in env_idx_list if _drained(e)
        ):
            self._rtc_collect(block=True)
        todo = [e for e in dict.fromkeys(env_idx_list) if _drained(e)]
        if not todo:
            return
        for env_idx, decoded in zip(todo, self._infer_chunks(todo)):
            self.action_chunks_by_env[env_idx] = decoded
            self.rtc_consumed_by_env[env_idx] = 0

    def get_action(self):
        if not self._latest_env_idx_list:
            raise AssertionError("update_obs must be called before get_action.")
        return self.get_action_batch(env_idx_list=[self._latest_env_idx_list[0]])[0]

    def get_action_batch(self, env_idx_list=None):
        # An EMPTY list is not the same as "not given": a bench whose last
        # env just finished mid-chunk asks for zero actions, and falling back
        # to the cached list there would step envs that already left.
        if env_idx_list is None:
            env_idx_list = self._latest_env_idx_list
        env_idx_list = [int(env_idx) for env_idx in env_idx_list]
        if len(set(env_idx_list)) != len(env_idx_list):
            raise ValueError(
                f"get_action_batch got a repeated env index in {env_idx_list}: each env "
                "consumes one row of its own chunk per control step, so a duplicate "
                "would silently advance that env twice and desynchronize its queue."
            )
        if not env_idx_list:
            return []
        vectors = self._next_action_vectors(env_idx_list)
        return [
            [
                unpack_robot_state(
                    vector,
                    self.action_type,
                    self.robot_action_dim_info,
                    source_type="obs",
                )
            ]
            for vector in vectors
        ]

    def reset(self):
        # Drain the in-flight RTC prefetch first so a stale chunk from the
        # previous episode can never be spliced into the new one.
        pending = self._rtc_pending_future
        if pending is not None:
            try:
                pending.result()
            except Exception:
                pass
        self._rtc_pending_future = None
        self._rtc_pending_envs = []
        # Snapshots from the previous episode must never reach the new one,
        # and the round bookkeeping is re-learned from the first steps.
        self._rtc_pending_entries = []
        self._rtc_round_envs = []
        self._rtc_round_size = 1
        self.obs_by_env.clear()
        self.raw_state_by_env.clear()
        self.state_history_by_env.clear()
        self._tactile_cache_gen = int(getattr(self, "_tactile_cache_gen", 0)) + 1
        self.camera_pose_by_env.clear()
        self.action_chunks_by_env.clear()
        self.step_by_env.clear()
        getattr(self, "_obs_step_by_env", {}).clear()
        self.last_action_by_env.clear()
        self.rtc_consumed_by_env.clear()
        self.rtc_fire_index_by_env.clear()
        self.rtc_delay_window_by_env.clear()
        self.rtc_submit_wall_by_env.clear()
        self._rtc_step_wall_by_env.clear()
        self.rtc_failures_by_env.clear()
        self.rtc_failed_envs.clear()
        self.rtc_saturated_warned_envs.clear()
        getattr(self, "_stale_obs_warned_envs", set()).clear()
        self._latest_env_idx_list = [0]

    def close(self) -> None:
        """Orderly shutdown: drain the in-flight prefetch, stop the pool,
        close the websocket.

        ``ModelTemplate`` has no close hook, so the bench never calls this;
        scripts that build several ``Model`` instances in one process (e.g.
        sequential checkpoint evals) should, or each instance keeps its RTC
        thread pool and websocket open until interpreter exit. The model is
        not usable afterwards.
        """
        pending = self._rtc_pending_future
        if pending is not None:
            try:
                pending.result()
            except Exception:
                pass
        self._rtc_pending_future = None
        self._rtc_pending_envs = []
        self._rtc_pending_entries = []
        if self._rtc_pool is not None:
            self._rtc_pool.shutdown(wait=False)
            self._rtc_pool = None
        close = getattr(self.client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass

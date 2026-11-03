"""XPolicyLab adapter for flex-pi (Flex-Pi: a multi-stream world-action model).

The upstream policy is vendored under ``FlexPi/`` and reached through its
RoboTwin deploy entry point, ``experiments.robotwin.flexpi_policy.deploy_policy``.
Nothing in the upstream tree is modified: this file only translates between the
XPolicyLab observation/action contract and the dict shape that entry point
expects, exactly as ``policy/FastWAM/model.py`` does for its own upstream.

XPolicyLab owns the rollout loop and the replanning cadence, so the upstream
``step()`` / ``pending_actions`` queue is bypassed and ``_infer_action_chunk()``
is called directly.
"""

import inspect
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.FlexPi.robodojo_real_tasks import TRAINING_INSTRUCTIONS
from XPolicyLab.policy.FlexPi.state_bridge import (
    ACTION_TYPES,
    DEFAULT_EEF_FRAME,
    EE_POSE_DIM,
    EEF_FRAMES,
    build_yam32_state,
    ee_action_dim_info,
    eef_block_to_pose7,
    load_base_transforms,
    pose7_error,
    relabel_rot,
    resolve_bridge,
    yam32_to_rd14,
    yam32_to_rd_ee16,
)
from XPolicyLab.utils.load_file import load_json
from XPolicyLab.utils.process_data import (
    get_robot_action_dim_info,
    pack_robot_state,
    unpack_robot_state,
)

POLICY_DIR = Path(__file__).resolve().parent
FLEXPI_ROOT = POLICY_DIR / "FlexPi"
FLEXPI_SRC = FLEXPI_ROOT / "src"

# XPolicyLab camera name -> the obs key the upstream deploy path reads.
# Mirrors _CAM_NAME_TO_OBS in the upstream deploy_policy.py, one hop earlier:
# upstream maps its dataset cam names (cam_high/...) onto these same obs keys.
_CAM_TO_OBS_KEY = {
    "cam_head": "head_camera",
    "cam_left_wrist": "left_camera",
    "cam_right_wrist": "right_camera",
}

# Resolution the training videos were stored at. The upstream per-cam pipeline
# resizes from this to its own per-cam sizes (_PER_CAM_HW) with one antialiased
# bilinear step, so standardizing here reproduces the training resize chain
# rather than adding a second, differently-sized one.
_DEFAULT_OBS_HW = (240, 320)


def _is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _standardize_rgb(image: np.ndarray, target_hw: Optional[tuple]) -> np.ndarray:
    """Return a contiguous uint8 RGB array, resized to ``target_hw`` unless it is None.

    No channel conversion: the policy server hands back RGB and it stays RGB
    end to end. The explicit copy matters — decoded observation arrays are
    read-only views, and the upstream depth path calls torch.from_numpy().
    """
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Expected an HWC 3-channel image, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if target_hw is None:
        return np.ascontiguousarray(image)
    h, w = target_hw
    if image.shape[:2] != (h, w):
        # INTER_AREA is the correct kernel for the downsample this usually is.
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(image)


def _standardize_depth(depth: np.ndarray, scale: float) -> np.ndarray:
    """Return a contiguous uint16 millimetre depth map, shape (H, W).

    The upstream builder assumes uint16 **millimetres** (its encoder divides by
    1000) and casts with ``.to(torch.int32).clamp_(0, 65535).to(torch.uint16)``,
    so metres fed straight through would floor to 0 or 1 without raising.

    RoboDojo pops Isaac's ``distance_to_image_plane`` and emits up to two keys
    (env/observation_manager/obs_manager.py), so the scale depends on which one
    is enabled in the env_cfg ``observation.vision`` block:

      ``depth``              float32 metres            -> scale 1000.0
      ``approximate_depth``  uint16 mm, already scaled -> scale 1.0

    ``_pick_depth`` below chooses the key and the matching scale.
    """
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim == 3 and depth.shape[0] == 1:
        depth = depth[0]
    if depth.ndim != 2:
        raise ValueError(f"Expected depth of shape (H, W), got {depth.shape}")
    scaled = np.asarray(depth, dtype=np.float64) * float(scale)
    return np.ascontiguousarray(np.clip(scaled, 0, 65535).astype(np.uint16))


def _pick_depth(cam: dict, configured_scale: float):
    """Return ``(depth_array, scale)`` for whichever RoboDojo depth key is on.

    ``depth`` (float32 metres) wins when both are present, since it has not been
    quantized to integer millimetres. ``approximate_depth`` is uint16 mm by
    construction, so its scale is 1.0 regardless of ``depth_input_scale`` —
    that knob only describes the raw ``depth`` key. Returns ``(None, None)``
    when neither key is present.
    """
    if cam.get("depth") is not None:
        return cam["depth"], configured_scale
    if cam.get("approximate_depth") is not None:
        return cam["approximate_depth"], 1.0
    return None, None


_dump_counter = 0


def _flatten_obs(obs: dict, instruction: str) -> Dict[str, np.ndarray]:
    """``_encode_obs`` output -> flat npz fields (``<obs_key>.<field>``, state, instruction)."""
    flat: Dict[str, Any] = {}
    for obs_key, entry in (obs.get("observation") or {}).items():
        for field, value in entry.items():
            flat[f"{obs_key}.{field}"] = np.asarray(value)
    flat["state"] = np.asarray(obs["joint_action"]["vector"])
    flat["instruction"] = np.array(instruction)
    return flat


def _dump_observation(obs: dict, instruction: str) -> None:
    """Save the exact observation handed to the model, gated on ``FLEXPI_DUMP_OBS``.

    Set ``FLEXPI_DUMP_OBS=<dir>`` on the SERVER to write one ``.npz`` per
    inference call into that directory; unset (the default) this is a no-op and
    costs one env lookup. The point is to compare what the policy actually sees
    at eval against what it saw in training — resolution, channel order, value
    range, depth units, state layout — without trusting any config to describe
    it. ``obs`` here is ``_encode_obs``'s output, i.e. the literal argument of
    ``_infer_action_chunk``, so nothing downstream has touched it yet.

    Failures never propagate: a debug aid must not be able to kill an eval.
    """
    out_dir = os.environ.get("FLEXPI_DUMP_OBS")
    if not out_dir:
        return
    global _dump_counter
    try:
        os.makedirs(out_dir, exist_ok=True)
        flat = _flatten_obs(obs, instruction)
        flat["call_index"] = np.array(_dump_counter)
        path = os.path.join(out_dir, f"obs_{_dump_counter:05d}.npz")
        np.savez_compressed(path, **flat)
        if _dump_counter == 0:
            shapes = ", ".join(
                f"{k}{tuple(v.shape)}:{v.dtype}" for k, v in sorted(flat.items())
            )
            print(f"[FlexPi dump] -> {out_dir}\n[FlexPi dump] {shapes}", flush=True)
        _dump_counter += 1
    except Exception as exc:  # never break an eval for a debug aid
        print(f"[FlexPi dump] skipped ({type(exc).__name__}: {exc})", file=sys.stderr)


def _ee16_poses(row16: np.ndarray) -> Dict[str, np.ndarray]:
    """``{side: pose7}`` of one ``yam32_to_rd_ee16`` row, [L_pose7 L_grip R_pose7 R_grip]."""
    width = EE_POSE_DIM + 1
    return {"left": row16[:EE_POSE_DIM], "right": row16[width:width + EE_POSE_DIM]}


def _sent_arrays(action_type: str, sent: np.ndarray, joint_head: np.ndarray) -> Dict[str, np.ndarray]:
    """Trace keys for what went to the simulator.

    ``rd14`` means "the commanded joint targets" to every analysis script that
    reads these dumps, so it is written only when joints WERE commanded. Under
    ``ee`` the joint head is still recorded, as ``joint_head14``, but a reader
    that asks for ``rd14`` fails instead of analysing targets that never ran.
    """
    if action_type == "ee":
        return {"sent16": sent, "joint_head14": joint_head}
    return {"rd14": sent}


class _ActionTrace:
    """Full action chunks plus every per-step state, gated on ``FLEXPI_DUMP_ACTIONS``.

    ``log_action_chunk`` only prints a summary; this keeps everything needed to
    measure jitter offline. Set ``FLEXPI_DUMP_ACTIONS=<dir>`` on the SERVER and
    each (episode, env) gets one ``ep<NNN>_env<i>.npz``, rewritten at every
    inference and at reset() so the file is always complete up to the last call:

      states       [N, D]      the encoded state at every update_obs, i.e. the
                               state after each executed step (25 Hz)
      chunk_at     [K]         index into ``states`` each chunk was inferred at
      anchor       [K, D]      state the chunk was decoded against
      chunk        [K, T, D']  model-layout chunk (32D for the yam32 bridge)
      action_type  ()          ``joint`` or ``ee``: which half of ``chunk`` ran
      rd14         [K, T, 14]  joint: what was sent to the simulator
      sent16       [K, T, 16]  ee: what was sent, [L_pose7 L_grip R_pose7 R_grip]
      joint_head14 [K, T, 14]  ee: the joint half, predicted but not executed

    ``episode`` counts reset() calls, i.e. one per eval batch. Unset (the
    default) this is a no-op. Failures never propagate, as with the obs dump.
    """

    def __init__(self, action_type: str = "joint"):
        self.out_dir = os.environ.get("FLEXPI_DUMP_ACTIONS") or None
        self.action_type = action_type
        self.episode = -1
        self._envs: Dict[Any, Dict[str, Any]] = {}

    def new_episode(self):
        if not self.out_dir:
            return
        self._flush()
        self.episode += 1
        self._envs = {}

    def add_state(self, env_idx, state, instruction):
        if not self.out_dir:
            return
        env = self._envs.setdefault(env_idx, {
            "states": [], "chunk_at": [], "anchor": [], "chunk": [], "sent": [], "joint_head": [],
            "instruction": instruction,
        })
        env["states"].append(np.array(state, dtype=np.float32))

    def add_chunk(self, env_idx, anchor, chunk, sent, joint_head):
        if not self.out_dir or env_idx not in self._envs:
            return
        env = self._envs[env_idx]
        env["chunk_at"].append(len(env["states"]) - 1)
        env["anchor"].append(np.array(anchor, dtype=np.float32))
        env["chunk"].append(np.array(chunk, dtype=np.float32))
        env["sent"].append(np.array(sent, dtype=np.float32))
        env["joint_head"].append(np.array(joint_head, dtype=np.float32))
        self._write(env_idx)

    def _flush(self):
        for env_idx in list(self._envs):
            self._write(env_idx)

    def _write(self, env_idx):
        env = self._envs[env_idx]
        if not env["chunk"]:
            return
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            tag = 0 if env_idx is None else int(env_idx)
            path = os.path.join(self.out_dir, f"ep{max(self.episode, 0):03d}_env{tag}.npz")
            np.savez(
                path,
                states=np.stack(env["states"]),
                chunk_at=np.asarray(env["chunk_at"], dtype=np.int64),
                anchor=np.stack(env["anchor"]),
                chunk=np.stack(env["chunk"]),
                action_type=np.array(self.action_type),
                **_sent_arrays(self.action_type, np.stack(env["sent"]), np.stack(env["joint_head"])),
                instruction=np.array(env["instruction"]),
                episode=np.array(self.episode),
                env_idx=np.array(tag),
            )
        except Exception as exc:  # never break an eval for a debug aid
            print(f"[FlexPi action dump] skipped ({type(exc).__name__}: {exc})", file=sys.stderr)


_STREAM_KEYS = ("video_latents", "dino_latents", "pointmap_latents")


class _StreamTrace:
    """Predicted future video / DINO / pointmap of every inference, gated on ``FLEXPI_DUMP_STREAMS``.

    A joint_* checkpoint already denoises all streams together with the action,
    but the deploy path asks for ``return_stream_latents=False`` and keeps only
    the action. ``attach()`` wraps the FlexPi module's ``infer_action`` to ask
    for the latents as well; that adds device-to-host copies and changes nothing
    about the action. Set ``FLEXPI_DUMP_STREAMS=<dir>`` on the SERVER and each
    (episode, env) gets a directory ``ep<NNN>_env<i>/`` with

      call_<KKKK>.pt   step (index of update_obs calls), the three stream latents
                       as returned (CPU fp32), the model-layout chunk and what was
                       sent (``rd14``, or ``sent16`` + ``joint_head14`` under ee;
                       see ``_sent_arrays``)
      obs_<SSSSS>.npz  the encoded observation (``_infer_action_chunk``'s input)
                       at every inference step and every ``stride`` steps after
                       it up to the horizon -- the frames the predicted video
                       covers, i.e. its ground truth

    ``tools/render_streams.py`` turns a run into pred-vs-actual mp4s. Episode
    numbering follows ``_ActionTrace``. Unset (the default) this is a no-op;
    failures never propagate.
    """

    def __init__(self, action_type: str = "joint"):
        self.out_dir = os.environ.get("FLEXPI_DUMP_STREAMS") or None
        self.action_type = action_type
        self.episode = -1
        self.stride = None
        self.horizon = None
        self._envs: Dict[Any, Dict[str, Any]] = {}
        self._pending: Optional[Dict[str, Any]] = None
        self._warned = False

    def attach(self, flexpi_module, num_video_frames: int, horizon: int):
        if not self.out_dir:
            return
        import functools

        orig = flexpi_module.infer_action
        if "return_stream_latents" not in inspect.signature(orig).parameters:
            raise RuntimeError(
                "FLEXPI_DUMP_STREAMS is set but this checkpoint's infer_action has no "
                "return_stream_latents; there are no predicted streams to dump."
            )
        self.horizon = int(horizon)
        self.stride = self.horizon // (int(num_video_frames) - 1)

        # functools.wraps keeps the signature visible to inspect.signature, which
        # the upstream deploy uses to decide which kwargs to pass.
        @functools.wraps(orig)
        def infer_action(*args, **kwargs):
            kwargs["return_stream_latents"] = True
            out = orig(*args, **kwargs)
            self._pending = {k: out.get(k) for k in _STREAM_KEYS}
            return out

        flexpi_module.infer_action = infer_action
        print(
            f"[FlexPi] FLEXPI_DUMP_STREAMS -> {self.out_dir} "
            f"({num_video_frames} video frames, one every {self.stride} steps)"
        )

    def new_episode(self):
        if not self.out_dir:
            return
        self.episode += 1
        self._envs = {}

    def _dir(self, env_idx):
        tag = 0 if env_idx is None else int(env_idx)
        path = os.path.join(self.out_dir, f"ep{max(self.episode, 0):03d}_env{tag}")
        os.makedirs(path, exist_ok=True)
        return path

    def _save_obs(self, env_idx, env):
        path = os.path.join(self._dir(env_idx), f"obs_{env['step']:05d}.npz")
        if not os.path.exists(path):
            np.savez_compressed(path, step=np.array(env["step"]), **_flatten_obs(env["obs"], env["instruction"]))

    def add_obs(self, env_idx, obs, instruction):
        if not self.out_dir or self.stride is None:
            return
        env = self._envs.setdefault(env_idx, {"step": -1, "calls": 0, "last_call": None})
        env["step"] += 1
        env["obs"], env["instruction"] = obs, instruction
        try:
            since = None if env["last_call"] is None else env["step"] - env["last_call"]
            if since is not None and 0 < since <= self.horizon and since % self.stride == 0:
                self._save_obs(env_idx, env)
        except Exception as exc:  # never break an eval for a debug aid
            print(f"[FlexPi stream dump] obs skipped ({type(exc).__name__}: {exc})", file=sys.stderr)

    def add_call(self, env_idx, chunk, sent, joint_head):
        pending, self._pending = self._pending, None
        if not self.out_dir or env_idx not in self._envs:
            return
        if (pending is None or pending.get("video_latents") is None) and not self._warned:
            self._warned = True
            print(
                "[FlexPi stream dump] WARNING: infer_action returned no video latents -- "
                "the action-only path ran (joint_* off?), nothing to show.",
                file=sys.stderr,
            )
        env = self._envs[env_idx]
        env["last_call"] = env["step"]
        try:
            import torch

            self._save_obs(env_idx, env)
            torch.save(
                {
                    "step": env["step"],
                    "stride": self.stride,
                    "horizon": self.horizon,
                    **(pending or {}),
                    "chunk": np.array(chunk, dtype=np.float32),
                    "action_type": self.action_type,
                    **_sent_arrays(
                        self.action_type,
                        np.array(sent, dtype=np.float32),
                        np.array(joint_head, dtype=np.float32),
                    ),
                    "instruction": env["instruction"],
                },
                os.path.join(self._dir(env_idx), f"call_{env['calls']:04d}.pt"),
            )
            env["calls"] += 1
        except Exception as exc:  # never break an eval for a debug aid
            print(f"[FlexPi stream dump] call skipped ({type(exc).__name__}: {exc})", file=sys.stderr)


_PAD34 = (26, 33)  # left/right_joint_6: identically 0 on the 6-DoF ARX-X5


class _GTReplay:
    """Dataset GT actions in place of model output, gated on ``FLEXPI_REPLAY_GT``.

    Checks whether the eval controller (per 25 Hz target: 8 interpolated substeps
    from the measured joints, then 2 hold substeps) moves the arm the way the
    expert data did. Set ``FLEXPI_REPLAY_GT=<ep>[,<ep>...]`` (RoboDojo 34D episode
    indices; dataset root ``FLEXPI_REPLAY_GT_ROOT``) on the SERVER together with
    ``allow_dummy_policy`` and ``state_bridge=yam32``. Each (episode, env) that
    reaches get_action takes the next index from the list, cycling. A chunk asked
    for at state index t is ``action[t : t + n]`` (== ``state[t+1 : t+n+1]``) with
    the pad joints dropped to the 32D bridge layout, held at the last action once
    the episode runs out, so GT targets take exactly the rd14 / client /
    controller path a policy chunk does -- or, under ``action_type: ee``, the
    EE pose targets and the simulator's IK. Unset (the default) this is a no-op.

    Under ``action_type: ee`` the GT EEF block goes back through ``eef_frame``
    as a policy chunk's does, which sends the recorded pose only when
    ``eef_frame`` is the frame the root's EEF block is in, ``eef_frame()``:
    ``relabeled`` for a root whose ``meta/info.json`` carries ``eef_frame`` (the
    relabeled RoboDojo_real corpus), ``world_link6`` for any other (the RoboDojo
    sim corpus, the raw RoboDojo_real corpus). Model refuses any other pairing at
    init; the ``.npz`` entries below count as being in the root's frame.

    An entry ending in ``.npz`` is a ``tools/openloop_replay.py --stride 32``
    output instead: its ``executed34`` (the stitched open-loop prediction, indexed
    like ``action``) is served the same way, so a prediction made from dataset
    frames can be executed in the sim next to its GT episode.
    """

    def __init__(self):
        spec = os.environ.get("FLEXPI_REPLAY_GT") or ""
        self.episodes = [e if e.endswith(".npz") else int(e) for e in spec.replace(",", " ").split()]
        root = os.environ.get("FLEXPI_REPLAY_GT_ROOT")
        if self.episodes and not root:
            raise ValueError("FLEXPI_REPLAY_GT is set but FLEXPI_REPLAY_GT_ROOT (the 34D dataset root) is not")
        self.root = Path(root) if root else None
        self._next = 0
        self._steps: Dict[Any, int] = {}
        self._actions: Dict[Any, np.ndarray] = {}

    @property
    def active(self) -> bool:
        return bool(self.episodes)

    def eef_frame(self) -> str:
        """The frame of the root's EEF block (see the class docstring)."""
        info = self.root / "meta" / "info.json"
        recorded = load_json(str(info)).get("eef_frame") if info.is_file() else None
        if recorded is None:
            return "world_link6"
        # a corpus relabeled with any other table would be replayed through the wrong constants
        table = recorded.get("P") or {}
        if not table or any(not np.array_equal(np.asarray(p, dtype=np.float64), relabel_rot(r)) for r, p in table.items()):
            raise ValueError(f"{info} records an eef_frame table that differs from state_bridge's relabeled table")
        return "relabeled"

    def new_episode(self):
        self._steps = {}
        self._actions = {}

    def add_state(self, env_idx):
        self._steps[env_idx] = self._steps.get(env_idx, 0) + 1

    def chunk(self, env_idx, n: int) -> np.ndarray:
        if env_idx not in self._actions:
            import pyarrow.parquet as pq

            ep = self.episodes[self._next % len(self.episodes)]
            self._next += 1
            if isinstance(ep, str):
                path = Path(ep)
                with np.load(path) as z:
                    a34 = z["executed34"].astype(np.float32)
                    if not np.array_equal(z["executed_idx"], np.arange(len(a34))):
                        raise ValueError(f"{path}: executed34 does not tile the episode from action 0")
                what = "open-loop prediction"
            else:
                path = self.root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
                col = pq.read_table(path, columns=["action"]).column("action")
                a34 = np.stack(col.to_numpy(zero_copy_only=False)).astype(np.float32)
                what = f"GT episode {ep}"
            self._actions[env_idx] = np.delete(a34, _PAD34, axis=1)
            print(f"[FlexPi replay] env {env_idx} <- {what} ({len(a34)} steps) {path}")
        actions = self._actions[env_idx]
        t = self._steps.get(env_idx, 1) - 1
        return actions[np.minimum(np.arange(t, t + n), len(actions) - 1)]


def _get_instruction(obs: dict, fallback: str) -> str:
    value = obs.get("instruction")
    if value is None:
        value = obs.get("instructions")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return fallback
    if hasattr(value, "item"):
        try:
            value = value.item()
        except Exception:
            pass
    # str() of bytes is the text "b'...'", a prompt no checkpoint was trained on.
    if isinstance(value, (bytes, bytearray)):
        value = bytes(value).decode("utf-8", "replace")
    text = str(value).strip()
    return text if text else fallback


# dataset_stats.json `coverage` key prefix of the relabeled RoboDojo_real corpus:
# `..._relabeled_<robot>` for a per-robot checkpoint, bare for the pooled one.
_RELABELED_CORPUS = "robodojo_realworld_34d_relabeled"
# Corpora that store the observed EE pose unchanged (eef_frame world_link6): the
# RoboDojo sim 34-D one and the raw-frame RoboDojo_real one.
_WORLD_LINK6_CORPORA = ("robodojo_34d", "robodojo_realworld_34d")


def _check_stats_coverage(stats: dict, eef_frame: str, env_cfg_type: str) -> Optional[str]:
    """Raise if the settings contradict the corpus the checkpoint's stats were fit on.

    A checkpoint trained on the relabeled corpus needs ``eef_frame: relabeled``,
    and a per-robot one needs its own robot's ``env_cfg_type``; one trained on
    the sim or the raw-frame corpus needs ``world_link6``. Each mistake otherwise
    serves without an error and moves the arms wrongly. Returns the last corpus
    checked, or None when ``coverage`` names none of these (other checkpoints,
    older stats).
    """
    skip = " (check_stats_coverage=false skips this check)."
    checked = None
    for name in stats.get("coverage") or {}:
        name = str(name)
        if name in _WORLD_LINK6_CORPORA:
            if eef_frame != "world_link6":
                raise ValueError(
                    f"dataset_stats.json covers {name}, whose EE rotations are the observed ones, "
                    f"but eef_frame={eef_frame!r}. Set eef_frame=world_link6" + skip
                )
            checked = name
            continue
        if not name.startswith(_RELABELED_CORPUS):
            continue
        if eef_frame != "relabeled":
            raise ValueError(
                f"dataset_stats.json covers {name}, so this checkpoint was trained with every EE "
                f"rotation in the relabeled tool frame, but eef_frame={eef_frame!r}. Set "
                "eef_frame=relabeled" + skip
            )
        robot = name[len(_RELABELED_CORPUS) + 1:]
        if robot and robot != env_cfg_type:
            raise ValueError(
                f"dataset_stats.json covers {name}, the {robot} checkpoint, but "
                f"env_cfg_type={env_cfg_type!r}. Serve each robot with its own checkpoint "
                "folder" + skip
            )
        checked = name
    return checked


# Encoded prompts kept per server; a RoboDojo_real server needs its robot's six.
_PROMPT_CACHE_SIZE = 32


def _cache_prompt_encodes(flexpi_module) -> None:
    """Memoize ``encode_prompt`` by the full prompt, across resets.

    Upstream keeps one encoded prompt and drops it on every reset, so each
    episode re-runs umt5-xxl on the CPU for its first call (25 s on a slow
    host). The encoder is deterministic, so a reused encode is the one a fresh
    encode returns; upstream already reuses one for a whole episode.
    """
    import functools

    orig = flexpi_module.encode_prompt
    cache: Dict[Any, Any] = {}

    @functools.wraps(orig)
    def encode_prompt(prompt):
        key = prompt if isinstance(prompt, str) else tuple(prompt)
        if key in cache:
            print(f"[FlexPi] prompt encode reused ({len(cache)} cached)")
            return cache[key]
        if len(cache) >= _PROMPT_CACHE_SIZE:
            cache.pop(next(iter(cache)))
        cache[key] = orig(prompt)
        return cache[key]

    flexpi_module.encode_prompt = encode_prompt


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = dict(model_cfg)
        self.action_type = str(self.model_cfg["action_type"]).strip().lower()
        self.env_cfg_type = self.model_cfg["env_cfg_type"]

        # What is SENT to RoboDojo; the state fed to the model is the same either
        # way. `joint` executes the checkpoint's joint half, the 14D [L_arm6 L_grip
        # R_arm6 R_grip] layout; `ee` executes its EEF half as EE pose targets,
        # which RoboDojo turns into joints with cuRobo IK (state_bridge.py). The
        # simulator tells the two apart by the action dict keys: --action-type on
        # the CLIENT only labels the result dir, so it must match the server's.
        if self.action_type not in ACTION_TYPES:
            raise ValueError(
                f"FlexPi action_type must be one of {ACTION_TYPES}, got {self.action_type!r}."
            )

        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self._ee_action_dim_info = ee_action_dim_info(self.robot_action_dim_info)
        self.default_instruction = str(
            self.model_cfg.get("default_instruction")
            or self.model_cfg.get("prompt")
            or "complete the task"
        )

        # Key absent -> the RoboDojo training resolution; explicit null -> pass
        # native frames through, for checkpoints trained on other-resolution data
        # whose deploy path resizes from native itself.
        obs_hw = self.model_cfg.get("obs_image_hw", _DEFAULT_OBS_HW)
        self.obs_hw = None if _is_none_like(obs_hw) else (int(obs_hw[0]), int(obs_hw[1]))
        self.depth_input_scale = float(self.model_cfg.get("depth_input_scale", 1000.0))
        self.use_depth = _is_true(self.model_cfg.get("use_depth", False))

        self.last_obs: Optional[Dict[str, Any]] = None
        self.last_instruction = self.default_instruction
        self._batch_obs: Dict[int, Dict[str, Any]] = {}
        self._batch_instruction: Dict[int, str] = {}
        self.model = None
        self._needs_depth = False

        self.allow_dummy_policy = _is_true(self.model_cfg.get("allow_dummy_policy", False))
        self.replan_steps = int(self.model_cfg.get("replan_steps") or 32)

        self._robot_dim = sum(self.robot_action_dim_info["arm_dim"]) + sum(
            self.robot_action_dim_info["ee_dim"]
        )
        self._bridge_mode = self.model_cfg.get("state_bridge") or "auto"
        # Which frame the EEF block (32D dims 0:18) is expressed in. Defaults to
        # the RoboDojo sim convention; see state_bridge.py's module docstring.
        self.eef_frame = self.model_cfg.get("eef_frame") or DEFAULT_EEF_FRAME
        if self.eef_frame not in EEF_FRAMES:
            raise ValueError(
                f"deploy.yml eef_frame must be one of {EEF_FRAMES}, got {self.eef_frame!r}"
            )
        if self.eef_frame == "relabeled":
            # Its tool-frame constant is per robot: a robot without one fails
            # here rather than at the first observation.
            relabel_rot(self.env_cfg_type)
        # A wrong eef_frame or another robot's checkpoint folder serves without
        # an error; when the stats name their corpus, both fail here instead.
        stats_path = self.model_cfg.get("dataset_stats_path")
        if (
            _is_true(self.model_cfg.get("check_stats_coverage", True))
            and not _is_none_like(stats_path)
            and Path(str(stats_path)).is_file()
        ):
            corpus = _check_stats_coverage(load_json(str(stats_path)), self.eef_frame, self.env_cfg_type)
            if corpus:
                print(
                    f"[FlexPi] stats coverage {corpus}: eef_frame={self.eef_frame} "
                    f"env_cfg_type={self.env_cfg_type} match"
                )
        # The RoboDojo_real checkpoints saw one instruction per task, verbatim.
        self._training_instructions = (
            set(TRAINING_INSTRUCTIONS.get(self.env_cfg_type, {}).values())
            if self.model_cfg.get("bench_name") == "RoboDojo_real"
            else set()
        )
        self._warned_instructions: set = set()
        # hold: the model runs on every call, but each chunk sent is the state
        # measured at the episode's first call, latched per env.
        self.hold = _is_true(self.model_cfg.get("hold", False))
        self._hold_rows: Dict[Any, np.ndarray] = {}
        if self.hold and self.allow_dummy_policy:
            # The dummy policy's chunk is sent before hold applies: zero joint
            # targets and closed grippers under joint, not the measured state.
            raise ValueError(
                "hold=true needs the model; with allow_dummy_policy the dummy chunk is sent "
                "instead (zeros under joint). Unset FLEXPI_ALLOW_DUMMY_POLICY or hold."
            )
        if self.hold:
            print(
                "[FlexPi] hold=true: every chunk sent is the state measured at the episode's "
                "first call; the arms stay put while the model runs and logs as usual."
            )
        self.state_bridge = "none"
        self._base_tf = None
        self._bridge_logged = False
        self.log_action_chunk = _is_true(self.model_cfg.get("log_action_chunk", False))
        # Per-env, because get_action_batch runs one inference PER ENV before
        # any of them steps: a single shared slot makes `tracking` compare two
        # different robots and read ~2 rad when the controller is actually
        # within 0.01 rad of its target.
        self._chunk_calls: Dict[Any, int] = {}
        self._prev_chunk_last_target: Dict[Any, np.ndarray] = {}
        self._prev_chunk_state: Dict[Any, np.ndarray] = {}
        # ee: the last pose7 of the previous chunk per env, {side: pose7}; the
        # arm should be there when the next chunk is asked for.
        self._prev_sent_pose7: Dict[Any, Dict[str, np.ndarray]] = {}
        self._ee_held = 0
        self._action_trace = _ActionTrace(self.action_type)
        if self._action_trace.out_dir:
            print(f"[FlexPi] FLEXPI_DUMP_ACTIONS -> {self._action_trace.out_dir}")
        self._stream_trace = _StreamTrace(self.action_type)
        self._replay = _GTReplay()
        if self._replay.active:
            print(f"[FlexPi] FLEXPI_REPLAY_GT -> episodes {self._replay.episodes} from {self._replay.root}")

        if self.allow_dummy_policy:
            # No processor to read the checkpoint dim from: auto means none here,
            # an explicit yam32 still exercises the state-building path.
            self._setup_bridge(ckpt_state_dim=None)
            if self._replay.active:
                served = "dataset GT (FLEXPI_REPLAY_GT)"
            else:
                served = "zeros" if self.action_type == "joint" else "a hold of the current EE pose"
            print(
                "[FlexPi] allow_dummy_policy=true; skipping checkpoint load. "
                f"Debug wiring only — actions are {served}. action_type={self.action_type} "
                f"state_bridge={self.state_bridge}"
            )
            return

        checkpoint_path = self.model_cfg.get("checkpoint_path") or self.model_cfg.get("ckpt_setting")
        if _is_none_like(checkpoint_path):
            raise FileNotFoundError(
                "FlexPi needs checkpoint_path (or ckpt_setting) for a real run. "
                "Set FLEXPI_ALLOW_DUMMY_POLICY=true to exercise the wiring without weights."
            )

        for path in (str(FLEXPI_ROOT), str(FLEXPI_SRC)):
            if path not in sys.path:
                sys.path.insert(0, str(path))

        from experiments.robotwin.flexpi_policy.deploy_policy import get_model

        self.model = get_model(self._build_upstream_cfg(checkpoint_path))
        _cache_prompt_encodes(self.model.model)
        self.replan_steps = int(self.model.replan_steps)
        self.action_horizon = int(self.model.action_horizon)
        self._stream_trace.attach(self.model.model, self.model._num_video_frames, self.action_horizon)
        state_meta = self.model.processor.shape_meta["state"]
        if len(state_meta) == 1 and not isinstance(state_meta[0]["raw_shape"], int):
            # Mixed-embodiment checkpoint: a list of admissible widths, e.g. [32, 34].
            ckpt_state_dim = tuple(int(w) for w in state_meta[0]["raw_shape"])
        else:
            ckpt_state_dim = sum(int(m["raw_shape"]) for m in state_meta)
        self._setup_bridge(ckpt_state_dim=ckpt_state_dim)
        if self.action_type == "ee":
            # A checkpoint trained with its EEF channels padded out (StaticActionDimPad,
            # e.g. RoboDojo ABS_JOINTONLY) never learned that block; ee would execute it.
            trained_pad = set()
            for trans in self.model.processor.action_state_transforms or []:
                trained_pad.update(getattr(trans, "pad_ids", []) if type(trans).__name__ == "StaticActionDimPad" else [])
            if trained_pad & set(range(18)):
                raise ValueError(
                    "action_type=ee executes the EEF block (34D channels 0:18), but this checkpoint "
                    f"was trained with those channels padded out (StaticActionDimPad pad ids "
                    f"{sorted(trained_pad)}). Use action_type=joint."
                )

        # The upstream chunk builder feeds per_cam_depth + intrinsics whenever
        # the loaded model's infer_action accepts them and the pointmap stream
        # is not dropped (infer_present_pointmap: false), and KeyErrors deep in
        # its builders if the observation lacks them. Detect that here so the
        # failure surfaces at this boundary with something actionable.
        self._needs_depth = (
            "per_cam_depth" in inspect.signature(self.model.model.infer_action).parameters
            and self.model.infer_present_pointmap is not False
        )
        if self._needs_depth and not self.use_depth:
            print(
                "[FlexPi] WARNING: the loaded checkpoint's infer_action accepts "
                "per_cam_depth (pointmap stream), but use_depth=false in deploy.yml. "
                "Set use_depth=true and make sure the RoboDojo env_cfg emits depth "
                "and intrinsic_matrix, or set infer_present_pointmap: false and "
                "infer_joint_pointmap: false to run this flex checkpoint without depth.",
                file=sys.stderr,
            )

        print(
            f"[FlexPi] ready: action_type={self.action_type} env_cfg_type={self.env_cfg_type} "
            f"replan_steps={self.replan_steps} action_horizon={self.action_horizon} "
            f"use_depth={self.use_depth} needs_depth={self._needs_depth} "
            f"state_bridge={self.state_bridge} eef_frame={self.eef_frame} "
            f"diffusion_seed={self.model.seed}"
        )

    def _setup_bridge(self, ckpt_state_dim):
        self.state_bridge = resolve_bridge(self._bridge_mode, ckpt_state_dim, self._robot_dim)
        # The arm mount poses are read by arm_base_grasp only, and an env_cfg
        # without sim mount poses (a real robot's) need not carry them.
        if self.state_bridge == "yam32" and self.eef_frame == "arm_base_grasp":
            self._base_tf = load_base_transforms(self.env_cfg_type)
        if self.action_type == "ee" and self.state_bridge != "yam32":
            raise ValueError(
                "action_type=ee executes the checkpoint's EEF block (32D dims 0:18), which "
                f"only a yam32-layout checkpoint has; state_bridge resolved to {self.state_bridge!r}. "
                "Use action_type=joint for this checkpoint"
                + (", or set state_bridge: yam32 (under allow_dummy_policy auto means none)."
                   if ckpt_state_dim is None else ".")
            )
        if self.action_type == "ee" and self._replay.active:
            replay_frame = self._replay.eef_frame()
            if self.eef_frame != replay_frame:
                raise ValueError(
                    f"FLEXPI_REPLAY_GT under action_type=ee needs eef_frame={replay_frame}, the frame "
                    f"of the EEF block in {self._replay.root}: relabeled for a root whose "
                    "meta/info.json carries eef_frame (the relabeled RoboDojo_real corpus), "
                    "world_link6 for any other."
                )

    def _build_upstream_cfg(self, checkpoint_path: Any) -> Dict[str, Any]:
        """Map deploy.yml onto the keys upstream get_model() reads.

        Passed through verbatim where the names already agree; only
        checkpoint_path -> ckpt_setting is renamed. Unset keys are dropped so
        upstream falls back to its own config defaults rather than seeing None.
        """
        cfg = dict(self.model_cfg)
        cfg["ckpt_setting"] = str(Path(str(checkpoint_path)).expanduser().resolve())

        dataset_stats_path = self.model_cfg.get("dataset_stats_path")
        if not _is_none_like(dataset_stats_path):
            cfg["dataset_stats_path"] = str(Path(str(dataset_stats_path)).expanduser().resolve())
        else:
            cfg.pop("dataset_stats_path", None)

        # `seed` in model_cfg is the eval/layout seed the launcher injects. Upstream
        # turns an integer seed into a freshly re-seeded generator on every
        # infer_action call, so identical observations get identical diffusion
        # noise — at a static start pose one "hold still" sample then repeats
        # forever. The real-robot serve passes seed=None (fresh noise per call);
        # do the same unless diffusion_seed pins it for reproducible debugging.
        cfg["seed"] = self.model_cfg.get("diffusion_seed")
        cfg.setdefault("sim_cfg_name", "sim_robotwin.yaml")
        cfg.setdefault("sim_task", "robodojo_unified_flex_3cam_384_1e-4")
        return {k: v for k, v in cfg.items() if not _is_none_like(v) or k == "ckpt_setting"}

    def _encode_obs(self, obs: dict) -> Dict[str, Any]:
        """XPolicyLab observation -> the dict shape upstream deploy reads."""
        vision = obs.get("vision")
        if not vision:
            raise KeyError("Observation has no 'vision' field.")

        observation: Dict[str, Any] = {}
        for cam_name, obs_key in _CAM_TO_OBS_KEY.items():
            cam = vision.get(cam_name)
            if cam is None:
                raise KeyError(
                    f"FlexPi needs all three cameras {sorted(_CAM_TO_OBS_KEY)}; "
                    f"'{cam_name}' is missing from this observation. The vendored "
                    "per-cam pipeline is fixed at 3 cameras."
                )
            entry: Dict[str, Any] = {
                "rgb": _standardize_rgb(cam["color"], self.obs_hw),
            }
            if self.use_depth or self._needs_depth:
                depth_raw, depth_scale = _pick_depth(cam, self.depth_input_scale)
                if depth_raw is None or cam.get("intrinsic_matrix") is None:
                    missing = (
                        "depth (nor approximate_depth)" if depth_raw is None
                        else "intrinsic_matrix"
                    )
                    raise KeyError(
                        f"use_depth is on (or the checkpoint requires the pointmap "
                        f"stream) but vision['{cam_name}'] has no {missing}. "
                        "Enable them under observation.vision in the RoboDojo "
                        "env_cfg: arx_x5.yml ships depth/intrinsic_matrix/"
                        "extrinsic_matrix all false, and approximate_depth false."
                    )
                # Depth and K must stay on one grid: the upstream pointmap
                # encoder unprojects with this K, so depth is passed at its
                # native resolution, unresized, unlike rgb above.
                entry["depth"] = _standardize_depth(depth_raw, depth_scale)
                entry["intrinsic_cv"] = np.asarray(
                    cam["intrinsic_matrix"], dtype=np.float32
                ).copy()
            observation[obs_key] = entry

        # Always the joint layout, whatever is executed: the 32D state needs the
        # joints and grippers, and its EEF block comes from *_ee_pose below.
        # Packing "ee" would read the 7D pose at arm_dim 6 and raise.
        state_vector = pack_robot_state(
            obs,
            "joint",
            self.robot_action_dim_info,
            source_type="obs",
            state_type="state",
        ).astype(np.float32)
        if self.state_bridge == "yam32":
            state_vector = build_yam32_state(
                obs["state"], state_vector, self._base_tf, eef_frame=self.eef_frame,
                env_cfg_type=self.env_cfg_type,
            )

        return {"observation": observation, "joint_action": {"vector": state_vector}}

    def _check_instruction(self, instruction: str) -> None:
        """Warn once per instruction that is not one of this robot's RoboDojo_real training sentences."""
        known = self._training_instructions
        if not known or instruction in known or instruction in self._warned_instructions:
            return
        self._warned_instructions.add(instruction)
        why = (
            "the observation carries no instruction, so this is the fallback"
            if instruction == self.default_instruction
            else "it is not one of the training sentences"
        )
        print(
            f"[FlexPi] WARNING: instruction {instruction!r}: {why}. The {self.env_cfg_type} "
            "checkpoint was trained on exactly these, character for character: "
            + " | ".join(sorted(known)),
            file=sys.stderr,
        )

    def update_obs(self, obs):
        self.last_obs = self._encode_obs(obs)
        self.last_instruction = _get_instruction(obs, self.default_instruction)
        self._check_instruction(self.last_instruction)
        self._action_trace.add_state(
            None, self.last_obs["joint_action"]["vector"], self.last_instruction
        )
        self._replay.add_state(None)
        self._stream_trace.add_obs(None, self.last_obs, self.last_instruction)

    def update_obs_batch(self, obs_list):
        if not obs_list:
            raise ValueError("update_obs_batch received an empty observation list.")
        self._batch_obs = {}
        self._batch_instruction = {}
        for obs in obs_list:
            env_idx = int(obs["env_idx"])
            self._batch_obs[env_idx] = self._encode_obs(obs)
            self._batch_instruction[env_idx] = _get_instruction(obs, self.default_instruction)
            self._check_instruction(self._batch_instruction[env_idx])
            self._action_trace.add_state(
                env_idx,
                self._batch_obs[env_idx]["joint_action"]["vector"],
                self._batch_instruction[env_idx],
            )
            self._replay.add_state(env_idx)
            self._stream_trace.add_obs(
                env_idx, self._batch_obs[env_idx], self._batch_instruction[env_idx]
            )

    def _zero_actions(self):
        zeros = np.zeros((self.replan_steps, self._robot_dim), dtype=np.float32)
        return unpack_robot_state(zeros, self.action_type, self.robot_action_dim_info, source_type="obs")

    def _pose7s(self, row32) -> Dict[str, np.ndarray]:
        """``{side: pose7}`` of a 32D state row: the EE pose as RoboDojo reports it (sim: X5 link6)."""
        return {
            side: eef_block_to_pose7(row32[sl], side, self.eef_frame, self._base_tf, self.env_cfg_type)
            for side, sl in (("left", slice(0, 9)), ("right", slice(9, 18)))
        }

    def _log_bridge_once(self, state32: np.ndarray, action32: np.ndarray, sent: np.ndarray):
        """First-call sanity print for the yam32 bridge; see state_bridge.py."""
        if self._bridge_logged:
            return
        self._bridge_logged = True
        np.set_printoptions(precision=3, suppress=True)
        q01 = q99 = None
        stats_path = self.model_cfg.get("dataset_stats_path")
        if not _is_none_like(stats_path):
            try:
                stats = load_json(str(stats_path))["state"]["default"]
                q01, q99 = np.asarray(stats["global_q01"]), np.asarray(stats["global_q99"])
            except Exception as exc:
                print(f"[FlexPi bridge] could not read state stats for the sanity print: {exc}")
        for side, sl in (("left", slice(0, 3)), ("right", slice(9, 12))):
            ranges = "" if q01 is None else f" (ckpt q01 {q01[sl]} q99 {q99[sl]})"
            print(f"[FlexPi bridge] {side} ee pos ({self.eef_frame}) {state32[sl]}{ranges}")
        print(f"[FlexPi bridge] state grip [18:20] {state32[18:20]}  joints [20:32] {state32[20:32]}")
        print(
            "[FlexPi bridge] action[0] joints - state joints (should be small, not ~ -state): "
            f"{action32[0, 20:32] - state32[20:32]}"
        )
        print(f"[FlexPi bridge] action[0] grip {action32[0, 18:20]}")
        if self.action_type == "ee":
            measured = self._pose7s(state32)
            for side, target in _ee16_poses(sent[0]).items():
                pos_mm, rot_deg = pose7_error(target, measured[side])
                print(
                    f"[FlexPi bridge] action[0] {side} EE target - measured link6: {pos_mm:.1f} mm "
                    f"{rot_deg:.2f} deg (should be mm, not cm)"
                )

    def _log_ee_chunk(self, prefix: str, state32: np.ndarray, sent: np.ndarray, env_idx=None):
        """ee half of ``_log_action_chunk``: what the EE targets ask for and how well the last chunk landed.

        ``tracking`` is the EE pose measured now against the previous chunk's
        last target. RoboDojo holds an arm whose IK failed at its previous
        command without saying so, which shows up here as a tracking error of
        about the distance that command was short of the target; the exact count
        is in ``_timing.jsonl`` (deploy.py wraps the IK solve).
        """
        measured = self._pose7s(state32)
        prev = self._prev_sent_pose7.get(env_idx)
        for i, side in enumerate(("left", "right")):
            targets = [_ee16_poses(row)[side] for row in sent]
            step0 = pose7_error(targets[0], measured[side])
            reach = max(pose7_error(t, measured[side])[0] for t in targets)
            line = (
                f"{prefix} EE {side}: step0 target - measured {step0[0]:.1f} mm {step0[1]:.2f} deg, "
                f"chunk reach {reach:.1f} mm"
            )
            if prev is not None:
                trk = pose7_error(measured[side], prev[side])
                line += f", tracking (now - prev chunk last target) {trk[0]:.1f} mm {trk[1]:.2f} deg"
            print(line)

    def _log_action_chunk(self, state32: np.ndarray, action32: np.ndarray, sent: np.ndarray, env_idx=None):
        """Per-inference chunk summary, enabled by deploy.yml log_action_chunk.

        Joint deltas are absolute targets minus the anchor (current state), so
        they show how far the whole chunk asks the arms to move. ``tracking``
        compares the previous chunk's last executed target with the state seen
        now: small deltas plus small tracking error means the policy itself is
        asking for little motion; large tracking error means targets are not
        being reached. Under ``ee`` the joint head is not executed, so joint
        tracking would compare the arm with targets nobody sent; the EE lines
        (``_log_ee_chunk``) replace it.
        """
        self._chunk_calls[env_idx] = self._chunk_calls.get(env_idx, 0) + 1
        np.set_printoptions(precision=3, suppress=True, linewidth=200)
        state_j = np.asarray(state32[20:32], dtype=np.float32)
        delta = action32[:, 20:32] - state_j[None, :]
        prefix = f"[FlexPi chunk env{env_idx} #{self._chunk_calls[env_idx]}]"
        head = "" if self.action_type == "joint" else "  (joint head, not executed)"
        print(
            f"{prefix} T={action32.shape[0]} max|delta| {np.abs(delta).max():.4f} rad  "
            f"per-joint max|delta| {np.abs(delta).max(axis=0)}{head}"
        )
        print(f"{prefix} delta step0 {delta[0]}")
        print(f"{prefix} delta step{action32.shape[0] - 1} {delta[-1]}")
        print(
            f"{prefix} grip L [{action32[:, 18].min():.3f}, {action32[:, 18].max():.3f}] "
            f"R [{action32[:, 19].min():.3f}, {action32[:, 19].max():.3f}]  state grip {state32[18:20]}"
        )
        print(f"{prefix} joints now {state_j}")
        if self.action_type == "ee":
            self._log_ee_chunk(prefix, state32, sent, env_idx=env_idx)
            return
        # Under hold the chunk's targets were never sent, so there is nothing to track.
        prev = None if self.hold else self._prev_chunk_last_target.get(env_idx)
        if prev is not None:
            tracking = state_j - prev
            print(
                f"{prefix} tracking (state now - prev chunk last target) "
                f"max {np.abs(tracking).max():.4f} rad {tracking}"
            )
            # A joint pinned by a limit, a self-collision or contact reads as
            # "commanded hard, did not move". Joint limits live in the USD asset
            # on the simulator side and never reach this process, so infer the
            # saturation instead of comparing against a bound we do not have.
            moved = np.abs(state_j - self._prev_chunk_state.get(env_idx, state_j))
            asked = np.abs(prev - self._prev_chunk_state.get(env_idx, state_j))
            stuck = np.where((asked > 0.2) & (moved < 0.05 * asked))[0]
            if stuck.size:
                print(
                    f"{prefix} STUCK joints {stuck.tolist()}: asked "
                    f"{np.round(asked[stuck], 3).tolist()} rad, moved "
                    f"{np.round(moved[stuck], 3).tolist()} rad"
                )
        self._prev_chunk_last_target[env_idx] = action32[-1, 20:32].astype(np.float32).copy()
        self._prev_chunk_state[env_idx] = state_j.copy()

    def _emit(self, env_idx, anchor, model_chunk, from_model: bool = False, hold_row=None):
        """Model-layout chunk -> the simulator's per-step action dicts, logged and traced.

        The one place ``action_type`` picks which half of the chunk runs: the
        joint half (``yam32_to_rd14``) or the EEF half as EE pose targets
        (``yam32_to_rd_ee16``). Model output, GT replay and the dummy policy all
        come through here, so each exercises exactly the path a policy chunk takes.
        ``hold_row`` (a model-layout state row) is sent at every step instead of
        the chunk, which is still logged as what the model asked for.
        """
        model_chunk = np.asarray(model_chunk, dtype=np.float32)
        executed = model_chunk if hold_row is None else np.repeat(hold_row[None], len(model_chunk), axis=0)
        joint_head = yam32_to_rd14(executed) if self.state_bridge == "yam32" else executed
        if self.action_type == "ee":
            sent, held = yam32_to_rd_ee16(
                executed, self.eef_frame, self._base_tf, hold_pose7=self._pose7s(anchor),
                env_cfg_type=self.env_cfg_type,
            )
            if held:
                self._ee_held += held
                print(
                    f"[FlexPi] WARNING env{env_idx}: {held} EE target(s) had no valid rotation; "
                    f"sent a hold of the current pose instead ({self._ee_held} so far)",
                    file=sys.stderr,
                )
            actions = unpack_robot_state(sent, "ee", self._ee_action_dim_info, source_type="obs")
        else:
            sent = joint_head
            actions = unpack_robot_state(sent, "joint", self.robot_action_dim_info, source_type="obs")
        # Joint mode logs model chunks only, as it always has; ee logs every
        # chunk, so a GT replay or a dummy hold shows its EE tracking too.
        if self.state_bridge == "yam32" and (from_model or self.action_type == "ee"):
            self._log_bridge_once(anchor, model_chunk, sent)
            if self.log_action_chunk:
                self._log_action_chunk(anchor, model_chunk, sent, env_idx=env_idx)
        if self.action_type == "ee":
            self._prev_sent_pose7[env_idx] = {
                side: pose.copy() for side, pose in _ee16_poses(sent[-1]).items()
            }
        self._action_trace.add_chunk(env_idx, anchor, model_chunk, sent, joint_head)
        if from_model:
            self._stream_trace.add_call(env_idx, model_chunk, sent, joint_head)
        return actions

    def _infer_actions(self, obs, instruction, env_idx=None):
        if self.allow_dummy_policy and not self._replay.active and self.action_type == "joint":
            return self._zero_actions()
        if obs is None:
            raise ValueError("No observation available; call update_obs() before get_action().")
        anchor = obs["joint_action"]["vector"]
        if self._replay.active:
            if self.state_bridge != "yam32":
                raise ValueError("FLEXPI_REPLAY_GT needs state_bridge=yam32 (GT actions are 34D -> 32D).")
            return self._emit(env_idx, anchor, self._replay.chunk(env_idx, self.replan_steps))
        if self.allow_dummy_policy:
            # ee: hold the current EE pose and gripper. A zero pose would be an
            # unreachable IK target, and a hold doubles as a frame round trip:
            # the arms stay put only if RoboDojo reads the target in the frame
            # the observation reports it in.
            hold = np.repeat(np.asarray(anchor, dtype=np.float32)[None], self.replan_steps, axis=0)
            return self._emit(env_idx, anchor, hold)

        _dump_observation(obs, instruction)

        action_chunk = np.asarray(
            self.model._infer_action_chunk(obs, instruction), dtype=np.float32
        )
        if action_chunk.ndim == 1:
            action_chunk = action_chunk[None, :]
        action_chunk = action_chunk[: min(self.replan_steps, action_chunk.shape[0])]
        hold_row = None
        if self.hold:
            # Latched rather than re-read, so a sag under gravity is never
            # commanded back as the next target.
            hold_row = self._hold_rows.setdefault(env_idx, np.asarray(anchor, dtype=np.float32).copy())
        return self._emit(env_idx, anchor, action_chunk, from_model=True, hold_row=hold_row)

    def get_action(self):
        return self._infer_actions(self.last_obs, self.last_instruction)

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is None:
            env_idx_list = sorted(self._batch_obs)
        if not self._batch_obs:
            raise ValueError("No batch observation available; call update_obs_batch() first.")
        return [
            self._infer_actions(
                self._batch_obs[int(env_idx)],
                self._batch_instruction.get(int(env_idx), self.default_instruction),
                env_idx=int(env_idx),
            )
            for env_idx in env_idx_list
        ]

    def reset(self):
        self.last_obs = None
        self.last_instruction = self.default_instruction
        self._batch_obs = {}
        self._batch_instruction = {}
        # A new episode starts from a reset pose: the previous chunk's target
        # says nothing about tracking there.
        self._prev_chunk_last_target = {}
        self._prev_chunk_state = {}
        self._prev_sent_pose7 = {}
        self._chunk_calls = {}
        self._hold_rows = {}
        self._action_trace.new_episode()
        self._stream_trace.new_episode()
        self._replay.new_episode()
        if self.model is not None:
            self.model.reset()

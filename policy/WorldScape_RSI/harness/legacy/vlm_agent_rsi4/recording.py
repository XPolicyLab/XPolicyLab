"""Capture real control-step observations through RoboDojo's native video hook."""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path


_MISSING = object()


def control_period(env, fallback_hz=25.0):
    manager = getattr(env, "obs_manager", None)
    dt = float(getattr(manager, "dt", 0) or 0)
    interval = float(getattr(manager, "collect_interval", 0) or 0)
    sim = getattr(env, "sim", None)
    decimation = float(getattr(getattr(sim, "cfg", None), "decimation", 1) or 1)
    period = dt * interval * decimation if dt > 0 and interval > 0 else 1.0 / float(
        getattr(manager, "collect_freq", fallback_hz) or fallback_hz
    )
    if not math.isfinite(period) or period <= 0:
        raise ValueError("recording requires a positive simulation control period")
    return period


class EpisodeRecording:
    def __init__(self, env, config):
        self.env = env
        self.control_dt = control_period(env, (config.get("controller") or {}).get("control_hz", 25.0))
        self.start_steps = list(env.take_action_cnt)
        self.seen = set()
        self.cache = {}
        self.frames = {}
        self.first_steps = {}
        self.last_steps = {}
        self.duplicates = 0
        self.errors = []
        self.index = None
        self.finished = False
        self.video_failed = False
        self.reason = None
        self.originals = {}
        self.instance_values = {}
        episode = int(getattr(env, "success_nums", 0) + getattr(env, "fail_nums", 0))
        directory = Path(getattr(env, "save_dir", None)
                         or config.get("log_dir", "eval_result/_vlm_logs")).expanduser().resolve()
        self.index_path = directory / f"episode_{episode:07d}_frames.jsonl"
        self.metadata_path = directory / f"episode_{episode:07d}_recording.json"

    def error(self, phase, exc):
        message = f"{phase}: {type(exc).__name__}: {exc}"
        if message not in self.errors:
            self.errors.append(message)
            print(f"[vlm_agent] recording {message}", flush=True)

    def __enter__(self):
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.index = self.index_path.open("x", encoding="utf-8")
        except OSError as exc:
            self.error("index", exc)
        for name in ("_stream_vision", "get_obs", "take_action"):
            self.originals[name] = getattr(self.env, name, None)
            self.instance_values[name] = vars(self.env).get(name, _MISSING)
        if self.originals["_stream_vision"] is None:
            self.error("native_hook", RuntimeError("environment has no _stream_vision hook"))
        else:
            self.env._stream_vision = self.stream
        self.env.get_obs = self.get_obs
        self.env.take_action = self.take_action
        # Normal RoboDojo reset leaves no writers. Unknown pre-existing frames
        # cannot be assigned invented timestamps or silently recorded twice.
        for env_idx, writers in getattr(self.env, "video_writers", {}).items():
            if any(getattr(writer, "n_frames", 0) for writer in writers.values()):
                self.seen.add((int(env_idx), int(self.env.take_action_cnt[env_idx])))
                self.error("initial_frame", RuntimeError("video already contains unindexed frames"))
        try:
            self.get_obs()
        except Exception as exc:  # noqa: BLE001 - metadata must expose capture failure
            self.error("initial_capture", exc)
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            self.finish("exception" if exc_type else self.reason or "complete")
        finally:
            for name, value in self.instance_values.items():
                if value is _MISSING:
                    if name in vars(self.env):
                        delattr(self.env, name)
                else:
                    setattr(self.env, name, value)
        return False

    def _writer_counts(self, env_idx):
        return {name: int(writer.n_frames) for name, writer in
                getattr(self.env, "video_writers", {}).get(env_idx, {}).items()}

    def stream(self, env_idx, frame):
        step = int(self.env.take_action_cnt[env_idx])
        key = (int(env_idx), step)
        if key in self.seen:
            self.duplicates += 1
            return
        cached = deepcopy(frame)
        cached["env_idx"] = env_idx
        self.cache[env_idx] = (step, cached)
        if self.video_failed or self.originals["_stream_vision"] is None:
            return
        try:
            before = self._writer_counts(env_idx)
            self.originals["_stream_vision"](env_idx, frame)
            after = self._writer_counts(env_idx)
            cameras = sorted(name for name in after if after[name] > before.get(name, 0))
            if not cameras:
                raise RuntimeError("native writer appended no camera frame")
            self.seen.add(key)
            label = str(env_idx)
            self.frames[label] = self.frames.get(label, 0) + 1
            self.first_steps.setdefault(label, step)
            self.last_steps[label] = step
            entry = {"env_idx": env_idx, "frame_index": self.frames[label] - 1,
                     "control_step": step, "simulated_time_s": round(step * self.control_dt, 9),
                     "cameras": cameras, "camera_frame_indices": {name: after[name] - 1 for name in cameras}}
            if self.index is not None:
                self.index.write(json.dumps(entry, allow_nan=False) + "\n")
                self.index.flush()
        except Exception as exc:  # noqa: BLE001 - preserve policy execution and flag invalid video
            self.video_failed = True
            self.error("frame", exc)

    def get_obs(self):
        step = int(self.env.take_action_cnt[0])
        cached = self.cache.get(0)
        if cached is not None and cached[0] == step:
            return deepcopy(cached[1])
        obs = self.originals["get_obs"]()
        # get_obs skips native streaming after an episode ends; explicitly
        # retain its real final image, deduplicating any native last_frame call.
        if (0, step) not in self.seen:
            self.stream(0, obs)
        self.cache[0] = (step, deepcopy(obs))
        return deepcopy(obs)

    def take_action(self, action):
        before = int(self.env.take_action_cnt[0])
        try:
            return self.originals["take_action"](action)
        finally:
            if int(self.env.take_action_cnt[0]) > before:
                try:
                    self.get_obs()
                except Exception as exc:  # noqa: BLE001
                    self.error("control_capture", exc)

    def metadata(self):
        missing = {}
        for env_idx, start in enumerate(self.start_steps):
            missing[str(env_idx)] = [step for step in range(int(start), int(self.env.take_action_cnt[env_idx]) + 1)
                                     if (env_idx, step) not in self.seen]
        return {"control_dt_s": self.control_dt,
                "frames_by_env": dict(self.frames), "first_step_by_env": dict(self.first_steps),
                "last_step_by_env": dict(self.last_steps),
                "missing_control_steps": sum(map(len, missing.values())),
                "missing_steps_by_env": missing, "duplicate_frames_skipped": self.duplicates,
                "errors": list(self.errors), "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path), "termination_reason": self.reason}

    def finish(self, reason):
        if self.finished:
            return
        self.reason = reason
        try:
            self.get_obs()
        except Exception as exc:  # noqa: BLE001
            self.error("final_capture", exc)
        if self.index is not None:
            try:
                self.index.close()
            except OSError as exc:
                self.error("index_close", exc)
        try:
            self.metadata_path.write_text(json.dumps(self.metadata(), indent=2, allow_nan=False), encoding="utf-8")
        except OSError as exc:
            self.error("metadata", exc)
        self.finished = True

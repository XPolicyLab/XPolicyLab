"""Opt-in lossless capture; default inference does not import this module."""
from collections import deque
from functools import wraps
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def encode(value, arrays=None):
    torch = sys.modules.get("torch")
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu().contiguous()
        if value.dtype == torch.bfloat16:
            return {"bfloat16_bits": encode(value.view(torch.uint16).numpy(), arrays)}
        value = value.numpy()
    if isinstance(value, np.ndarray):
        value = np.array(value, copy=True, order="C")
        if value.dtype.hasobject:
            raise TypeError("Object arrays are not valid replay evidence")
        header = json.dumps([value.dtype.str, list(value.shape)]).encode()
        digest = hashlib.sha256(header + value.tobytes()).hexdigest()
        if arrays is not None:
            arrays = Path(arrays)
            arrays.mkdir(parents=True, exist_ok=True)
            target = arrays / (digest + ".npy")
            if not target.exists():
                with target.open("xb") as stream:
                    np.save(stream, value, allow_pickle=False)
        return {"array": digest, "dtype": value.dtype.str, "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return encode(value.item(), arrays)
    if isinstance(value, dict):
        return {str(k): encode(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple, deque)):
        return [encode(v, arrays) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported evidence type: {type(value).__name__}")


def decode(value, arrays):
    if isinstance(value, dict) and "array" in value:
        digest = value["array"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid array identifier")
        result = np.load(Path(arrays) / (digest + ".npy"), allow_pickle=False)
        if encode(result) != value:
            raise ValueError("Replay array integrity check failed")
        return result
    if isinstance(value, dict):
        return {k: decode(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [decode(v, arrays) for v in value]
    return value


def records(directory):
    with (Path(directory) / "trace.jsonl").open() as stream:
        return [json.loads(line) for line in stream]


def history_state(model):
    # Normalize virtual IDs: routing bookkeeping is not a numerical input.
    native = model.model
    return {name: [state[k] for k in sorted(state)]
            for name in native._PER_ENV_STATE
            for state in [getattr(native, name)]}


class Recorder:
    def __init__(self, model, root, limit=209):
        self.model, self.root, self.limit = model, Path(root), limit
        self.root.mkdir(parents=True, exist_ok=False)
        self.episode = -1
        self.stream = None
        self.observations = self.replans = 0
        self.directory = None

    @property
    def active(self):
        return self.stream is not None and self.observations <= self.limit

    def event(self, kind, data, *, save=False):
        if self.stream is not None:
            row = {"kind": kind, "data": encode(data, self.directory / "arrays" if save else None)}
            self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.stream.flush()

    def start(self):
        if self.stream is not None:
            self.stream.close()
        self.episode += 1
        self.directory = self.root / f"episode-{self.episode:03d}"
        self.directory.mkdir()
        self.stream = (self.directory / "trace.jsonl").open("x")
        self.observations = self.replans = 0

    def install(self):
        model = self.model
        original_reset = model.reset
        original_client_reset = model.reset_client
        original_update = model.update_obs
        original_action = model.get_action
        original_infer = model.model.model.infer_action

        @wraps(original_reset)
        def reset():
            result = original_reset()
            self.start()
            return result

        @wraps(original_client_reset)
        def reset_client(obs=None):
            result = original_client_reset(obs=obs)
            if self.stream is None:
                self.start()
            sizes = {name: len(getattr(model.model, name)) for name in model.model._PER_ENV_STATE}
            adapter_sizes = {name: len(getattr(model, name)) for name in (
                "_last_infer_step", "_infer_count_by_virtual", "_ensemble_by_virtual",
                "_last_obs_by_client", "_last_instruction_by_client", "_last_virtual_by_client")}
            if any(sizes.values()) or any(adapter_sizes.values()) or model.last_obs is not None:
                raise RuntimeError("Episode reset leaked numerical state")
            self.event("reset", {"native": sizes, "adapter": adapter_sizes})
            return result

        @wraps(original_update)
        def update_obs(obs):
            if self.stream is None:
                self.start()
            result = original_update(obs)
            if self.observations < self.limit:
                numerical = {k: v for k, v in obs.items() if k != model.CLIENT_ID_FIELD}
                self.event("observation", {"step": self.observations, "observation": numerical}, save=True)
            self.observations += 1
            return result

        @wraps(original_infer)
        def infer_action(**kwargs):
            if self.active:
                self.event("inputs", kwargs)
                self.event("history", {"state": history_state(model),
                           "mature": any(s >= 175 for s in model.model._history_step_by_env.values())})
            result = original_infer(**kwargs)
            if self.active:
                self.event("prediction", result["action"], save=True)
            return result

        @wraps(original_action)
        def get_action(obs=None):
            result = original_action(obs=obs)
            if model.history_guard_mismatches:
                raise RuntimeError("Observation history cadence mismatch")
            if self.active:
                self.event("ensemble", model._last_ensemble, save=True)
                self.event("commands", {"replan": self.replans,
                           "observations": self.observations, "actions": result}, save=True)
                (self.directory / "coverage.json").write_text(json.dumps({
                    "observations": self.observations, "action_chunks": self.replans + 1,
                    "history_mature": any(s >= 175 for s in model.model._history_step_by_env.values()),
                    "capture_limit": self.limit}, indent=2) + "\n")
            self.replans += 1
            return result

        model.reset = reset
        model.reset_client = reset_client
        model.update_obs = update_obs
        model.get_action = get_action
        model.model.model.infer_action = infer_action
        return self


def attach(model, root):
    return Recorder(model, root).install()

"""Synthetic and HDF5 replay checks over the policy websocket interface."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

from client_server.ws import WsModelClient
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit

_MMABC = Path(os.environ.get("MMABC_ROOT", Path(__file__).resolve().parent / "MM_ABC"))
if str(_MMABC) not in sys.path:
    sys.path.insert(0, str(_MMABC))

from mmabc.embodiments import mobile  # noqa: E402

def _str2bool(v) -> bool:
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _encode_camera(rgb: np.ndarray, raw: bytes | None, mode: str):
    if mode == "array":
        return rgb
    if mode == "jpeg_rgb":
        return encode_image_bit(rgb)
    if mode == "raw_hdf5":
        return raw
    raise ValueError(mode)

class SyntheticSource:
    def __init__(self, seed: int, image_hw=(720, 1280)):
        self.rng = np.random.default_rng(seed)
        self.hw = image_hw
        self.instruction = "Navigate to the table, pick up the kettle, carry it to the TV stand."

    def __len__(self):
        return 10**9

    def obs(self, t: int, env_idx: int, send: str) -> dict:
        vision = {}
        for cam in mobile.CAMERAS.values():
            rgb = self.rng.integers(0, 256, size=(*self.hw, 3), dtype=np.uint8)
            vision[cam] = {"color": _encode_camera(rgb, None, "array" if send == "raw_hdf5" else send)}
        state = {}
        for key, dim, kind in mobile.KEYS:
            value = self.rng.normal(0.0, 0.05, dim)
            if kind == "pose":
                value[3:] = [1.0, 0.0, 0.0, 0.0]
            state[mobile.singular(key)] = value.astype(np.float32)
        return {"vision": vision, "state": state, "instruction": self.instruction, "env_idx": env_idx}

class ReplaySource:
    def __init__(self, path: str):
        import h5py
        self.f = h5py.File(path, "r")
        self.n = self.f["action/left_arm_joint_states"].shape[0]
        raw = self.f["instruction"][()]
        self.instruction = raw.decode() if isinstance(raw, bytes) else str(raw)
        self.state = {k: self.f["state"][k][:] for k in self.f["state"]}
        self.action = {k: self.f["action"][k][:] for k in self.f["action"]}

    def __len__(self):
        return self.n

    def obs(self, t: int, env_idx: int, send: str) -> dict:
        vision = {}
        for cam in mobile.CAMERAS.values():
            raw = bytes(self.f[f"vision/{cam}/colors"][t])
            rgb = decode_image_bit(raw)
            vision[cam] = {"color": _encode_camera(rgb, raw, send)}
        state = {mobile.singular(k): v[t].astype(np.float32) for k, v in self.state.items()}
        return {"vision": vision, "state": state, "instruction": self.instruction, "env_idx": env_idx}

def validate_action(action: dict, key_style: str) -> None:
    if not isinstance(action, dict):
        raise TypeError(f"action must be a dict, got {type(action)}")
    expected = {(mobile.singular(k) if key_style == "singular" else k): (d, kind) for k, d, kind in mobile.KEYS}
    extra = set(action) - set(expected)
    if extra:
        raise KeyError(f"unexpected action keys {sorted(extra)}")
    for key, (dim, kind) in expected.items():
        if key not in action:
            raise KeyError(f"action missing key {key!r}")
        arr = np.asarray(action[key])
        if arr.shape != (dim,):
            raise ValueError(f"action[{key!r}] expected shape ({dim},), got {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"action[{key!r}] is not finite")
        if kind == "pose" and abs(np.linalg.norm(arr[3:]) - 1.0) > 1e-3:
            raise ValueError(f"action[{key!r}] quaternion is not unit: {arr[3:]}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--hdf5", nargs="*", default=None, help="replay these Mobile episodes")
    ap.add_argument("--start", type=int, default=0, help="first frame of each replay")
    ap.add_argument("--eval_episode_num", type=int, default=2)
    ap.add_argument("--episode_step_limit", type=int, default=48)
    ap.add_argument("--eval_batch", type=_str2bool, default=False)
    ap.add_argument("--batch_size", type=int, default=3)
    ap.add_argument("--send", default=None, choices=["array", "jpeg_rgb", "raw_hdf5"])
    ap.add_argument("--action_key_style", default="singular")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--request_timeout_s", type=float, default=600.0)
    args = ap.parse_args()
    send = args.send or ("jpeg_rgb" if _str2bool(os.environ.get("DEBUG_OBS_ENCODED", "0")) else "array")

    client = WsModelClient(
        url=f"ws://{args.host}:{args.port}",
        evaluation_id="mmabc-debug",
        trial_id="mmabc-debug",
        request_timeout_s=args.request_timeout_s,
    )
    sources = [ReplaySource(p) for p in args.hdf5] if args.hdf5 else [
        SyntheticSource(args.seed + i) for i in range(args.eval_episode_num)
    ]
    errors: dict[str, list[float]] = {}
    hold: dict[str, list[float]] = {}
    latencies = []
    try:
        for ep, src in enumerate(sources):
            print(f"\033[94m[episode {ep}] {type(src).__name__} send={send}\033[0m", flush=True)
            client.call(func_name="reset")
            t = args.start if args.hdf5 else 0
            end = min(len(src) - 1, t + args.episode_step_limit)
            while t < end:
                n_env = args.batch_size if args.eval_batch else 1
                t0 = time.time()
                if args.eval_batch:
                    client.call(func_name="update_obs_batch", obs=[src.obs(t, i, send) for i in range(n_env)])
                    chunks = client.call(func_name="get_action_batch", obs=list(range(n_env)))
                else:
                    client.call(func_name="update_obs", obs=src.obs(t, 0, send))
                    chunks = [client.call(func_name="get_action")]
                latencies.append(time.time() - t0)
                for chunk in chunks:
                    assert isinstance(chunk, list) and chunk, "get_action must return a non-empty list"
                    for action in chunk:
                        validate_action(action, args.action_key_style)
                k = len(chunks[0])
                if isinstance(src, ReplaySource):
                    for step, action in enumerate(chunks[0][: end - t]):
                        for key, rec in src.action.items():
                            name = mobile.singular(key) if args.action_key_style == "singular" else key
                            pred = np.asarray(action[name], dtype=np.float64)
                            ref, now = rec[t + step], src.state[key][t]
                            if key.endswith("poses"):
                                sign = np.sign(np.dot(pred[3:], ref[3:])) or 1.0
                                pred = np.concatenate([pred[:3], pred[3:] * sign])
                                now = np.concatenate([now[:3], now[3:] * (np.sign(np.dot(now[3:], ref[3:])) or 1.0)])
                            errors.setdefault(key, []).append(float(np.abs(pred - ref).mean()))
                            hold.setdefault(key, []).append(float(np.abs(now - ref).mean()))
                print(f"  t={t:5d} got {len(chunks)} chunk(s) x {k} steps, {latencies[-1] * 1000:.0f} ms", flush=True)
                t += k
        print(f"\nmedian round trip {np.median(latencies) * 1000:.0f} ms over {len(latencies)} calls")
        if errors:
            print(f"\n{'key':28s} {'model MAE':>10s} {'hold MAE':>10s}")
            for key in errors:
                print(f"{key:28s} {np.mean(errors[key]):10.4f} {np.mean(hold[key]):10.4f}")
            m, h = np.mean([np.mean(v) for v in errors.values()]), np.mean([np.mean(v) for v in hold.values()])
            print(f"{'mean':28s} {m:10.4f} {h:10.4f}")
        print("\n\033[92mMM_ABC CLIENT PASSED: every action chunk matched the Mobile contract.\033[0m")
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()

if __name__ == "__main__":
    main()

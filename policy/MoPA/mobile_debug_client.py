"""Exercise mobile observations and the extended 90-D action packet over WebSocket."""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from client_server.ws import WsModelClient
from XPolicyLab.utils.process_data import encode_image_bit

from mopa.data import mobile


CAMERAS = ("cam_left_wrist", "cam_right_wrist", "head_infra1")
STATE_KEYS = (*mobile.EXTENDED_ACTION_KEYS, "root_poses")
STATE_DIMS = {**mobile.EXTENDED_ACTION_DIMS, "root_poses": 7}


def _singular(key: str) -> str:
    return key[:-1] if key.endswith("states") or key.endswith("poses") else key


def _state(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    output = {}
    for key in STATE_KEYS:
        value = np.zeros(STATE_DIMS[key], dtype=np.float32)
        if key.endswith("poses"):
            value[3] = 1.0
        else:
            value[:] = rng.normal(0.0, 0.01, value.shape)
        output[_singular(key)] = value
    return output


class SyntheticMobile:
    def __init__(self, seed: int, *, encoded: bool = False, batch_size: int = 3):
        self.rng = np.random.default_rng(seed)
        self.encoded = encoded
        self.batch_size = batch_size
        self.t = 0

    def obs(self, env_idx: int = 0) -> dict:
        vision = {}
        for camera in CAMERAS:
            image = self.rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)
            vision[camera] = {"color": encode_image_bit(image) if self.encoded else image}
        return {
            "vision": vision,
            "state": _state(self.t + env_idx * 1000),
            "instruction": "Move the object to the target.",
            "data_format_version": "v1.0",
            "additional_info": {"frequency": 30},
            "env_idx": env_idx,
        }


def _validate_action(action: dict, key_style: str) -> None:
    if not isinstance(action, dict):
        raise TypeError(f"Expected action dict, got {type(action).__name__}")
    for key, dim in mobile.EXTENDED_ACTION_DIMS.items():
        name = _singular(key) if key_style == "singular" else key
        if name not in action:
            raise KeyError(f"Missing extended mobile action key {name!r}")
        value = np.asarray(action[name])
        if value.shape != (dim,) or not np.isfinite(value).all():
            raise ValueError(f"Action {name!r}: expected finite shape ({dim},), got {value.shape}")


def _run_single(client: WsModelClient, source: SyntheticMobile, limit: int, key_style: str) -> int:
    t = 0
    while t < limit:
        client.call(func_name="update_obs", obs=source.obs(0))
        chunks = client.call(func_name="get_action")
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("get_action returned an empty action chunk")
        for action in chunks:
            _validate_action(action, key_style)
            t += 1
            if t >= limit:
                break
    return t


def _run_batch(client: WsModelClient, source: SyntheticMobile, limit: int, key_style: str) -> int:
    envs = list(range(source.batch_size))
    t = 0
    while t < limit:
        client.call(func_name="update_obs_batch", obs=[source.obs(i) for i in envs])
        chunks = client.call(func_name="get_action_batch", obs=envs)
        if len(chunks) != len(envs) or any(not chunk for chunk in chunks):
            raise ValueError("get_action_batch returned an invalid batch")
        for step in range(min(len(chunk) for chunk in chunks)):
            for chunk in chunks:
                _validate_action(chunk[step], key_style)
            t += 1
            if t >= limit:
                break
    return t


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--eval-batch", action="store_true")
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--episode-step-limit", type=int, default=32)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoded", action="store_true", help="send marked RGB JPEG buffers")
    parser.add_argument("--action-key-style", choices=("singular", "plural"), default="singular")
    args = parser.parse_args()

    client = WsModelClient(
        url=f"ws://{args.host}:{args.port}",
        evaluation_id="mopa-extended-debug",
        trial_id="mopa-extended-debug",
        request_timeout_s=600.0,
    )
    try:
        for episode in range(args.episodes):
            client.call(func_name="reset")
            source = SyntheticMobile(args.seed + episode, encoded=args.encoded,
                                          batch_size=args.batch_size)
            started = time.perf_counter()
            if args.eval_batch:
                steps = _run_batch(client, source, args.episode_step_limit, args.action_key_style)
            else:
                steps = _run_single(client, source, args.episode_step_limit, args.action_key_style)
            elapsed = time.perf_counter() - started
            print(f"episode={episode} steps={steps} elapsed={elapsed:.2f}s", flush=True)
        print("MOPA EXTENDED PROTOCOL PASSED", flush=True)
    finally:
        client.close()


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

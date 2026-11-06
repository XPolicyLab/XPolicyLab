#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
SRC_ROOT = REPO_ROOT / "src"

for candidate in [THIS_DIR, SRC_ROOT, REPO_ROOT]:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:
    from .remote_client import RealLift2RemoteClient
    from .websocket_client import WebsocketClientPolicy
except ImportError:
    from remote_client import RealLift2RemoteClient
    from websocket_client import WebsocketClientPolicy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test connectivity between the robot-side client and a remote WSABase websocket server."
    )
    parser.add_argument("--ws_url", default="ws://127.0.0.1:8000", help="Remote WSABase websocket URL.")
    parser.add_argument(
        "--prompt",
        default="Clear the junk and items off the desktop.",
        help="Prompt used for the optional smoke inference request.",
    )
    parser.add_argument("--state_dim", type=int, default=14, help="State dimension for the smoke test request.")
    parser.add_argument(
        "--image_history_interval",
        type=int,
        default=15,
        help="History interval used by the remote client when --smoke_infer is enabled.",
    )
    parser.add_argument("--image_height", type=int, default=480, help="Dummy image height for smoke inference.")
    parser.add_argument("--image_width", type=int, default=640, help="Dummy image width for smoke inference.")
    parser.add_argument(
        "--smoke_infer",
        action="store_true",
        help="In addition to reading server metadata, send a dummy all-zero observation and check the returned action chunk.",
    )
    parser.add_argument(
        "--expect_action_dim",
        type=int,
        default=14,
        help="Expected action dimension for the optional smoke inference response.",
    )
    return parser.parse_args()


def print_metadata(ws_url: str) -> dict:
    client = WebsocketClientPolicy(host=ws_url)
    try:
        metadata = client.get_server_metadata()
    finally:
        client.close()

    print("[Connect] Connected to remote WSABase server.")
    print("[Metadata]")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


def run_smoke_infer(args) -> None:
    client = RealLift2RemoteClient(
        host=args.ws_url,
        prompt=args.prompt,
        image_history_interval=args.image_history_interval,
        state_dim=args.state_dim,
    )
    try:
        client.reset()
        blank = np.zeros((args.image_height, args.image_width, 3), dtype=np.uint8)
        qpos = np.zeros((args.state_dim,), dtype=np.float32)
        response = client.infer_step(
            images={
                "head": blank,
                "left_wrist": blank,
                "right_wrist": blank,
            },
            qpos=qpos,
            timestep=0,
            prompt=args.prompt,
        )
    finally:
        client.close()

    if not isinstance(response, dict):
        raise RuntimeError(f"Unexpected smoke inference response type: {type(response)!r}")
    if "actions" not in response:
        raise RuntimeError("Smoke inference response is missing `actions`.")

    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.ndim != 2:
        raise RuntimeError(f"Expected actions with shape [T, D], got {actions.shape}")
    if actions.shape[1] != args.expect_action_dim:
        raise RuntimeError(
            f"Expected action dim {args.expect_action_dim}, got {actions.shape[1]} with shape {actions.shape}"
        )

    print("[SmokeInfer] Success.")
    print(f"[SmokeInfer] actions.shape = {tuple(actions.shape)}")

    server_timing = response.get("server_timing")
    if server_timing is not None:
        print(f"[SmokeInfer] server_timing = {json.dumps(server_timing, ensure_ascii=False)}")
    client_timing = response.get("client_timing")
    if client_timing is not None:
        print(f"[SmokeInfer] client_timing = {json.dumps(client_timing, ensure_ascii=False)}")


def main():
    args = parse_args()
    print(f"[Connect] Testing remote WSABase server: {args.ws_url}")
    print_metadata(args.ws_url)

    if args.smoke_infer:
        run_smoke_infer(args)
    else:
        print("[Connect] Metadata check passed. Use --smoke_infer for an end-to-end dummy request test.")


if __name__ == "__main__":
    main()

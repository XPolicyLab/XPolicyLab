#!/usr/bin/env python3

from __future__ import annotations

import argparse
import asyncio
import hashlib
import http
import importlib.util
import json
import logging
import os
import pathlib
import sys
import threading
import time
import traceback
from typing import Any

from policy_core import msgpack_numpy
from go3_nano_preview.ttc_config import TTCConfig
from go3_nano_preview.memory_config import MemoryConfig
import websockets
from websockets.legacy.server import serve as websocket_serve

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_cfg_module(cfg_path: str):
    path = pathlib.Path(cfg_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"cfg not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load cfg: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(mod)
    finally:
        if sys.path and sys.path[0] == str(path.parent):
            sys.path.pop(0)
    return mod


def validate_robot_checkpoint(ckpt: pathlib.Path, robot_name: str) -> None:
    robot = str(robot_name).strip().lower().replace("-", "_")
    if robot != 'robotwin':
        raise ValueError('Only robotwin robot type is supported')
    from policy_core.utils.WAM_policy.body_embedding import resolve_body_emb_id

    config = json.loads((ckpt / "action_model" / "config.json").read_text(encoding="utf-8"))
    body_id = resolve_body_emb_id(robot)
    if not config.get("body_embedding_enabled", False) or int(config.get("num_body_embeddings", 0)) <= body_id:
        raise ValueError(
            f"robot={robot!r} requires a trained body embedding row {body_id}; "
            f"checkpoint has body_embedding_enabled={config.get('body_embedding_enabled')!r}, "
            f"num_body_embeddings={config.get('num_body_embeddings')!r}"
        )


def build_inferrer(
    cfg_path: str,
    ckpt_path: str,
    *,
    robot_name: str = "robotwin",
    stat_path: str = "",
    dense_action_steps: int | None = None,
    return_head_waist: bool = False,
    run_name: str = "WAM_policy_deploy",
    memory_config: MemoryConfig | None = None,
):
    from policy_core.settings import settings

    ckpt = pathlib.Path(ckpt_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt}")
    validate_robot_checkpoint(ckpt, robot_name)

    settings.mode = "infer"
    settings.infer_checkpoint_dir = str(ckpt)
    settings.run_name = run_name
    if robot_name:
        settings.WAM_policy_infer_robot_name = robot_name
    if stat_path:
        settings.WAM_policy_infer_stat_path = stat_path
    if dense_action_steps is not None:
        settings.WAM_policy_infer_dense_action_steps = dense_action_steps
    if return_head_waist:
        settings.WAM_policy_infer_return_head_waist = True

    mod = _load_cfg_module(cfg_path)
    if not hasattr(mod, "inferrer"):
        raise RuntimeError(
            f"cfg {cfg_path} under settings.mode='infer' must expose `inferrer`"
        )
    if memory_config is not None:
        memory_config.validate_task_flow(mod.inferrer._taskflow)
        memory_config.validate_checkpoint(str(ckpt), mod.inferrer._taskflow)
    return mod.inferrer


class WAM_policyPolicyServer:
    def __init__(self, inferrer: Any, *, ttc_config: TTCConfig | None = None, memory_config: MemoryConfig | None = None) -> None:
        self.inferrer = inferrer
        self.memory_config = memory_config
        self._lock = threading.Lock()
        task_flow = inferrer._taskflow
        self.robot_name = task_flow.cfg.infer.robot_name
        self.ttc_config = ttc_config or TTCConfig()
        if self.ttc_config.enabled:
            from policy_core.model.WAM_policy.pipeline.va_binding_pipeline_rcs_ttc import RCSTTCVABindingPipe

            task_flow._ensure_deploy_runtime()
            task_flow._deploy_action_pipeline = RCSTTCVABindingPipe.from_pipeline(
                task_flow._deploy_action_pipeline,
                num_proposals=self.ttc_config.num_proposals,
                rcs_timestep_ratios=self.ttc_config.timestep_ratios,
                rcs_noise_seed=self.ttc_config.noise_seed,
            )
        logger.info("TTC configuration: %s", self.ttc_config.metadata())

    def infer(self, params: dict[str, Any]) -> dict[str, Any]:
        if self.memory_config is not None:
            self.memory_config.validate_request(params)
        with self._lock:
            out = self.inferrer.infer(data=params)
        if not isinstance(out, dict):
            raise TypeError(f"deploy forward returned {type(out).__name__}, want dict")
        if "result" in out:
            return {"result": out["result"]}
        raise KeyError(
            "deploy output missing 'result' (output_transforms should format corobot)"
        )


async def _process_request(path, request_headers):
    if path == "/healthz":
        return http.HTTPStatus.OK, [], b"OK\n"
    return None


async def serve(args: argparse.Namespace) -> None:
    requested = {}
    if args.deploy_config:
        import yaml
        with open(args.deploy_config, encoding="utf-8") as stream:
            requested = yaml.safe_load(stream) or {}
    ttc_config = TTCConfig.from_mapping(requested)
    memory_config = MemoryConfig.from_mapping(requested)
    cfg_path = memory_config.resolve_config(args.cfg_path, args.deploy_config)
    logger.info("Memory configuration: %s; effective config=%s", memory_config.metadata(), cfg_path)
    if args.seed is not None:
        _set_seed(int(args.seed))
        logger.info("Global random seed set to %s", args.seed)

    inferrer = build_inferrer(
        cfg_path,
        args.ckpt_path,
        robot_name=args.robot_name,
        stat_path=args.stat_path,
        dense_action_steps=args.dense_action_steps,
        return_head_waist=args.return_head_waist,
        memory_config=memory_config,
    )
    server = WAM_policyPolicyServer(inferrer, ttc_config=ttc_config, memory_config=memory_config)
    video_cfg = inferrer._taskflow.cfg.model.video
    bbox_enabled = bool(getattr(video_cfg, "use_bbox_condition", False))
    if args.deploy_config:
        if bbox_enabled != bool(requested.get("bbox_enabled", False)):
            raise ValueError("cfg-path did not apply bbox_enabled from deploy-config")
    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    metadata = {
        **server.ttc_config.metadata(),
        **memory_config.metadata(),
        "cfg_path": str(pathlib.Path(cfg_path).resolve()),
        "schema_version": 1,
        "bbox_enabled": bbox_enabled,
        "bbox_input_mode": "red_reference" if bbox_enabled else None,
        "vlm_fixed_sequence_length": getattr(video_cfg, "vlm_fixed_sequence_length", None),
        "checkpoint": str(pathlib.Path(args.ckpt_path).resolve()),
        "checkpoint_files": {
            name: file_hash(pathlib.Path(args.ckpt_path) / name)
            for name in ("diffusion_pytorch_model.bin", "action_model/diffusion_pytorch_model.bin")
        },
        "cfg_sha256": file_hash(cfg_path),
        "stats_sha256": file_hash(args.stat_path),
        "robot_name": server.robot_name,
        "dense_action_steps": args.dense_action_steps,
        "state_layout": os.environ.get("WAM_policy_INFER_LEGACY_STATE_LAYOUT", ""),
        "startup_seed": args.seed,
    }

    async def handler(ws):
        logger.info("conn open: %s", ws.remote_address)
        try:
            async for msg in ws:
                t0 = time.monotonic()
                try:
                    req = msgpack_numpy.unpackb(msg)
                    if isinstance(req, dict) and req.get("method") == "metadata":
                        await ws.send(msgpack_numpy.packb({"metadata": metadata}))
                        continue
                    if not isinstance(req, dict) or req.get("method") != "infer":
                        method = req.get("method") if isinstance(req, dict) else None
                        await ws.send(
                            msgpack_numpy.packb(
                                {"error": f"unknown method: {method!r}"}
                            )
                        )
                        continue
                    resp = await asyncio.to_thread(server.infer, req["params"])
                    await ws.send(msgpack_numpy.packb(resp))
                    n = len(resp["result"]["left_arm"]["values"])
                    logger.info(
                        "robot=%s -> %d steps (%.1fms)",
                        server.robot_name,
                        n,
                        (time.monotonic() - t0) * 1000,
                    )
                except Exception:
                    err = traceback.format_exc()
                    logger.error("infer error: %s", err)
                    await ws.send(msgpack_numpy.packb({"error": err}))
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info("conn closed: %s", ws.remote_address)

    async with websocket_serve(
        handler,
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=_process_request,
    ):
        logger.info(
            "WAM_policy server ws://%s:%d robot=%s (TaskFlow transforms→step→out-tf)",
            args.host,
            args.port,
            server.robot_name,
        )
        await asyncio.Future()


def main() -> None:
    saved_argv = sys.argv[:]
    p = argparse.ArgumentParser(description="WAM_policy Robotwin policy server")
    p.add_argument("--deploy-config", default="", help="Shared WAM_policy deploy.yml (memory, bbox and TTC switches)")
    p.add_argument("--cfg-path", required=True)
    p.add_argument("--ckpt-path", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--robot-name", choices=["robotwin"], default="robotwin")
    p.add_argument("--stat-path", default="")
    p.add_argument("--dense-action-steps", type=int, default=None)
    p.add_argument("--return-head-waist", action="store_true")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global RNG seed at startup (align WAM_policy main_server_all --seed)",
    )
    args = p.parse_args(saved_argv[1:])
    if args.dense_action_steps is not None and args.dense_action_steps <= 0:
        p.error("--dense-action-steps must be positive")

    if args.deploy_config:
        os.environ["WAM_policy_DEPLOY_CONFIG"] = str(pathlib.Path(args.deploy_config).resolve())

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    sys.argv = [saved_argv[0]]
    try:
        asyncio.run(serve(args))
    finally:
        sys.argv = saved_argv


if __name__ == "__main__":
    main()

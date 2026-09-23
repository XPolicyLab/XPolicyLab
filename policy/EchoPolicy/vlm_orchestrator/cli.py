# SPDX-License-Identifier: Apache-2.0

"""Minimal command-line entry point for the Subgoal Agent VLA proxy."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys

from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.proxy import (
    OrchestratorProxy,
    ProxyConfig,
)
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.base import StrategyContext
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.passthrough import PassthroughStrategy
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.transforms import (
    BBox,
    BBoxImageTransform,
    PrefixPromptTransform,
    SuffixPromptTransform,
)
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.vlm import GoogleVLM, PassthroughVLM

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Subgoal Agent VLA proxy with pluggable transforms"
    )
    parser.add_argument("--vla-host", default="127.0.0.1")
    parser.add_argument("--vla-port", type=int, default=6000)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument(
        "--frontend-protocol",
        choices=["robodojo"],
        default="robodojo",
        help="Eval-client frontend protocol.",
    )
    parser.add_argument(
        "--backend-protocol",
        choices=["xpolicylab"],
        default="xpolicylab",
        help="Policy backend protocol/input schema.",
    )
    parser.add_argument(
        "--mode", choices=["subgoal", "passthrough"], default="subgoal"
    )
    parser.add_argument("--vlm-model", default="gemini-3.8-flash")
    parser.add_argument(
        "--vlm-base-url",
        default="https://generativelanguage.googleapis.com/v1beta",
    )
    parser.add_argument("--vlm-api-key", default=None)
    parser.add_argument(
        "--vlm-proxy-url",
        default=None,
        help="Explicit proxy for VLM requests; environment proxies stay disabled.",
    )
    parser.add_argument("--vlm-temperature", type=float, default=0.0)
    parser.add_argument(
        "--vlm-thinking-level",
        choices=["minimal", "low", "medium", "high"],
        default="low",
        help="Google Gemini thinking level.",
    )
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument(
        "--cfg-uncond-prompt",
        default="highlevel",
        help="CFG unconditional prompt; 'highlevel' uses the original task instruction.",
    )
    parser.add_argument(
        "--no-cfg",
        action="store_true",
        help="Do not send CFG scale or unconditional prompt to the VLA.",
    )
    parser.add_argument("--progress-interval", type=int, default=20)
    parser.add_argument("--action-chunk-size", type=int, default=10)
    parser.add_argument("--far-distance-m", type=float, default=0.15)
    parser.add_argument("--near-trajectory-steps", type=int, default=30)
    parser.add_argument("--far-trajectory-steps", type=int, default=30)
    parser.add_argument(
        "--vla-candidates",
        type=int,
        default=16,
        help="Total candidate pool size per environment, including an eligible previous trajectory.",
    )
    parser.add_argument(
        "--vla-image-noise-std",
        type=float,
        default=2.0,
        help="Independent Gaussian noise standard deviation for VLA RGB candidates (0 disables).",
    )
    parser.add_argument(
        "--no-vla-image-augmentation",
        action="store_true",
        help="Disable Pi05 training-equivalent image augmentation for VLA candidates.",
    )
    parser.add_argument(
        "--vla-joint-state-noise-std",
        type=float,
        default=0.01,
        help="Gaussian noise in arm joint radians for far-target VLA candidates.",
    )
    parser.add_argument(
        "--batch-vlm-concurrency",
        type=int,
        default=0,
        help="Maximum concurrent VLM calls per batch (0 means unlimited).",
    )
    parser.add_argument(
        "--global-vlm-concurrency",
        type=int,
        default=None,
        help="Maximum concurrent VLM calls across sessions (0 or omitted means unlimited).",
    )
    parser.add_argument(
        "--image-key", default="observation/exterior_image_1_left"
    )
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument(
        "--extra-image-key", action="append", default=None,
        help="Additional camera key; may be repeated.",
    )
    parser.add_argument(
        "--head-only", action="store_true",
        help="Send only the head camera to the VLM; keep VLA camera inputs unchanged.",
    )
    parser.add_argument("--prompt-prefix", default=None)
    parser.add_argument("--prompt-suffix", default=None)
    parser.add_argument(
        "--prompt-transform-consumer",
        action="append",
        choices=["vla", "vlm_decompose", "vlm_progress"],
        default=[],
        help="Consumer(s) receiving prompt transforms; omit to use all.",
    )
    parser.add_argument(
        "--image-transform-consumer",
        action="append",
        choices=["vla", "vlm_decompose", "vlm_progress"],
        default=[],
        help="Consumer(s) receiving the configured bbox image edit.",
    )
    parser.add_argument(
        "--image-edit-mode",
        choices=["none", "highlight", "dim", "both"],
        default="none",
    )
    parser.add_argument(
        "--image-bbox",
        metavar="X1,Y1,X2,Y2",
        default=None,
        help="Optional fixed bbox for the basic image edit transform.",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def _parse_bbox(value: str | None) -> BBox | None:
    if not value:
        return None
    try:
        parts = [int(part.strip()) for part in value.split(",")]
        if len(parts) != 4:
            raise ValueError
        return BBox(*parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--image-bbox must be X1,Y1,X2,Y2"
        ) from exc


def build_strategy(args: argparse.Namespace):
    from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.transforms import (
        ImageTransform,
        PromptTransform,
    )

    extra_keys = [] if args.head_only else (
        args.extra_image_key or ["observation/wrist_image_left"]
    )
    if (
        not args.head_only
        and args.frontend_protocol == "robodojo"
        and args.extra_image_key is None
    ):
        extra_keys.append("observation/wrist_image_right")
    prompt_transforms: list[PromptTransform] = []
    if args.prompt_prefix is not None:
        prompt_transforms.append(PrefixPromptTransform(args.prompt_prefix))
    if args.prompt_suffix is not None:
        prompt_transforms.append(SuffixPromptTransform(args.prompt_suffix))

    image_transforms: list[ImageTransform] = []
    bbox = _parse_bbox(args.image_bbox)
    if bbox is not None:
        image_transforms.append(BBoxImageTransform(bbox, args.image_edit_mode))

    if args.mode == "passthrough":
        vlm = PassthroughVLM()
    else:
        vlm = GoogleVLM(
            model=args.vlm_model,
            base_url=args.vlm_base_url,
            api_key=args.vlm_api_key,
            temperature=args.vlm_temperature,
            thinking_level=args.vlm_thinking_level,
            proxy_url=args.vlm_proxy_url,
        )
        logging.getLogger(__name__).info(
            "VLM backend=GoogleVLM model=%s base_url=%s thinking_level=%s",
            args.vlm_model,
            args.vlm_base_url,
            args.vlm_thinking_level,
        )

    ctx = StrategyContext(
        vlm=vlm,
        image_key=args.image_key,
        prompt_key=args.prompt_key,
        extra_image_keys=extra_keys,
        prompt_transforms=prompt_transforms,
        prompt_transform_consumers=set(args.prompt_transform_consumer),
        image_transforms=image_transforms,
        image_transform_consumers=set(args.image_transform_consumer),
        vla_host=args.vla_host,
        vla_port=args.vla_port,
    )
    if args.mode == "passthrough":
        return PassthroughStrategy(ctx)
    return SubgoalStrategy(ctx, SubgoalConfig(progress_interval=args.progress_interval))


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        stream=sys.stderr,
    )
    if args.log_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_dir = os.path.expanduser(
            f"~/vlm-orchestrator/results/{args.mode}_{stamp}"
        )
    os.makedirs(args.log_dir, exist_ok=True)

    strategy = build_strategy(args)
    from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.robodojo_ws import RoboDojoWsFrontend
    from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.xpolicylab_ws import XPolicyLabWsBackend
    frontend = RoboDojoWsFrontend()
    extra_image_keys = args.extra_image_key or [
        "observation/wrist_image_left",
        "observation/wrist_image_right",
    ]
    backend = XPolicyLabWsBackend(args.vla_host, args.vla_port)

    proxy = OrchestratorProxy(ProxyConfig(
        vla_host=args.vla_host,
        vla_port=args.vla_port,
        host=args.host,
        port=args.port,
        strategy=strategy,
        image_key=args.image_key,
        prompt_key=args.prompt_key,
        extra_image_keys=extra_image_keys,
        log_dir=args.log_dir,
        cfg_scale=args.cfg_scale,
        cfg_uncond_prompt=args.cfg_uncond_prompt,
        cfg_enabled=not args.no_cfg,
        action_chunk_size=args.action_chunk_size,
        vla_candidates=args.vla_candidates,
        vla_image_noise_std=args.vla_image_noise_std,
        vla_image_augmentation=not args.no_vla_image_augmentation,
        vla_joint_state_noise_std=args.vla_joint_state_noise_std,
        far_distance_m=args.far_distance_m,
        near_trajectory_steps=args.near_trajectory_steps,
        far_trajectory_steps=args.far_trajectory_steps,
        batch_vlm_concurrency=args.batch_vlm_concurrency,
        global_vlm_concurrency=args.global_vlm_concurrency,
        frontend=frontend,
        backend=backend,
    ))
    proxy.serve_forever()


if __name__ == "__main__":
    main()

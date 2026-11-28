#!/usr/bin/env python3
"""Offline A0/A1 inference from a recorded RoboDojo RGB window.

Does not execute robot actions, modify training, or enable legacy KV caching.
Run with python -m scripts.vapv2.infer_robodojo --help.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Recorded RGB window .npz; see INFERENCE.md",
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--stage", choices=("a0", "a1"), default="a0")
    p.add_argument(
        "--window-start-step",
        type=int,
        required=True,
        help="Control-step index of local I; episode begins at 0",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--normalizer-key")
    p.add_argument("--normalizer-stats", type=Path)
    p.add_argument(
        "--text-cache",
        type=Path,
        help="Relocate the checkpoint SAME Qwen cache, not T5 or a new cache identity",
    )
    p.add_argument(
        "--qwen-model",
        type=Path,
        help="Relocate identical Qwen weights for an online checkpoint",
    )
    p.add_argument(
        "--qwen-online-from-cache",
        action="store_true",
        help="Explicitly use the original Qwen weights instead of a cache-backed checkpoint's raw features; verify all source file hashes",
    )
    p.add_argument(
        "--action-steps", type=int, help="ODE evaluations, not action horizon"
    )
    p.add_argument("--action-shift", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--vae-tiled", action="store_true")
    p.add_argument("--kv-cache", action="store_true",
                   help="Request-local clean-prefix cache; rebuild every window")
    return p


def prepare_memory(runtime, payload, *, window_start_step):
    """Select O0 + exact earlier offsets, independently encode valid frames.

    Selection is relative to local I, including A1 calls. Never substitute a
    newer observation for missing history or use the end of V0 as the PE origin.
    """
    if window_start_step < 0:
        raise ValueError("window-start-step must be nonnegative")
    if not runtime.first_frame_memory_enabled:
        if "history_rgb" in payload or "history_steps" in payload:
            raise ValueError("History supplied to a checkpoint without Memory")
        return None, None
    from wall_x.model.vap.history import select_history

    if "history_steps" not in payload or "history_rgb" not in payload:
        raise ValueError(
            "Memory checkpoint requires history_steps and history_rgb, including episode frame 0"
        )
    steps, frames = payload["history_steps"], payload["history_rgb"]
    if (
        steps.ndim != 1
        or steps.dtype.kind not in "iu"
        or len(set(steps.tolist())) != len(steps)
    ):
        raise ValueError(
            "history_steps must be unique integer [N] control-step indices"
        )
    if frames.ndim != 5 or frames.shape[0] != len(steps) or frames.shape[1] != 3:
        raise ValueError("history_rgb must be [N,3,H,W,3] aligned to history_steps")
    if np.any(steps < 0) or np.any(steps > window_start_step):
        raise ValueError(
            "History cannot precede episode start or be newer than local I"
        )
    by_step = dict(zip(steps.tolist(), frames))
    if 0 not in by_step:
        raise ValueError(
            "Missing episode first frame 0; not interchangeable with current I"
        )
    indices, valid, ages = select_history(window_start_step, runtime.history_offsets)
    anchors = []
    for slot, index in enumerate(indices):
        valid[slot] = valid[slot] and index in by_step
        if valid[slot]:
            anchors.append(runtime.encode_first_frame_memory(by_step[index][:, None]))
        else:
            anchors.append(None)
            ages[slot] = 0
    if runtime.history_offsets:
        return runtime.assemble_history_memory(anchors, valid, ages), None
    age = (
        window_start_step
        if runtime.first_frame_memory_position_mode == "relative_age"
        else None
    )
    return anchors[0], age


def infer_window(runtime, payload, *, stage, instruction, window_start_step, seed):
    if "video" not in payload:
        raise ValueError("Input archive requires RGB video [3,T,H,W,3]")
    expected_frames = 1 if stage == "a0" else 1 + runtime.video_frames_per_chunk
    if payload["video"].ndim != 5 or payload["video"].shape[1] != expected_frames:
        raise ValueError(
            f"{stage} requires exactly {expected_frames} RGB frames per view"
        )
    if stage == "a0" and "previous_normalized_action" in payload:
        raise ValueError("A0 must not receive a previous action prefix")
    if stage == "a1" and "previous_normalized_action" not in payload:
        raise ValueError(
            "A1 requires previous_normalized_action [A,20] in the local A0 frame"
        )
    state = payload.get("physical_state")
    if runtime.state_conditioning_enabled != (state is not None):
        raise ValueError(
            "physical_state presence must match checkpoint State FiLM setting"
        )
    memory, age = prepare_memory(runtime, payload, window_start_step=window_start_step)
    text = runtime.encode_text(instruction)
    options = dict(
        text_context=text,
        seed=seed,
        physical_state=state,
        first_frame_memory=memory,
        first_frame_memory_age_steps=age,
    )
    if stage == "a0":
        result = runtime.sample_first_action(payload["video"], **options)
    else:
        result = runtime.sample_next_action(
            payload["video"],
            payload["previous_normalized_action"],
            **options,
        )
    normalized, physical = runtime.prepare_action_for_execution(result)
    return dict(
        raw_normalized_action=result.numpy(),
        normalized_action=normalized,
        relative_action=physical,
    )


def main():
    args = parser().parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    from wall_x.model.vapv2.serving import VAPv2ServingRuntime

    runtime = VAPv2ServingRuntime.from_paths(
        checkpoint_path=args.checkpoint,
        device=args.device,
        normalizer_key=args.normalizer_key,
        normalizer_stats_path=args.normalizer_stats,
        text_feature_cache_path=args.text_cache,
        qwen_model_path=args.qwen_model,
        qwen_online_from_cache=args.qwen_online_from_cache,
        action_inference_steps=args.action_steps,
        action_inference_shift=args.action_shift,
        vae_tiled=args.vae_tiled,
        use_kv_cache=args.kv_cache,
    )
    with np.load(args.input, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    result = infer_window(
        runtime,
        payload,
        stage=args.stage,
        instruction=args.instruction,
        window_start_step=args.window_start_step,
        seed=args.seed,
    )
    metadata = dict(
        model_type="vapv2",
        architecture=runtime.cfg.model.architecture_version,
        checkpoint=str(args.checkpoint.resolve()),
        stage=args.stage,
        window_start_step=args.window_start_step,
        seed=args.seed,
        history_offsets=runtime.history_offsets,
        kv_cache=runtime.use_kv_cache,
        action_flow=runtime.cfg.model.action_flow,
        normalizer_stats=runtime.normalizer_stats_file,
        normalizer_key=runtime.normalizer_key,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        np.savez_compressed(handle, **result, metadata=json.dumps(metadata))
    print(json.dumps(metadata, indent=2))
    print(f"Saved {args.output}; relative actions only, not executed")


if __name__ == "__main__":
    main()

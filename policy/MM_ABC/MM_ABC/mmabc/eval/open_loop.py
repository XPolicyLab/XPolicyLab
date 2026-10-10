"""Open-loop action errors in model space and physical command units."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from mmabc.canonical.layout import CanonicalLayout
from mmabc.data.dataset import ProfileConfig, ProfileDataset


def evaluate_profile(
    model,
    profile_cfg: ProfileConfig,
    layout: CanonicalLayout,
    *,
    chunk_size: int,
    image_size: int,
    future_offsets: tuple[int, ...],
    norm_stats,
    num_samples: int = 128,
    batch_size: int = 8,
    device: torch.device | None = None,
    seed: int = 0,
    prompt_header: bool = False,
) -> dict:
    """Mean absolute error per segment for one profile."""
    import random

    from mmabc.data.collate import collate, to_device

    device = device or torch.device("cuda")
    ds = ProfileDataset(
        profile_cfg,
        layout,
        chunk_size=chunk_size,
        image_size=image_size,
        future_offsets=future_offsets,
        norm_stats=norm_stats,
        prompt_header=prompt_header,
    )
    rng = random.Random(seed)

    seg_abs: dict[str, list[float]] = defaultdict(list)
    seg_norm: dict[str, list[float]] = defaultdict(list)
    scale = ds.normalizer.action_scale

    model.eval()
    done = 0
    while done < num_samples:
        n = min(batch_size, num_samples - done)
        batch = collate([ds.sample(rng) for _ in range(n)])
        batch = to_device(batch, device)
        # bf16 autocast: FA2 rejects fp32, and the model is built/loaded in fp32.
        # Mirrors training and the deployment policy so eval does not crash on
        # the first attention call.
        dev_type = device.type if hasattr(device, "type") else str(device).split(":")[0]
        autocast = (
            torch.autocast(device_type=dev_type, dtype=torch.bfloat16)
            if dev_type == "cuda"
            else torch.autocast(device_type=dev_type)
        )
        with torch.no_grad(), autocast:
            pred = model.predict_action(batch).float().cpu().numpy()
        target = batch["target"].cpu().numpy()
        mask = batch["target_mask"].cpu().numpy() > 0

        err = np.abs(pred - target)
        for seg in layout.segments:
            sl = seg.target_slice
            m = mask[..., sl]
            if not m.any():
                continue
            e = err[..., sl][m]
            seg_norm[seg.name].append(float(e.mean()))
            # De-normalised: the per-dim scale converts back to SI units.
            unit = np.broadcast_to(scale[sl], err[..., sl].shape)[m]
            seg_abs[seg.name].append(float((e * unit).mean()))
        done += n

    return {
        "profile": profile_cfg.name,
        "embodiment_tag": profile_cfg.embodiment_tag,
        "samples": done,
        "normalized_mae": {k: float(np.mean(v)) for k, v in sorted(seg_norm.items())},
        "unit_mae": {k: float(np.mean(v)) for k, v in sorted(seg_abs.items())},
        "overall_normalized_mae": float(
            np.mean([np.mean(v) for v in seg_norm.values()]) if seg_norm else float("nan")
        ),
    }


def report(results: list[dict], out: str | Path | None = None) -> str:
    lines = ["open-loop evaluation (mean absolute error)"]
    for r in results:
        lines.append(
            f"\n{r['profile']}  tag={r['embodiment_tag']}  n={r['samples']}  "
            f"overall_norm_mae={r['overall_normalized_mae']:.4f}"
        )
        for seg, val in r["normalized_mae"].items():
            unit = r["unit_mae"].get(seg, float("nan"))
            lines.append(f"    {seg:22s} norm={val:.4f}  unit={unit:.5f}")
    text = "\n".join(lines)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(results, indent=1))
    return text

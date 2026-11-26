"""Prepare portable metadata, critical sampling indices, and T5 context caches."""

from dataclasses import asdict
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following instruction: {task}"
)


def prepare(metadata, media_root, stats, output, holdout):
    """Keep the reference row order and exact split; never re-encode videos."""
    from .datasets.lerobot.critical_phase_sampling import (
        SCHEMA,
        SCORE_RECIPE,
        EventRules,
        detect_gripper_events,
        row_key,
        row_masks,
    )

    source, media_root, output = (
        Path(metadata).resolve(),
        Path(media_root).resolve(),
        Path(output).resolve(),
    )
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    df = pd.read_csv(source)
    required = {
        "episode_index",
        "task_name",
        "dimension",
        "video_path",
        "parquet_path",
        "fps",
        "trim_start",
        "trim_end",
        "training_prompt",
    }
    if required - set(df):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(df))}")
    if df.episode_index.duplicated().any() or not df.fps.eq(25).all():
        raise ValueError("Expected unique full episodes at 25 Hz")
    split = json.loads(Path(holdout).read_text())
    heldout = {int(r["episode_index"]): r["task_name"] for r in split["held_out"]}
    for episode, task in heldout.items():
        found = df[df.episode_index.eq(episode)]
        if len(found) != 1 or found.iloc[0].task_name != task:
            raise ValueError(f"Reference holdout mismatch: {episode}/{task}")
    for column in ("video_path", "parquet_path"):
        relative = []
        for value in df[column]:
            p = Path(value)
            if not p.is_absolute():
                p = media_root / p
            p = p.resolve()
            if not p.is_file():
                raise FileNotFoundError(p)
            relative.append(p.relative_to(media_root).as_posix())
        df[column] = relative
    prompts = list(dict.fromkeys(df.training_prompt))
    if any(not isinstance(p, str) or not p.strip() for p in prompts):
        raise ValueError("Each episode needs a nonempty training_prompt")
    prompt_files = {p: f"text_embeds/prompt_{i:05d}.pt" for i, p in enumerate(prompts)}
    df["text_embedding_path"] = df.training_prompt.map(prompt_files)
    is_val = df.episode_index.isin(heldout)
    train, val = df[~is_val].copy(), df[is_val].copy()
    if (len(train), len(val)) != (int(split["train_episodes"]), int(split["val_episodes"])):
        raise ValueError("Episode counts differ from the reference split")
    # Validate stats by parsing and checking the fields actually consumed by z-score.
    statistics = json.loads(Path(stats).read_text())
    for group in ("action", "state"):
        for field in ("global_mean", "global_std"):
            a = np.asarray(statistics[group]["default"][field])
            if a.shape != (16,) or not np.isfinite(a).all():
                raise ValueError(f"Invalid {group} normalization {field}")
    rules = EventRules(before=4, after=2)
    keys, masks, cores, offsets, report = [], [], [], [0], []
    for row in train.itertuples(index=False):
        table = pd.read_parquet(media_root / row.parquet_path)
        for col in ("frame_index", "raw_frame_index"):
            if col not in table or not np.array_equal(table[col], np.arange(len(table))):
                raise ValueError(
                    f"Expected identity native-frame mapping: {row.parquet_path}/{col}"
                )
        action, state = np.stack(table.action), np.stack(table["observation.state"])
        if (
            action.shape != (len(table), 16)
            or state.shape != action.shape
            or not np.isfinite(state).all()
        ):
            raise ValueError(f"Invalid EE16 action/state: {row.parquet_path}")
        start, end = int(row.trim_start), int(row.trim_end)
        if not 0 <= start < end <= len(table):
            raise ValueError(f"Invalid trim range: {row.parquet_path}")
        events = detect_gripper_events(action, rules)
        mask, core, _, accepted, _ = row_masks(events, start, end, rules, prefix=8)
        keys.append(
            row_key(row.parquet_path, row.video_path, start, end, start, end, row.training_prompt)
        )
        masks.append(mask)
        cores.append(core)
        offsets.append(offsets[-1] + len(mask))
        report.append(
            dict(episode=int(row.episode_index), events=len(accepted), frames=end - start)
        )
        if len(report) % 500 == 0:
            print(f"Validated {len(report)}/{len(train)} training episodes", flush=True)
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate sampling row identities")
    output.mkdir(parents=True)
    df.to_csv(output / "all.csv", index=False)
    train.to_csv(output / "train.csv", index=False)
    val.to_csv(output / "val.csv", index=False)
    shutil.copyfile(stats, output / "dataset_stats.json")
    (output / "prompts.json").write_text(
        json.dumps(prompt_files, ensure_ascii=False, indent=2) + "\n"
    )
    index = output / "critical_index"
    index.mkdir()
    np.savez_compressed(
        index / "start_weights.npz",
        keys=np.asarray(keys),
        offsets=np.asarray(offsets, dtype=np.int64),
        critical=np.concatenate(masks),
        core=np.concatenate(cores),
    )
    manifest = dict(
        schema=SCHEMA,
        rows=len(keys),
        rules=asdict(rules),
        sampling=SCORE_RECIPE,
        review_status="validated",
        validation="identity_frame_mapping_and_ee16_ranges",
        contract=dict(action_horizon=32, stride=1, action_offset=0, prefix=8, fps=25),
    )
    (index / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (index / "events_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = dict(
        train=len(train),
        val=len(val),
        prompts=len(prompts),
        media_root=str(media_root),
        statistics_source="supplied reference full-dataset statistics",
    )
    (output / "preparation.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def text_cache(data, wan, device="cuda", batch_size=16):
    import torch
    from .models.wan21_14b.helpers.io import load_state_dict
    from .models.wan21_14b.wan_video_text_encoder import WanTextEncoder, HuggingfaceTokenizer

    data, wan = Path(data), Path(wan)
    prompts = json.loads((data / "prompts.json").read_text())
    tokenizer = HuggingfaceTokenizer(
        name=str(wan / "google/umt5-xxl"), seq_len=128, clean="whitespace"
    )
    encoder = WanTextEncoder()
    encoder.load_state_dict(
        load_state_dict(str(wan / "models_t5_umt5-xxl-enc-bf16.pth"), torch_dtype=torch.bfloat16),
        strict=True,
    )
    encoder = encoder.to(device=device, dtype=torch.bfloat16).eval().requires_grad_(False)
    items = list(prompts)
    with torch.inference_mode():
        for start in range(0, len(items), batch_size):
            instructions = items[start : start + batch_size]
            texts = instructions  # training_prompt already contains the complete template
            ids, mask = tokenizer(texts, return_mask=True, add_special_tokens=True)
            mask = mask.to(device=device, dtype=torch.bool)
            context = encoder(ids.to(device), mask)
            for i, instruction in enumerate(instructions):
                path = data / prompts[instruction]
                path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "context": context[i].cpu().to(torch.bfloat16).contiguous(),
                        "mask": mask[i].cpu().contiguous(),
                    },
                    path,
                )
    return {"prompts": len(items), "context_length": 128}

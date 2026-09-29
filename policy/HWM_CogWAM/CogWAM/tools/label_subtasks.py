#!/usr/bin/env python3
"""Reference implementation for producing CogWAM subtask annotations.

Samples frames from an episode video, asks a vision-language model where the
stage boundaries fall, expands them into the ``subtask_text`` /
``complete_text`` columns, and writes them back into the episode parquet.

The contract these columns must satisfy is in the Data section of README.md. This tool
imports the loader's own normalisation helpers rather than reimplementing them,
so what it writes and what training reads cannot drift apart.

Three modes:

    --dry-run   show the proposed segmentation, write nothing
    (default)   label the selected episodes and write the two columns
    --check     validate existing annotations, make no model calls

Configuration comes from the environment; nothing about a provider is baked in:

    COGWAM_LABEL_BASE_URL   any OpenAI-compatible /chat/completions endpoint
    COGWAM_LABEL_MODEL      model name to send
    COGWAM_LABEL_API_KEY    bearer token
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from cogwam.data.event_memory import (
    finished_task_items,
    format_finished_task_items,
    normalize_n1,
    text_value,
)

SUBTASK_COLUMN = "subtask_text"
COMPLETE_COLUMN = "complete_text"
EMPTY_MEMORY = "None."

# Match the released recipe, so --check reports the number training will assert on.
REPLAN_INTERVAL = 10
REPLAN_PHASE = 0
EXPECTED_UPDATE_RATIO = 0.064
EXPECTED_UPDATE_TOLERANCE = 0.005

DEFAULT_CAMERA = "observation.images.cam_high"
DEFAULT_MAX_FRAMES = 16

PROMPT_TEMPLATE = """You are segmenting a robot manipulation episode into ordered stages.

Task instruction: {instruction}

The episode has {total_frames} frames. You are shown {num_images} frames sampled
from it, in order. Their frame indices are: {indices}.

The episode passes through exactly these {num_stages} stages, in this order:
{stage_list}

Report the frame index at which each stage AFTER the first one BEGINS. That is
{num_boundaries} strictly increasing integers in [1, {last_frame}]. A boundary is
the first frame on which the next stage is under way -- the moment the previous
stage is finished, not the moment the robot starts moving toward it.

Reply with JSON only, no prose:
{{"boundaries": [{example}]}}"""


# --------------------------------------------------------------------------- #
# LeRobot v2.1 layout
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    chunks_size: int
    data_path: str
    video_path: str

    def parquet(self, episode: int) -> Path:
        return self.root / self.data_path.format(
            episode_chunk=episode // self.chunks_size, episode_index=episode
        )

    def video(self, episode: int, camera: str) -> Path:
        return self.root / self.video_path.format(
            episode_chunk=episode // self.chunks_size, episode_index=episode, video_key=camera
        )


def load_layout(root: Path) -> DatasetLayout:
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    version = str(info.get("codebase_version", "")).lstrip("v")
    if version and not version.startswith("2.1"):
        raise SystemExit(f"{root}: expected a LeRobot v2.1 dataset, found codebase_version={version!r}")
    return DatasetLayout(
        root=root,
        chunks_size=int(info.get("chunks_size", 1000)),
        data_path=str(info["data_path"]),
        video_path=str(info["video_path"]),
    )


def episode_instruction(root: Path, episode: int) -> str | None:
    """Best-effort read of the episode's natural-language task."""
    episodes_file = root / "meta" / "episodes.jsonl"
    if not episodes_file.is_file():
        return None
    with episodes_file.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if int(record.get("episode_index", -1)) == episode:
                tasks = record.get("tasks") or []
                return str(tasks[0]) if tasks else None
    return None


def parse_episode_selector(selector: str) -> list[int]:
    episodes: list[int] = []
    for part in selector.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, _, end = part.partition("-")
            episodes.extend(range(int(start), int(end) + 1))
        else:
            episodes.append(int(part))
    return episodes


# --------------------------------------------------------------------------- #
# Frame sampling
# --------------------------------------------------------------------------- #


def sample_frames(video: Path, max_frames: int) -> tuple[list[Any], list[int], int]:
    """Return evenly spaced PIL frames, their indices, and the episode length."""
    import av

    with av.open(str(video)) as container:
        stream = container.streams.video[0]
        frames = [frame.to_image() for frame in container.decode(stream)]
    total = len(frames)
    if total == 0:
        raise SystemExit(f"{video}: decoded zero frames")
    count = min(max_frames, total)
    indices = [round(i * (total - 1) / max(count - 1, 1)) for i in range(count)]
    indices = sorted(dict.fromkeys(indices))
    return [frames[i] for i in indices], indices, total


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Model call
# --------------------------------------------------------------------------- #


def call_model(prompt: str, images: list[Any], *, timeout: int = 180) -> str:
    """POST to any OpenAI-compatible /chat/completions endpoint.

    Deliberately uses urllib rather than a vendor SDK: this keeps the annotation
    path free of a dependency that the training environment does not need, and
    makes it obvious that no credential is stored anywhere but the environment.
    """
    base_url = os.environ.get("COGWAM_LABEL_BASE_URL", "").rstrip("/")
    model = os.environ.get("COGWAM_LABEL_MODEL", "")
    api_key = os.environ.get("COGWAM_LABEL_API_KEY", "")
    missing = [
        name
        for name, value in (
            ("COGWAM_LABEL_BASE_URL", base_url),
            ("COGWAM_LABEL_MODEL", model),
            ("COGWAM_LABEL_API_KEY", api_key),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"set {', '.join(missing)} (see the Data section of README.md)")

    content: list[dict] = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": image_to_data_url(img)}} for img in images)
    payload = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": content}], "temperature": 0}
    ).encode("utf-8")

    # The URL comes from the operator's own environment, not from data.
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise SystemExit(f"annotation endpoint returned {error.code}: {error.read()[:400]!r}") from error
    choices = body.get("choices") or []
    if not choices:
        raise SystemExit(f"annotation endpoint returned no choices: {str(body)[:400]}")
    return str(choices[0]["message"].get("content") or "")


def extract_json(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, flags=re.S)
    if not match:
        raise ValueError(f"no JSON object in model reply: {raw[:300]!r}")
    return json.loads(match.group(0))


# --------------------------------------------------------------------------- #
# Boundaries -> columns
# --------------------------------------------------------------------------- #


def validate_boundaries(boundaries: list[int], stages: list[str], total_frames: int) -> list[int]:
    expected = len(stages) - 1
    if len(boundaries) != expected:
        raise ValueError(f"expected {expected} boundaries for {len(stages)} stages, got {len(boundaries)}")
    if any(not isinstance(value, int) for value in boundaries):
        raise ValueError(f"boundaries must be integers, got {boundaries}")
    if any(b <= 0 or b >= total_frames for b in boundaries):
        raise ValueError(f"boundaries must lie in [1, {total_frames - 1}], got {boundaries}")
    if any(later <= earlier for earlier, later in pairwise(boundaries)):
        raise ValueError(f"boundaries must be strictly increasing, got {boundaries}")
    return boundaries


def columns_from_boundaries(stages: list[str], boundaries: list[int], total_frames: int) -> tuple[list[str], list[str]]:
    """Expand stage boundaries into the two per-frame columns.

    Right-open spans: a boundary at frame ``b`` means the next stage applies
    from ``b`` inclusive, and the stage that just ended enters ``complete_text``
    on the same frame.
    """
    edges = [0, *boundaries, total_frames]
    subtask = [""] * total_frames
    complete = [""] * total_frames
    for stage_index, stage in enumerate(stages):
        start, end = edges[stage_index], edges[stage_index + 1]
        finished = format_finished_task_items(stages[:stage_index], empty_value=EMPTY_MEMORY)
        for frame in range(start, end):
            subtask[frame] = stage
            complete[frame] = finished
    return subtask, complete


def write_columns(parquet_path: Path, subtask: list[str], complete: list[str]) -> None:
    import pandas as pd

    frame = pd.read_parquet(parquet_path)
    if len(frame) != len(subtask):
        raise SystemExit(f"{parquet_path}: {len(frame)} rows but {len(subtask)} labels")
    frame[SUBTASK_COLUMN] = subtask
    frame[COMPLETE_COLUMN] = complete
    temporary = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, parquet_path)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def check_episode(subtask: list[str], complete: list[str]) -> list[str]:
    """Return the contract violations in one episode (the Data section of README.md)."""
    problems: list[str] = []
    empty = [i for i, value in enumerate(subtask) if not text_value(value)]
    empty += [i for i, value in enumerate(complete) if not text_value(value)]
    if empty:
        problems.append(f"empty annotation at frames {sorted(set(empty))[:5]}")

    subtask_edges = {i for i in range(1, len(subtask)) if normalize_n1(subtask[i]) != normalize_n1(subtask[i - 1])}
    complete_edges = {i for i in range(1, len(complete)) if normalize_n1(complete[i]) != normalize_n1(complete[i - 1])}
    if subtask_edges != complete_edges:
        only_subtask = sorted(subtask_edges - complete_edges)[:5]
        only_complete = sorted(complete_edges - subtask_edges)[:5]
        problems.append(f"columns flip on different frames (subtask-only {only_subtask}, complete-only {only_complete})")

    # complete_text must be the ordered concatenation of the earlier subtasks.
    stages: list[str] = []
    for frame in [0, *sorted(subtask_edges)]:
        expected = format_finished_task_items(stages, empty_value=EMPTY_MEMORY)
        if normalize_n1(complete[frame]) != normalize_n1(expected):
            problems.append(f"frame {frame}: complete_text is not the cumulative prefix of subtask_text")
            break
        stages.append(text_value(subtask[frame]))

    for stage in stages:
        if len(finished_task_items(stage, empty_value=EMPTY_MEMORY)) > 1:
            problems.append(f"subtask splits into several memory items: {stage!r}")
            break
    return problems


def phase_update_ratio(episodes: list[tuple[list[str], list[str]]]) -> tuple[int, int]:
    """Count UPDATE decisions over phase frames, the way the loader will."""
    updates = 0
    total = 0
    for subtask, complete in episodes:
        for frame in range(REPLAN_PHASE, len(subtask), REPLAN_INTERVAL):
            total += 1
            cached = frame - REPLAN_INTERVAL
            if cached < 0:
                updates += 1
                continue
            if normalize_n1(subtask[frame]) != normalize_n1(subtask[cached]) or normalize_n1(
                complete[frame]
            ) != normalize_n1(complete[cached]):
                updates += 1
    return updates, total


def run_check(layout: DatasetLayout, episodes: list[int], *, partial: bool) -> int:
    import pandas as pd

    failed = 0
    loaded: list[tuple[list[str], list[str]]] = []
    for episode in episodes:
        path = layout.parquet(episode)
        if not path.is_file():
            print(f"episode {episode}: MISSING {path}")
            failed += 1
            continue
        frame = pd.read_parquet(path, columns=[SUBTASK_COLUMN, COMPLETE_COLUMN])
        subtask = [text_value(value) for value in frame[SUBTASK_COLUMN].tolist()]
        complete = [text_value(value) for value in frame[COMPLETE_COLUMN].tolist()]
        problems = check_episode(subtask, complete)
        if problems:
            failed += 1
            print(f"episode {episode}: " + "; ".join(problems))
        loaded.append((subtask, complete))

    print(f"\n{len(episodes) - failed}/{len(episodes)} episodes satisfy the annotation contract")

    ratio_drifted = False
    if loaded:
        updates, total = phase_update_ratio(loaded)
        ratio = updates / max(total, 1)
        drift = abs(ratio - EXPECTED_UPDATE_RATIO)
        verdict = "within" if drift <= EXPECTED_UPDATE_TOLERANCE else "OUTSIDE"
        print(
            f"phase UPDATE ratio:  {updates}/{total} = {ratio:.4f} "
            f"({verdict} {EXPECTED_UPDATE_RATIO} +/- {EXPECTED_UPDATE_TOLERANCE})"
        )
        if partial:
            # The contract is a property of the whole mix. A subset is a
            # sanity check, not a verdict -- short episodes and long ones have
            # very different phase-frame counts.
            print("  (subset only -- the contract applies to the full dataset; re-run without --episodes)")
        elif drift > EXPECTED_UPDATE_TOLERANCE:
            ratio_drifted = True
            print("  training will refuse to start; see the Data section of README.md")

    return 1 if (failed or ratio_drifted) else 0


# --------------------------------------------------------------------------- #
# Labelling
# --------------------------------------------------------------------------- #


def label_episode(
    layout: DatasetLayout, episode: int, spec: dict, camera: str, max_frames: int
) -> tuple[list[str], list[str], list[int]]:
    stages = [str(stage) for stage in spec["stages"]]
    if len(stages) < 2:
        raise SystemExit(f"episode {episode}: a checklist needs at least two stages")
    instruction = spec.get("instruction") or episode_instruction(layout.root, episode) or "Complete the task."

    frames, indices, total = sample_frames(layout.video(episode, camera), max_frames)
    prompt = PROMPT_TEMPLATE.format(
        instruction=instruction,
        total_frames=total,
        num_images=len(frames),
        indices=", ".join(str(i) for i in indices),
        num_stages=len(stages),
        stage_list="\n".join(f"  {i}. {stage}" for i, stage in enumerate(stages)),
        num_boundaries=len(stages) - 1,
        last_frame=total - 1,
        example=", ".join(str(round(total * (i + 1) / len(stages))) for i in range(len(stages) - 1)),
    )
    raw = call_model(prompt, frames)
    boundaries = validate_boundaries([int(b) for b in extract_json(raw)["boundaries"]], stages, total)
    subtask, complete = columns_from_boundaries(stages, boundaries, total)
    return subtask, complete, boundaries


def resolve_spec(checklists: dict, layout: DatasetLayout, episode: int, task: str | None) -> dict:
    if task:
        if task not in checklists:
            raise SystemExit(f"checklist file has no entry for task {task!r}")
        return checklists[task]
    instruction = episode_instruction(layout.root, episode)
    for name, spec in checklists.items():
        if instruction and normalize_n1(spec.get("instruction", "")) == normalize_n1(instruction):
            return spec
        if instruction and name.replace("_", " ") in normalize_n1(instruction):
            return spec
    raise SystemExit(
        f"episode {episode}: cannot match instruction {instruction!r} to a checklist entry; pass --task explicitly"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--episodes", default=None, help="e.g. 0-99 or 3,7,12")
    parser.add_argument("--episode", type=int, default=None, help="shorthand for a single episode")
    parser.add_argument("--checklists", type=Path, default=None, help="JSON file of per-task stage lists")
    parser.add_argument("--task", default=None, help="force one checklist entry instead of matching by instruction")
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES)
    parser.add_argument("--dry-run", action="store_true", help="print the segmentation, write nothing")
    parser.add_argument("--check", action="store_true", help="validate existing annotations, make no model calls")
    args = parser.parse_args()

    layout = load_layout(args.dataset_root)

    selected_subset = args.episode is not None or bool(args.episodes)
    if args.episode is not None:
        episodes = [args.episode]
    elif args.episodes:
        episodes = parse_episode_selector(args.episodes)
    else:
        episodes = sorted(
            int(re.search(r"episode_(\d+)", path.stem).group(1))
            for path in (args.dataset_root / "data").rglob("episode_*.parquet")
        )
    if not episodes:
        raise SystemExit("no episodes selected")

    if args.check:
        return run_check(layout, episodes, partial=selected_subset)

    if args.checklists is None:
        raise SystemExit("--checklists is required unless --check is given")
    checklists = json.loads(args.checklists.read_text(encoding="utf-8"))

    for episode in episodes:
        spec = resolve_spec(checklists, layout, episode, args.task)
        subtask, complete, boundaries = label_episode(layout, episode, spec, args.camera, args.max_frames)
        problems = check_episode(subtask, complete)
        if problems:
            raise SystemExit(f"episode {episode}: generated labels violate the contract: {'; '.join(problems)}")
        print(f"episode {episode}: {len(subtask)} frames, boundaries {boundaries}")
        if args.dry_run:
            edges = [0, *boundaries]
            for start, stage in zip(edges, spec["stages"], strict=True):
                print(f"    t={start:5d}  {stage}")
        else:
            write_columns(layout.parquet(episode), subtask, complete)

    if args.dry_run:
        print("\ndry run: nothing written")
    return 0


if __name__ == "__main__":
    sys.exit(main())

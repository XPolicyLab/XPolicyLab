"""Episode-level instruction metadata used by GenieSim task-suite data."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path


def load_episode_instruction_map(
    dataset_root: str | Path,
    source_key: str,
) -> dict[int, str]:
    """Load one non-empty natural-language instruction for every episode.

    RoboColiseum's ``tasks.jsonl`` contains only the broad task family (for
    example ``pick_block_color``).  The language needed to solve each episode
    lives in ``meta/info.json["high_level_instruction"]``.  Treating the broad
    family as the prompt would remove the color/shape/arm binding that the
    benchmark is designed to test.
    """

    root = Path(dataset_root)
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    info = json.loads(info_path.read_text(encoding="utf-8"))
    raw = info.get(source_key)
    if not isinstance(raw, Mapping):
        raise KeyError(
            f"{info_path} is missing episode instruction mapping {source_key!r}"
        )

    instructions: dict[int, str] = {}
    for raw_episode, raw_value in raw.items():
        episode = int(raw_episode)
        value = raw_value
        if isinstance(value, Mapping):
            value = value.get(source_key)
        if value is None:
            raise ValueError(
                f"Missing {source_key!r} text for episode {episode} in {info_path}"
            )
        text = str(value).strip()
        if not text:
            raise ValueError(
                f"Empty {source_key!r} text for episode {episode} in {info_path}"
            )
        instructions[episode] = text

    expected = int(info.get("total_episodes", len(instructions)))
    if len(instructions) != expected or set(instructions) != set(range(expected)):
        missing = sorted(set(range(expected)) - set(instructions))
        raise ValueError(
            f"{source_key!r} in {info_path} covers {len(instructions)}/{expected} "
            f"episodes; first missing={missing[:5]}"
        )
    return instructions

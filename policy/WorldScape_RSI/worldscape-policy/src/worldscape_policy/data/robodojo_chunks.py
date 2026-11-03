"""Shared RoboDojo task identities and opt-in temporal packing policy."""

from pathlib import Path


MEMORY_TASKS = frozenset(
    {
        "cover_blocks",
        "match_and_pick_from_conveyor",
        "swap_T",
        "imitate_sorting_sequence",
        "press_by_number",
        "swap_blocks",
    }
)
MEMORY_CHUNKS = 24
DEFAULT_CHUNKS = 4


def task_id_from_path(path: Path) -> str:
    if path.parent.name == "data" and len(path.parents) >= 3:
        return path.parents[2].name
    return path.parent.name or "robodojo"


def task_chunk_counts(action_horizon: int = 24) -> tuple[int, int]:
    """Preserve memory coverage; use one non-memory chunk only at horizon 48."""
    if action_horizon not in {24, 48}:
        raise ValueError("RoboDojo task_aware requires action_horizon=24 or 48")
    scale = action_horizon // 24
    return MEMORY_CHUNKS // scale, 1 if action_horizon == 48 else DEFAULT_CHUNKS


def task_chunk_limit(path: Path, action_horizon: int = 24) -> int:
    memory_chunks, default_chunks = task_chunk_counts(action_horizon)
    return memory_chunks if task_id_from_path(path) in MEMORY_TASKS else default_chunks


def chunk_policy_options(policy: str, action_horizon: int = 24) -> dict[str, object]:
    """Omit uniform policy metadata to retain compatibility with old indices."""
    if policy == "uniform":
        return {}
    if policy != "task_aware":
        raise ValueError("RoboDojo chunk_policy must be uniform or task_aware")
    memory_chunks, default_chunks = task_chunk_counts(action_horizon)
    return {
        "chunk_policy": {
            "name": policy,
            "memory_tasks": sorted(MEMORY_TASKS),
            "memory_chunks": memory_chunks,
            "default_chunks": default_chunks,
        }
    }

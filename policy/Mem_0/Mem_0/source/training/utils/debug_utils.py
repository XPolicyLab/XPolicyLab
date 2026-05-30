from typing import List

from termcolor import cprint

from source.utils import BOLD, RESET_BOLD


def log_debug_batch(
    step_idx: int,
    sampler_round: int,
    rank: int,
    batch: List[dict],
) -> None:
    """
    Pretty-print per-sample batch info for debug mode to verify ordering across batches.
    - episode_pos is derived locally per episode using episode_offsets tracker.
    - global_idx falls back to per-rank running index when dataset does not expose it.
    """
    header = f"[debug] step {step_idx:03d} round {sampler_round:02d} | rank = {rank} batch = {len(batch)}"
    cprint(header, "magenta")
    for idx, sample in enumerate(batch):
        episode_id = sample.get("episode_id")
        episode_pos = sample.get("episode_pos", "?")
        global_idx = sample.get("global_idx", f"{idx}*")
        subtask_end = sample.get("subtask_end")
        lang = sample.get("lang")
        lang_prefix = (lang[:30] + "...") if isinstance(lang, str) and len(lang) > 30 else lang
        cprint(
            f"    [{idx:02d}] episode / global_idx / episode_pos = ({BOLD}{episode_id}{RESET_BOLD} / {BOLD}{global_idx}{RESET_BOLD} / {BOLD}{episode_pos}{RESET_BOLD}); "
            f"sub_end = {BOLD}{subtask_end}{RESET_BOLD}; "
            f"lang = {BOLD}{lang_prefix}{RESET_BOLD}",
            "magenta",
        )

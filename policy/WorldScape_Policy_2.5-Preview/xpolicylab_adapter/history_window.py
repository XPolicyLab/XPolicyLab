"""Recognize the six fixed RoboDojo memory instructions without simulator imports."""

import re
import unicodedata

# Source: RoboDojo/task/RoboDojo/tasks/<task>.py, gen_instruction().
# Keep complete instructions: words such as "remember", "swap", or "conveyor"
# alone cannot distinguish all memory tasks from the ordinary tasks.
MEMORY_TASK_INSTRUCTIONS = {
    "cover_blocks": (
        "Cover the blocks from left to right, remember their colors, then uncover "
        "them in the order: red, green, and blue."
    ),
    "imitate_sorting_sequence": (
        "Observe the object placement order, remember it, then place the "
        "corresponding objects into the basket in the same order."
    ),
    "match_and_pick_from_conveyor": (
        "Remember the first object on the conveyor, then pick the matching "
        "object when it appears again."
    ),
    "press_by_number": (
        "Press the two red buttons the required number of times according to "
        "the number cards, then press the blue button to confirm."
    ),
    "swap_T": (
        "Pick up the two T-shaped blocks, swap their positions, and place "
        "them back with the correct orientations."
    ),
    "swap_blocks": (
        "Swap the two blocks using the empty mat, pressing the button after each move."
    ),
}


def _instruction_tokens(instruction: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", instruction).casefold()
    return tuple(re.findall(r"[^\W_]+", normalized))


_MEMORY_TASK_BY_TOKENS = {
    _instruction_tokens(instruction): task
    for task, instruction in MEMORY_TASK_INSTRUCTIONS.items()
}


def match_memory_task(instruction: str) -> str | None:
    """Match a complete instruction, ignoring case, whitespace and punctuation.

    Unrecognized paraphrases deliberately return None, like ordinary tasks.
    The caller must still pass the original instruction to the policy model.
    """
    return _MEMORY_TASK_BY_TOKENS.get(_instruction_tokens(instruction))

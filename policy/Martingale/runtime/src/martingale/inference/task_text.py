# Copyright © 2026 Martingale. All Rights Reserved.

DETAILED = 'detailed_instruction'
CLIENT = 'instruction'


def official_text(obs):
    """The benchmark's official Detail Description carried by an observation, or None.

    RoboDojo supplies it as the string ``obs["extra_info"]["detailed_instruction"]``; older
    simulator builds omit the field. Anything other than a non-empty string counts as absent.
    """
    extra = obs.get('extra_info') if isinstance(obs, dict) else None
    text = extra.get(DETAILED) if isinstance(extra, dict) else None
    return text if isinstance(text, str) and text.strip() else None


def statement(obs, instruction):
    """Return ``(text, source)``: the lines of the prompt that state the task, and where they came from.

    With the official Detail Description present it is stated unchanged, and the client's current
    instruction follows as its own labelled line (several official descriptions refer to "the
    current language instruction", which only that observation field carries). Without it the
    client instruction alone is stated. ``source`` is ``'detailed_instruction'`` or
    ``'instruction'``; the session records it in the episode's status file. No other text in the
    prompt depends on the task.
    """
    official = official_text(obs)
    if official is None:
        return 'Instruction: ' + instruction, CLIENT
    return ('Official task description (obs["extra_info"]["detailed_instruction"], unchanged):\n' + official
            + '\n\nCurrent instruction (obs["instruction"]): ' + instruction), DETAILED

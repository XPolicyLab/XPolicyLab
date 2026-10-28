"""One packaged, human-reviewable source for the initial module/LLM protocol."""
from importlib.resources import files

REFERENCE_SECTIONS = ("q1-reference", "q2-experience", "q2-autorevision", "q3-interval")


def reference_section(name):
    """Extract one opt-in contract without introducing a second prompt source."""
    if name not in REFERENCE_SECTIONS:
        raise ValueError("unknown reference contract section")
    text = files('oeh_policy').joinpath('llm_interaction_contract.md').read_text(encoding='utf-8')
    start, end = f'<!-- {name}:start -->', f'<!-- {name}:end -->'
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(f"canonical {name} section missing or ambiguous")
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def has_v2_experience(runtime):
    store = getattr(runtime, "notes", None) or {}
    return any(note.get("schema") == "oeh.task_note.v2" for note in store.get("notes", []))


def interaction_contract(*, task_tree=False):
    text = files('oeh_policy').joinpath('llm_interaction_contract.md').read_text(encoding='utf-8')
    # Reference additions are enabled by their runtime feature, never by memory mode alone.
    for name in REFERENCE_SECTIONS:
        before, rest = text.split(f'<!-- {name}:start -->', 1)
        _, after = rest.split(f'<!-- {name}:end -->', 1)
        text = before + after
    start, end = '<!-- task-tree:start -->', '<!-- task-tree:end -->'
    before, rest = text.split(start, 1)
    task_section, after = rest.split(end, 1)
    return (before + (task_section if task_tree else '') + after).strip()

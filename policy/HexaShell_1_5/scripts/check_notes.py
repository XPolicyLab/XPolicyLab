"""Check the RoboDojo task notes: python3 -I scripts/check_notes.py skills/robodojo/*.md

Format: title "# <task>", the sections below in order (Flow, Verified parameters and Typical step costs may
carry a remark in brackets; With clutter is optional), at most 75 lines.
Errors (exit 1): wording that shows evaluator or simulator internals (10-26: none may show; such knowledge
only roughly or implicitly, as what to do or expect), seed numbers, LLM names.
Warnings: coordinates, and items in Verified parameters, Common failures and fixes and With clutter that
carry no evidence from our own runs."""
import os
import re
import sys

SECTIONS = ["## Judging", "## Flow", "## Verified parameters", "## Common failures and fixes", "## Final checks",
            "## Typical step costs", "## With clutter"]
REMARK = {"## Flow", "## Verified parameters", "## Typical step costs"}
TAGGED = {"## Verified parameters", "## Common failures and fixes", "## With clutter"}
INTERNAL = re.compile(
    r"task code|test code|evaluator|\breward|truth|\btrue (?:rim|top|height|size|position)|simulat|\bsim\b"
    r"|qpos|\blatch|threshold|distractor|decoy|bounding box|\bbbox|\bassets?\b|\bspawn|instant(?:ly)? fail"
    r"|\bjudged\b|the check runs|\bre-?checks\b|_random|score table", re.I)
TRUE = re.compile(r"\bTRUE\b")  # "TRUE top"; Python's True (contact=True) is fine
SEED = re.compile(r"\bseeds?\b|\bs\d{1,2}\b", re.I)
LLM = re.compile(r"\b(gpt|astra|claude|opus|sonnet|openai|llm)\b", re.I)
COORD = re.compile(r"\(\s*-?\d+\.\d+\s*,\s*-?\d+\.\d+(?:\s*,\s*-?\d+\.\d+)?\s*\)")
# counts (3/4, 3 of 4, (2), 12 drops), rounds (R0-R4, R0a), dev layouts and cells (l6, c39), method tags,
# the card, or a pointer to the step or section that carries the evidence
EVIDENCE = re.compile(
    r"\b\d+\s*/\s*\d+\b|\b~?\d+ of (?:the )?~?\d+\b|\(\d+\)|\b\d+ (?:grasps|holds|drops|moves|times)\b|\btwice\b"
    r"|\bR[0-4]a?\b|\b[lLc]\d{1,2}\b|\(all\b|\(\d+(?:-\d+)? steps\)"
    r"|\b(?:scripted|probes?|untested|harness|arm model|runs?|successes|wins?|tries|tried|worked|failed)\b"
    r"|\bthe card\b|\(steps? \d|\((?:Flow|Verified parameters|Common failures|Final checks)", re.I)


def section_of(head: str) -> str:
    for s in SECTIONS:
        if head == s or (s in REMARK and head.startswith(s + " (") and head.endswith(")")):
            return s
    return head


def check(path: str) -> tuple[list[str], list[str]]:
    task = os.path.basename(path)[:-3]
    lines = open(path, encoding="utf-8").read().splitlines()
    err, warn = [], []
    if not lines or lines[0] != f"# {task}":
        err.append(f"title is {lines[0] if lines else ''!r}, not '# {task}'")
    heads = [section_of(ln) for ln in lines if ln.startswith("## ")]
    if heads != (SECTIONS if "## With clutter" in heads else SECTIONS[:-1]):
        err.append(f"sections {heads}")
    if len(lines) > 75:
        err.append(f"{len(lines)} lines")
    item, no_ev, section, covered = [], [], "", False
    for n, ln in enumerate(lines, 1):
        if ln.startswith("## "):
            section = section_of(ln)
            covered = bool(EVIDENCE.search(ln[len(section):]))  # "(all runs)" covers the section
        for name, m in (("internal", INTERNAL.search(ln) or TRUE.search(ln)), ("seed", SEED.search(ln)),
                        ("LLM name", LLM.search(ln))):
            if m:
                err.append(f"line {n}: {name} {m.group(0)!r}")
        m = COORD.search(ln)
        if m:
            warn.append(f"line {n}: coordinates {m.group(0)!r}")
        # an item runs until the next one, a heading or a blank line
        if re.match(r"^(- |\d+\. )", ln) or ln.startswith("#") or not ln.strip():
            if item and not EVIDENCE.search(" ".join(t for _, t in item)):
                no_ev.append(item[0][0])
            # "- None ..." / "- No clutter ..." in With clutter claim nothing
            item = ([(n, ln)] if re.match(r"^(- |\d+\. )", ln) and section in TAGGED and not covered
                    and not re.match(r"^- (None|No clutter)\b", ln) else [])
        elif item:
            item.append((n, ln))
    if item and not EVIDENCE.search(" ".join(t for _, t in item)):
        no_ev.append(item[0][0])
    if no_ev:
        warn.append(f"items without evidence at lines {no_ev}")
    return err, warn


if __name__ == "__main__":
    bad = 0
    for p in sys.argv[1:]:
        err, warn = check(p)
        n = len(open(p, encoding="utf-8").read().splitlines())
        print(f"{os.path.basename(p):36s} {n:2d} lines  " + ("ok" if not err else "; ".join(err)))
        for w in warn:
            print(f"{'':36s}           warn: {w}")
        bad += bool(err)
    sys.exit(1 if bad else 0)

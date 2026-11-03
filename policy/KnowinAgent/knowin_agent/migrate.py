"""Import frozen task knowledge from the previous harness generations (H4/H5) into KnowinAgent's frozen format.

Task skills and lessons are copied as they are (the program namespace keeps the H4/H5 function surface), the
embedded shared-library copy is matched against KnowinAgent's library versions, wording that referred to the old
workspace commands is rewritten, and every skill file is checked by the program sandbox.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
from pathlib import Path
from typing import Optional

REWRITES = [
    (re.compile(r"`?python scripts/env\.py exec[^`\n]*`?"), "the `exec` tool"),
    (re.compile(r"`?python scripts/env\.py status`?"), "the budget panel in every receipt"),
    (re.compile(r"`?python scripts/env\.py observe`?"), "the images in every receipt"),
    (re.compile(r"`?python primitives/rgb_perception/cli\.py (\w+)[^`\n]*`?"), r"`\1(...)` in a program"),
    (re.compile(r"\bsubmissions\b"), "exec programs"),
    (re.compile(r"\bsubmission\b"), "exec program"),
    (re.compile(r"\bPlayground\b"), "practice"),
    (re.compile(r"\bPG\b"), "practice"),
]


def _digest(path: Path) -> str:
    """Hash of the code structure without docstrings (doc rewording does not change behaviour)."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr) and \
                isinstance(getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()


def _match_library(shared: Path) -> Optional[str]:
    """The KnowinAgent library version whose files equal the embedded copy, if any."""
    from knowin_agent.workspace.knowledge import library_dir, load_versions

    files = {p.name: _digest(p) for p in shared.glob("*") if p.is_file() and p.suffix == ".py"}
    for version in (load_versions().get("versions") or {}):
        try:
            candidate = library_dir(version)
        except Exception:  # noqa: BLE001 - unknown or candidate-only versions
            continue
        theirs = {p.name: _digest(p) for p in candidate.glob("*.py")}
        if theirs and theirs == files:
            return version
    return None


def rewrite_text(text: str) -> str:
    for pattern, replacement in REWRITES:
        text = pattern.sub(replacement, text)
    return text


def check_skills(skills: Path) -> dict:
    from knowin_agent.program.sandbox import SandboxError, compile_program

    report = {}
    for path in sorted(skills.rglob("*.py")):
        try:
            compile_program(path.read_text(encoding="utf-8"), filename=str(path.relative_to(skills)))
            report[str(path.relative_to(skills))] = "ok"
        except (SandboxError, SyntaxError) as exc:
            report[str(path.relative_to(skills))] = f"rejected: {exc}"
    return report


def import_frozen(source: Path, destination: Path, *, label: Optional[str] = None) -> dict:
    """source: an H4/H5 run's frozen_harness/ (or a dir with skills/ and lessons/)."""
    from knowin_agent.workspace import freeze_dirs

    source = Path(source)
    if (source / "frozen_harness").is_dir():
        source = source / "frozen_harness"
    if not (source / "skills").is_dir() and not (source / "lessons").is_dir():
        raise FileNotFoundError(f"{source} has no skills/ or lessons/")
    with tempfile.TemporaryDirectory(prefix="knowin_agent-import-") as tmp:
        tmp = Path(tmp)
        skills, lessons, shared = tmp / "skills", tmp / "lessons", None
        if (source / "skills").is_dir():
            shutil.copytree(source / "skills", skills, ignore=shutil.ignore_patterns("shared", "__pycache__"))
        else:
            skills.mkdir()
        if (source / "lessons").is_dir():
            shutil.copytree(source / "lessons", lessons, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            lessons.mkdir()
        rewritten = []
        for path in list(skills.rglob("*.md")) + list(lessons.rglob("*.md")):
            text = path.read_text(encoding="utf-8")
            new = rewrite_text(text)
            if new != text:
                path.write_text(new, encoding="utf-8")
                rewritten.append(str(path.relative_to(tmp)))
        version = None
        embedded = source / "skills" / "shared"
        if embedded.is_dir():
            version = _match_library(embedded)
            if version is None:
                shared = tmp / "shared"
                shutil.copytree(embedded, shared, ignore=shutil.ignore_patterns("__pycache__"))
        check = check_skills(skills)
        manifest = freeze_dirs(destination, skills=skills, lessons=lessons, shared=shared,
                               source_run=label or source.parent.name, library_version=version,
                               extra={"imported_from": str(source), "generation": "H4/H5",
                                      "rewritten_docs": rewritten, "sandbox_check": check})
    return {"manifest": str(manifest), "library_version": version, "embedded_shared": shared is not None,
            "rewritten_docs": rewritten, "sandbox_check": check}

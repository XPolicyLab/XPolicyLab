"""Per-episode workspace (docs/INTERFACES.md, M4).

    workspace/
      instruction.md   read-only
      skills/          task skills (.py + .md); practice writable, test read-only
      skills/shared/   cross-task library, staged read-only from a version
      lessons/         PLAN.md, EVIDENCE.md, RECOVERY.md, PERCEPTION.md, shared_proposals.md ...
      scratch/         temporary files, writable in both phases

The receipt (phase, seed knowledge hashes, library version and hashes) is written next to the workspace,
not inside it, so the agent never sees host bookkeeping.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from knowin_agent.workspace import knowledge, permissions
from knowin_agent.workspace.files import read_json, sha256_file, tree_hashes, write_json

LESSON_FILES = ("PLAN.md", "EVIDENCE.md", "RECOVERY.md", "PERCEPTION.md", "shared_proposals.md")

PLAN_TEMPLATE = """# PLAN

One page. Each row is one sub-goal. Numbers, scores and traces go in EVIDENCE.md.

| # | What | Verification | Recovery |
|---|------|--------------|----------|
|   | arm, object, from → to, heading, skill/include, native est. | image / holding / checkpoint that proves it | what to change if it fails |
"""


@dataclass
class Workspace:
    root: Path
    phase: str
    receipt: dict = field(default_factory=dict)

    @property
    def instruction(self) -> Path:
        return self.root / "instruction.md"

    @property
    def skills(self) -> Path:
        return self.root / "skills"

    @property
    def shared(self) -> Path:
        return self.root / knowledge.SHARED_REL

    @property
    def lessons(self) -> Path:
        return self.root / "lessons"

    @property
    def scratch(self) -> Path:
        return self.root / "scratch"

    @property
    def shared_version(self) -> Optional[str]:
        return (self.receipt.get("shared_library") or {}).get("version")

    def path(self, rel_path: Union[str, Path]) -> Path:
        return self.root / permissions.relative_path(self.root, rel_path)

    def assert_writable(self, rel_path: Union[str, Path]) -> Path:
        return permissions.assert_writable(self.root, rel_path, self.phase)

    def write_text(self, rel_path: Union[str, Path], text: str) -> Path:
        path = self.assert_writable(rel_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def shared_unchanged(self) -> bool:
        expected = (self.receipt.get("shared_library") or {}).get("files") or {}
        return tree_hashes(self.shared) == expected


def receipt_path_for(root: Path) -> Path:
    root = Path(root)
    return root.with_name(root.name + ".json")


def create_workspace(root: Union[str, Path], *, instruction: str, phase: str,
                     knowledge_from: Optional[Union[str, Path]] = None,
                     shared_library: Optional[Union[str, Path]] = None,
                     library_root: Optional[Union[str, Path]] = None,
                     templates: bool = True, receipt_path: Optional[Path] = None) -> Workspace:
    """Create a fresh workspace.

    knowledge_from: a frozen dir; its skills/ and lessons/ seed the workspace (its skills/shared/ copy is
    restored unless shared_library names a version to stage instead).
    """
    permissions.check_phase(phase)
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"workspace already exists: {root}")
    for name in ("skills", "lessons", "scratch"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "instruction.md").write_text(instruction, encoding="utf-8")
    receipt: dict = {"phase": phase, "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                     "instruction_sha256": sha256_file(root / "instruction.md")}
    if knowledge_from:
        frozen = Path(knowledge_from)
        receipt["seed_knowledge"] = knowledge.seed_knowledge(root, frozen)
        manifest = read_json(frozen / "manifest.json", {}) or {}
        receipt["seed_knowledge"]["source_run"] = manifest.get("source_run")
    elif templates and phase == "practice":
        (root / "lessons" / "PLAN.md").write_text(PLAN_TEMPLATE, encoding="utf-8")
    if shared_library:
        receipt["shared_library"] = knowledge.stage_library(root, shared_library, root=library_root)
        if knowledge_from and (Path(knowledge_from) / knowledge.SHARED_REL).is_dir():
            receipt["shared_library"]["replaced_frozen_copy"] = True
    elif knowledge_from:
        restored = knowledge.restore_shared(root, Path(knowledge_from))
        if restored:
            receipt["shared_library"] = restored
    permissions.apply_permissions(root, phase)
    workspace = Workspace(root, phase, receipt)
    write_json(receipt_path or receipt_path_for(root), receipt)
    return workspace


def open_workspace(root: Union[str, Path], phase: Optional[str] = None) -> Workspace:
    root = Path(root)
    receipt = read_json(receipt_path_for(root), {}) or {}
    return Workspace(root, phase or receipt.get("phase", "practice"), receipt)

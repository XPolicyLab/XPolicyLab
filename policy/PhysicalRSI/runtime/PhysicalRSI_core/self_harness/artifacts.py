"""Frozen F/H closure, extracted from the reviewed v1 Self-Harness."""

from pathlib import Path

from PhysicalRSI_core.infra.storage import (
    digest,
    file_digest,
    identifier,
    relative_path,
)

CLOSURE = {
    "foundation",
    "skill_selection",
    "skills",
    "tools",
    "control",
    "prompts",
    "memory_rules",
    "dependencies",
    "weights",
    "assets",
    "configuration",
}


def verify_harness(h):
    """Verify all declared physical inputs, including referenced asset manifests.

    Closure completeness is independently attested by admission, not inferred
    from file extensions. Large assets may be covered by a pinned manifest, whose
    physical verification must be performed by the dependency_closure gate.
    """
    identifier(h["id"])
    source_root = Path(h["root"])
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("Harness root must be a physical directory")
    root = source_root.resolve()
    if set(h["components"]) != CLOSURE:
        raise ValueError("Incomplete F/H component declaration")
    for kind, entries in h["components"].items():
        if not isinstance(entries, dict) or not entries:
            raise ValueError(f"Empty harness component: {kind}")
        for name, sha in entries.items():
            declared = root / relative_path(name)
            if any(
                part.is_symlink()
                for part in [declared, *declared.parents]
                if part != root.parent
            ):
                raise ValueError("Harness inputs must be physical files")
            p = declared.resolve()
            if not p.is_relative_to(root) or not p.is_file() or file_digest(p) != sha:
                raise ValueError(f"Changed or missing harness input: {name}")
    return digest({"components": h["components"]})

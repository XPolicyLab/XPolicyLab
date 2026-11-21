"""Resolve the include list of an `exec` submission into (name, source) pairs.

Allowed entries, in the order given (duplicates keep their first position):
  skills/<name>.py          a task skill written by the agent
  skills/shared/<name>.py   a helper from the read-only shared library
  knowin_agent:<module>             a built-in library module (source text from knowin_agent/library/builtin/<module>.py,
                            or registered by the program layer with register_builtin)
Anything else (lessons, scratch, other suffixes, absolute paths, '..', links out of the workspace) is refused.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping, Optional, Union

from knowin_agent.workspace.permissions import relative_path

BUILTIN_PREFIX = "knowin_agent:"
BUILTIN_DIR = Path(__file__).resolve().parents[1] / "library" / "builtin"
MAX_INCLUDES = 32
MAX_INCLUDE_BYTES = 400_000
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_\-]*\.py")
MODULE_NAME = re.compile(r"[a-z][a-z0-9_]*")

Provider = Union[str, Path, Callable[[], str]]
_registry: dict[str, Provider] = {}
_lock = threading.Lock()


class IncludeError(ValueError):
    pass


def register_builtin(name: str, source: Provider) -> None:
    """Register `knowin_agent:<name>`: source text, a file path, or a zero-argument callable returning text."""
    key = name[len(BUILTIN_PREFIX):] if name.startswith(BUILTIN_PREFIX) else name
    if not MODULE_NAME.fullmatch(key):
        raise ValueError(f"invalid built-in module name: {name!r}")
    with _lock:
        _registry[key] = source


def builtin_modules(extra: Optional[Mapping[str, Provider]] = None) -> dict[str, Provider]:
    """Available built-in modules keyed without the `knowin_agent:` prefix."""
    found: dict[str, Provider] = {}
    if BUILTIN_DIR.is_dir():
        for path in sorted(BUILTIN_DIR.glob("*.py")):
            if MODULE_NAME.fullmatch(path.stem):
                found[path.stem] = path
    with _lock:
        found.update(_registry)
    for name, provider in (extra or {}).items():
        found[name[len(BUILTIN_PREFIX):] if name.startswith(BUILTIN_PREFIX) else name] = provider
    return found


def _builtin_source(provider: Provider) -> str:
    if callable(provider):
        return provider()
    if isinstance(provider, Path):
        return provider.read_text(encoding="utf-8")
    return str(provider)


def resolve_includes(workspace: Union[str, Path], include: Optional[Iterable[str]], *,
                     builtins: Optional[Mapping[str, Provider]] = None) -> list[tuple[str, str]]:
    if include is None:
        return []
    if isinstance(include, (str, bytes)) or not all(isinstance(i, str) for i in include):
        raise IncludeError("include must be a list of strings")
    entries = [i.strip() for i in include]
    if len(entries) > MAX_INCLUDES:
        raise IncludeError(f"at most {MAX_INCLUDES} includes per submission")
    root = Path(workspace)
    modules = None
    out, seen, total = [], set(), 0
    for entry in entries:
        if entry.startswith(BUILTIN_PREFIX):
            key = entry[len(BUILTIN_PREFIX):]
            modules = builtin_modules(builtins) if modules is None else modules
            if key not in modules:
                raise IncludeError(f"unknown built-in module {entry!r}; available: "
                                   f"{sorted(BUILTIN_PREFIX + m for m in modules) or 'none'}")
            name, source = BUILTIN_PREFIX + key, _builtin_source(modules[key])
        else:
            try:
                rel = relative_path(root, entry)
            except PermissionError as exc:
                raise IncludeError(str(exc)) from None
            parts = PurePosixPath(rel).parts
            allowed = (len(parts) == 2 and parts[0] == "skills") or \
                      (len(parts) == 3 and parts[:2] == ("skills", "shared"))
            if not allowed or not FILE_NAME.fullmatch(parts[-1]):
                raise IncludeError(f"cannot include {entry!r}: only skills/<name>.py, skills/shared/<name>.py "
                                   f"and knowin_agent:<module> are includable")
            path = root / rel
            if path.is_symlink() or not path.is_file():
                raise IncludeError(f"include not found: {rel}")
            name, source = rel, path.read_text(encoding="utf-8")
        if name in seen:
            continue
        seen.add(name)
        total += len(source.encode("utf-8"))
        if total > MAX_INCLUDE_BYTES:
            raise IncludeError(f"included sources exceed {MAX_INCLUDE_BYTES} bytes")
        out.append((name, source))
    return out

"""Load exact, content-addressed shared tool sources for a qualified skill.

Only the tool package gets a private import namespace. Public robot/transport
dependencies remain shared and must match the declared hashes; incompatible
dependency upgrades fail closed. This is version isolation, NOT a sandbox.
"""

import builtins
from copy import deepcopy
import importlib
import importlib.abc
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
import uuid

from orimapi.evaluation.storage import atomic_json, digest, file_digest

TOOL_PACKAGE = "orimapi.skills.code_tools"
ROOT = Path(__file__).resolve().parents[2]


def _relative(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError("Invalid frozen tool source path")
    if len(path.parts) == 1 and path.suffix == ".py":
        return Path("orimapi/skills/code_tools") / name
    if path.parts[0] == "dependencies" and path.suffix == ".py":
        rest = Path(*path.parts[1:])
        if len(path.parts) == 2:
            return Path("orimapi") / rest
        return Path("orimapi/skills/code_skills") / rest
    raise ValueError("Unsupported frozen tool source path")


def _identity(value):
    if (
        not isinstance(value, dict)
        or set(value) != {"sha256", "files"}
        or not isinstance(value["files"], dict)
        or not 8 <= len(value["files"]) <= 100
        or value["sha256"] != digest(value["files"])
    ):
        raise ValueError("Invalid frozen tool identity")
    for name, sha in value["files"].items():
        _relative(name)
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise ValueError("Invalid frozen source hash")
    if (
        not {"__init__.py", "controller.py", "prompt.py", "session.py", "recipes.py"}
        <= value["files"].keys()
    ):
        raise ValueError("Frozen tool snapshot lacks required execution sources")
    return deepcopy(value)


def export_snapshot(destination, native_snapshot, expected):
    """Copy the audited native-run snapshot, never today's mutable tool sources."""
    destination, native_snapshot = Path(destination), Path(native_snapshot)
    expected = _identity(expected)
    if json.loads((native_snapshot / "index.json").read_text()) != expected:
        raise ValueError("Native tool snapshot does not match qualified evidence")
    if destination.exists():
        snapshot = FrozenTools(destination / "index.json")
        if snapshot.identity != expected:
            raise ValueError("Refusing to overwrite a different shared tool version")
        return snapshot.index
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tool-version-", dir=destination.parent))
    try:
        for name, sha in expected["files"].items():
            source = native_snapshot / name
            if source.stat().st_size > 2_000_000 or file_digest(source) != sha:
                raise ValueError("Native tool source changed before publication")
            target = temporary / _relative(name)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        atomic_json(temporary / "index.json", expected)
        FrozenTools(temporary / "index.json").verify()
        temporary.rename(destination)
    finally:
        if temporary.exists():
            # Only our newly created unpublished temporary directory.
            shutil.rmtree(temporary)
    return destination / "index.json"


class _ExactSourceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """A unique namespace only; compile verified bytes, never cached .pyc files."""

    def __init__(self, owner):
        self.owner = owner

    def find_spec(self, fullname, path=None, target=None):
        prefix = self.owner.namespace
        if fullname != prefix and not fullname.startswith(prefix + "."):
            return None
        suffix = fullname[len(prefix) :].lstrip(".").replace(".", "/")
        folder = self.owner.package / suffix
        source = (
            folder / "__init__.py" if folder.is_dir() else folder.with_suffix(".py")
        )
        if source not in self.owner.tool_files:
            raise ImportError("Module is absent from the frozen tool version")
        return importlib.util.spec_from_file_location(
            fullname,
            source,
            loader=self,
            submodule_search_locations=[str(folder)] if folder.is_dir() else None,
        )

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.owner.verify()
        path = Path(module.__spec__.origin)
        content = path.read_bytes()
        if digest_source(content) != self.owner.tool_files[path]:
            raise ValueError("Frozen source changed before compilation")
        exec(compile(content, str(path), "exec"), module.__dict__)


def digest_source(content):
    import hashlib

    return hashlib.sha256(content).hexdigest()


class FrozenTools:
    def __init__(self, index):
        self.index = Path(index).resolve(strict=True)
        self.root = self.index.parent
        self.identity = _identity(json.loads(self.index.read_text()))
        self.package = self.root / "orimapi/skills/code_tools"
        self.files = {
            name: self.root / _relative(name) for name in self.identity["files"]
        }
        self.tool_files = {
            self.files[name]: sha
            for name, sha in self.identity["files"].items()
            if not name.startswith("dependencies/")
        }
        self.namespace = "_orimapi_frozen_tools_" + uuid.uuid4().hex
        self.loader = None
        self.verify()

    def verify(self):
        if json.loads(self.index.read_text()) != self.identity:
            raise ValueError("Frozen tool identity changed")
        for name, path in self.files.items():
            if not path.resolve().is_relative_to(self.root):
                raise ValueError("Frozen source escapes its version directory")
            if (
                path.stat().st_size > 2_000_000
                or file_digest(path) != self.identity["files"][name]
            ):
                raise ValueError("Frozen shared tool source changed: " + name)
            if name.startswith("dependencies/"):
                live = ROOT / _relative(name)
                if file_digest(live) != self.identity["files"][name]:
                    raise ValueError(
                        "Shared dependency no longer matches qualified tools: " + name
                    )

    def load(self):
        self.verify()
        if self.loader is None:
            self.loader = _ExactSourceLoader(self)
            sys.meta_path.insert(0, self.loader)
        try:
            return importlib.import_module(self.namespace)
        except BaseException:
            self.close()
            raise

    def importer(self, name, globals=None, locals=None, fromlist=(), level=0):
        if not level and (name == TOOL_PACKAGE or name.startswith(TOOL_PACKAGE + ".")):
            if not fromlist:
                raise ImportError("Frozen controllers must use explicit from-imports")
            self.load()
            return importlib.import_module(self.namespace + name[len(TOOL_PACKAGE) :])
        return builtins.__import__(name, globals, locals, fromlist, level)

    def close(self):
        if self.loader in sys.meta_path:
            sys.meta_path.remove(self.loader)
        for name in list(sys.modules):
            if name == self.namespace or name.startswith(self.namespace + "."):
                sys.modules.pop(name, None)
        self.loader = None

"""Run the unchanged official client with scoped physical layout selection."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import runpy
import stat
import sys

try:
    from .native_layout import NativeSelection, bind_native_environment
except ImportError:  # Executed as a file by the official shell launcher.
    from native_layout import NativeSelection, bind_native_environment


TARGET = "src.eval_client.eval_env"


def wrap_eval_module(module, selection):
    original = module.create_eval_env

    def create_eval_env(config, app, resume_state=None, **kwargs):
        env = original(config, app, resume_state=resume_state, **kwargs)
        return bind_native_environment(env, selection, resume_state=resume_state)

    module.create_eval_env = create_eval_env


class _Loader(importlib.abc.Loader):
    def __init__(self, delegate, selection):
        self.delegate, self.selection = delegate, selection

    def create_module(self, spec):
        create = getattr(self.delegate, "create_module", None)
        return create(spec) if create else None

    def exec_module(self, module):
        self.delegate.exec_module(module)
        wrap_eval_module(module, self.selection)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, selection):
        self.selection = selection

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, self.selection)
        return spec


NATIVE_KIT_EXTENSION_ENV = "ROBORSI_NATIVE_KIT_EXTENSION_PARENT"
NATIVE_KIT_PORTABLE_ENV = "ROBORSI_NATIVE_KIT_PORTABLE_ROOT"
NATIVE_KIT_EXTENSION_CONFIGS = {
    "isaacsim.asset.importer.urdf-2.4.31+107.3.3.lx64.r.cp311":
        "1984a4ecf3134745215638420e449c197083e97a3932901656479a7677f9a3fb",
    "omni.kit.pip_archive-d38fa9ecd1fb6df4":
        "824447f8269824374116ec9ee8f12f8e83a5bca49f85b2a564b1f9678a6a541d",
}


def _check_native_extension_config(root, expected_sha256, error):
    """Accept only the preserved public extension metadata at an owned real root."""
    for directory in (root, root / "config"):
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise error("native extension metadata requires owned real directories")
    config = root / "config/extension.toml"
    info = config.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid():
        raise error("native extension config must be an owned regular file")
    if hashlib.sha256(config.read_bytes()).hexdigest() != expected_sha256:
        raise error("native extension metadata differs from the preserved source")


def _native_kit_arguments(native_args):
    """Forward opt-in SDK discovery and per-run storage through official CLI."""
    additions = []
    requested = os.environ.get(NATIVE_KIT_EXTENSION_ENV)
    if requested is not None:
        parent = Path(requested)
        if (not parent.is_absolute() or ".." in parent.parts
                or any(character.isspace() for character in requested) or "\0" in requested):
            raise ValueError("native extension parent must be an absolute path without whitespace")
        if parent.resolve(strict=True) != parent:
            raise ValueError("native extension parent must not traverse symbolic links")
        info = parent.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
            raise ValueError("native extension parent must be an owned real directory")
        for name, expected in NATIVE_KIT_EXTENSION_CONFIGS.items():
            _check_native_extension_config(parent / name, expected, ValueError)
        additions.append(("--ext-folder", parent))

    requested = os.environ.get(NATIVE_KIT_PORTABLE_ENV)
    if requested is not None:
        state_value = os.environ.get("ROBORSI_RUNTIME_STATE_DIR")
        if state_value is None:
            raise ValueError("native Kit storage requires ROBORSI_RUNTIME_STATE_DIR")
        root, state = Path(requested), Path(state_value)
        for path in (root, state):
            if (not path.is_absolute() or ".." in path.parts
                    or any(character.isspace() for character in str(path)) or "\0" in str(path)):
                raise ValueError("native Kit storage requires clean absolute paths without whitespace")
        if state.name != "state" or root != state.parent / "cache/native-kit":
            raise ValueError("native Kit storage must use RUN_ROOT/cache/native-kit")
        for directory in (state.parent, state, root.parent, root):
            if directory.resolve(strict=True) != directory:
                raise ValueError("native Kit storage must not traverse symbolic links")
            info = directory.lstat()
            if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
                raise ValueError("native Kit storage requires owned real directories")
        additions.append(("--portable-root", root))
    if not additions:
        return native_args

    # Official main uses argparse.parse_args(), so raw Kit-only options cannot
    # be appended to its argv. AppLauncher forwards the last --kit_args value.
    effective = None
    for index, argument in enumerate(native_args):
        if argument == "--kit_args":
            if index + 1 >= len(native_args):
                raise ValueError("--kit_args requires a value")
            effective = (index + 1, native_args[index + 1], False)
        elif argument.startswith("--kit_args="):
            effective = (index, argument.split("=", 1)[1], True)
    existing = effective[1] if effective is not None else ""
    tokens = existing.split()  # Exactly the official AppLauncher's splitting.
    extra = []
    for flag, path in additions:
        present = False
        for index, argument in enumerate(tokens):
            value = None
            if argument == flag:
                if index + 1 >= len(tokens) or tokens[index + 1].startswith("--"):
                    raise ValueError(flag + " requires a path in --kit_args")
                value = tokens[index + 1]
            elif argument.startswith(flag + "="):
                if flag == "--portable-root":
                    raise ValueError("--portable-root requires the separate documented path token")
                value = argument.split("=", 1)[1]
                if not value:
                    raise ValueError(flag + " requires a path in --kit_args")
            if value is not None:
                if Path(value) == path:
                    present = True
                elif flag == "--portable-root":
                    raise ValueError("--kit_args conflicts with this run's owned Kit storage")
        if not present:
            extra.extend([flag, str(path)])
    if not extra:
        return native_args
    combined = existing + (" " if existing else "") + " ".join(extra)
    result = list(native_args)
    if effective is None:
        result.extend(["--kit_args", combined])
    else:
        index, _, equals = effective
        result[index] = ("--kit_args=" if equals else "") + combined
    return result


def _open_native_diagnostics():
    """Open an opt-in infrastructure-only stack sink inside this run's state."""
    requested = os.environ.get("ROBORSI_NATIVE_DIAGNOSTICS_DIR")
    if not requested:
        return None
    state_value = os.environ.get("ROBORSI_RUNTIME_STATE_DIR")
    if not state_value:
        raise ValueError("Native diagnostics requires ROBORSI_RUNTIME_STATE_DIR")
    state = Path(state_value)
    destination = Path(requested)
    if (not state.is_absolute() or not destination.is_absolute()
            or ".." in state.parts or ".." in destination.parts
            or state.name != "state"
            or destination != state / "native-infrastructure"):
        raise ValueError("Native diagnostics must use RUN_ROOT/state/native-infrastructure")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    opened = []
    try:
        root_fd = os.open(str(state.parent), directory_flags)
        opened.append(root_fd)
        state_fd = os.open("state", directory_flags, dir_fd=root_fd)
        opened.append(state_fd)
        sink_fd = os.open("native-infrastructure", directory_flags, dir_fd=state_fd)
        opened.append(sink_fd)
        if any(os.fstat(fd).st_uid != os.geteuid() for fd in opened):
            raise ValueError("Native diagnostics directories must belong to the current user")
        # execv preserves the PID. Keep each predecessor's stack file intact.
        for generation in range(10000):
            suffix = "" if generation == 0 else "-" + str(generation)
            name = "python-stack-" + str(os.getpid()) + suffix + ".log"
            try:
                fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                             | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=sink_fd)
            except FileExistsError:
                continue
            try:
                os.fchmod(fd, 0o600)
                return os.fdopen(fd, "wb", buffering=0)
            except BaseException:
                os.close(fd)
                raise
        raise RuntimeError("Native diagnostics exhausted its per-process file generations")
    finally:
        for fd in reversed(opened):
            os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-entry", required=True)
    parser.add_argument("--layout-id", type=int)
    parser.add_argument("native_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    native_args = args.native_args
    if native_args[:1] == ["--"]:
        native_args = native_args[1:]
    native_args = _native_kit_arguments(native_args)
    native = argparse.ArgumentParser(add_help=False)
    native.add_argument("--task_name", required=True)
    native.add_argument("--seed", required=True, type=int)
    identity, _ = native.parse_known_args(native_args)
    selection = NativeSelection.from_launch(identity.task_name, identity.seed, args.layout_id)
    if selection.run_id and not os.environ.get("ROBODOJO_RUN_ID"):
        os.environ["ROBODOJO_RUN_ID"] = selection.run_id
    if selection.layout is not None:
        # Native main compares its requested cumulative result count with the
        # existing manifest before asking SeedManager for pending scenes. The
        # exact one-scene queue bounds this invocation; a stale EVAL_NUM=1 must
        # not suppress an interrupted layout after other layouts were scored.
        os.environ["EVAL_NUM"] = "native"
        os.environ["ROBORSI_NATIVE_LAYOUT_ID"] = str(selection.layout)
    entry = Path(args.official_entry)
    if not entry.is_file():
        parser.error("Official evaluation entry is unavailable")
    finder = _Finder(selection)
    if TARGET in sys.modules:
        raise RuntimeError("Native EvalEnv was imported before scoped layout hook")
    sys.meta_path.insert(0, finder)
    previous_argv = sys.argv
    previous_path = list(sys.path)
    sys.path.insert(0, str(entry.resolve().parent))
    original_execv = os.execv

    def restart(executable, restart_args):
        if len(restart_args) >= 2 and Path(restart_args[1]).resolve() == entry.resolve():
            restart_args = [restart_args[0], "-u", str(Path(__file__).resolve()),
                            "--official-entry", str(entry), "--", *restart_args[2:]]
        return original_execv(executable, restart_args)

    os.execv = restart
    sys.argv = [str(entry), *native_args]
    diagnostics = None
    try:
        diagnostics = _open_native_diagnostics()
        if diagnostics is not None:
            faulthandler.dump_traceback_later(120, repeat=True, file=diagnostics, exit=False)
        # run_path preserves official imports, CLI, scoring and resume handling.
        runpy.run_path(str(entry), run_name="__main__")
    finally:
        if diagnostics is not None:
            faulthandler.cancel_dump_traceback_later()
            diagnostics.close()
        sys.argv = previous_argv
        sys.path[:] = previous_path
        os.execv = original_execv
        sys.meta_path.remove(finder)


if __name__ == "__main__":
    main()

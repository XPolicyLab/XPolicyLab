"""Version-compatible Codex launch prefix; mandatory isolation flags stay explicit."""
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess

DISABLED_FEATURES = (
    'shell_tool', 'unified_exec', 'shell_snapshot', 'memories', 'apps', 'plugins',
    'browser_use', 'computer_use', 'view_image', 'multi_agent', 'multi_agent_v2',
    'code_mode', 'browser_use_external', 'browser_use_full_cdp_access',
    'in_app_browser', 'image_generation',
)


@lru_cache(maxsize=16)
def _global_options(executable, file_identity):
    # Include file identity in the cache key so an in-place CLI upgrade is probed
    # again. Help performs no provider request or credential operation.
    try:
        result = subprocess.run(
            [executable, "--help"], stdin=subprocess.DEVNULL, capture_output=True,
            text=True, timeout=15, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Cannot inspect Codex CLI capabilities using --help") from exc
    if result.returncode:
        raise RuntimeError(f"Codex --help failed with exit status {result.returncode}")
    return ("--no-daemon",) if re.search(r"(?:^|\s)--no-daemon(?:\s|=|$)", result.stdout) else ()


def exec_prefix(binary):
    """Return [binary, optional --no-daemon, exec] using cached CLI help.

    Some supported Codex builds have no daemon option and reject --no-daemon.
    This is the only optional flag: callers retain every isolation/config flag.
    """
    executable = Path(shutil.which(str(binary)) or str(binary)).expanduser().resolve()
    try:
        info = executable.stat()
    except OSError as exc:
        raise RuntimeError("Configured Codex executable is unavailable: " + str(binary)) from exc
    identity = (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)
    return [str(binary), *_global_options(str(executable), identity), "exec"]

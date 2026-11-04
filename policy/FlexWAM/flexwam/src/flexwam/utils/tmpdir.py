"""Choose TMPDIR for DataLoader multiprocessing listener dirs.

Workers create tiny `pymp-*` dirs here (locks/sockets for fd handoff), not
videos or checkpoints. Override with `FLEXWAM_TMPDIR`.
"""

from __future__ import annotations

import errno
import os
from pathlib import Path


_HOME_TMP = Path.home() / "tmp"


def resolve_tmpdir() -> Path:
    explicit = os.environ.get("FLEXWAM_TMPDIR")
    if explicit:
        path = Path(explicit).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    _HOME_TMP.mkdir(parents=True, exist_ok=True)
    return _HOME_TMP


def _patch_multiprocessing_temp_cleanup() -> None:
    try:
        import multiprocessing.util as mp_util
    except Exception:
        return

    def _remove_temp_dir(rmtree, tempdir):
        ignored = {errno.EBUSY, errno.ENOTEMPTY, errno.ENOENT}

        def _onerror(func, path, exc_info):
            err = exc_info[1]
            if isinstance(err, OSError) and err.errno in ignored:
                return
            raise err

        try:
            rmtree(tempdir, onerror=_onerror)
        except OSError as err:
            if err.errno not in ignored:
                raise
        current_process = mp_util.process.current_process()
        if current_process is not None:
            current_process._config["tempdir"] = None

    mp_util._remove_temp_dir = _remove_temp_dir


def configure_tmpdir() -> Path:
    tmpdir = resolve_tmpdir()
    os.environ["TMPDIR"] = str(tmpdir)
    os.environ["TEMP"] = str(tmpdir)
    os.environ["TMP"] = str(tmpdir)
    os.environ.setdefault("MPLCONFIGDIR", str(tmpdir / "matplotlib-flexwam"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    _patch_multiprocessing_temp_cleanup()
    return tmpdir

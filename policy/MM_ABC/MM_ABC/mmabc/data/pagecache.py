"""Optional filesystem page-cache management for large training datasets.

Dropping cached pages releases only clean file-backed pages; unsupported
platforms and disabled settings leave the cache unchanged."""

from __future__ import annotations

import os

_ENABLED = os.environ.get("MMABC_FADVISE_DONTNEED", "0").lower() not in (
    "0",
    "false",
    "no",
)

_HAS_FADVISE = hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED")


def drop_page_cache(path: str) -> None:
    """Best-effort ``posix_fadvise(DONTNEED)`` over the whole file at ``path``.

    Opens a throwaway read-only fd (page cache is keyed on the inode, not the
    fd, so this evicts the pages even while another open handle -- e.g. a pooled
    PyAV container -- keeps the file open). Silent on any error.
    """
    if not (_ENABLED and _HAS_FADVISE):
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        # offset=0, length=0 means "from here to end of file".
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except OSError:
        pass
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

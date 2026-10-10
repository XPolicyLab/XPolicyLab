#!/usr/bin/env python3
"""Fail early when Isaac Sim's host OpenGL runtime is unavailable."""

from __future__ import annotations

import ctypes
import sys


def main() -> int:
    failures: list[str] = []
    for library in ("libGLU.so.1", "libOpenGL.so.0"):
        try:
            ctypes.CDLL(library)
        except OSError as error:
            failures.append(f"{library}: {error}")

    if not failures:
        return 0

    print(
        "[starVLA HF][ISAAC][ERROR] required host OpenGL libraries are not "
        "loadable from the RoboDojo environment:",
        file=sys.stderr,
    )
    for failure in failures:
        print(f"[starVLA HF][ISAAC][ERROR]   {failure}", file=sys.stderr)
    print(
        "[starVLA HF][ISAAC][ERROR] Install compatible system OpenGL/GLU "
        "runtime packages (or configure an administrator-provided "
        "LD_LIBRARY_PATH) before evaluation.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

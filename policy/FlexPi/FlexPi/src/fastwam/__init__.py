"""Alias package: resolves legacy `fastwam.*` config targets onto FlexPi.

FastWAM-lineage checkpoints carry a `config.yaml` whose `_target_`s point at
`fastwam.*`. This package exists only so those strings resolve, leaving the
checkpoint and its config untouched. Every module here is a re-export of the
FlexPi implementation — no FastWAM source is vendored or imported, and deleting
the FastWAM codebase from the machine changes nothing.

See `flexpi/_legacy_compat.py` for what is translated and what is refused.

NOTE: a real `fastwam` package may also exist on a developer machine. Which one
wins is a sys.path question, so the compat entry point logs the line
"[FlexPi compat] legacy FastWAM config -> flexpi.runtime.create_flexpi"
whenever it runs. If you do not see it, the real FastWAM was imported instead.
"""

"""Strict recursive composition for WorldScape YAML configurations."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, ListConfig, OmegaConf


class ConfigCompositionError(ValueError):
    """Raised when an ``includes`` graph is invalid."""


def load_composed_config(path: str | Path) -> DictConfig:
    """Load a mapping config and recursively merge its top-level includes.

    Include paths are relative to the file that declares them. Includes are
    merged in listed order and the declaring file is applied last.
    """

    return _Composer().load(Path(path))


class _Composer:
    def __init__(self) -> None:
        self._seen: dict[Path, Path | None] = {}
        self._stack: list[Path] = []

    def load(self, path: Path) -> DictConfig:
        return self._load(path, declared_by=None)

    def _load(self, path: Path, *, declared_by: Path | None) -> DictConfig:
        candidate = path if path.is_absolute() else (
            declared_by.parent / path if declared_by is not None else path
        )
        resolved = candidate.expanduser().resolve()
        if not resolved.exists():
            context = f" (included by {declared_by})" if declared_by else ""
            raise FileNotFoundError(
                f"Configuration file does not exist: {resolved}{context}"
            )
        if not resolved.is_file():
            context = f" (included by {declared_by})" if declared_by else ""
            raise ConfigCompositionError(
                f"Configuration path is not a file: {resolved}{context}"
            )
        if resolved in self._stack:
            start = self._stack.index(resolved)
            chain = self._stack[start:] + [resolved]
            raise ConfigCompositionError(
                "Config include cycle detected: "
                + " -> ".join(str(item) for item in chain)
            )
        if resolved in self._seen:
            first = self._seen[resolved]
            first_context = str(first) if first is not None else "<root>"
            current_context = str(declared_by) if declared_by is not None else "<root>"
            raise ConfigCompositionError(
                f"Duplicate config include {resolved}; first declared by "
                f"{first_context}, declared again by {current_context}"
            )

        self._seen[resolved] = declared_by
        self._stack.append(resolved)
        try:
            raw = OmegaConf.load(resolved)
            if not isinstance(raw, DictConfig):
                raise ConfigCompositionError(
                    f"Configuration root must be a mapping: {resolved}"
                )
            includes = raw.get("includes", [])
            if not isinstance(includes, (list, tuple, ListConfig)):
                raise ConfigCompositionError(
                    f"Top-level 'includes' must be a list in {resolved}"
                )

            merged = OmegaConf.create({})
            for index, include in enumerate(includes):
                if not isinstance(include, str) or not include:
                    raise ConfigCompositionError(
                        f"includes[{index}] must be a non-empty path string in "
                        f"{resolved}"
                    )
                merged = OmegaConf.merge(
                    merged,
                    self._load(Path(include), declared_by=resolved),
                )

            overlay = OmegaConf.create(OmegaConf.to_container(raw, resolve=False))
            if "includes" in overlay:
                del overlay["includes"]
            result = OmegaConf.merge(merged, overlay)
            if not isinstance(result, DictConfig):
                raise AssertionError("OmegaConf mapping merge returned a non-mapping")
            return result
        finally:
            self._stack.pop()


__all__ = ["ConfigCompositionError", "load_composed_config"]

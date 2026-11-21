# Copyright 2026 The HEART Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit filesystem skill loading for runtime agents."""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from roborsi.data_convert import TextDocument


class _SkillTooLarge(ValueError):
    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        super().__init__(f"skill resource is {size_bytes} bytes; limit is {max_bytes}")


def load_skill(path: str | Path, *, max_bytes: int | None = None) -> TextDocument:
    """Read one skill file as UTF-8, preserving its content and resolved source.

    Args:
        path: Explicit Markdown file path. SKILL.md uses its parent directory
            name as the title; other files use their filename stem.
        max_bytes: Optional bound for on-demand resource reads.

    Returns:
        A document that can be supplied to ``convert_planner_input(skills=...)``.

    File access and UTF-8 decoding errors propagate to the caller.
    """
    skill = Path(path)
    if max_bytes is None:
        text = skill.read_text(encoding="utf-8")
    else:
        with skill.open("rb") as stream:
            data = stream.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise _SkillTooLarge(skill.stat().st_size, max_bytes)
        text = data.decode("utf-8")
    return TextDocument(
        title=skill.parent.name if skill.name == "SKILL.md" else skill.stem,
        text=text,
        source=str(skill.resolve()),
    )


@dataclass(frozen=True)
class SkillEntry:
    """Public metadata and the explicitly authorized resource directory."""

    name: str
    description: str
    directory: Path


class SkillCatalog:
    """Index explicit skill directories and read their text resources on demand.

    Args:
        paths: Directories containing SKILL.md; no recursive discovery occurs.
        max_bytes: Maximum UTF-8 resource size, enforced before returning content.
    """

    def __init__(self, paths: Sequence[str | Path], *, max_bytes: int = 262144):
        import yaml

        if (
            isinstance(max_bytes, bool)
            or not isinstance(max_bytes, int)
            or max_bytes < 1
        ):
            raise ValueError("skill max_bytes must be a positive integer")
        self.max_bytes = max_bytes
        self.entries: dict[str, SkillEntry] = {}
        for path in paths:
            directory = Path(path).resolve(strict=True)
            if not directory.is_dir():
                raise ValueError(f"skill path must be a directory: {directory}")
            source = (directory / "SKILL.md").resolve(strict=True)
            if not source.is_relative_to(directory):
                raise ValueError(f"SKILL.md is outside skill directory: {directory}")
            text = load_skill(source).text
            metadata: dict[str, Any] = {}
            body = text
            lines = text.splitlines()
            if lines and lines[0].strip() == "---":
                end = next(
                    (
                        index
                        for index, line in enumerate(lines[1:], 1)
                        if line.strip() == "---"
                    ),
                    None,
                )
                if end is None:
                    raise ValueError(f"unclosed skill frontmatter: {source}")
                try:
                    metadata = yaml.safe_load("\n".join(lines[1:end])) or {}
                except yaml.YAMLError as exc:
                    raise ValueError(f"invalid skill frontmatter: {source}") from exc
                if not isinstance(metadata, dict):
                    raise ValueError(f"skill frontmatter must be a mapping: {source}")
                body = "\n".join(lines[end + 1 :])
            name = metadata.get("name", directory.name)
            description = metadata.get("description") or next(
                (
                    line.strip().lstrip("# ")
                    for line in body.splitlines()
                    if line.strip()
                ),
                name,
            )
            if not isinstance(name, str) or not name.strip() or name != name.strip():
                raise ValueError(f"skill name must be non-empty text: {source}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"skill description must be non-empty text: {source}")
            if name in self.entries:
                raise ValueError(f"duplicate skill name: {name}")
            self.entries[name] = SkillEntry(name, description.strip(), directory)

    @property
    def instructions(self) -> str:
        """Render only skill names and descriptions, without skill bodies."""
        if not self.entries:
            return ""
        return (
            "Available skills (use read_skill to read a skill or its resources):\n"
            + "\n".join(
                f"- {entry.name}: {entry.description}"
                for entry in self.entries.values()
            )
        )

    def read_skill(self, name: str, resource: str = "SKILL.md") -> dict[str, Any]:
        """Read a UTF-8 file within the named skill's explicit directory.

        Args:
            name: A name from the available skills index.
            resource: Relative text-file path inside that skill directory.

        Returns:
            Content, resolved source and SHA-256, or a structured read error.
        """
        result: dict[str, Any] = {"name": name, "resource": resource}
        entry = self.entries.get(name)
        if entry is None:
            return {**result, "error": {"code": "unknown_skill"}}
        try:
            path = (entry.directory / resource).resolve()
            if Path(resource).is_absolute() or not path.is_relative_to(entry.directory):
                return {**result, "error": {"code": "outside_skill"}}
            document = load_skill(path, max_bytes=self.max_bytes)
        except _SkillTooLarge as exc:
            return {
                **result,
                "error": {
                    "code": "too_large",
                    "size_bytes": exc.size_bytes,
                    "max_bytes": exc.max_bytes,
                },
            }
        except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
            return {**result, "error": {"code": "read_error", "message": str(exc)}}
        return {
            **result,
            "content": document.text,
            "source": document.source,
            "sha256": hashlib.sha256(document.text.encode("utf-8")).hexdigest(),
        }

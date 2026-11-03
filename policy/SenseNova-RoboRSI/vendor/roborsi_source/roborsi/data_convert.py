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

"""Pure conversion of caller-provided inputs to planner messages."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai import BinaryContent


@dataclass(frozen=True)
class TextDocument:
    """Resolved text and its provenance, used as memory or a skill.

    Attributes:
        title: Human-readable name used in the context section heading.
        text: Document content, already selected by the caller.
        source: Optional source identifier. Assembly never reads this location.
    """

    title: str
    text: str
    source: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.title, str) or not isinstance(self.text, str):
            raise TypeError("context document title and text must be strings")
        if self.source is not None and not isinstance(self.source, str):
            raise TypeError("context document source must be a string or None")


@dataclass(frozen=True)
class PlannerInput:
    """Separate context sources with projections for the existing planners.

    Use :func:`convert_planner_input` to snapshot and validate input collections.
    The projections preserve the current planner protocol; SDK adapters still
    own their message-role mapping. History and tool schemas remain owned by
    the planner and toolkit, respectively.
    """

    prompt: str
    query: str
    memory: tuple[TextDocument, ...] = ()
    skills: tuple[TextDocument, ...] = ()
    initial_context: tuple[str | BinaryContent, ...] = ()

    @property
    def system_prompt(self) -> str:
        """Render instructions and skills without including memory or observations."""
        return "\n\n".join(
            [
                self.prompt,
                *(f"## Skill: {doc.title}\n\n{doc.text}" for doc in self.skills),
            ]
        )

    @property
    def user_message(self) -> str | list[str | BinaryContent]:
        """Render the query and memory, followed by the initial content parts.

        A fresh list is returned when initial content is present. Image objects
        are passed through, preserving their bytes, media type, and metadata.
        """
        sections = [self.query]
        for doc in self.memory:
            source = f"\nSource: {doc.source}" if doc.source is not None else ""
            sections.append(f"## Memory: {doc.title}{source}\n\n{doc.text}")
        message = "\n\n".join(sections)
        if self.initial_context:
            return [message, *self.initial_context]
        return message


def convert_planner_input(
    *,
    prompt: str,
    query: str,
    memory: Sequence[TextDocument] = (),
    skills: Sequence[TextDocument] = (),
    initial_context: Sequence[str | BinaryContent] = (),
) -> PlannerInput:
    """Assemble resolved context without filesystem, network, or model access.

    Args:
        prompt: Rendered system instructions, including any robot-specific rules.
        query: The current task or continuation message.
        memory: Already selected, authorized memory excerpts. They appear in
            user context, separately from system instructions.
        skills: Resolved skill documents, appended to the system instructions
            in caller order. Read filesystem skills with runtime.skills.load_skill.
        initial_context: Text and BinaryContent observations after the query
            and memory. Planner-specific multimodal limits still apply.

    Returns:
        A bundle retaining the separate sources and their original text.

    Raises:
        TypeError: If a context input has an unsupported type.
    """
    if not isinstance(prompt, str) or not isinstance(query, str):
        raise TypeError("prompt and query must be strings")
    memory_docs = tuple(memory)
    skill_docs = tuple(skills)
    for name, documents in (("memory", memory_docs), ("skills", skill_docs)):
        if any(not isinstance(doc, TextDocument) for doc in documents):
            raise TypeError(f"{name} entries must be TextDocument")
    parts = tuple(initial_context)
    if any(not isinstance(part, str) for part in parts):
        from pydantic_ai import BinaryContent

        if any(not isinstance(part, (str, BinaryContent)) for part in parts):
            raise TypeError("initial_context parts must be text or BinaryContent")
    return PlannerInput(
        prompt=prompt,
        query=query,
        memory=memory_docs,
        skills=skill_docs,
        initial_context=parts,
    )

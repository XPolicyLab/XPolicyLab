"""The conversation sent to the model, and its picture window.

Pictures travel as JPEG data URIs inside ``user`` messages (a ``tool`` message cannot carry
them), each after a short caption so the model knows what it is looking at.  The gateway takes
8-10 MB per request, so only a window of pictures stays in the conversation: the task card's
pictures are pinned; when the live ones pass ``images_max`` or their data URIs pass ``bytes_max``
(counted as sent: base64 is 4/3 of the JPEG), the oldest unpinned are replaced by a placeholder
until ``images_low`` and 70% of ``bytes_max`` remain.  Dropping several at once means the cached
prefix of the conversation is invalidated less often.

Claude's thinking blocks (kept in a reply's ``_blocks``) are bound to everything before them, and
the API refuses one sent back after an earlier message changed.  So when pictures are dropped, the
replies after the first changed message lose their thinking blocks; their text and tool calls stay.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..vision.overlay import jpeg

THINKING = ("thinking", "redacted_thinking")


@dataclass
class Picture:
    number: int
    caption: str
    data: bytes  # JPEG
    pinned: bool = False
    kind: str = "look"
    file: str | None = None  # where the record keeps it

    @property
    def url(self) -> str:
        return "data:image/jpeg;base64," + base64.b64encode(self.data).decode("ascii")

    @property
    def wire_size(self) -> int:
        """len(self.url), without encoding it."""
        return 23 + 4 * ((len(self.data) + 2) // 3)


class Conversation:
    def __init__(self, system: str, *, images_max: int = 90, images_low: int = 60,
                 bytes_max: int = 10_000_000, quality: int = 90):
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.images_max = images_max
        self.images_low = min(images_low, images_max)
        self.bytes_max = bytes_max
        self.quality = quality
        self.pictures: dict[int, Picture] = {}  # number -> picture (all, for the record)
        self._live: list[int] = []  # numbers of pictures still in the conversation, oldest first
        self._count = 0
        self.dropped = 0
        self.thinking_cleared = 0  # thinking blocks left out because pictures before them were dropped

    # -- building ----------------------------------------------------------------------------------
    def picture(self, image: np.ndarray, caption: str, *, pinned: bool = False, kind: str = "look") -> Picture:
        self._count += 1
        pic = Picture(self._count, caption, jpeg(image, self.quality), pinned=pinned, kind=kind)
        self.pictures[pic.number] = pic
        return pic

    def user(self, text: str, pictures: list[Picture] = ()) -> None:
        if not pictures:
            self.messages.append({"role": "user", "content": text})
            return
        parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for pic in pictures:
            parts.append({"type": "text", "text": f"[picture {pic.number}: {pic.caption}]"})
            parts.append({"type": "image_url", "image_url": {"url": pic.url}, "_picture": pic.number})
            self._live.append(pic.number)
        self.messages.append({"role": "user", "content": parts})

    def assistant(self, text: str, calls: list[dict[str, Any]], blocks: list[dict[str, Any]] | None = None,
                  reasoning: str | None = None) -> None:
        """``blocks``: the reply's content blocks from an Anthropic gateway (thinking with its signature,
        text, tool calls), sent back unchanged with every later request.  ``reasoning``: the reply's
        thinking from an OpenAI-compatible gateway, sent back as the message's ``reasoning``."""
        message: dict[str, Any] = {"role": "assistant", "content": text or ""}
        if reasoning:
            message["reasoning"] = reasoning
        if calls:
            message["tool_calls"] = calls
        if blocks is not None:
            message["_blocks"] = blocks
        self.messages.append(message)

    def tool(self, call_id: str, text: str) -> None:
        self.messages.append({"role": "tool", "tool_call_id": call_id, "content": text})

    # -- the window --------------------------------------------------------------------------------
    def live_counts(self) -> tuple[int, int]:
        """(pictures, bytes of their data URIs) still in the conversation."""
        return len(self._live), sum(self.pictures[n].wire_size for n in self._live)

    def trim(self) -> list[int]:
        """Drop the oldest unpinned pictures when the window is over its limits; their numbers."""
        count, size = self.live_counts()
        if count <= self.images_max and size <= self.bytes_max:
            return []
        low_bytes = int(0.7 * self.bytes_max)
        drop: list[int] = []
        for number in list(self._live):
            if count <= self.images_low and size <= low_bytes:
                break
            pic = self.pictures[number]
            if pic.pinned:
                continue
            drop.append(number)
            count -= 1
            size -= pic.wire_size
        if drop:
            gone = set(drop)
            self._live = [n for n in self._live if n not in gone]
            first = len(self.messages)  # the first message that changes
            for i, message in enumerate(self.messages):
                content = message.get("content")
                if isinstance(content, list) and any(p.get("_picture") in gone for p in content):
                    message["content"] = [self._placeholder(p) if p.get("_picture") in gone else p for p in content]
                    first = min(first, i)
            self.thinking_cleared += self._clear_thinking(after=first)
            self.dropped += len(drop)
        return drop

    def _clear_thinking(self, after: int) -> int:
        """Leave out the thinking blocks of the replies after message ``after``; how many there were."""
        cleared = 0
        for message in self.messages[after + 1:]:
            blocks = message.get("_blocks")
            if blocks:
                kept = [b for b in blocks if b.get("type") not in THINKING]
                cleared += len(blocks) - len(kept)
                message["_blocks"] = kept
        return cleared

    def _placeholder(self, part: dict[str, Any]) -> dict[str, Any]:
        pic = self.pictures[part["_picture"]]
        return {"type": "text", "text": f"[picture {pic.number} removed to keep the conversation small; "
                                        "call look() again for a current one]"}

    def wire(self) -> list[dict[str, Any]]:
        """The messages as the gateway takes them (without our bookkeeping keys)."""
        out = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                message = dict(message)
                message["content"] = [{k: v for k, v in p.items() if not k.startswith("_")} for p in content]
            out.append(message)
        return out

    def for_record(self) -> list[dict[str, Any]]:
        """The messages with pictures replaced by references to their files."""
        out = []
        for message in self.messages:
            content = message.get("content")
            if isinstance(content, list):
                message = dict(message)
                parts = []
                for p in content:
                    if p.get("type") == "image_url":
                        pic = self.pictures[p["_picture"]]
                        parts.append({"type": "picture", "number": pic.number, "file": pic.file})
                    else:
                        parts.append({k: v for k, v in p.items() if not k.startswith("_")})
                message["content"] = parts
            out.append(message)
        return out

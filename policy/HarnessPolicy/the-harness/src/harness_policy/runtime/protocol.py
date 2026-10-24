"""The high-level reply format: a short thought and exactly one action block."""

from dataclasses import dataclass
import re


KINDS = ("python", "verify", "finish")
DIRECT_KINDS = ("action", "finish")
_FENCE = re.compile(r"^\s*```[a-zA-Z0-9_-]*\s*\n(.*?)\n\s*```\s*$", re.DOTALL)
# DeepSeek models sometimes end a block with their own tool-call markup instead of its
# closing tag ("</\uff5cDSML\uff5c parameter>", "</\uff5cDSML\uff5c invoke>", ...): 50 of
# agent 1.4.0's 677 development replies, each a call that ran nothing. Lines of it are
# dropped, and a block it left open is closed at the end of the reply.
_LEAKED = re.compile(r"^\s*</?\s*\uff5cDSML\uff5c[^>\n]*>\s*$", re.MULTILINE)


class ProtocolError(ValueError):
    pass


@dataclass
class Reply:
    thought: str
    kind: str
    body: str


def _without_leaked_markup(text: str, kinds) -> str:
    if "\uff5cDSML\uff5c" not in text:
        return text
    text = _LEAKED.sub("", text).rstrip()
    for kind in kinds:
        if text.count(f"<{kind}>") == 1 and f"</{kind}>" not in text:
            return text + f"\n</{kind}>"
    return text


def parse_reply(text: str, kinds=KINDS) -> Reply:
    text = _without_leaked_markup(text or "", kinds)
    pattern = re.compile(r"<(%s)>(.*?)</\1>" % "|".join(kinds), re.DOTALL)
    blocks = pattern.findall(text)
    if len(blocks) != 1:
        names = ", ".join(f"<{kind}>" for kind in kinds)
        raise ProtocolError(f"Reply must contain exactly one of {names}; found {len(blocks)}")
    kind, body = blocks[0]
    thought = text[:text.index(f"<{kind}>")].strip()
    body = body.strip("\n")
    if kind in ("python", "action"):
        fenced = _FENCE.match(body)
        if fenced:
            body = fenced.group(1)
        if not body.strip():
            raise ProtocolError(f"<{kind}> block is empty")
    return Reply(thought=thought, kind=kind, body=body if kind == "python" else body.strip())

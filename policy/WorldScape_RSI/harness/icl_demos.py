"""In-context demonstrations.

A demonstration is one successful episode of the task, shown before every turn
of the current episode: the messages sit between the system prompt and the
current turn, so every turn sees them in the same place.

A demonstration bank is a directory with one entry per task,
``<bank>/<task>/`` (further entries of the same task are ``<task>+2/``,
``<task>+3/`` ...). An entry holds ``demo.json`` plus the JPEG frames it
names: per frame the label, the state line, the plan, the commands and their
net effect, exactly as the model reads them. Numbers belong to the
demonstration scene; the system prompt tells the model to copy strategy, not
coordinates.

Only the standard library is imported here.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class DemoFrame:
    turn: int
    label: str
    images: list[tuple[str, bytes]] = field(default_factory=list)
    state_line: str = ""
    plan: str = ""
    commands: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    effect: str = ""
    source_index: Optional[int] = None


@dataclass
class Demo:
    task: str
    instruction: str
    source: dict
    frames: list[DemoFrame] = field(default_factory=list)
    kind: str = "task"

    def n_images(self) -> int:
        return sum(len(frame.images) for frame in self.frames)


def looks_like_case(path: Path) -> bool:
    path = Path(path)
    return path.is_dir() and (path / "demo.json").is_file()


def resolve_icl_path(raw: str | Path) -> Path:
    """Resolve a bank path against the repo root, then cwd.

    Relative paths in deploy.yml / CLI are written from the repository root,
    but the policy server's CWD is ``XPolicyLab/policy/vlm_agent``.
    """
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    repo = _repo_root()
    for candidate in (repo / path, Path.cwd() / path):
        if candidate.exists():
            return candidate.resolve()
    return (repo / path)


def load_demos(cfg: dict, task: str = "") -> list[Demo]:
    """Load the demonstrations of ``task`` according to the ``icl`` config block."""
    dirs = resolve_demo_dirs(cfg.get("demo_bank"), task, int(cfg.get("demo_count") or 1))
    if not dirs:
        raise FileNotFoundError(f"no demonstration for task {task or '?'} in {cfg.get('demo_bank')}")
    demos = [build_demo(path) for path in dirs]
    if any(demo.kind != "official_video" for demo in demos):
        raise ValueError("Ken collection requires official RoboDojo video ICL; legacy RoboDawn trace ICL is disabled")
    return demos


def resolve_demo_dirs(demo_bank: Optional[str], task: str, count: int = 1) -> list[Path]:
    """The bank entries of ``task``: ``<task>/`` and then ``<task>+k/`` up to ``count``."""
    bank = resolve_icl_path(str(demo_bank or ""))
    if not bank.exists():
        raise FileNotFoundError(f"ICL demo bank not found: {bank}")
    return _bank_dirs(bank, task, count)


def build_demo(episode_dir: Path) -> Demo:
    """Load one bank entry (``demo.json`` plus its frames)."""
    episode_dir = Path(episode_dir)
    packed = episode_dir / "demo.json"
    if packed.is_file():
        return _load_packed_demo(episode_dir, packed)
    raise FileNotFoundError(f"{episode_dir} is not a demonstration entry (no demo.json)")


def demo_messages(demos: Demo | list[Demo]) -> list[dict]:
    """One user message plus a short assistant acknowledgement."""
    demos = [demos] if isinstance(demos, Demo) else list(demos)
    content: list[dict] = []
    tasks = [demo for demo in demos if demo.kind != "primer"]
    for demo in demos:
        content.extend(_demo_content(demo, tasks.index(demo) + 1 if demo in tasks else 0, len(tasks)))
    content.append(_text(
        "--- END OF THE DEMONSTRATION" + ("S" if len(demos) > 1 else "") + ". Your own episode starts "
        "with the next message; its scene, object positions and table height may differ, so measure "
        "everything again from your own images."
    ))
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": (
            "Understood. I will follow the demonstrated strategy and read all positions from my own images and state."
        )},
    ]


def save_demo(demo: Demo, out_dir: Path) -> None:
    """Write the frames the model saw, for offline inspection."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for frame in demo.frames:
        files = []
        for view, blob in frame.images:
            name = f"frame{frame.turn:03d}_{view}.jpg"
            (out_dir / name).write_bytes(blob)
            files.append(name)
        frames.append({
            "turn": frame.turn, "label": frame.label, "images": files, "state": frame.state_line,
            "plan": frame.plan, "commands": frame.commands, "failed": frame.failed, "effect": frame.effect,
            "source_index": frame.source_index,
        })
    source = dict(demo.source)
    source["images_scaled"] = True
    (out_dir / "demo.json").write_text(json.dumps({
        "task": demo.task, "instruction": demo.instruction, "kind": demo.kind,
        "source": source, "frames": frames,
    }, indent=1), encoding="utf-8")


# --------------------------------------------------------------------------- bank entries


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "RoboDojo").is_dir() and (parent / "evaluation" / "policies").is_dir():
            return parent
    return here.parent.parent


def _bank_dirs(bank: Path, task: str, count: int) -> list[Path]:
    if not task:
        return []
    out = []
    for index in range(1, count + 1):
        path = bank / (task if index == 1 else f"{task}+{index}")
        if looks_like_case(path):
            out.append(path)
    return out


def _load_packed_demo(episode_dir: Path, packed: Path) -> Demo:
    # Bank JPEGs are already display-sized; do not scale them again.
    data = _read_json(packed)
    source = dict(data.get("source") or {"episode_dir": str(episode_dir)})
    frames = []
    for index, raw in enumerate(data.get("frames") or []):
        images = []
        for name in raw.get("images") or []:
            blob = _load_image_bytes(episode_dir / name)
            if blob:
                images.append((Path(name).stem, blob))
        frames.append(DemoFrame(
            turn=int(raw.get("turn") or index + 1),
            label=str(raw.get("label") or f"turn {index + 1}"),
            images=images,
            state_line=str(raw.get("state") or ""),
            plan=str(raw.get("plan") or ""),
            commands=list(raw.get("commands") or []),
            failed=list(raw.get("failed") or []),
            effect=str(raw.get("effect") or ""),
            source_index=raw.get("source_index"),
        ))
    return Demo(
        task=str(data.get("task") or episode_dir.name),
        instruction=str(data.get("instruction") or ""),
        source=source,
        frames=frames,
        kind=str(data.get("kind") or "task"),
    )


def _demo_content(demo: Demo, index: int, total: int) -> list[dict]:
    if demo.kind == "official_video":
        content = [_text(
            "OFFICIAL ROBODOJO VIDEO REFERENCE (visual-only keyframes, not a logged policy trajectory). "
            f"Task: {demo.task}. Source: {demo.source.get('page_url', '')}. "
            "Frames show temporal order and visible manipulation strategy. They contain NO robot coordinates, "
            "joint actions, rewards, or verified native step count. Video timestamps are not control-step budgets. "
            "Infer current geometry and language targets from CURRENT observations; never copy reference colors or positions. "
            "Do not assume a motion is feasible merely because it appears in the reference."
        )]
        for frame in demo.frames:
            content.append(_text(frame.label))
            content.extend(_image(blob) for _view, blob in frame.images)
        return content
    n_img = demo.n_images()
    label = f"DEMONSTRATION {index} of {total}" if total > 1 else "DEMONSTRATION"
    header = (
        f"{label} (a successful episode of a similar task in a reference scene; {len(demo.frames)} selected "
        f"phases, {n_img} images). Its instruction was: \"{demo.instruction}\".\n"
        "For each selected phase you see its reference state, phase description and commands. "
        "Phases in between may be omitted. Positions are fingertip "
        "centre, same frame as your CURRENT STATE. Command numbers belong to the demonstration's "
        "object positions — copy the order of sub-goals, not the centimetres."
    )
    content: list[dict] = [_text(header)]
    tag = f"DEMO {index} " if total > 1 else "DEMO "
    for frame in demo.frames:
        lines = [f"--- {tag}{frame.label}"]
        if frame.source_index is not None:
            lines.append(f"source: expert trace[{frame.source_index}]")
        for _view, blob in frame.images:
            content.append(_image(blob))
        if frame.state_line:
            lines.append(f"state: {frame.state_line}")
        if frame.plan:
            lines.append(f"plan: {frame.plan}")
        if frame.commands:
            lines.append("commands: " + json.dumps(frame.commands))
        if frame.effect:
            lines.append("net effect: " + frame.effect)
        if frame.failed:
            lines.append("(Command reported failure; motion may be partial. Use net effect and subsequent observed state, not the requested target: " + ", ".join(frame.failed) + ")")
        content.append(_text("\n".join(lines)))
    return content


def _text(text: str) -> dict:
    return {"type": "text", "text": text}


def _image(jpeg_or_png: bytes) -> dict:
    import base64
    mime = "image/jpeg" if jpeg_or_png[:2] == b"\xff\xd8" else "image/png"
    url = f"data:{mime};base64," + base64.b64encode(jpeg_or_png).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def _load_image_bytes(path: Path) -> Optional[bytes]:
    return path.read_bytes() if path.is_file() else None


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data

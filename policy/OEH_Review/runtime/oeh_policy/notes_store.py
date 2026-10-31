"""A task-notes store (G7): a directory holding ``notes.jsonl`` and ``ledger.jsonl``.

``notes.jsonl`` holds one note per line, a JSON object with exactly NOTE_KEYS, written with sorted keys and compact
separators, so the same notes are the same bytes. ``ledger.jsonl`` holds one line per write
(``{"utc", "op", "note_index", "sha256_before", "sha256_after", "by"}``). The store's sha256 is the sha256 of
``notes.jsonl``'s bytes. ``seal_store`` writes a ``sealed`` marker holding that sha; a sealed store refuses every
write. A scored round reads a store through ``load_store`` with the sha its lock sealed and refuses any other bytes.

Who writes the notes, and from which exploration episodes, is not decided here (P10, the user's): this module stores,
hashes, scopes, scans and renders. Standard library only; every output is deterministic given its inputs.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

NOTES_FILE = "notes.jsonl"
LEDGER_FILE = "ledger.jsonl"
SEALED_FILE = "sealed"
NOTE_KEYS = ("task", "signature", "instead", "evidence", "status", "layout", "written_utc", "source_attempt")
TEXT_KEYS = ("signature", "instead", "evidence")
PACKET_KEYS = ("signature", "instead", "evidence", "status")  # what a packet or lessons file may carry of a note
V2_FIELDS = ("schema", "preconditions", "verification", "not_applicable", "epistemic_status")
V2_PACKET_KEYS = PACKET_KEYS + V2_FIELDS[1:]
STATUSES = ("active", "retired")
SCOPES = ("global", "task", "layout")
RANDOM_SUFFIX = "_random"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _line(value: Mapping) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _notes_path(path: str | Path) -> Path:
    return Path(path) / NOTES_FILE


def _validate_note(value: Any, where: str) -> dict:
    v2 = isinstance(value, Mapping) and value.get("schema") == "oeh.task_note.v2"
    keys = NOTE_KEYS + V2_FIELDS if v2 else NOTE_KEYS
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise ValueError(f"{where} must be an object with exactly the keys {list(NOTE_KEYS)}")
    for key in ("task", *TEXT_KEYS, "written_utc"):
        if not isinstance(value[key], str) or not value[key].strip():
            raise ValueError(f"{where}.{key} must be a non-empty string")
    if value["status"] not in STATUSES:
        raise ValueError(f"{where}.status must be one of {list(STATUSES)}")
    for key in ("layout", "source_attempt"):
        if value[key] is not None and (not isinstance(value[key], str) or not value[key]):
            raise ValueError(f"{where}.{key} must be a non-empty string or null")
    if v2:
        for key in ("preconditions", "verification", "not_applicable"):
            if not isinstance(value[key], str) or not value[key].strip():
                raise ValueError(f"{where}.{key} must be non-empty text")
        if value["epistemic_status"] not in {"verified", "scene_specific", "hypothesis"}:
            raise ValueError(f"{where}.epistemic_status is unknown")
    return {key: value[key] for key in keys}


def _parse(data: bytes, where: str) -> list[dict]:
    notes = []
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{where} is not UTF-8") from error
    if text and not text.endswith("\n"):
        raise ValueError(f"{where} does not end with a newline")
    for number, line in enumerate(text.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where} line {number} is not JSON: {error.msg}") from error
        note = _validate_note(value, f"{where} line {number}")
        if note.get("schema") == "oeh.task_note.v2" and scan_notes({"notes": [note]}):
            raise ValueError(f"{where} line {number} contains forbidden public content")
        if _line(note).decode("utf-8") != line + "\n":
            raise ValueError(f"{where} line {number} is not in the store's canonical form")
        notes.append(note)
    return notes


# Assets already verified in this process, by file version: a compile loads one store once per Explore config.
_VERIFIED_ASSETS: set[tuple[str, int, int]] = set()


def store_sha256(path: str | Path) -> str:
    """v1: notes bytes; v2: domain-separated notes + complete audit bytes."""
    notes = _notes_path(path)
    if not notes.is_file():
        raise ValueError(f"notes store {path} has no {NOTES_FILE}")
    return _store_digest(Path(path), notes.read_bytes())


def _store_digest(root: Path, data: bytes) -> str:
    manifest = root / "store.json"
    if not manifest.exists():
        # v2 notes cannot silently lose their audit binding by deleting store.json.
        if any(note.get("schema") == "oeh.task_note.v2" for note in _parse(data, str(root))):
            raise ValueError("v2 notes require store.json and audit.json")
        return _sha(data)
    if json.loads(manifest.read_text()) != {"schema": "oeh.notes_store.v2"}:
        raise ValueError("unknown notes store manifest")
    audit = (root / "audit.json").read_bytes()
    parsed = json.loads(audit)
    if parsed.get("schema") != "oeh.notes.audit.v2":
        raise ValueError("invalid v2 audit")
    for digest in parsed.get("evidence", {}).get("assets", []):
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("v2 observation asset missing or changed")
        asset = root / "assets" / digest
        try:
            stat = asset.stat()
        except OSError:
            raise ValueError("v2 observation asset missing or changed") from None
        key = (str(asset.resolve()), stat.st_size, stat.st_mtime_ns)
        if key not in _VERIFIED_ASSETS:
            if _sha(asset.read_bytes()) != digest:
                raise ValueError("v2 observation asset missing or changed")
            _VERIFIED_ASSETS.add(key)
    return _sha(b"oeh.notes_store.v2\0" + data + b"\0" + audit)


def load_store(path: str | Path, sha256: str) -> dict:
    """Read a store whose notes.jsonl hashes ``sha256``; refuse a missing, malformed or changed store."""
    root = Path(path)
    if not root.is_dir():
        raise ValueError(f"notes store {path} is not a directory")
    notes_file = root / NOTES_FILE
    if not notes_file.is_file():
        raise ValueError(f"notes store {path} has no {NOTES_FILE}")
    data = notes_file.read_bytes()
    got = _store_digest(root, data)
    if got != sha256:
        raise ValueError(f"notes store {path} hashes {got}, the lock sealed {sha256}: the scored round refuses a "
                         "changed store")
    sealed = root / SEALED_FILE
    if (root / "store.json").exists() and not sealed.is_file():
        raise ValueError("v2 notes store must be sealed before scoring")
    if sealed.exists() and sealed.read_text(encoding="utf-8").strip() != got:
        raise ValueError(f"notes store {path}: its sealed marker names another sha than notes.jsonl's {got}")
    notes = _parse(data, f"notes store {path} {NOTES_FILE}")
    return {"path": str(root), "sha256": got, "notes": notes, "count": len(notes)}


def _read_for_write(path: str | Path) -> tuple[Path, bytes, list[dict]]:
    root = Path(path)
    if (root / SEALED_FILE).exists():
        raise ValueError(f"notes store {path} is sealed: a sealed store is read-only")
    if (root / "store.json").exists():
        raise ValueError("v2 stores are exported by the offline exploration session, not edited in place")
    notes_file = root / NOTES_FILE
    data = notes_file.read_bytes() if notes_file.is_file() else b""
    return root, data, _parse(data, f"notes store {path} {NOTES_FILE}")


def _write(root: Path, notes: list[dict], before: bytes, op: str, index: int, by: str, utc: str) -> dict:
    if not isinstance(by, str) or not by.strip():
        raise ValueError("a notes store write names who made it (by)")
    root.mkdir(parents=True, exist_ok=True)
    data = b"".join(_line(note) for note in notes)
    temporary = root / (NOTES_FILE + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, root / NOTES_FILE)
    entry = {"utc": utc, "op": op, "note_index": index, "sha256_before": _sha(before), "sha256_after": _sha(data),
             "by": by}
    with (root / LEDGER_FILE).open("ab") as stream:
        stream.write(_line(entry))
    return entry


def add_note(path: str | Path, note: Mapping, by: str) -> dict:
    """Append a note (task, signature, instead, evidence; optional layout, source_attempt, status, written_utc) and
    its ledger line; returns the ledger entry. Refused on a sealed store. Creates the store when it does not exist."""
    unknown = set(note) - set(NOTE_KEYS)
    if unknown:
        raise ValueError(f"unknown note keys: {sorted(unknown)}")
    utc = _now()
    full = {"status": "active", "layout": None, "source_attempt": None, "written_utc": utc, **dict(note)}
    root, before, notes = _read_for_write(path)
    notes.append(_validate_note(full, "note"))
    return _write(root, notes, before, "add", len(notes) - 1, by, utc)


def retire_note(path: str | Path, index: int, by: str) -> dict:
    """Mark note ``index`` retired (notes are never removed, so indices stay stable); returns the ledger entry."""
    root, before, notes = _read_for_write(path)
    if type(index) is not int or not 0 <= index < len(notes):
        raise ValueError(f"notes store {path} has no note {index}")
    if notes[index]["status"] == "retired":
        raise ValueError(f"note {index} of notes store {path} is already retired")
    notes[index] = {**notes[index], "status": "retired"}
    return _write(root, notes, before, "retire", index, by, _now())


def seal_store(path: str | Path) -> str:
    """Write the ``sealed`` marker holding the store's sha256; from then on the store refuses writes."""
    sha = store_sha256(path)
    _parse(_notes_path(path).read_bytes(), f"notes store {path} {NOTES_FILE}")
    sealed = Path(path) / SEALED_FILE
    if sealed.exists():
        if sealed.read_text(encoding="utf-8").strip() != sha:
            raise ValueError(f"notes store {path} is sealed with another sha than notes.jsonl's {sha}")
        return sha
    sealed.write_text(sha + "\n", encoding="utf-8")
    return sha


def source_attempts(path: str | Path) -> set[str]:
    """The source_attempt ids the store's notes name (retired notes included); empty for a store not yet written.
    Reads a sealed store too (oeh-eval explore-notes skips an attempt that already has a note)."""
    notes_file = _notes_path(path)
    if not notes_file.is_file():
        return set()
    notes = _parse(notes_file.read_bytes(), f"notes store {path} {NOTES_FILE}")
    return {note["source_attempt"] for note in notes if note["source_attempt"] is not None}


def _base_task(task: str) -> str:
    return task[:-len(RANDOM_SUFFIX)] if task.endswith(RANDOM_SUFFIX) else task


def _active(store: Mapping, task: str | None, scope: str, layout: str | None) -> list[dict]:
    if scope not in SCOPES:
        raise ValueError(f"notes scope must be one of {list(SCOPES)}")
    if scope != "global" and task is None:
        raise ValueError(f"notes scope {scope} needs a task")
    if scope == "layout" and layout is None:
        raise ValueError("notes scope layout needs a layout")
    chosen = []
    for note in store["notes"]:
        if note["status"] != "active":
            continue
        if scope != "global" and _base_task(note["task"]) != _base_task(task):
            continue
        if scope == "layout" and note["layout"] != layout:
            continue
        chosen.append(note)
    return chosen


def notes_for(store: Mapping, task: str, scope: str, layout: str | None = None) -> list[dict]:
    """The active notes a packet may carry, in store order, each reduced to PACKET_KEYS: every note (global), the
    task's (task; ``_random`` stripped on both sides), or the task's at one layout (layout)."""
    return [{key: note[key] for key in (V2_PACKET_KEYS if note.get("schema") == "oeh.task_note.v2" else PACKET_KEYS)}
            for note in _active(store, task, scope, layout)]


def _word_pattern() -> re.Pattern:
    from .contracts import PACKET_FORBIDDEN_KEYS
    words = "|".join(sorted((re.escape(word) for word in PACKET_FORBIDDEN_KEYS), key=lambda w: (-len(w), w)))
    return re.compile(rf"(?<![A-Za-z0-9_])({words})(?![A-Za-z0-9_])", re.IGNORECASE)


def scan_notes(store: Mapping) -> list[str]:
    """The forbidden-key scan of note text: every contracts.PACKET_FORBIDDEN_KEYS word found as a whole word
    (case-insensitive; an underscore joins a word) in a note's signature, instead or evidence, retired notes included.
    Returns ``"note <i> <field>: <word>"`` per hit, in store and text order; empty when clean."""
    pattern, hits = _word_pattern(), []
    for index, note in enumerate(store["notes"]):
        keys = TEXT_KEYS + ("preconditions", "verification", "not_applicable") if note.get("schema") == \
            "oeh.task_note.v2" else TEXT_KEYS
        for key in keys:
            hits += [f"note {index} {key}: {match.group(1).lower()}" for match in pattern.finditer(note[key])]
    return hits


def render_lessons(store: Mapping, task: str | None = None) -> str:
    """The active notes as markdown (the read path of behavior.lessons_file): every note, or the task's
    (``_random`` stripped). Grouped by task in first-appearance order; carries only PACKET_KEYS' text, never the
    layout, write time or source attempt. LF line endings and one trailing newline."""
    notes = _active(store, task, "global" if task is None else "task", None)
    lines = ["# Task notes", ""]
    if not notes:
        lines += ["(no notes)", ""]
    groups: dict[str, list[dict]] = {}
    for note in notes:
        groups.setdefault(_base_task(note["task"]), []).append(note)
    for name, members in groups.items():
        lines += [f"## {name}", ""]
        for number, note in enumerate(members, 1):
            lines += [f"{number}. Signature: {_one_line(note['signature'])}",
                      f"   Instead: {_one_line(note['instead'])}",
                      f"   Evidence: {_one_line(note['evidence'])}", ""]
            if note.get("schema") == "oeh.task_note.v2":
                lines[-1:] = [f"   Preconditions: {_one_line(note['preconditions'])}",
                              f"   Verify: {_one_line(note['verification'])}",
                              f"   Do not apply: {_one_line(note['not_applicable'])}",
                              f"   Status: {note['epistemic_status']} (evidence in exploration; transfer unverified)", ""]
    return "\n".join(lines).rstrip("\n") + "\n"


def _one_line(text: str) -> str:
    return " ".join(text.split())


def check_disjoint(exploration_ids: list[str], *other_lists: list[str]) -> list[str]:
    """The exploration case ids (or layout hashes) that also appear in any other list, sorted; empty when disjoint."""
    others = set()
    for items in other_lists:
        others |= set(items)
    return sorted(set(exploration_ids) & others)


def read_id_list(path: str | Path) -> list[str]:
    """A case-id list file: one id per line; blank lines and ``#`` comments ignored."""
    ids = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids

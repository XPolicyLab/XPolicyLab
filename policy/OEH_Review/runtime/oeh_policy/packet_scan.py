"""G1 tooling: the packet scan and the packet diff over recorded attempts (reads only).

``scan_attempt`` applies ``contracts.check_packet`` to every teacher packet an attempt wrote
(``<attempt>/teacher/packets/NNNN.json``): forbidden keys at any depth always, and the top-level field set when one is
known — the attempt's own ``identity.json`` ``review.context.field_set`` (a lock that declared a manifest), or one
given by the caller. An attempt without a manifest gets the forbidden-key scan only: the "shadow scan" of a legacy
attempt, whose packets the runtime never checked while it ran.

``diff_packets`` compares two attempts' packets decision by decision, byte for byte after removing the named
top-level keys, and reports the first differing decision and its differing keys.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

from .contracts import check_packet, encoded

PACKET_NAME = re.compile(r"^\d{4}\.json$")


def packet_files(attempt: str | Path) -> list[Path]:
    """An attempt's teacher packet files in decision order (the images beside them are not packets)."""
    directory = Path(attempt) / "teacher" / "packets"
    if not directory.is_dir():
        return []
    return sorted(path for path in directory.iterdir() if PACKET_NAME.match(path.name) and path.is_file())


def declared_field_set(attempt: str | Path) -> str | None:
    """The field set an attempt's identity.json records for its declared manifest, or None (no manifest, or no
    identity.json)."""
    identity = Path(attempt) / "identity.json"
    if not identity.is_file():
        return None
    review = json.loads(identity.read_text(encoding="utf-8")).get("review") or {}
    return (review.get("context") or {}).get("field_set")


def scan_attempt(attempt: str | Path, field_set: str = "auto") -> dict:
    """Scan every packet of one attempt. ``field_set``: "auto" (the identity's declared field set, else the shadow
    scan) or an explicit field-set name. Returns {attempt, packets, field_set, field_set_source, violations}; a
    violation is {packet (path relative to the attempt), message}. No packet at all is a violation: an empty scan
    shows nothing."""
    root = Path(attempt)
    if field_set == "auto":
        used = declared_field_set(root)
        source = "identity.json review.context.field_set" if used is not None else "shadow"
    else:
        used, source = field_set, "argument"
    files = packet_files(root)
    violations = []
    if not files:
        violations.append({"packet": "teacher/packets", "message": "no teacher packets"})
    for path in files:
        where = str(path.relative_to(root))
        try:
            packet = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(packet, dict):
                raise ValueError("teacher packet is not a JSON object")
            check_packet(packet, used)
        except (ValueError, UnicodeDecodeError) as error:  # json.JSONDecodeError is a ValueError
            violations.append({"packet": where, "message": str(error)})
    return {"attempt": str(root), "packets": len(files), "field_set": used, "field_set_source": source,
            "violations": violations}


def format_scan(result: dict) -> str:
    """The scan's text report: one header line per attempt, then one line per violation."""
    if result["field_set"] is None:
        used = "none (shadow scan: forbidden keys only; the attempt declared no manifest)"
    else:
        used = f"{result['field_set']} ({result['field_set_source']})"
    lines = [f"{result['attempt']}: {result['packets']} packets scanned, field set {used}, "
             f"{len(result['violations'])} violations"]
    lines += [f"  {item['packet']}: {item['message']}" for item in result["violations"]]
    return "\n".join(lines)


def _stripped(path: Path, ignore: frozenset) -> tuple[bytes, dict | None]:
    data = path.read_bytes()
    if not ignore:
        try:
            return data, json.loads(data)
        except ValueError:
            return data, None
    packet = json.loads(data)
    packet = {key: value for key, value in packet.items() if key not in ignore}
    return encoded(packet), packet


def diff_packets(a: str | Path, b: str | Path, ignore: Iterable[str] = ()) -> dict:
    """Compare two attempts' packets decision by decision. Returns {identical, packets_a, packets_b, ignored,
    first_difference}; first_difference is None or {decision (the file name), keys (top-level keys whose values
    differ, or that one side lacks), reason}."""
    ignored = frozenset(ignore)
    left, right = packet_files(a), packet_files(b)
    first = None
    for one, other in zip(left, right):
        if one.name != other.name:
            first = {"decision": min(one.name, other.name), "keys": [], "reason": "decision missing on one side"}
            break
        bytes_a, packet_a = _stripped(one, ignored)
        bytes_b, packet_b = _stripped(other, ignored)
        if bytes_a != bytes_b:
            keys = []
            if isinstance(packet_a, dict) and isinstance(packet_b, dict):
                keys = sorted(key for key in set(packet_a) | set(packet_b)
                              if key not in packet_a or key not in packet_b or packet_a[key] != packet_b[key])
            first = {"decision": one.name, "keys": keys,
                     "reason": "values differ" if keys else "bytes differ (key order or encoding)"}
            break
    if first is None and len(left) != len(right):
        longer = left if len(left) > len(right) else right
        first = {"decision": longer[min(len(left), len(right))].name, "keys": [],
                 "reason": "decision missing on one side"}
    if first is None and not left:
        first = {"decision": None, "keys": [], "reason": "no teacher packets"}
    return {"identical": first is None, "packets_a": len(left), "packets_b": len(right), "ignored": sorted(ignored),
            "first_difference": first}


def format_diff(result: dict) -> str:
    ignored = f" (ignoring {result['ignored']})" if result["ignored"] else ""
    if result["identical"]:
        return f"identical: {result['packets_a']} packets byte for byte{ignored}"
    first = result["first_difference"]
    return (f"differ: {result['packets_a']} vs {result['packets_b']} packets{ignored}; first difference at "
            f"{first['decision']}: {first['reason']}" + (f", keys {first['keys']}" if first["keys"] else ""))

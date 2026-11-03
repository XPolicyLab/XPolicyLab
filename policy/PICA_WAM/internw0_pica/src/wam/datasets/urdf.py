"""Kinematic chains from user-supplied URDF descriptions."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class Joint:
    name: str
    kind: str
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float]


def _vector(value: str) -> tuple[float, float, float]:
    values = tuple(float(v) for v in value.split())
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Expected three finite URDF coordinates: {value!r}")
    return values


def load_chain(path: str, base_link: str, end_link: str) -> tuple[Joint, ...]:
    source = Path(path).expanduser().resolve()
    stat = source.stat()
    return _load_chain(str(source), stat.st_mtime_ns, stat.st_size, base_link, end_link)


@lru_cache(maxsize=32)
def _load_chain(path, mtime_ns, size, base_link, end_link):
    root = ET.parse(path).getroot()
    links = {link.attrib["name"] for link in root.findall("link")}
    if base_link not in links or end_link not in links:
        raise ValueError(f"Unknown base/end link: {base_link!r}, {end_link!r}")
    parents = {}
    for element in root.findall("joint"):
        parent, child = element.find("parent"), element.find("child")
        if parent is None or child is None:
            raise ValueError("URDF joint is missing its parent or child link")
        child_name = child.attrib["link"]
        if child_name in parents:
            raise ValueError(f"Multiple parent joints for {child_name!r}")
        parents[child_name] = (parent.attrib["link"], element)
    chain, seen = [], set()
    link = end_link
    while link != base_link:
        if link in seen or link not in parents:
            raise ValueError(f"No acyclic chain from {base_link!r} to {end_link!r}")
        seen.add(link)
        link, element = parents[link]
        if element.find("mimic") is not None:
            raise ValueError("FK chains require independent joints; expand mimic joints first")
        origin, axis_node = element.find("origin"), element.find("axis")
        axis = _vector(axis_node.get("xyz", "1 0 0") if axis_node is not None else "1 0 0")
        norm = math.sqrt(sum(v * v for v in axis))
        if norm == 0:
            raise ValueError("URDF joint axis must be nonzero")
        chain.append(Joint(
            element.attrib["name"], element.attrib["type"],
            _vector(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0"),
            _vector(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"),
            tuple(v / norm for v in axis),
        ))
    return tuple(reversed(chain))

"""Drop-in XPolicyLab file loaders without an eager HDF5 dependency.

RoboDojo's online policy adapter only needs YAML/JSON to resolve the robot
dimensions. Upstream imports h5py at module import time even when no HDF5 is
read, which can prevent the policy bridge from starting in an otherwise valid
evaluation environment. Keep the upstream API and import h5py only for the one
function that actually needs it.
"""

from __future__ import annotations

import json

import yaml


def load_hdf5(path: str) -> dict:
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - not used by online eval
        raise RuntimeError("load_hdf5 requires h5py; install it before reading HDF5 data.") from exc

    def _read(obj):
        if isinstance(obj, h5py.Dataset):
            value = obj[()]
            if isinstance(value, (bytes, bytearray)):
                return value.decode("utf-8", errors="replace")
            try:
                return value.item()
            except Exception:
                return value
        return {key: _read(value) for key, value in obj.items()}

    with h5py.File(path, "r") as handle:
        data = _read(handle)
        if handle.attrs:
            data["_attrs"] = {key: handle.attrs[key] for key in handle.attrs.keys()}
        return data


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = ["load_hdf5", "load_json", "load_yaml"]

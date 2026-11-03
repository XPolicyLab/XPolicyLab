"""Persistent operator host exclusions, shared by all campaign snapshots."""

import json
import socket

from orchestrator.config import REPO


def canonical_host(name):
    return name.rsplit("@", 1)[-1].split(".", 1)[0].lower().replace("-", "")


def exclusion(name):
    if name == "local":
        name = socket.gethostname()
    path = REPO / "configs/host_blacklist.json"
    policy = json.loads(path.read_text()) if path.exists() else {}
    return next((reason for host, reason in policy.items()
                 if canonical_host(host) == canonical_host(name)), None)


def require_allowed(name):
    if reason := exclusion(name):
        raise ValueError(f"Host {name} is blacklisted: {reason}")

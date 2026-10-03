"""Durable call identities: uncertain effects are never automatically replayed."""

import json
from pathlib import Path
from threading import Lock

from PhysicalRSI_core.infra.storage import (
    atomic_json,
    digest,
    identifier,
    locked,
    read_json,
)


class RequestJournal:
    def __init__(self, root):
        self.root = Path(root)
        self._lock = Lock()

    def status(self, request_id):
        path = self.root / (identifier(request_id) + ".json")
        return read_json(path) if path.exists() else {"state": "missing"}

    def execute(self, request_id, binding, function):
        from .http_rpc import _from_json, _NumpyEncoder

        signature = digest(json.loads(json.dumps(binding, cls=_NumpyEncoder)))
        path = self.root / (identifier(request_id) + ".json")
        with self._lock, locked(self.root / ".claims.lock"):
            record = self.status(request_id)
            if record["state"] != "missing":
                if record["binding"] != signature:
                    raise ValueError(
                        "Request identity reused with different inputs/session"
                    )
                if record["state"] == "completed":
                    return _from_json(record["result"])
                raise RuntimeError(
                    "Request requires reconciliation: " + record["state"]
                )
            atomic_json(path, {"state": "started", "binding": signature})
        try:
            result = function()
            encoded = json.loads(json.dumps(result, cls=_NumpyEncoder))
        except BaseException as error:
            atomic_json(
                path,
                {
                    "state": "uncertain",
                    "binding": signature,
                    "error": type(error).__name__ + ": " + str(error),
                },
            )
            raise
        atomic_json(
            path, {"state": "completed", "binding": signature, "result": encoded}
        )
        return result

"""Persist large/binary observations separately from compact execution records."""

import hashlib
import os
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path


class Artifacts:
    def __init__(self, root):
        self.root = Path(root)

    def encode(self, value):
        if is_dataclass(value) and not isinstance(value, type):
            return self.encode(asdict(value))
        if isinstance(value, bytes):
            return self._bytes(value, ".bin")
        if isinstance(value, dict):
            return {str(key): self.encode(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [self.encode(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if type(value).__module__.split(".")[0] == "numpy":
            import io

            import numpy as np

            if isinstance(value, np.ndarray):
                buffer = io.BytesIO()
                np.save(buffer, value, allow_pickle=False)
                return self._bytes(buffer.getvalue(), ".npy")
            return value.item()
        if value is None or isinstance(value, (str, bool, int, float)):
            return value
        raise TypeError(f"Unsupported evidence type: {type(value).__name__}")

    def _bytes(self, content, suffix):
        sha = hashlib.sha256(content).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / (sha + suffix)
        if path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
                raise ValueError("Existing evidence artifact is corrupt")
        else:
            fd, temporary = tempfile.mkstemp(dir=self.root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return {"artifact": str(path.resolve()), "sha256": sha, "bytes": len(content)}

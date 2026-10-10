#!/usr/bin/env python
"""Extract model-only tensors from a distributed resume checkpoint.

Pass --src for the source checkpoint and --out for the destination directory."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="resume checkpoint directory")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    meta = FileSystemReader(args.src).read_metadata().state_dict_metadata
    model = {
        k[len("model.") :]: torch.empty(tuple(m.size), dtype=m.properties.dtype)
        for k, m in meta.items()
        if k.startswith("model.") and isinstance(m, TensorStorageMetadata)
    }
    dcp.load({"model": model}, checkpoint_id=args.src)
    n = sum(v.numel() for v in model.values())
    print(f"loaded {len(model)} tensors ({n / 1e9:.2f}B params) in {time.time() - t0:.0f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dcp.save({"model": model}, checkpoint_id=str(out))
    (out / "SOURCE.json").write_text(json.dumps({"source": str(args.src), "tensors": len(model),
                                                  "params": n, "dtype": "float32"}, indent=1))
    print(f"wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

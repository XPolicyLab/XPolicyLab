"""Exact comparison of captured or replayed model evidence."""
import argparse
import json
from pathlib import Path

import numpy as np

from validation import decode, records

KINDS = {"reset", "observation", "inputs", "history", "prediction", "ensemble", "commands"}


def compare(left, right):
    left, right = Path(left), Path(right)
    a = [r for r in records(left) if r["kind"] in KINDS]
    b = [r for r in records(right) if r["kind"] in KINDS]
    differences = []
    max_abs = 0.0

    def visit(x, y, path):
        nonlocal max_abs
        if x == y:
            return
        if isinstance(x, dict) and isinstance(y, dict):
            if "array" in x and "array" in y:
                item = {"path": path, "left": x, "right": y}
                # Inputs/history store fingerprints only; actions are lossless.
                if (left / "arrays" / (x["array"] + ".npy")).exists() and (right / "arrays" / (y["array"] + ".npy")).exists():
                    av, bv = decode(x, left / "arrays"), decode(y, right / "arrays")
                    if av.shape == bv.shape and av.dtype == bv.dtype:
                        delta = float(np.max(np.abs(av.astype(np.float64) - bv.astype(np.float64)), initial=0))
                        item["max_abs_difference"] = delta
                        max_abs = max(max_abs, delta)
                differences.append(item)
                return
            if set(x) != set(y):
                differences.append({"path": path, "reason": "different keys"})
                return
            for key in x:
                visit(x[key], y[key], path + "/" + key)
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (av, bv) in enumerate(zip(x, y)):
                visit(av, bv, path + "/" + str(i))
        else:
            differences.append({"path": path, "left": x, "right": y})

    if not a or not b:
        differences.append({"reason": "empty evidence is not a passing comparison"})
    if len(a) != len(b):
        differences.append({"reason": "different event counts", "left": len(a), "right": len(b)})
    for i, (av, bv) in enumerate(zip(a, b)):
        visit(av, bv, f"event-{i}")
    return {"passed": not differences, "events": [len(a), len(b)],
            "max_abs_stored_array_difference": max_abs,
            "difference_count": len(differences), "differences": differences[:20]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--packaged", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fixtures = json.loads(args.fixtures.read_text())["fixtures"]
    checks = {}
    for i in range(3):
        checks[f"reference_vs_packaged_{i}"] = compare(
            args.reference / f"episode-{i:03d}", args.packaged / f"episode-{i:03d}")
    for i in range(2):
        checks[f"captured_vs_packaged_{i}"] = compare(fixtures[i], args.packaged / f"episode-{i:03d}")
    for name, root in [("reference", args.reference), ("packaged", args.packaged)]:
        checks[f"{name}_reset_A_to_B_to_A"] = compare(root / "episode-000", root / "episode-002")
    a = json.loads((args.reference / "environment.json").read_text())
    b = json.loads((args.packaged / "environment.json").read_text())
    keys = ["torch", "cuda", "cudnn", "gpu", "deterministic", "warn_only", "cudnn_benchmark",
            "cudnn_deterministic", "cudnn_tf32", "matmul_tf32", "matmul_precision"]
    checks["matching_backend"] = {"passed": all(a[k] == b[k] for k in keys),
                                 "reference": {k: a[k] for k in keys}, "packaged": {k: b[k] for k in keys}}
    report = {"passed": all(c["passed"] for c in checks.values()), "checks": checks,
              "scope": "Same observations, seed 1, hardware and backend; not a benchmark score."}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["passed"], "checks": {k: v["passed"] for k, v in checks.items()}}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()

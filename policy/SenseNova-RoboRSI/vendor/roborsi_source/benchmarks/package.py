# Copyright 2026 The HEART Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export a minimal Docker build context without Agent code or credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from benchmarks.build_inputs import build_inputs, file_hashes, include_inputs
from roborsi.benchmarks.runtime_source import (
    benchmark_source_digest,
    benchmark_source_files,
)


def prepare_context(benchmark: str, destination: Path, *, source_root: Path) -> Path:
    """Copy the declared server bundle into a fresh build context."""
    source_root = source_root.resolve()
    destination = destination.absolute()
    source = source_root / "benchmarks" / benchmark
    before = benchmark_source_digest(source_root, benchmark)
    if destination.resolve().is_relative_to(source):
        raise ValueError("build context must be outside the Benchmark input directory")
    inputs = build_inputs(source)
    hashes = file_hashes(source, inputs)
    destination.mkdir(parents=True, exist_ok=False)
    try:
        runtime = destination / "benchmark-runtime"
        for name in benchmark_source_files(benchmark):
            target = runtime / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root / name, target)
        for path in inputs:
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        if (
            file_hashes(source, build_inputs(source)) != hashes
            or file_hashes(destination, tuple(destination / p for p in hashes))
            != hashes
            or benchmark_source_digest(runtime, benchmark) != before
            or benchmark_source_digest(source_root, benchmark) != before
        ):
            raise ValueError("Benchmark source or build inputs changed during export")
        ignore = destination / "Dockerfile.dockerignore"
        ignore.write_text(include_inputs(ignore.read_text(), tuple(hashes)))
        (destination / "source.json").write_text(
            json.dumps(
                {
                    "benchmark": benchmark,
                    "sha256": before,
                    "files": benchmark_source_files(benchmark),
                },
                indent=2,
            )
            + "\n"
        )
        exported = file_hashes(
            destination, tuple(sorted(p for p in destination.rglob("*") if p.is_file()))
        )
        context_hash = hashlib.sha256(
            json.dumps(exported, sort_keys=True).encode()
        ).hexdigest()
        (destination / "build.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "benchmark": benchmark,
                    "context_sha256": context_hash,
                    "files": exported,
                },
                indent=2,
            )
            + "\n"
        )
        return destination
    except BaseException:
        shutil.rmtree(destination)
        raise


def main() -> None:
    """Prepare an image context; build and launch tools remain operator-selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", required=True, help="registered Benchmark source bundle name"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        prepare_context(
            args.benchmark,
            args.output.absolute(),
            source_root=Path(__file__).resolve().parents[1],
        )
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

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

"""Export a run's trajectory as a portable, self-contained HTML report."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from roborsi.evaluation.trajectory import TrajectoryReader
from roborsi.utils.logging import get_logger

logger = get_logger("trajectory")


def export_trajectory(run_dir: str | Path, output_dir: str | Path) -> Path:
    """Write HTML and referenced media to an empty report directory."""
    reader = TrajectoryReader(run_dir)
    if not (reader.output_dir / "trace/events.jsonl").is_file():
        raise ValueError("run has no trajectory; enable runtime trace before recording")
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError("report output directory must be empty")
    data = reader.index()
    data["details"] = {key: reader.turn(key) for key in reader.projection.turns}
    output.mkdir(parents=True, exist_ok=True)
    references: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            reference = value.get("artifact_ref")
            if isinstance(reference, str):
                references.add(reference)
            for child in value.values():
                collect(child)
        elif isinstance(value, list):
            for child in value:
                collect(child)

    collect(data)
    data["media_urls"] = {}
    for reference in sorted(references):
        source = reader.artifact_path(reference)
        if source.is_file():
            destination = output / "media" / reference
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            data["media_urls"][reference] = destination.relative_to(output).as_posix()
    static = Path(__file__).parents[1] / "dashboard/static"
    script = (static / "trajectory.js").read_text(encoding="utf-8")
    style = (static / "trajectory.css").read_text(encoding="utf-8")
    serialized = (
        json.dumps(data, ensure_ascii=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>RoboRSI trajectory</title><style>" + style + "</style>"
        '<main id="trajectory"></main><script id="trajectory-data" type="application/json">'
        + serialized
        + "</script><script>"
        + script
        + "\n"
        'const data = JSON.parse(document.getElementById("trajectory-data").textContent);'
        'const view = new RoboRSITrajectory(document.getElementById("trajectory"), {'
        "loadTurn: async id => data.details[id], resolveMedia: ref => data.media_urls[ref] || null});"
        "view.update(data);</script></html>"
    )
    index = output / "index.html"
    index.write_text(html, encoding="utf-8")
    return index


def main() -> None:
    """Run the offline trajectory exporter."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logger.info("Trajectory report: %s", export_trajectory(args.run_dir, args.output))


if __name__ == "__main__":
    main()

"""Stage 0: the index line of every head frame of one capture, from its tapes.

usage: python scripts/trim_tapes.py CAPTURE_FRAMES_DIR OUT_JSONL --set NAME

CAPTURE_FRAMES_DIR holds env-*/episode-*/tape.jsonl and env-*/episode-*/frames/*_cam_head.png. For each
head frame it writes one line: the neutral set name, the view (environment and episode), the frame
file stem, and both arms' joint positions from the tape's start line, which is the pose the frame was
taken in. Nothing else of the tape is kept.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out")
    ap.add_argument("--set", required=True)
    a = ap.parse_args()
    root = Path(a.capture)
    lines = []
    for tape in sorted(root.glob("env-*/episode-*/tape.jsonl")):
        env, ep = tape.parts[-3], tape.parts[-2]
        start = None
        with open(tape) as fh:
            for raw in fh:
                r = json.loads(raw)
                if r.get("kind") == "start":
                    start = r
                    break
        if start is None:
            continue
        for png in sorted((tape.parent / "frames").glob("*_cam_head.png")):
            lines.append({"set": a.set, "view": f"{env}_{ep}", "frame": png.name.split("_")[0],
                          "q": {k: [round(float(x), 6) for x in start["q"][k]] for k in ("left", "right")}})
    with open(a.out, "w") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    print(f"{len(lines)} head frames -> {a.out}")


if __name__ == "__main__":
    main()

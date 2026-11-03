#!/usr/bin/env python3
"""Turn a recorded episode into a replay script for `harness run --script`.

    python scripts/replay_script.py RUN/stack_blocks/seed-0 > replies.json
    harness run configs/experiments/dev.toml --tasks stack_blocks --seeds 0 --script replies.json \
        --output /tmp/replay

The model replies are replayed in order (no model calls) against the same task
and dev seed, so a code change can be checked end to end in the simulator; the
verifier answers are replayed as recorded. Prints the script as JSON and,
on stderr, the recorded outcome to compare with. Output: {"replies": [...],
"verifier": [...]}.
"""

import argparse
import json
from pathlib import Path
import sys


def replies(run: Path):
    texts, answers = [], []
    for line in (run / "events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "model_reply":
            texts.append(event["text"])
        elif event["event"] == "verify":
            answers.append(event["answer"])
    return texts, answers


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path, help="episode directory with events.jsonl")
    args = parser.parse_args()
    texts, answers = replies(args.run)
    json.dump({"replies": texts, "verifier": answers}, sys.stdout, indent=1)
    sys.stdout.write("\n")
    summary = json.loads((args.run / "summary.json").read_text()) if (args.run / "summary.json").exists() else {}
    print(f"{len(texts)} replies, {len(answers)} verifier answers; recorded outcome: "
          f"status={summary.get('status')} reward={summary.get('reward')} steps={summary.get('steps')}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

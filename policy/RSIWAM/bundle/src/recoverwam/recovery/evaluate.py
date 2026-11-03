"""Report paired recovery only after matching frozen state and runtime evidence."""
import argparse
import json
from pathlib import Path

from .protocol import paired_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--training-layouts", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    results = [json.loads(line) for line in args.results.read_text().splitlines() if line.strip()]
    layouts = json.loads(args.training_layouts.read_text()) if args.training_layouts else []
    report = paired_metrics(cases["cases"] if isinstance(cases,dict) else cases,
                            results,args.candidate,layouts)
    with args.output.open("x") as f:
        json.dump(report,f,indent=2,allow_nan=False)
        f.write("\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()

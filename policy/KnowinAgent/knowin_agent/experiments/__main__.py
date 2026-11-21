"""python -m knowin_agent.experiments {switches,resolve,splits,tables,ingest} ..."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence


def _switches(argv: Sequence[str]) -> int:
    from knowin_agent.experiments.switches import table_rows

    for row in table_rows():
        print(f"{row['switch']:32} {row['values']:28} default={row['default']!s:12} {row['module']:6} "
              f"{row['compliance']:10} {','.join(row['experiments'])}")
    return 0


def _resolve(argv: Sequence[str]) -> int:
    from knowin_agent.experiments.switches import apply, compliance_report, resolve

    parser = argparse.ArgumentParser(prog="python -m knowin_agent.experiments resolve")
    parser.add_argument("profile")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=JSON", help="override one switch")
    parser.add_argument("--config", help="run config to apply the switches to (prints the result)")
    args = parser.parse_args(argv)
    overrides = {}
    for item in args.set:
        name, _, text = item.partition("=")
        try:
            overrides[name] = json.loads(text)
        except json.JSONDecodeError:
            overrides[name] = text
    values = resolve(args.profile, overrides)
    if args.config:
        with open(args.config, encoding="utf-8") as handle:
            print(json.dumps(apply(json.load(handle), values), indent=2))
    else:
        print(json.dumps({"switches": values, "compliance": compliance_report(values)}, indent=2))
    return 0


def _ingest(argv: Sequence[str]) -> int:
    from knowin_agent.experiments.results import ingest

    parser = argparse.ArgumentParser(prog="python -m knowin_agent.experiments ingest")
    parser.add_argument("--results", required=True)
    parser.add_argument("--runs", required=True, help="directory searched for run manifests")
    args = parser.parse_args(argv)
    print(f"appended {ingest(args.results, args.runs)} manifests to {args.results}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"switches": _switches, "resolve": _resolve, "ingest": _ingest}
    if argv and argv[0] == "splits":
        from knowin_agent.experiments.splits import main as splits_main

        return splits_main(argv[1:])
    if argv and argv[0] == "tables":
        from knowin_agent.experiments.make_tables import main as tables_main

        return tables_main(argv[1:])
    if not argv or argv[0] not in commands:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    return commands[argv[0]](argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())

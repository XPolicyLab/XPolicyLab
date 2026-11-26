"""Prepare a pinned policy configuration and install its separate XPolicyLab adapter."""
import argparse
import json
from pathlib import Path
import shutil
import sys

from .contracts import digest, encoded, file_digest, validate_config
from .takeover import SCHEDULE_FILES

ADAPTER_NAME = "OEHInternalPolicy"
OFFICIAL_DEPLOY_SHA256 = "a114b1f8168b3c170d2f20389cca63de6e9ef413b12a859e8129923c031cf685"


def install(root: Path) -> Path:
    original = root.resolve() / "policy/OpenWAM"
    if file_digest(original / "deploy.py") != OFFICIAL_DEPLOY_SHA256:
        raise ValueError("official episode loop changed; review the gateway before installing")
    target = original.parent / ADAPTER_NAME
    required = ("eval.sh", "setup_eval_policy_server.sh", "setup_eval_env_client.sh", "deploy.yml")
    for name in required:
        if not (original / name).is_file():
            raise FileNotFoundError(original / name)
    if not (original / "OpenWAM").is_dir():
        raise FileNotFoundError(original / "OpenWAM")
    deploy = (original / "deploy.yml").read_text(encoding="utf-8")
    if "policy_name: OpenWAM" not in deploy or "eval_batch: true" not in deploy:
        raise ValueError("unexpected official deployment configuration")
    target.mkdir(exist_ok=False)
    for name in required[:3]:
        shutil.copy2(original / name, target / name)
    # The campaign's XPolicyLab request_timeout_s, the same for every RoboDojo adapter (oeh_eval.install).
    from oeh_eval.install import with_request_timeout
    (target / "deploy.yml").write_text(with_request_timeout(
        deploy.replace("policy_name: OpenWAM", f"policy_name: {ADAPTER_NAME}").replace("eval_batch: true",
                                                                                     "eval_batch: false")), encoding="utf-8")
    (target / "model.py").write_text("from oeh_policy.model import Model\n", encoding="utf-8")
    (target / "deploy.py").write_text("from oeh_policy.deploy import eval_one_episode, eval_one_episode_batch\n",
                                      encoding="utf-8")
    (target / "__init__.py").write_text("", encoding="utf-8")
    (target / "OpenWAM").symlink_to(original / "OpenWAM", target_is_directory=True)
    return target


_DIGESTS: dict[tuple[str, int, int], str] = {}


def _checkpoint_digest(path: Path) -> str:
    """file_digest, once per file version in this process: a matrix compile freezes one config per case, and the
    RoboDojo checkpoint is 25 GB."""
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in _DIGESTS:
        _DIGESTS[key] = file_digest(path)
    return _DIGESTS[key]


def freeze(template: Path, output: Path) -> dict:
    cfg = json.loads(template.read_text(encoding="utf-8"))
    repo = Path(cfg["repository"]).resolve()
    cfg["repository"] = str(repo)
    cfg["kernel"] = str(Path(cfg["kernel"]).resolve())
    cfg["output"] = str(Path(cfg["output"]).resolve())
    official = cfg["openwam"]
    checkpoint = Path(official["ckpt_dir"]).resolve()
    source = Path(official["openwam_root"]).resolve()
    official["ckpt_dir"], official["openwam_root"] = str(checkpoint), str(source)
    files = {str(p.relative_to(checkpoint)): _checkpoint_digest(p) for p in sorted(checkpoint.rglob("*"))
             if p.is_file()}
    if not files:
        raise ValueError("empty checkpoint directory")
    cfg["checkpoint_sha256"] = digest(files)
    cfg["normalization_sha256"] = file_digest(checkpoint / "normalization_stats.npy")
    cfg["openwam_source_sha256"] = digest({str(p.relative_to(source)): file_digest(p)
                                          for p in sorted((source / "openwam").rglob("*.py"))})
    paths = [Path(path).resolve() for path in cfg.pop("identity_paths")]
    paths.extend([Path(cfg["kernel"]), Path(__file__).with_name("method.txt")])
    paths.extend(Path(__file__).parent.glob("*.py"))
    if cfg.get("mode") == "review-stmem":
        stmem = cfg["stmem"]
        paths.append(Path(stmem["calibration_file"]).resolve())
        paths.append(Path(stmem["sam3_checkpoint"]).resolve())
        depth_root = Path(stmem["da3_checkpoint"]).resolve()
        depth_files = sorted(p for p in depth_root.rglob("*") if p.is_file())
        if not depth_files:
            raise ValueError("review-stmem requires local pinned DA3 checkpoint files before freezing")
        paths.extend(depth_files)
    if cfg.get("mode") in {"review", "review-stmem"}:
        paths.append(Path(__file__).with_name("llm_interaction_contract.md"))
        # The teacher's instructions, gate and task notes are part of the method: a lock detects any change. A
        # schedule's prompt addendum only where that schedule runs: every_chunk configs freeze the same files as before.
        # Files only: teacher/<benchmark>/ holds another benchmark's prompt set (oeh_policy.teacher.prompt_dir).
        schedule = SCHEDULE_FILES.get((cfg.get("review") or {}).get("schedule"))
        paths.extend(sorted(path for path in (Path(__file__).parent / "teacher").glob("*") if path.is_file()
                            and (path.name == schedule or path.name not in SCHEDULE_FILES.values())))
    if cfg.get("brain"):
        provider = Path(cfg["brain"]["providers"]).resolve()
        cfg["brain"]["providers"] = str(provider)
        paths.append(provider)
    # Stat-keyed once per file version (as the checkpoint): Explore configs list every experience asset.
    cfg["identity_files"] = {str(path): _checkpoint_digest(path) for path in paths}
    cfg["kernel_sha256"] = file_digest(Path(cfg["kernel"]))
    validated = validate_config(cfg)
    if cfg.get("mode") in {"review", "review-stmem"}:
        cfg = validated  # the review block's defaults are frozen explicitly: the lock records every effective value
    with output.open("xb") as stream:
        stream.write(encoded(cfg))
    return {"configuration_sha256": digest(cfg), "checkpoint_sha256": cfg["checkpoint_sha256"],
            "normalization_sha256": cfg["normalization_sha256"]}


def _notes_parser(commands) -> None:
    notes = commands.add_parser("notes", help="the G7 task-notes store (oeh_policy.notes_store)")
    sub = notes.add_subparsers(dest="notes_command", required=True)
    add = sub.add_parser("add")
    add.add_argument("--store", type=Path, required=True)
    add.add_argument("--task", required=True)
    add.add_argument("--signature", required=True)
    add.add_argument("--instead", required=True)
    add.add_argument("--evidence", required=True)
    add.add_argument("--layout")
    add.add_argument("--source-attempt")
    add.add_argument("--by", required=True)
    retire = sub.add_parser("retire")
    retire.add_argument("--store", type=Path, required=True)
    retire.add_argument("--index", type=int, required=True)
    retire.add_argument("--by", required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--store", type=Path, required=True)
    show = sub.add_parser("show")
    show.add_argument("--store", type=Path, required=True)
    show.add_argument("--task")
    show.add_argument("--scope", choices=("global", "task", "layout", "lessons"), default=None,
                      help="global/task/layout: the notes a packet would carry (JSON); lessons: the markdown render "
                           "(default: lessons)")
    show.add_argument("--layout")
    scan = sub.add_parser("scan")
    scan.add_argument("--store", type=Path, required=True)
    disjoint = sub.add_parser("check-disjoint")
    disjoint.add_argument("--exploration", type=Path, required=True)
    disjoint.add_argument("--against", type=Path, nargs="+", required=True)


def _notes(args) -> int:
    from . import notes_store as ns
    command = args.notes_command
    if command == "add":
        note = {"task": args.task, "signature": args.signature, "instead": args.instead, "evidence": args.evidence,
                "layout": args.layout, "source_attempt": args.source_attempt}
        print(json.dumps(ns.add_note(args.store, note, args.by), indent=2))
    elif command == "retire":
        print(json.dumps(ns.retire_note(args.store, args.index, args.by), indent=2))
    elif command == "seal":
        print(ns.seal_store(args.store))
    elif command == "check-disjoint":
        exploration = ns.read_id_list(args.exploration)
        overlap = ns.check_disjoint(exploration, *(ns.read_id_list(path) for path in args.against))
        print(json.dumps({"exploration": str(args.exploration), "ids": len(exploration),
                          "against": [str(path) for path in args.against], "overlap": overlap}, indent=2))
        return 1 if overlap else 0
    else:
        store = ns.load_store(args.store, ns.store_sha256(args.store))
        if command == "scan":
            hits = ns.scan_notes(store)
            print(f"{args.store}: {store['count']} notes, sha256 {store['sha256']}, {len(hits)} forbidden-word hits")
            for hit in hits:
                print(f"  {hit}")
            return 1 if hits else 0
        if args.scope in (None, "lessons"):
            print(ns.render_lessons(store, args.task), end="")
        else:
            print(json.dumps(ns.notes_for(store, args.task, args.scope, args.layout), indent=2, ensure_ascii=False))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    setup = commands.add_parser("install")
    setup.add_argument("xpolicylab_root", type=Path)
    pin = commands.add_parser("freeze")
    pin.add_argument("template", type=Path)
    pin.add_argument("output", type=Path)
    check = commands.add_parser("check")
    check.add_argument("configuration", type=Path)
    evidence = commands.add_parser("check-evidence")
    evidence.add_argument("attempt", type=Path)
    evidence.add_argument("--sidecar", type=Path)
    _notes_parser(commands)
    scan = commands.add_parser("scan-packets", help="G1: check_packet over every packet of recorded attempts")
    scan.add_argument("attempts", type=Path, nargs="+")
    scan.add_argument("--field-set", default="auto",
                      help="auto (identity.json review.context.field_set, else the shadow scan) or a field-set name")
    scan.add_argument("--json", action="store_true")
    diff = commands.add_parser("diff-packets", help="G1: compare two attempts' packets decision by decision")
    diff.add_argument("a", type=Path)
    diff.add_argument("b", type=Path)
    diff.add_argument("--ignore", nargs="*", default=[], metavar="KEY")
    diff.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "notes":
        try:
            code = _notes(args)
        except ValueError as error:
            print(f"error: {error}", file=sys.stderr)
            code = 2
        if code:
            raise SystemExit(code)
    elif args.command == "scan-packets":
        from .packet_scan import format_scan, scan_attempt
        results = [scan_attempt(attempt, args.field_set) for attempt in args.attempts]
        print(json.dumps(results, indent=2) if args.json else "\n".join(map(format_scan, results)))
        if any(result["violations"] for result in results):
            raise SystemExit(1)
    elif args.command == "diff-packets":
        from .packet_scan import diff_packets, format_diff
        result = diff_packets(args.a, args.b, args.ignore)
        print(json.dumps(result, indent=2) if args.json else format_diff(result))
        if not result["identical"]:
            raise SystemExit(1)
    elif args.command == "install":
        print(install(args.xpolicylab_root))
    elif args.command == "freeze":
        print(json.dumps(freeze(args.template, args.output), indent=2))
    elif args.command == "check-evidence":
        from .evidence import validate_attempt
        result = validate_attempt(args.attempt, args.sidecar)
        print(json.dumps(result, indent=2))
        if not result["integrity_ok"]:
            raise SystemExit(1)
    else:
        cfg = validate_config(json.loads(args.configuration.read_text(encoding="utf-8")))
        print(json.dumps({"configuration_sha256": digest(cfg), "mode": cfg["mode"]}, indent=2))


if __name__ == "__main__":
    main()

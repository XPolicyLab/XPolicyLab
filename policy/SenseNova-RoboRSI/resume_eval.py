"""Validate an own-episode resume, then invoke the standard ten-argument eval.sh."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from .hybrid_resume import digest, load_resume, read_json
except ImportError:
    from hybrid_resume import digest, load_resume, read_json


def launch_environment(request_path: Path, arguments: list[str], environ: dict) -> dict:
    if len(arguments) != 10:
        raise ValueError("Use the ten standard eval.sh arguments after --")
    path = request_path.absolute()
    request = read_json(path)
    identity = request.get("identity", {})
    request = load_resume(path, identity)
    if (arguments[0], arguments[1], arguments[3], arguments[4], int(arguments[5])) != (
        "RoboDojo", identity["task"], "arx_x5", "joint", identity["eval_seed"]
    ):
        raise ValueError("Evaluation arguments differ from the stopped native episode")
    if not identity.get("run_id"):
        raise ValueError("Resume needs its original native run ID")
    if path.with_suffix(path.suffix + ".claimed").exists():
        raise ValueError("Resume request already started a controller")
    if environ.get("ROBORSI_RESUME_MANIFEST"):
        raise ValueError("Unset the obsolete ROBORSI_RESUME_MANIFEST variable")
    result = dict(environ)
    bindings = {
        "ROBORSI_RESUME_REQUEST": str(path),
        "ROBORSI_RESUME_REQUEST_SHA256": digest(path),
        "ROBORSI_NATIVE_LAYOUT_ID": str(identity["layout"]),
        "ROBORSI_NATIVE_SCENE_SHA256": request["native_identity"]["scene_sha256"],
        "ROBODOJO_RUN_ID": identity["run_id"],
    }
    for key, value in bindings.items():
        if key in result and result[key] and result[key] != value:
            raise ValueError(f"Conflicting resume launch variable: {key}")
        result[key] = value
    result["ROBORSI_REQUIRE_NATIVE_IDENTITY"] = "1"
    result["EVAL_ENV_TYPE"] = "sim"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("eval_arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    arguments = args.eval_arguments
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    environment = launch_environment(args.request, arguments, os.environ)
    if args.check_only:
        print("Resume source, task, seed, layout, scene and evaluation arguments verified")
        return
    script = Path(__file__).resolve().parent / "eval.sh"
    os.execvpe("bash", ["bash", str(script), *arguments], environment)


if __name__ == "__main__":
    main()

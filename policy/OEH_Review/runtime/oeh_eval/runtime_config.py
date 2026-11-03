"""What each entry point actually ran with, reported by the entry point itself.

The campaign freezes declared values; these reports are the effective ones. The
policy server reports the exact arguments it hands the official OpenWAM Model
(RoboDojo) or `build_server_from_config` (LIBERO, RoboTwin), and the simulator-side
boundary reports the eval client's effective settings before the first policy
call: RoboDojo's launch flags and step limit, LIBERO's merged `single_eval.py`
config, and what RoboTwin's pinned `eval_policy()` received. Both files sit next
to the attempt's policy event sidecar.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

SCHEMA = "oeh.eval.runtime_config.v1"
POLICY_FILE = "runtime-config-policy.json"
ENV_FILE = "runtime-config-env.json"
# Written by a policy server that can serve no further episode in this attempt; the evaluator stops the attempt
# when it appears instead of letting an official client cycle seeds until the wall (oeh_eval.campaign).
FATAL_FILE = "policy-fatal.json"
EXPECTED_ENV = "OEH_EVAL_EXPECTED_RUNTIME"
EXPECTED_POLICY_ENV = "OEH_EVAL_EXPECTED_POLICY_RUNTIME"
# Wiring the XPolicyLab launcher assigns per process; none of it changes what OpenWAM computes.
TRANSPORT_KEYS = frozenset({"policy_name", "protocol", "host", "port", "eval_batch", "gpu_id"})
ENV_FLAGS = ("task_name", "env_cfg_type", "num_envs", "seed")
# LIBERO entries. The client's transport and output location differ per attempt by design.
LIBERO_CLIENT_ENTRY = "libero_single_eval"
OPENWAM_SERVER_ENTRY = "openwam_server"
# A session of the GPU's shared OpenWAM service (oeh_eval.serving): reported by the launcher that leased it, with what
# the service's engine was built from, in the same structure as OPENWAM_SERVER_ENTRY's constructor_args.
OPENWAM_SHARED_ENTRY = "openwam_shared_session"
CLIENT_TRANSPORT_KEYS = frozenset({"host", "port", "request_timeout", "result_dir", "debug_dir"})
# Official server flags that the campaign pins; the rest of the parsed CLI is reported as is.
SERVER_PATH_FLAGS = frozenset({"config", "ckpt_dir", "ckpt_name", "device", "host", "port"})
# RoboTwin's client: what the pinned eval_policy() received. Only the policy route differs between conditions.
ROBOTWIN_CLIENT_ENTRY = "robotwin_client"
ROBOTWIN_ROUTE_KEYS = frozenset({"policy_name", "ckpt_setting", "policy_module", "policy_module_file"})
# EBench's client: what OpenWAM's released bridge loop ran with. Only the driver and the run id differ between
# conditions (oeh_eval.ebench_client).
EBENCH_CLIENT_ENTRY = "ebench_client"
EBENCH_ROUTE_KEYS = frozenset({"policy_module", "policy_module_file", "run_id"})


def _simple(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _simple(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_simple(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _write(name: str, payload: Mapping[str, Any]) -> Path | None:
    events = os.environ.get("OEH_EVAL_EVENTS")
    if not events:
        return None
    path = Path(events).parent / name
    path.parent.mkdir(parents=True, exist_ok=True)
    # One report per attempt: a second writer means a reused process or a restart.
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"schema": SCHEMA, **payload}, stream, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    return path


def report_policy(constructor_args: Mapping[str, Any]) -> None:
    """Record the argument mapping passed to the official OpenWAM Model."""
    _write(POLICY_FILE, {"entry": "openwam_model", "constructor_args": _simple(constructor_args),
                         "environment": {key: os.environ.get(key) for key in
                                         ("OPENWAM_CKPT_DIR", "OPENWAM_ROOT", "OPENWAM_DEPLOY_CONFIG")}})


def report_policy_fatal(reason: str, *, provider_error: Mapping[str, Any] | None = None) -> Path | None:
    """This policy server can serve no further episode in the attempt (its runtime failed to start, the official
    loop asked it for a second episode, or a hosted model's provider refused it). ``provider_error`` is that
    provider's own answer, as its HTTP status, error type and code (never its message): the campaign counts the
    attempt as infrastructure when the answer is one of the refusals in ``provider_refusal``. The first report is
    kept; later ones find the file and change nothing, so a refusal is reported before anything raises."""
    report = {"entry": "openwam_model", "reason": str(reason)}
    if provider_error is not None:
        report["provider_error"] = {key: provider_error.get(key) for key in ("status", "type", "code")}
    try:
        return _write(FATAL_FILE, report)
    except FileExistsError:
        return None


# A hosted model's provider refusing the operator's account, key, model or budget. Seen 2026-09-28 (the OEH
# baseline line's review teacher): HTTP 429, type insufficient_quota, code project_spend_limit_exceeded. Not
# refusals: rate limits, server errors and an unreachable API, which a policy retries or falls back through as
# designed, and a budget the study itself declares (OEH's own call budgets), which is the policy's to keep.
PROVIDER_REFUSAL_STATUS = frozenset({401, 403, 404})
PROVIDER_REFUSAL_TYPES = frozenset({"insufficient_quota", "authentication_error", "permission_error"})
PROVIDER_REFUSAL_CODES = frozenset({"insufficient_quota", "invalid_api_key", "model_not_found", "account_deactivated",
                                    "project_spend_limit_exceeded", "billing_hard_limit_reached",
                                    "billing_not_active"})


# A hosted model the policy runtime could not reach at its preflight, before the episode's first step (the policy's
# own code, oeh_policy.teacher.PREFLIGHT_UNREACHABLE): no episode ran, so neither arm's policy failed. An outage
# during the episode is not on this list.
PREFLIGHT_UNREACHABLE = "unreachable_before_episode"


def provider_refusal(report: Any) -> str | None:
    """The provider refusal a policy-fatal report records, as text; None when it records none of the refusals above.
    The evaluator's list decides, not the policy: a policy cannot excuse other failures by reporting them."""
    error = report.get("provider_error") if isinstance(report, Mapping) else None
    if not isinstance(error, Mapping):
        return None
    status, kind, code = error.get("status"), error.get("type"), error.get("code")
    if code == PREFLIGHT_UNREACHABLE:
        return f"unreachable at the policy's preflight, before the episode ({kind})"
    if status in PROVIDER_REFUSAL_STATUS or kind in PROVIDER_REFUSAL_TYPES or code in PROVIDER_REFUSAL_CODES:
        return f"HTTP {status}, type {kind}, code {code}"
    return None


def report_robotwin_env(effective: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Record what the RoboTwin client's `eval_policy()` received; refuse to start on a mismatch."""
    view = _simple(effective)
    differences = env_differences(view, expected)
    _write(ENV_FILE, {"entry": ROBOTWIN_CLIENT_ENTRY, "effective": view, "expected": _simple(expected),
                      "differences": differences})
    if differences:
        raise RuntimeError("effective runtime config differs from campaign: " + "; ".join(differences))


def report_robotwin_refusal(reason: str, expected: Mapping[str, Any]) -> None:
    """An entry point that stops before `eval_policy()` (bad configuration) says why; the campaign seals preflight_failed."""
    events = os.environ.get("OEH_EVAL_EVENTS")
    if events and (Path(events).parent / ENV_FILE).exists():
        return
    _write(ENV_FILE, {"entry": ROBOTWIN_CLIENT_ENTRY, "effective": None, "refused": str(reason),
                      "expected": _simple(expected), "differences": [f"refused to start: {reason}"]})


def report_ebench_env(effective: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Record what the EBench client's bridge loop is about to run with; refuse to start on a mismatch."""
    view = _simple(effective)
    differences = env_differences(view, expected)
    _write(ENV_FILE, {"entry": EBENCH_CLIENT_ENTRY, "effective": view, "expected": _simple(expected),
                      "differences": differences})
    if differences:
        raise RuntimeError("effective runtime config differs from campaign: " + "; ".join(differences))


def report_ebench_refusal(reason: str, expected: Mapping[str, Any]) -> None:
    """An EBench entry point that stops before the bridge loop (bad configuration) says why."""
    events = os.environ.get("OEH_EVAL_EVENTS")
    if events and (Path(events).parent / ENV_FILE).exists():
        return
    _write(ENV_FILE, {"entry": EBENCH_CLIENT_ENTRY, "effective": None, "refused": str(reason),
                      "expected": _simple(expected), "differences": [f"refused to start: {reason}"]})


def parse_flags(argv: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    for index, item in enumerate(argv):
        if item.startswith("--") and index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            flags[item[2:]] = argv[index + 1]
    return flags


def env_view(argv: list[str], step_lim: Any, num_envs: Any = None) -> dict[str, Any]:
    """Task, config and seed come from the eval client's argv; the environment count and step limit from the live
    task env, because RoboDojo forces one environment under ``eval_batch: false`` whatever ``--num_envs`` said."""
    flags = parse_flags(argv)
    view: dict[str, Any] = {key: flags.get(key) for key in ENV_FLAGS}
    for key in ("num_envs", "seed"):
        if view[key] is not None and view[key].lstrip("-").isdigit():
            view[key] = int(view[key])
    view["argv_num_envs"] = view["num_envs"]
    if type(num_envs) is int:
        view["num_envs"] = num_envs
    view["step_lim"] = step_lim if type(step_lim) is int else None
    return view


def env_differences(view: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    return [f"{key}: ran {view.get(key)!r}, frozen {value!r}"
            for key, value in sorted(expected.items()) if view.get(key) != value]


def report_env(task_env: Any) -> None:
    """Record the eval client's effective settings; refuse to start on a mismatch."""
    if not os.environ.get("OEH_EVAL_EVENTS"):
        return
    argv = [str(item) for item in sys.argv]
    view = env_view(argv, getattr(task_env, "step_lim", None), getattr(task_env, "num_envs", None))
    expected = json.loads(os.environ.get(EXPECTED_ENV) or "{}")
    differences = env_differences(view, expected)
    _write(ENV_FILE, {"entry": "eval_client", "argv": argv, "effective": view,
                      "expected": expected, "differences": differences})
    if differences:
        raise RuntimeError("effective runtime config differs from campaign: " + "; ".join(differences))


def single_eval_view(cfg: Mapping[str, Any], resolved: Mapping[str, Any]) -> dict[str, Any]:
    """The merged `single_eval.py` config as `run_eval` receives it, plus the official resolutions."""
    return {**_simple(cfg), "resolved": _simple(resolved)}


def report_single_eval(benchmark: str, argv: list[str], cfg: Mapping[str, Any],
                       resolved: Mapping[str, Any]) -> None:
    """Record LIBERO's effective client config; refuse to start on a campaign mismatch."""
    if not os.environ.get("OEH_EVAL_EVENTS"):
        return
    view = single_eval_view(cfg, resolved)
    expected = json.loads(os.environ.get(EXPECTED_ENV) or "{}")
    differences = env_differences(view, expected)
    _write(ENV_FILE, {"entry": LIBERO_CLIENT_ENTRY, "benchmark": benchmark, "argv": list(argv),
                      "effective": view, "expected": expected, "differences": differences})
    if differences:
        raise RuntimeError("effective single_eval config differs from campaign: " + "; ".join(differences))


def _plain_config(cfg: Any) -> Any:
    try:
        from omegaconf import OmegaConf
        if OmegaConf.is_config(cfg):
            return OmegaConf.to_container(cfg, resolve=True)
    except ImportError:
        pass
    return _simple(cfg)


def server_view(parsed: Mapping[str, Any], cfg: Any, *, ckpt_dir: Any, device: Any,
                ckpt_name: Any) -> dict[str, Any]:
    """Arguments handed to OpenWAM's `build_server_from_config`, and the CLI that produced them."""
    config = Path(parsed["config"]).resolve() if parsed.get("config") else None
    return {"ckpt_dir": str(Path(ckpt_dir).resolve()) if ckpt_dir else None,
            "ckpt_name": ckpt_name, "device": device,
            "deploy_config": str(config) if config else None,
            "deploy_config_sha256": hashlib.sha256(config.read_bytes()).hexdigest()
            if config and config.is_file() else None,
            "deploy_cfg": _plain_config(cfg),
            "server_args": {k: _simple(v) for k, v in parsed.items() if k not in SERVER_PATH_FLAGS},
            "host": parsed.get("host"), "port": parsed.get("port")}


def policy_differences(args: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    out = []
    for key, value in sorted(expected.items()):
        if key == "server_args" and isinstance(value, Mapping):
            ran = args.get("server_args") if isinstance(args.get("server_args"), Mapping) else {}
            out.extend(f"server_args.{flag}: ran {ran.get(flag)!r}, frozen {want!r}"
                       for flag, want in sorted(value.items()) if ran.get(flag) != want)
        elif args.get(key) != value:
            out.append(f"{key}: ran {args.get(key)!r}, frozen {value!r}")
    return out


def report_openwam_server(argv: list[str], parsed: Mapping[str, Any], cfg: Any, *, ckpt_dir: Any,
                          device: Any, ckpt_name: Any) -> None:
    """Record what the official OpenWAM server is built from; refuse before loading weights on a mismatch."""
    if not os.environ.get("OEH_EVAL_EVENTS"):
        return
    args = server_view(parsed, cfg, ckpt_dir=ckpt_dir, device=device, ckpt_name=ckpt_name)
    expected = json.loads(os.environ.get(EXPECTED_POLICY_ENV) or "{}")
    differences = policy_differences(args, expected)
    _write(POLICY_FILE, {"entry": OPENWAM_SERVER_ENTRY, "argv": list(argv), "constructor_args": args,
                         "expected": expected, "differences": differences,
                         "environment": {"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES")}})
    if differences:
        raise RuntimeError("effective OpenWAM server config differs from campaign: " + "; ".join(differences))


def report_shared_session(identity: Mapping[str, Any], lease: Mapping[str, Any],
                          expected: Mapping[str, Any] | None = None) -> None:
    """Record the shared OpenWAM service a launcher leased a session of; refuse on a campaign mismatch. The service
    was built once for many attempts, so its build arguments come from its identity, not from this process."""
    if not os.environ.get("OEH_EVAL_EVENTS"):
        return
    args = dict(identity.get("server_view") or {})
    expected = dict(expected) if expected is not None else json.loads(os.environ.get(EXPECTED_POLICY_ENV) or "{}")
    differences = policy_differences(args, expected)
    _write(POLICY_FILE, {"entry": OPENWAM_SHARED_ENTRY, "constructor_args": args, "expected": expected,
                         "differences": differences, "service": _simple(lease)})
    if differences:
        raise RuntimeError("the shared OpenWAM service's config differs from campaign: " + "; ".join(differences))


def load(directory: Path, expected_env: Mapping[str, Any],
         expected_policy: Mapping[str, Any] | None = None,
         serving_mode: str = "exclusive") -> tuple[dict[str, Any], list[str], list[str]]:
    """Read both reports for one attempt: (effective, missing or malformed, differences). A shared-serving condition
    reports its OpenWAM as OPENWAM_SHARED_ENTRY, an exclusive one as OPENWAM_SERVER_ENTRY; each refuses the other."""
    issues: list[str] = []
    differences: list[str] = []
    effective: dict[str, Any] = {"env": None, "openwam": None}
    for name, key in ((ENV_FILE, "env"), (POLICY_FILE, "openwam")):
        path = directory / name
        if not path.is_file():
            issues.append(f"effective runtime config not reported: {name}")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            issues.append(f"effective runtime config unreadable: {name}")
            continue
        if not isinstance(data, dict) or data.get("schema") != SCHEMA:
            issues.append(f"effective runtime config has unknown schema: {name}")
            continue
        if key == "env" and data.get("entry") == ROBOTWIN_CLIENT_ENTRY:
            # Compare the reported values against the campaign here, not the reporter's own verdict.
            if data.get("refused"):
                differences.append(f"effective runtime config differs: entry point refused to start: {data['refused']}")
            elif not isinstance(data.get("effective"), dict):
                issues.append("effective RoboTwin client config lacks its values")
            else:
                view = data["effective"]
                effective["env"] = view
                effective["robotwin_client"] = {k: v for k, v in view.items() if k not in ROBOTWIN_ROUTE_KEYS}
                differences.extend(f"effective runtime config differs: {item}"
                                   for item in env_differences(view, expected_env))
        elif key == "env" and data.get("entry") == ROBOCASA_GR1_CLIENT_ENTRY:
            # As RoboTwin's: the reported values against the campaign, the policy route and transport left out of
            # the pair comparison (the merged single_eval.py config, as LIBERO's).
            if data.get("refused"):
                differences.append(f"effective runtime config differs: entry point refused to start: {data['refused']}")
            elif not isinstance(data.get("effective"), dict):
                issues.append("effective RoboCasa GR1 client config lacks its values")
            else:
                view = data["effective"]
                effective["env"] = view
                effective["single_eval"] = {k: v for k, v in view.items()
                                            if k not in CLIENT_TRANSPORT_KEYS | ROBOCASA_GR1_ROUTE_KEYS}
                differences.extend(f"effective runtime config differs: {item}"
                                   for item in env_differences(view, expected_env))
        elif key == "env" and data.get("entry") == VLABENCH_CLIENT_ENTRY:
            # Compare the reported values against the campaign here, not the reporter's own verdict.
            if data.get("refused"):
                differences.append(f"effective runtime config differs: entry point refused to start: {data['refused']}")
            elif not isinstance(data.get("effective"), dict):
                issues.append("effective VLABench client config lacks its values")
            else:
                view = data["effective"]
                effective["env"] = view
                effective["single_eval"] = {k: v for k, v in view.items() if k not in VLABENCH_ROUTE_KEYS}
                differences.extend(f"effective runtime config differs: {item}"
                                   for item in env_differences(view, expected_env))
        elif key == "env" and data.get("entry") == EBENCH_CLIENT_ENTRY:
            if data.get("refused"):
                differences.append(f"effective runtime config differs: entry point refused to start: {data['refused']}")
            elif not isinstance(data.get("effective"), dict):
                issues.append("effective EBench client config lacks its values")
            else:
                view = data["effective"]
                effective["env"] = view
                effective["ebench_client"] = {k: v for k, v in view.items() if k not in EBENCH_ROUTE_KEYS}
                differences.extend(f"effective runtime config differs: {item}"
                                   for item in env_differences(view, expected_env))
        elif key == "env" and data.get("entry") == LIBERO_CLIENT_ENTRY:
            view = data.get("effective")
            if not isinstance(view, dict) or not isinstance(data.get("argv"), list):
                issues.append("effective single_eval config lacks its merged view or argv")
                continue
            # Compare the reported view against the campaign here, not the reporter's own verdict.
            effective["env"] = view
            effective["single_eval"] = {k: v for k, v in view.items() if k not in CLIENT_TRANSPORT_KEYS}
            differences.extend(f"effective runtime config differs: {item}"
                               for item in env_differences(view, expected_env))
        elif key == "env":
            if not isinstance(data.get("argv"), list):
                issues.append("effective eval client config lacks argv")
                continue
            # Recompute from argv rather than trusting the reporter's own verdict.
            reported = data.get("effective") if isinstance(data.get("effective"), dict) else {}
            effective["env"] = env_view([str(item) for item in data["argv"]], reported.get("step_lim"), reported.get("num_envs"))
            differences.extend(f"effective runtime config differs: {item}"
                               for item in env_differences(effective["env"], expected_env))
        else:
            args = data.get("constructor_args")
            if not isinstance(args, dict):
                issues.append("effective OpenWAM config lacks constructor arguments")
                continue
            effective["openwam"] = {k: v for k, v in args.items() if k not in TRANSPORT_KEYS}
            if expected_policy is not None:
                entry = OPENWAM_SHARED_ENTRY if serving_mode == "shared" else OPENWAM_SERVER_ENTRY
                if data.get("entry") != entry:
                    issues.append(f"effective OpenWAM config has entry {data.get('entry')!r}, "
                                  f"expected {entry!r}")
                    continue
                differences.extend(f"effective runtime config differs: {item}"
                                   for item in policy_differences(args, expected_policy))
    return effective, issues, differences


def _differences(prefix: str, left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"field": f"{prefix}.{key}", "left": left.get(key, "<absent>"), "right": right.get(key, "<absent>")}
            for key in sorted(set(left) | set(right)) if left.get(key, "<absent>") != right.get(key, "<absent>")]


def openwam_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _differences("openwam", left, right)


def client_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _differences("single_eval", left, right)


def robotwin_client_differences(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Two RoboTwin client reports, the policy route (policy_name, ckpt_setting, module) excluded."""
    return _differences("robotwin_client", left, right)


# RoboCasa GR1's client: the pinned single_eval.py's merged config, the seed its env was made with and the files the
# loop loaded (oeh_eval.robocasa_gr1_client). Only the policy route differs between conditions.
ROBOCASA_GR1_CLIENT_ENTRY = "robocasa_gr1_client"
ROBOCASA_GR1_ROUTE_KEYS = frozenset({"policy_factory", "policy_module"})


def report_robocasa_gr1_env(effective: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Record what the RoboCasa GR1 loop is about to run its episode with; refuse to start on a mismatch."""
    view = _simple(effective)
    differences = env_differences(view, expected)
    _write(ENV_FILE, {"entry": ROBOCASA_GR1_CLIENT_ENTRY, "effective": view, "expected": _simple(expected),
                      "differences": differences})
    if differences:
        raise RuntimeError("effective runtime config differs from campaign: " + "; ".join(differences))


def report_robocasa_gr1_refusal(reason: str, expected: Mapping[str, Any]) -> None:
    """An entry point that stops before the policy episode (bad configuration) says why; the campaign seals
    preflight_failed."""
    events = os.environ.get("OEH_EVAL_EVENTS")
    if events and (Path(events).parent / ENV_FILE).exists():
        return
    _write(ENV_FILE, {"entry": ROBOCASA_GR1_CLIENT_ENTRY, "effective": None, "refused": str(reason),
                      "expected": _simple(expected), "differences": [f"refused to start: {reason}"]})


# VLABench's client: what the pinned Evaluator was about to run its one episode with (oeh_eval.vlabench_client). Only
# the policy route, the per-attempt transport and the output location differ between conditions; the rest is compared
# between a pair's two reports under single_eval (it is OpenWAM's single_eval.py in both).
VLABENCH_CLIENT_ENTRY = "vlabench_single_eval"
VLABENCH_ROUTE_KEYS = frozenset({"transport", "save_dir", "policy_module", "policy_module_file"})


def report_vlabench_env(effective: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    """Record what VLABench's Evaluator is about to run with; refuse to start on a mismatch."""
    view = _simple(effective)
    differences = env_differences(view, expected)
    _write(ENV_FILE, {"entry": VLABENCH_CLIENT_ENTRY, "effective": view, "expected": _simple(expected),
                      "differences": differences})
    if differences:
        raise RuntimeError("effective runtime config differs from campaign: " + "; ".join(differences))


def report_vlabench_refusal(reason: str, expected: Mapping[str, Any]) -> None:
    """An entry point that stops before the Evaluator runs (bad configuration) says why; the campaign seals
    preflight_failed."""
    events = os.environ.get("OEH_EVAL_EVENTS")
    if events and (Path(events).parent / ENV_FILE).exists():
        return
    _write(ENV_FILE, {"entry": VLABENCH_CLIENT_ENTRY, "effective": None, "refused": str(reason),
                      "expected": _simple(expected), "differences": [f"refused to start: {reason}"]})

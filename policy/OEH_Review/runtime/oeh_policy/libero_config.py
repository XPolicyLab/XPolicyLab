"""Freeze LIBERO, LIBERO-Plus and LIBERO-Pro policy identities before a pilot attempt.

This is a separate configuration schema because the released LIBERO action is
single-arm native OSC7 (the single_arm_eef profile), while the original OEH
policy schema is RoboDojo EE. The fast, hybrid and review conditions share every
native and model identity; hybrid adds the frozen slow brain, escalation schedule,
primitive limits and the robosuite OSC_POSE scale read from the LIBERO env;
review adds the teacher's review block, its workspace box, the same OSC_POSE
scale and the benchmark's teacher prompt files (oeh_policy.libero_review).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from .contracts import digest, encoded, file_digest
from .escalation import validate_escalation
from .libero_boundary import BENCHMARKS
from .osc_controller import freeze_osc_controller
from .profiles import SINGLE_ARM_EEF, check_embodiment, check_primitive_limits

# LIBERO 8f1084e libero/libero/envs/env_wrapper.py:27 and LIBERO-Plus 4976dc3
# env_wrapper.py:191: ControlEnv(control_freq=20); the released single_eval.py
# _make_env passes none, so every native step is 1/20 s of simulated time.
CONTROL_HZ = 20
HYBRID_ONLY = ("escalation", "primitives", "osc_controller")
REVIEW_ONLY = ("review", "review_workspace")
BRAIN_FIELDS = {"provider", "model", "reasoning_effort", "providers_path", "max_calls", "max_tokens"}

# Flags the pinned OpenWAM LIBERO/LIBERO-Plus scheduler passes every managed server
# (benchmarks/libero*/scheduler.py::_server_command defaults, XPolicyLab fa431ec).
OFFICIAL_SERVER = {"denoise_steps": 10, "denoise_mode": "sync", "inference_mode": "sync", "inference_horizon": 10}
LARGE_FILE = 64 * 1024 * 1024
# LIBERO's four released suites and their official step budgets (OpenWAM policy_config.yml max_steps_by_suite).
BASE_SUITES = {"libero_spatial": 600, "libero_object": 600, "libero_goal": 600, "libero_10": 700}
# LIBERO-Pro's suites the pinned loop can run (oeh_eval.libero.pro_suite: `lan` and `task` perturb only an instruction
# the loop never sends; no HF revision publishes the `env` suites' files). policy_config.yml names none of them, so
# each gets its top-level max_steps, 600.
PRO_SUITES = {f"{base}_{kind}": 600 for base in BASE_SUITES for kind in ("object", "swap")}
_large_digests: dict[tuple, str] = {}


def _digest(path: Path) -> str:
    """file_digest, memoized per content signature for checkpoints (~25 GB, checked several times)."""
    stat = path.stat()
    if stat.st_size < LARGE_FILE:
        return file_digest(path)
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ino)
    if key not in _large_digests:
        _large_digests[key] = file_digest(path)
    return _large_digests[key]


def _git_identity(root: Path) -> dict:
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(root), "diff", "--binary", "HEAD"])
    return {"commit": commit, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest()}


def _native(value: dict, benchmark: str) -> dict:
    native = copy.deepcopy(value)
    required = {"suite", "task_id", "seed", "rng_mode", "settle_steps", "settle_action",
                "max_steps", "camera_height", "camera_width", "head_camera_key",
                "left_wrist_camera_key", "right_wrist_camera_key", "image_transform",
                "action_mode", "action_indices", "action_clip", "num_trials",
                "send_state", "state_dim", "reseed_each_trial", "fail_on_incomplete", "debug"}
    if required - native.keys():
        raise ValueError(f"missing native protocol settings: {sorted(required - native.keys())}")
    if native["num_trials"] != 1 or native["settle_steps"] != 30 or native["action_mode"] != "eef":
        raise ValueError("released one-task EEF pilot protocol required")
    if native["settle_action"] != [0, 0, 0, 0, 0, 0, -1]:
        raise ValueError("released settling action mismatch")
    if native["action_indices"] is not None or native["action_clip"] is not None:
        raise ValueError("extra action projection would change direct parity")
    if native["send_state"] is not True or native["state_dim"] != 10 or \
            native["reseed_each_trial"] is not False or native["fail_on_incomplete"] is not False or \
            native["debug"] is not False:
        raise ValueError("released one-trial state/RNG/reporting protocol mismatch")
    if native["camera_height"] != 256 or native["camera_width"] != 256:
        raise ValueError("released LIBERO camera resolution required")
    if native["image_transform"] != "rotate_180" or native["head_camera_key"] != "agentview_image" or \
            native["left_wrist_camera_key"] != "robot0_eye_in_hand_image" or \
            native["right_wrist_camera_key"] is not None:
        raise ValueError("released LIBERO observation projection required")
    if benchmark == "libero_plus":
        expected = (10000, "official_global")
    else:  # LIBERO-Pro runs OpenWAM's LIBERO loop and its seed
        expected = (42, "environment")
    if (native["seed"], native["rng_mode"]) != expected:
        raise ValueError("released benchmark seed/RNG protocol mismatch")
    if type(native["task_id"]) is not int or native["task_id"] < 0 or \
            type(native["max_steps"]) is not int or native["max_steps"] < 1:
        raise ValueError("native task and step limit required")
    suites = PRO_SUITES if benchmark == "libero_pro" else BASE_SUITES
    if native["suite"] not in suites or native["max_steps"] != suites[native["suite"]]:
        raise ValueError("released suite step budget mismatch")
    return native


def _provider_declaration(path: Path, provider: str) -> dict:
    """The declared provider entry (env-variable names only, never a secret)."""
    text = path.read_text(encoding="utf-8")
    try:
        import yaml
        data = yaml.safe_load(text)
    except ImportError:
        data = json.loads(text)
    entry = ((data or {}).get("providers") or {}).get(provider)
    if not isinstance(entry, dict):
        raise ValueError(f"provider {provider!r} is not declared in {path}")
    return entry


def _hybrid(cfg: dict) -> None:
    brain = cfg.get("brain")
    if not isinstance(brain, dict) or BRAIN_FIELDS - brain.keys() or \
            brain.keys() - BRAIN_FIELDS - {"providers", "providers_sha256", "provider_declaration"}:
        raise ValueError(f"hybrid brain requires exactly {sorted(BRAIN_FIELDS)}")
    if not all(isinstance(brain[key], str) and brain[key].strip()
               for key in ("provider", "model", "reasoning_effort", "providers_path")):
        raise ValueError("hybrid provider, model, reasoning effort and declarations required")
    if any(type(brain[key]) is not int or brain[key] < 1 for key in ("max_calls", "max_tokens")):
        raise ValueError("hybrid call and token budgets required")
    cfg["escalation"] = validate_escalation(cfg.get("escalation"))
    if cfg["escalation"]["control_hz"] != CONTROL_HZ:
        raise ValueError(f"LIBERO escalation runs on the pinned {CONTROL_HZ} Hz control clock")
    cfg["primitives"] = check_primitive_limits(SINGLE_ARM_EEF, cfg.get("primitives"))
    cfg["osc_controller"] = freeze_osc_controller(cfg.get("osc_controller"))


def _review(cfg: dict) -> None:
    """Review mode: the teacher's block (oeh_policy.contracts.validate_review, its defaults frozen explicitly), the
    workspace box its corrections stay in and the OSC_POSE scale its rows are computed with. No slow brain, no
    escalation and no primitives: the teacher is the only model beside OpenWAM. A review call is bounded by
    review.call_seconds (and first_call_extra_seconds on the first call), so first_decision_seconds does not apply.
    One candidate is the official server's executed chunk: freeze holds max_chunk_steps to its inference horizon."""
    from .contracts import check_review_benchmark, validate_review
    from .libero_review import check_review_workspace
    if cfg.get("brain") is not None or any(key in cfg for key in ("escalation", "primitives",
                                                                     "first_decision_seconds")):
        raise ValueError("review condition has no slow brain, escalation, primitives or first_decision_seconds")
    check_review_benchmark(cfg["benchmark"], cfg.get("review"))
    cfg["review"] = validate_review(cfg.get("review"))
    if cfg["review"]["control_hz"] != CONTROL_HZ:
        raise ValueError(f"LIBERO review runs on the pinned {CONTROL_HZ} Hz control clock")
    task = cfg["review"]["task"]
    if task is not None and (not isinstance(task, str) or not task.strip()):
        raise ValueError("review.task must name the task or be null")
    from .teacher import task_context
    if task is not None and task_context(task, cfg["benchmark"]) is None:
        # A task without a note would reach the teacher with no task context and no error.
        raise ValueError(f"review.task {task!r} has no note in the {cfg['benchmark']} teacher's task_context.json")
    cfg["review_workspace"] = check_review_workspace(cfg.get("review_workspace"))
    cfg["osc_controller"] = freeze_osc_controller(cfg.get("osc_controller"))


def _teacher_files(benchmark: str) -> list[Path]:
    """The teacher prompt files a review condition's method includes (a lock detects any change, as RoboDojo's)."""
    from .teacher import prompt_dir, prompt_files
    return [*prompt_files(benchmark).values(), prompt_dir(benchmark) / "task_context.json"]


def freeze(template: dict) -> dict:
    cfg = copy.deepcopy(template)
    if cfg.get("schema") != "oeh.policy.libero.v1" or cfg.get("benchmark") not in BENCHMARKS:
        raise ValueError("LIBERO-family policy schema and benchmark (LIBERO, LIBERO-Plus or LIBERO-Pro) required")
    benchmark, mode = cfg["benchmark"], cfg.get("mode")
    check_embodiment(cfg.get("embodiment"), benchmark)
    if mode not in ("fast", "hybrid", "review"):
        raise ValueError("fast, hybrid or review condition required")
    if cfg.get("official_server") != OFFICIAL_SERVER:
        raise ValueError(f"official_server must be the pinned scheduler's server flags {OFFICIAL_SERVER}")
    cfg["native"] = _native(cfg["native"], benchmark)
    if not isinstance(cfg.get("model_id"), str) or not cfg["model_id"].strip() or \
            not isinstance(cfg.get("compute"), dict) or not cfg["compute"]:
        raise ValueError("model and compute identities required")
    if not all(isinstance(cfg["compute"].get(key), str) and cfg["compute"][key].strip()
               for key in ("device", "gpu", "server_env")):
        raise ValueError("compute device, GPU and environment lock required")
    # One native action per call; review's candidate is the official executed chunk instead (_review).
    if type(cfg.get("max_chunk_steps")) is not int or cfg["max_chunk_steps"] != (
            OFFICIAL_SERVER["inference_horizon"] if mode == "review" else 1) or \
            type(cfg.get("native_step_limit")) is not int or \
            cfg["native_step_limit"] != cfg["native"]["max_steps"]:
        raise ValueError("native 7D OSC endpoint requires one action per call and matching step budget")
    # first_decision_seconds (optional; absent = decision_seconds) bounds the first decision, which pays
    # OpenWAM's warm-up; the runtime and the client transport read it.
    for key in ("wall_seconds", "decision_seconds", "first_decision_seconds"):
        if key == "first_decision_seconds" and key not in cfg:
            continue
        value = cfg.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or \
                not math.isfinite(value) or value <= 0:
            raise ValueError(f"finite positive {key} required")
    if mode != "review" and any(key in cfg for key in REVIEW_ONLY):
        raise ValueError("only the review condition configures a teacher review block")
    if mode == "fast":
        if cfg.get("brain") is not None or any(key in cfg for key in HYBRID_ONLY):
            raise ValueError("fast condition must have no slow model, escalation or primitives")
    elif mode == "review":
        _review(cfg)
    else:
        _hybrid(cfg)
    for key in ("source_root", "benchmark_root", "checkpoint_dir", "official_deploy_config",
                "official_client_config", "kernel", "output"):
        cfg[key] = str(Path(cfg[key]).resolve())
    source = Path(cfg["source_root"])
    folder = BENCHMARKS[benchmark]
    required = [f"benchmarks/{folder}/single_eval.py",
                f"benchmarks/{folder}/openwam2libero_interface.py",
                f"benchmarks/{folder}/policy_config.yml",
                "benchmarks/utils/client.py", "benchmarks/utils/transport.py",
                "benchmarks/utils/action_conversion.py", "openwam/deploy/server.py",
                "openwam/deploy/obs_preprocess.py", "openwam/deploy/policy.py"]
    cfg["source_hashes"] = {name: file_digest(source / name) for name in required}
    if cfg["source_hashes"][f"benchmarks/{folder}/policy_config.yml"] != \
            file_digest(Path(cfg["official_client_config"])):
        raise ValueError("official client config must come from the pinned source")
    checkpoint = Path(cfg["checkpoint_dir"])
    name = cfg["checkpoint_name"]
    if not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors"):
        raise ValueError("explicit checkpoint safetensors filename required")
    cfg["checkpoint_sha256"] = _digest(checkpoint / name)
    cfg["normalization_sha256"] = file_digest(checkpoint / "normalization_stats.npy")
    cfg["training_config_sha256"] = file_digest(checkpoint / "config.yaml")
    cfg["deploy_config_sha256"] = file_digest(Path(cfg["official_deploy_config"]))
    cfg["kernel_sha256"] = file_digest(Path(cfg["kernel"]))
    cfg["source_git"] = _git_identity(source)
    cfg["benchmark_git"] = _git_identity(Path(cfg["benchmark_root"]))
    if mode == "hybrid":
        brain = cfg["brain"]
        brain["providers_path"] = str(Path(brain["providers_path"]).resolve())
        brain["providers_sha256"] = file_digest(Path(brain["providers_path"]))
        brain["providers"] = brain["providers_path"]
        brain["provider_declaration"] = _provider_declaration(Path(brain["providers_path"]),
                                                              brain["provider"])
        if brain["provider_declaration"].get("reasoning_effort") != brain["reasoning_effort"]:
            raise ValueError("declared provider reasoning effort differs from the frozen brain")
    cfg["repository"] = cfg["benchmark_root"]
    cfg["action_limits"] = {"native": "libero_osc7", "low": -1.0, "high": 1.0}
    identity_paths = [Path(cfg["kernel"]), Path(cfg["official_deploy_config"]),
                      Path(cfg["official_client_config"])]
    if mode == "hybrid":
        identity_paths.extend([Path(cfg["brain"]["providers_path"]), SINGLE_ARM_EEF.persona_path,
                               Path(cfg["osc_controller"]["path"])])
    elif mode == "review":
        identity_paths.extend([Path(cfg["osc_controller"]["path"]), *_teacher_files(benchmark)])
    cfg["identity_files"] = {str(p): file_digest(p) for p in identity_paths}
    cfg["policy_code_hashes"] = {p.name: file_digest(p) for p in
                                 sorted(Path(__file__).parent.glob("*.py"))}
    cfg["configuration_sha256"] = digest({k: v for k, v in cfg.items() if k != "configuration_sha256"})
    return cfg


def resolve_config_path(argument: str | None) -> Path:
    """`--config`, or the campaign's `OEH_EVAL_INTERNAL_CONFIG.policy_config`; both must agree."""
    internal = json.loads(os.environ.get("OEH_EVAL_INTERNAL_CONFIG") or "{}").get("policy_config")
    if argument and internal and Path(argument).resolve() != Path(internal).resolve():
        raise ValueError("--config differs from the campaign's policy_config")
    if not (argument or internal):
        raise ValueError("a frozen OEH policy config is required (--config or OEH_EVAL_INTERNAL_CONFIG)")
    return Path(argument or internal).resolve()


def check_client(cfg: dict, names) -> None:
    """Simulator-side check: the frozen record is intact and the named loop sources unchanged.

    The server process runs the full `check` (checkpoint, kernel, git identities).
    """
    if cfg.get("configuration_sha256") != digest({k: v for k, v in cfg.items() if k != "configuration_sha256"}):
        raise ValueError("frozen LIBERO policy configuration record changed")
    source = Path(cfg["source_root"])
    for name in names:
        if cfg["source_hashes"].get(name) != file_digest(source / name):
            raise ValueError(f"pinned official source identity mismatch: {name}")


def check(cfg: dict) -> None:
    expected = cfg.get("configuration_sha256")
    rebuilt = freeze({k: v for k, v in cfg.items() if k not in {
        "source_hashes", "checkpoint_sha256", "normalization_sha256", "training_config_sha256",
        "deploy_config_sha256", "kernel_sha256", "source_git", "benchmark_git",
        "policy_code_hashes", "configuration_sha256", "repository", "action_limits",
        "identity_files"}})
    if expected != rebuilt["configuration_sha256"] or cfg != rebuilt:
        raise ValueError("frozen LIBERO policy identities changed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest="command", required=True)
    pin = command.add_parser("freeze")
    pin.add_argument("template", type=Path)
    pin.add_argument("output", type=Path)
    verify = command.add_parser("check")
    verify.add_argument("configuration", type=Path)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        frozen = freeze(json.loads(args.template.read_text(encoding="utf-8")))
        with args.output.open("xb") as stream:
            stream.write(encoded(frozen))
        print(json.dumps({"configuration_sha256": frozen["configuration_sha256"]}))
    else:
        cfg = json.loads(args.configuration.read_text(encoding="utf-8"))
        check(cfg)
        print(json.dumps({"configuration_sha256": cfg["configuration_sha256"], "status": "valid"}))


if __name__ == "__main__":
    main()

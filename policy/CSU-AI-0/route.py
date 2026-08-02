from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import pathlib
import shlex
from typing import Any


HERE = pathlib.Path(__file__).resolve().parent
MANIFEST = HERE / "adapter_manifest.json"
EXPERTS = HERE / "experts.json"


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_router() -> tuple[dict[str, Any], pathlib.Path, str]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    model_record = manifest["artifacts"]["QRouter.joblib"]
    bundle_path = HERE / model_record["path"]
    bundle_sha = sha256(bundle_path)
    if bundle_sha != model_record["sha256"]:
        raise RuntimeError(
            f"QRouter SHA256 mismatch: {bundle_sha} != {model_record['sha256']}"
        )

    portable_record = manifest["artifacts"]["QRouter.portable.json"]
    portable_path = HERE / portable_record["path"]
    portable_sha = sha256(portable_path)
    if portable_sha != portable_record["sha256"]:
        raise RuntimeError(
            "portable QRouter SHA256 mismatch: "
            f"{portable_sha} != {portable_record['sha256']}"
        )
    router = json.loads(portable_path.read_text(encoding="utf-8"))
    if router.get("source_bundle_sha256") != bundle_sha:
        raise RuntimeError("portable QRouter is not derived from this QRouter.joblib")
    if router.get("model_name") != "QRouter":
        raise RuntimeError("portable router model_name is not QRouter")
    if router.get("posthoc_override_used"):
        raise RuntimeError("post-hoc routing override is forbidden")
    if router.get("hard_rule_used_at_inference"):
        raise RuntimeError("hard-rule inference is forbidden")
    return router, bundle_path, bundle_sha


def predict(router: dict[str, Any], task_index: int) -> tuple[int, list[float]]:
    raw = [float(value) for value in router["baseline_prediction"]]
    for iteration in router["trees"]:
        for class_index, nodes in enumerate(iteration):
            node_index = 0
            while True:
                node = nodes[node_index]
                if node["is_leaf"]:
                    raw[class_index] += float(node["value"])
                    break
                value = 1.0 if int(node["feature_idx"]) == task_index else 0.0
                node_index = int(
                    node["left"]
                    if value <= float(node["num_threshold"])
                    else node["right"]
                )

    maximum = max(raw)
    exponentials = [math.exp(value - maximum) for value in raw]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]
    raw_class_index = max(range(len(raw)), key=raw.__getitem__)
    return int(router["classes"][raw_class_index]), probabilities


def _load_expert_config() -> tuple[dict[str, Any], pathlib.Path]:
    payload = json.loads(EXPERTS.read_text(encoding="utf-8"))
    experts = payload.get("experts", payload)
    bundle = payload.get("bundle", {})
    root_override = os.environ.get("CSU_CHECKPOINT_ROOT")
    root = pathlib.Path(
        root_override or bundle.get("root") or "checkpoints/CSU-AI-0-v1"
    ).expanduser()
    if not root.is_absolute():
        root = HERE / root
    return experts, root.resolve()


def _resolve_expert_spec(expert: str) -> dict[str, Any]:
    experts, bundle_root = _load_expert_config()
    if expert not in experts:
        raise RuntimeError(f"QRouter selected unconfigured expert={expert!r}")
    spec = dict(experts[expert])
    env_prefix = {
        "xiaomi": "XIAOMI",
        "spatial_forcing": "SPATIAL",
        "hunyuan": "HUNYUAN",
    }[expert]

    checkpoint_override = os.environ.get(f"CSU_{env_prefix}_CHECKPOINT")
    if checkpoint_override:
        checkpoint = pathlib.Path(checkpoint_override).expanduser().resolve()
        checkpoint_origin = "environment_override"
    else:
        relpath = pathlib.Path(str(spec.pop("checkpoint_relpath")))
        if relpath.is_absolute() or ".." in relpath.parts:
            raise RuntimeError(f"unsafe checkpoint_relpath for {expert}: {relpath}")
        checkpoint = (bundle_root / relpath).resolve()
        try:
            checkpoint.relative_to(bundle_root)
        except ValueError as error:
            raise RuntimeError(f"checkpoint escapes bundle root: {checkpoint}") from error
        checkpoint_origin = "submission_bundle"

    policy_env = os.environ.get(
        f"CSU_{env_prefix}_POLICY_ENV", str(spec.get("policy_env", ""))
    )
    spec["policy_env"] = policy_env
    spec["checkpoint_path"] = str(checkpoint)
    spec["checkpoint_origin"] = checkpoint_origin
    spec["checkpoint_bundle_root"] = str(bundle_root)

    hf_relpath = spec.pop("hf_home_relpath", None)
    if expert == "xiaomi":
        hf_override = os.environ.get("CSU_XIAOMI_HF_HOME")
        if hf_override:
            spec["hf_home"] = str(pathlib.Path(hf_override).expanduser().resolve())
        elif hf_relpath:
            spec["hf_home"] = str((bundle_root / str(hf_relpath)).resolve())

    skip_runtime_checks = os.environ.get("CSU_SKIP_RUNTIME_CHECKS", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not skip_runtime_checks and not checkpoint.is_dir():
        raise RuntimeError(f"selected expert checkpoint is missing: {checkpoint}")
    return spec


def infer(task_name: str) -> dict[str, Any]:
    router, bundle_path, bundle_sha = load_router()
    vocabulary = list(router["task_vocabulary"])
    try:
        task_index = vocabulary.index(task_name)
    except ValueError as error:
        raise RuntimeError(f"QRouter does not know task_name={task_name!r}") from error

    prediction, probabilities = predict(router, task_index)
    expert = str(router["experts"][prediction])
    spec = _resolve_expert_spec(expert)
    return {
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "submission_policy": "CSU-AI-0",
        "router_name": router["model_name"],
        "router_family": router["model_family"],
        "router_bundle": str(bundle_path),
        "router_sha256": bundle_sha,
        "route_origin": "qrouter_histgbdt_prediction",
        "task_name": task_name,
        "selected_expert": expert,
        "prediction_confidence": max(probabilities),
        **spec,
    }


def append_audit(path: pathlib.Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n"
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)


def shell_output(result: dict[str, Any]) -> str:
    mapping = {
        "CSU_SELECTED_EXPERT": result["selected_expert"],
        "CSU_TARGET_POLICY": result["policy_name"],
        "CSU_TARGET_POLICY_DIR": result.get("policy_dir", ""),
        "CSU_TARGET_ACTION_TYPE": result["action_type"],
        "CSU_TARGET_CKPT_NAME": result["checkpoint_name"],
        "CSU_TARGET_CKPT_PATH": result["checkpoint_path"],
        "CSU_TARGET_POLICY_ENV": result["policy_env"],
        "CSU_TARGET_HF_HOME": result.get("hf_home", ""),
        "CSU_ROUTE_CONFIDENCE": str(result["prediction_confidence"]),
        "CSU_ROUTER_SHA256": result["router_sha256"],
    }
    return "\n".join(
        f"{key}={shlex.quote(str(value))}" for key, value in mapping.items()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--format", choices=("json", "shell", "expert"), default="json")
    parser.add_argument("--audit-jsonl", type=pathlib.Path)
    args = parser.parse_args()

    result = infer(args.task_name)
    if args.audit_jsonl:
        append_audit(args.audit_jsonl.expanduser().resolve(), result)
    if args.format == "shell":
        print(shell_output(result))
    elif args.format == "expert":
        print(result["selected_expert"])
    else:
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

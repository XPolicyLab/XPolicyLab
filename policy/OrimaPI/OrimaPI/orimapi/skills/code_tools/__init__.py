"""Model-selected public-observation tools; no task-specific success claims."""

import hashlib
import json
from pathlib import Path
import re

from orimapi.evaluation.storage import atomic_json, digest


def settings(parameters):
    value = parameters.get("tool_limits", {})
    if not isinstance(value, dict) or value.keys() - {
        "max_calls",
        "max_tools_per_turn",
        "history_size",
        "require_cartesian_completion",
        "report_observation_budget",
        "camera_pointing",
        "point_pursuit",
        "sam_proposals",
    }:
        raise ValueError("Invalid tool_limits settings")
    result = {"max_calls": 60, "max_tools_per_turn": 8, "history_size": 12, **value}
    if "require_cartesian_completion" in value and type(value["require_cartesian_completion"]) is not bool:
        raise ValueError("tool_limits.require_cartesian_completion must be boolean")
    if "report_observation_budget" in value and type(value["report_observation_budget"]) is not bool:
        raise ValueError("tool_limits.report_observation_budget must be boolean")
    if "camera_pointing" in value and type(value["camera_pointing"]) is not bool:
        raise ValueError("tool_limits.camera_pointing must be boolean")
    if "point_pursuit" in value and type(value["point_pursuit"]) is not bool:
        raise ValueError("tool_limits.point_pursuit must be boolean")
    if "sam_proposals" in value:
        from .segmentation import validate_config

        result["sam_proposals"] = validate_config(value["sam_proposals"])
    for key, maximum in (
        ("max_calls", 500),
        ("max_tools_per_turn", 16),
        ("history_size", 32),
    ):
        if type(result[key]) is not int or not 1 <= result[key] <= maximum:
            raise ValueError(f"tool_limits.{key} must be in 1..{maximum}")
    depth = parameters.get("depth_estimation")
    if depth is not None:
        if (
            not isinstance(depth, dict)
            or depth.keys() - {"model_path", "device"}
            or not isinstance(depth.get("model_path"), str)
        ):
            raise ValueError(
                "depth_estimation requires a local model_path and optional device"
            )
        device = depth.get("device", "cpu")
        if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda:\d+", device):
            raise ValueError("Depth device must be cpu or cuda:N")
        result["depth"] = {"model_path": depth["model_path"], "device": device}
    return result


def depth_identity(config):
    root = Path(config["model_path"]).resolve(strict=True)
    files = [
        root / name
        for name in ("config.json", "preprocessor_config.json", "model.safetensors")
    ]
    model = json.loads(files[0].read_text())
    if (
        model.get("model_type") != "depth_anything"
        or model.get("depth_estimation_type") != "metric"
    ):
        raise ValueError(
            "Depth tools require a metric Depth Anything checkpoint, not relative depth"
        )
    return {
        "model_path": str(root),
        "device": config["device"],
        "units": "estimated_meters",
        "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files
        },
    }


def depth_availability(config):
    if config is None:
        return {
            "available": False,
            "reason": "No local metric depth checkpoint configured",
        }
    root = Path(config["model_path"])
    required = ("config.json", "preprocessor_config.json", "model.safetensors")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        return {
            "available": False,
            "reason": "Missing local depth resources: " + ", ".join(missing),
        }
    try:
        metadata = json.loads((root / "config.json").read_text())
        if (
            metadata.get("model_type") != "depth_anything"
            or metadata.get("depth_estimation_type") != "metric"
        ):
            raise ValueError("Metric Depth Anything is required, not relative depth")
    except (ValueError, AttributeError) as error:
        return {"available": False, "reason": str(error)}
    return {
        "available": True,
        "reason": "Metric model configured; loaded only on a depth tool request",
    }


def sources():
    root = Path(__file__).parent
    files = {p.name: p for p in root.glob("*.py")}
    common = root.parent / "code_skills"
    for relative in (
        "common/geometry.py",
        "play_tic_tac_toe/motion.py",
        "play_tic_tac_toe/sandbox.py",
        "align_blocks/sandbox.py",
        "align_blocks/sandbox_worker.py",
        "align_blocks/generator.py",
    ):
        files["dependencies/" + relative] = common / relative
    # Outbound image preparation changes model inputs and is part of the frozen
    # toolchain, even though local perception continues using the original RGB.
    for name in ("llm_images.py", "supervisor.py", "state.py"):
        files["dependencies/" + name] = root.parents[1] / name
    return files


def identity():
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in sources().items()
    }
    return {"sha256": digest(hashes), "files": hashes}


def snapshot(trace):
    record = identity()
    root = trace.root / "code_tools" / record["sha256"]
    for name, path in sources().items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["files"][name]:
            raise ValueError("Code tools changed while snapshotting")
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError("Code tools snapshot changed")
        else:
            with target.open("xb") as stream:
                stream.write(content)
    atomic_json(root / "index.json", record)
    return record

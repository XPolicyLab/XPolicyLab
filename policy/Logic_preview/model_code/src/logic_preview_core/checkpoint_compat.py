"""Resolve the four Hydra targets embedded in the published step-10000 bundle."""


def action_config(payload):
    prefix, suffix, tail = str(payload.get("format", "")).partition("_wan21_action_v1")
    if not prefix.isidentifier() or not suffix or tail:
        raise ValueError("Unsupported action checkpoint format")
    if payload.get("training_mode") != "joint" or payload.get("step") != 10000:
        raise ValueError("Expected the released paired step-10000 checkpoint")
    processor = "datasets.lerobot.processors"
    transforms = "datasets.lerobot.transforms"
    targets = {
        f"{prefix}.runtime.create_{prefix}_wan21_14b":
            "logic_preview_core.runtime.create_logic_preview_wan21_14b",
        f"{prefix}.{processor}.{prefix}_processor.{prefix.upper()}Processor":
            f"logic_preview_core.{processor}.logic_preview_processor.LogicPreviewProcessor",
        f"{prefix}.{transforms}.image.ToTensor":
            f"logic_preview_core.{transforms}.image.ToTensor",
        f"{prefix}.{transforms}.action_state_merger.ConcatLeftAlign":
            f"logic_preview_core.{transforms}.action_state_merger.ConcatLeftAlign",
    }

    def resolve(value):
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, dict):
            return {k: targets[v] if k == "_target_" else resolve(v) for k, v in value.items()}
        return value

    return resolve(payload["resolved_config"])


def validate_video_checkpoint(payload, video, path):
    if not video.get("dit") or video.get("step") != payload.get("step"):
        raise ValueError(f"Video tensors missing or paired checkpoint step mismatch: {path}")

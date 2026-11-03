"""Read historical checkpoints while emitting only RecoverWAM configuration names.

Legacy identifiers below describe serialized input formats, not current methods.
No legacy package is imported and tensor keys are unchanged.
"""

ACTION_FORMAT = "recoverwam_wan21_action_v1"
_LEGACY_ACTION_FORMATS = {"fastwam_wan21_action_v1"}
_LEGACY_NAMESPACES = ("fastwam.", "robodojo_joint2b.")
_LEGACY_TARGET_PARTS = {
    "fastwam": "recoverwam",
    "FastWAM": "RecoverWAM",
    "fastwam_processor": "recoverwam_processor",
    "FastWAMProcessor": "RecoverWAMProcessor",
    "create_fastwam_wan21_14b": "create_recoverwam_wan21_14b",
}


def canonicalize_config(value):
    """Convert Hydra targets recursively, preserving paths and other values."""
    if isinstance(value, list):
        return [canonicalize_config(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: canonicalize_config(item) for key, item in value.items()}
    target = result.get("_target_")
    if isinstance(target, str):
        for prefix in _LEGACY_NAMESPACES:
            if target.startswith(prefix):
                target = "recoverwam." + target[len(prefix) :]
                break
        if target.startswith("recoverwam."):
            target = ".".join(_LEGACY_TARGET_PARTS.get(part, part) for part in target.split("."))
        result["_target_"] = target
    return result


def action_config(payload):
    name = payload.get("format", "")
    suffix = "_wan21_action_v1"
    if not isinstance(name, str) or not name.endswith(suffix):
        raise ValueError("Expected an exported RecoverWAM action checkpoint")
    namespace = name[:-len(suffix)]
    if not namespace.isidentifier():
        raise ValueError("Invalid serialized checkpoint namespace")
    def migrate(value):
        if isinstance(value, list):
            return [migrate(item) for item in value]
        if not isinstance(value, dict):
            return value
        out = {k: migrate(v) for k, v in value.items()}
        target = out.get("_target_")
        if isinstance(target, str) and target.startswith(namespace + "."):
            parts = target.split(".")
            parts[0] = "recoverwam"
            for i, part in enumerate(parts[1:], 1):
                if part == namespace or part == namespace + "_processor":
                    parts[i] = "recoverwam" + ("_processor" if part.endswith("_processor") else "")
                elif part.lower() == namespace.lower() + "processor":
                    parts[i] = "RecoverWAMProcessor"
                elif part.startswith("create_" + namespace + "_"):
                    parts[i] = "create_recoverwam_" + part[len("create_" + namespace + "_"):]
            out["_target_"] = ".".join(parts)
        return out
    return canonicalize_config(migrate(payload["resolved_config"]))

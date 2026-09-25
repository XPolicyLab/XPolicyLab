import torch
import torch.nn as nn

from wam.utils.logging_config import get_logger

logger = get_logger(__name__)


def save_wam_checkpoint(model, path, step=None):
    payload = {
        "mot": model.mot.state_dict(),
        "step": step,
        "torch_dtype": str(model.torch_dtype),
    }
    if model.proprio_encoder is not None:
        payload["proprio_encoder"] = model.proprio_encoder.state_dict()
    if model.action_proprio_encoder is not None:
        payload["action_proprio_encoder"] = (
            model.action_proprio_encoder.state_dict()
        )
    if model.understanding is not None:
        payload["understanding"] = model.understanding.adapter_state_dict()
    torch.save(payload, path)


def format_skipped_state_preview(skipped: list[tuple[str, tuple[int, ...], tuple[int, ...]]]) -> str:
    preview = [f"{key}: ckpt{ckpt_shape} != model{model_shape}" for key, ckpt_shape, model_shape in skipped[:8]]
    if len(skipped) > len(preview):
        preview.append(f"... +{len(skipped) - len(preview)} more")
    return "; ".join(preview)


def load_compatible_state_dict(module: nn.Module, checkpoint_state: dict[str, torch.Tensor], module_name: str, *, skip_all_on_shape_mismatch: bool = False) -> dict[str, int]:
    model_state = module.state_dict()
    compatible = {}
    skipped_shape: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    skipped_type: list[str] = []
    unexpected: list[str] = []

    for key, value in checkpoint_state.items():
        target_key = key
        if target_key not in model_state:
            if ".base." in key:
                alt_key = key.replace(".base.", ".")
                if alt_key in model_state:
                    target_key = alt_key
            elif key.endswith(".weight"):
                alt_key = f"{key[:-len('.weight')]}.base.weight"
                if alt_key in model_state:
                    target_key = alt_key
            elif key.endswith(".bias"):
                alt_key = f"{key[:-len('.bias')]}.base.bias"
                if alt_key in model_state:
                    target_key = alt_key
        if target_key not in model_state:
            unexpected.append(key)
            continue
        target = model_state[target_key]
        if not torch.is_tensor(value) or not torch.is_tensor(target):
            skipped_type.append(key)
            continue
        if tuple(value.shape) != tuple(target.shape):
            skipped_shape.append((target_key, tuple(value.shape), tuple(target.shape)))
            continue
        compatible[target_key] = value

    if skip_all_on_shape_mismatch and skipped_shape:
        logger.warning(
            "Skipped all %s checkpoint tensors because this module has incompatible shapes: %s",
            module_name,
            format_skipped_state_preview(skipped_shape),
        )
        return {
            "compatible": 0,
            "missing": len(model_state),
            "skipped_shape": len(skipped_shape),
            "skipped_type": len(skipped_type),
            "unexpected": len(unexpected),
        }

    incompat = module.load_state_dict(compatible, strict=False)
    missing = list(incompat.missing_keys)
    logger.info(
        "Loaded %s checkpoint tensors: compatible=%d/%d missing=%d skipped_shape=%d unexpected=%d",
        module_name,
        len(compatible),
        len(checkpoint_state),
        len(missing),
        len(skipped_shape),
        len(unexpected),
    )
    if skipped_shape:
        logger.warning(
            "Skipped %s checkpoint tensors with incompatible shapes: %s",
            module_name,
            format_skipped_state_preview(skipped_shape),
        )
    if skipped_type:
        logger.warning(
            "Skipped %d non-tensor %s checkpoint entries: %s",
            len(skipped_type),
            module_name,
            ", ".join(skipped_type[:8]),
        )
    if unexpected:
        suffix = "" if len(unexpected) <= 8 else f", ... +{len(unexpected) - 8} more"
        logger.warning(
            "Ignored %d unexpected %s checkpoint tensors: %s%s",
            len(unexpected),
            module_name,
            ", ".join(unexpected[:8]),
            suffix,
        )
    return {
        "compatible": len(compatible),
        "missing": len(missing),
        "skipped_shape": len(skipped_shape),
        "skipped_type": len(skipped_type),
        "unexpected": len(unexpected),
    }


def load_wam_checkpoint(model, path):
    payload = torch.load(path, map_location="cpu")
    mot_state = payload.get("mot")
    if not isinstance(mot_state, dict):
        raise ValueError(f"Checkpoint has no valid `mot` state: {path}")

    load_compatible_state_dict(model.mot, mot_state, "mot")

    optional_modules = (
        ("proprio_encoder", model.proprio_encoder),
        ("action_proprio_encoder", model.action_proprio_encoder),
    )
    for name, module in optional_modules:
        state = payload.get(name)
        if module is not None and isinstance(state, dict):
            load_compatible_state_dict(
                module,
                state,
                name,
                skip_all_on_shape_mismatch=True,
            )

    understanding_state = payload.get("understanding")
    if model.understanding is not None and isinstance(understanding_state, dict):
        model.understanding.load_adapter_state_dict(understanding_state, strict=False)
        logger.info("Loaded understanding adapter checkpoint.")

    return payload

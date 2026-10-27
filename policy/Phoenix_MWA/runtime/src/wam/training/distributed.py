import torch


def deepspeed_zero_stage(accelerator) -> int:
    plugin = getattr(accelerator.state, "deepspeed_plugin", None)
    config = getattr(plugin, "deepspeed_config", None) if plugin is not None else None
    if not isinstance(config, dict):
        return 0
    zero_config = config.get("zero_optimization", {})
    if not isinstance(zero_config, dict):
        return 0
    try:
        return int(zero_config.get("stage", 0))
    except (TypeError, ValueError):
        return 0


def model_init_device(accelerator) -> torch.device:
    if deepspeed_zero_stage(accelerator) == 3:
        return torch.device("cpu")
    return torch.device(accelerator.device)

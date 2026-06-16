"""Config loading and training entry point."""

import importlib


def load_config(config_path: str) -> dict:
    """Import config dict from a dotted module path like 'configs.xpolicylab_gigaworld.config'."""
    parts = config_path.rsplit(".", 1)
    if len(parts) == 2:
        module_path, attr = parts
    else:
        module_path, attr = config_path, "config"

    module = importlib.import_module(module_path)
    config = getattr(module, attr)
    if not isinstance(config, dict):
        raise TypeError(f"Expected config to be a dict, got {type(config)}")
    return config


def resolve_runner(runner_name: str):
    """Resolve a trainer class by short name or fully-qualified dotted path."""
    import world_action_model as _wam

    if hasattr(_wam, runner_name):
        return getattr(_wam, runner_name)

    parts = runner_name.rsplit(".", 1)
    if len(parts) == 2:
        module = importlib.import_module(parts[0])
        return getattr(module, parts[1])

    raise AttributeError(f"Cannot resolve runner '{runner_name}'")


def run_training(config_path: str):
    """End-to-end: load config -> create trainer -> train."""
    config = load_config(config_path)

    runners = config.get("runners", [])
    if not runners:
        raise ValueError("No runners specified in config")

    runner_cls = resolve_runner(runners[0])
    trainer = runner_cls(config)
    trainer.run()

"""
Utility functions for automatically updating action horizon and related configurations.
"""

from omegaconf import DictConfig, open_dict


def update_action_horizon_configs(cfg: DictConfig, action_horizon: int) -> DictConfig:
    """
    Automatically update action_horizon and corresponding delta_indices in all modality configs.

    Args:
        cfg: The hydra configuration
        action_horizon: The desired action horizon (e.g., 30)

    Returns:
        Updated configuration with action_horizon and delta_indices set appropriately
    """
    # Generate delta_indices for the given action_horizon [0, 1, 2, ..., action_horizon-1]
    delta_indices = list(range(action_horizon))

    # Update the global action_horizon
    with open_dict(cfg):
        cfg.action_horizon = action_horizon
        if hasattr(cfg.model, "vla_override_kwargs"):
            cfg.model.vla_override_kwargs.action_horizon = action_horizon
        if hasattr(cfg.model, "action_head_override_kwargs"):
            cfg.model.action_head_override_kwargs.action_horizon = action_horizon

    # Update delta_indices for all action modalities in modality_configs
    if hasattr(cfg, "modality_configs"):
        for embodiment_name, modality_config in cfg.modality_configs.items():
            if hasattr(modality_config, "action"):
                # Update the action delta_indices
                modality_config.action.delta_indices = delta_indices
                print(f"Updated {embodiment_name}.action.delta_indices to {delta_indices}")

    return cfg


def update_action_dim_configs(cfg: DictConfig, new_action_dim: int) -> DictConfig:
    """
    Update the action dimension in all modality configs.
    """
    with open_dict(cfg):
        cfg.max_action_dim = new_action_dim
    return cfg


def apply_action_overrides(cfg: DictConfig) -> DictConfig:
    """
    Apply action horizon overrides if action_horizon is specified in config.
    This function should be called after the config is loaded but before model instantiation.
    """
    if hasattr(cfg.model, "action_head_override_kwargs"):
        action_horizon = cfg.action_horizon
        print(f"Applying action_horizon={action_horizon} overrides...")
        cfg = update_action_horizon_configs(cfg, action_horizon)

    if hasattr(cfg.model, "expand_action_head_kwargs"):
        expand_action_head_kwargs = cfg.model.expand_action_head_kwargs
        if "expand_action_dim" in expand_action_head_kwargs:
            old_action_dim = expand_action_head_kwargs.expand_action_dim.old_action_dim
            new_action_dim = expand_action_head_kwargs.expand_action_dim.new_action_dim
            print(f"Applying expand_action_dim={old_action_dim}->{new_action_dim} overrides...")
            cfg = update_action_dim_configs(cfg, new_action_dim)

    return cfg

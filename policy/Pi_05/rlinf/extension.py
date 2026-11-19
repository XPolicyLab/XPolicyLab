"""Use the native OpenPI v0.4 training path with Pi_05's task transforms."""

from omegaconf import OmegaConf


def register():
    from rlinf.models import register_model

    register_model("openpi", build_openpi, category="embodied", force=True)


def build_openpi(cfg, torch_dtype):
    if cfg.get("xpolicylab", {}).get("policy_name") == "Pi_05":
        from XPolicyLab.infra.RLinf import get_model
    else:
        from rlinf.models.embodiment.openpi import get_model
    return get_model(cfg, torch_dtype)


def configure_model(cfg, model_cfg):
    if cfg.get("model_type") != "openpi":
        return
    options = OmegaConf.to_container(cfg.openpi, resolve=True)
    if options.pop("task") != "rl":
        raise ValueError("The Pi_05 RL bridge requires openpi.task=rl.")
    model_cfg["train_config_name"] = options.pop("config_name")
    model_cfg["embed_dtype"] = options.pop("embed_dtype", None)
    model_cfg["rl_head"] = {
        **model_cfg.get("rl_head", {}),
        "num_steps": cfg.num_steps,
        **options,
    }

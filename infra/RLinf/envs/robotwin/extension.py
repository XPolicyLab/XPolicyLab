"""Use the full RoboTwin seed files for RLinf rollout and evaluation."""

ENV_KEY = ("robotwin", None)


def env_class(cfg):
    if cfg.get("seeds_path"):
        from .env import RoboTwinEnv

        return RoboTwinEnv

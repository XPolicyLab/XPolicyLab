"""Public exports for the Abot policy and evaluation client."""
from .deploy import eval_one_episode, eval_one_episode_batch

__all__ = ["Model", "get_model", "eval_one_episode", "eval_one_episode_batch"]


def __getattr__(name):
    if name == "Model":
        from .model import Model
        return Model
    raise AttributeError(name)


def get_model(deploy_cfg):
    from .model import Model
    return Model(deploy_cfg)

from .model import Model


def get_model(deploy_cfg):
    return Model(deploy_cfg)

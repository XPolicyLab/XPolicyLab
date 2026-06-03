try:
    from .deploy import *
except ImportError:
    pass

try:
    from .model import *
except ImportError:
    pass


def get_model(deploy_cfg):
    return Model(deploy_cfg)
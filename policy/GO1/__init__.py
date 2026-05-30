try:
    from .deploy import *
except ImportError as e:
    pass
try:
    from .model import *
except ImportError as e:
    pass


def get_model(deploy_cfg):
    return Model(deploy_cfg)
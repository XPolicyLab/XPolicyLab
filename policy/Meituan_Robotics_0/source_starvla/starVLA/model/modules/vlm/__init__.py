def get_vlm_model(config):
    from .QWen3_5 import _QWen3_5_VL_Interface
    return _QWen3_5_VL_Interface(config)

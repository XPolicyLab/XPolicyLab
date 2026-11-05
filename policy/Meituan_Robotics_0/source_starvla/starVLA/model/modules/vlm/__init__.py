def get_vlm_model(config):
    vlm_name = config.framework.qwenvl.base_vlm

    if "Qwen3.5" in vlm_name:
        from .QWen3_5 import _QWen3_5_VL_Interface

        return _QWen3_5_VL_Interface(config)
    else:
        raise NotImplementedError(f"VLM model {vlm_name} not implemented")

"""M42 inference-only framework construction."""
from transformers import PretrainedConfig, PreTrainedModel

class baseframework(PreTrainedModel):
    def __init__(self, hf_config=None):
        super().__init__(hf_config or PretrainedConfig())

def build_framework(cfg):
    if cfg.framework.name != "QwenMMDiT":
        raise ValueError("This evaluation runtime supports QwenMMDiT only")
    from starVLA.model.framework.VLM4A.QwenMMDiT import Qwen_MMDiT
    return Qwen_MMDiT(cfg)

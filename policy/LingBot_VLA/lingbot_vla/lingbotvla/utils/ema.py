import torch
import torch.nn as nn

def ema_update(model_dest: nn.Module, model_src: nn.Module, rate):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_dest in model_dest.named_parameters():
        # p_src = param_dict_src[p_name].clone()
        p_src = param_dict_src[p_name]
        assert p_src is not p_dest
        assert  p_dest.data.dtype == torch.float32
        p_dest.data.mul_(rate).add_((1 - rate) * p_src.data.float())
        # p_dest.data.mul_(rate).add_((1 - rate) * p_src.data)
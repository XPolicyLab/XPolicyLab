import torch

def prepare_optimizer_precision(model):
    promoted = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad and parameter.dtype in (torch.float16, torch.bfloat16):
                promoted += parameter.numel()
                parameter.data = parameter.data.float()
                if parameter.grad is not None:
                    parameter.grad.data = parameter.grad.data.float()
    return {'version': 1, 'owner': 'pytorch', 'master_dtype': 'float32', 'promoted_elements': promoted}

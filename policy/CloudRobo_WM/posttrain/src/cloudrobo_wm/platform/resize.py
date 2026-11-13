import torch
import torch.nn.functional as F

def interpolate(value,*args,**kwargs):
    # Frozen image/mask preprocessing uses the same antialias kernel on CPU.
    # BF16 values are expanded to FP32 before that kernel and cast back after it.
    antialias=bool(kwargs.get('antialias',False))
    hosted=antialias and (value.device.type=='npu' or (value.device.type=='cpu' and value.dtype==torch.bfloat16))
    if hosted:
        if value.requires_grad:raise ValueError('Host antialias adapter is restricted to frozen preprocessing')
        with torch.autocast(device_type=value.device.type,enabled=False):
            result=F.interpolate(value.detach().to(device='cpu',dtype=torch.float32),*args,**kwargs)
        return result.to(device=value.device,dtype=value.dtype)
    return F.interpolate(value,*args,**kwargs)

import torch

def _staged_randn(shape, generator, rand_device, out_device, out_dtype):
    pin = str(rand_device) == 'cpu' and torch.cuda.is_available()
    t = torch.randn(shape, generator=generator, device=rand_device, dtype=torch.float32, pin_memory=pin)
    return t.to(device=out_device, dtype=out_dtype, non_blocking=pin)

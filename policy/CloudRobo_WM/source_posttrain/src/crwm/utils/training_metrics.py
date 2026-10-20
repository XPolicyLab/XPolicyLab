import torch

def reduce_training_metrics(accelerator, loss, metrics, grad_norm):
    names = sorted(metrics)
    values = torch.stack([loss.detach().float().reshape(()), torch.as_tensor(grad_norm, device=loss.device, dtype=torch.float32).detach().reshape(()), *[torch.as_tensor(metrics[name], device=loss.device, dtype=torch.float32).detach().reshape(()) for name in names]])
    averaged = accelerator.reduce(values, reduction='mean').cpu().tolist()
    return (averaged[0], dict(zip(names, averaged[2:])), averaged[1])

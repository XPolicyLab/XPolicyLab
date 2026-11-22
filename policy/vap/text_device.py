"""Optional device routing; leave Qwen weights, tokenization and dtype untouched."""
import torch

class DeviceRoutedQwenPrompter:
    def __init__(self, provider, device):
        self.provider = provider
        self.target_device = torch.device(device)
    def __getattr__(self, name):
        return getattr(self.provider, name)
    @torch.no_grad()
    def encode_prompt(self, prompt, positive=True, device='cpu', **kwargs):
        destination = torch.device(device)
        if self.target_device.type == 'cuda':
            with torch.cuda.device(self.target_device):
                result = self.provider.encode_prompt(prompt, positive=positive, device=self.target_device, **kwargs)
        else:
            result = self.provider.encode_prompt(prompt, positive=positive, device=self.target_device, **kwargs)
        return result.to(device=destination)

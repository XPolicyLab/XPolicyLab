import torch

class WanContinuousFlowMatchScheduler:

    def __init__(self, timestep_scale=1000, shift=5.0):
        if timestep_scale <= 0 or shift <= 0:
            raise ValueError('Timestep scale and sigma shift must be positive')
        self.timestep_scale = int(timestep_scale)
        self.shift = float(shift)

    def build_inference_schedule(self, num_inference_steps, device, dtype, shift_override=None):
        if num_inference_steps <= 0:
            raise ValueError('num_inference_steps must be positive')
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError('sigma shift must be positive')
        u = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma = shift * u / (1.0 + (shift - 1.0) * u)
        return ((sigma[:-1] * float(self.timestep_scale)).to(dtype=dtype), (sigma[1:] - sigma[:-1]).to(dtype=dtype))

    @staticmethod
    def step(model_output, delta, sample):
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim > 0:
            delta = delta.view(-1, *[1] * (sample.ndim - 1))
        return sample + model_output * delta

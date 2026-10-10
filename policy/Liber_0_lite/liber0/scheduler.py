import torch
class FlowScheduler:

    def __init__(self, num_train_timesteps, shift):
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def build_inference_schedule(self, num_inference_steps: int, device: torch.device, dtype: torch.dtype, shift_override: float | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        if num_inference_steps <= 0:
            raise ValueError(f'`num_inference_steps` must be positive, got {num_inference_steps}')
        shift = self.shift if shift_override is None else float(shift_override)
        if shift <= 0:
            raise ValueError(f'`shift` must be positive, got {shift}')
        u_steps = torch.linspace(1.0, 0.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        sigma_steps = self._phi(u_steps, shift)
        timesteps = sigma_steps[:-1] * float(self.num_train_timesteps)
        deltas = sigma_steps[1:] - sigma_steps[:-1]
        return (timesteps.to(dtype=dtype), deltas.to(dtype=dtype))

    @staticmethod
    def step(model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if delta.ndim == 0:
            return sample + model_output * delta
        delta = delta.view(-1, *[1] * (sample.ndim - 1))
        return sample + model_output * delta

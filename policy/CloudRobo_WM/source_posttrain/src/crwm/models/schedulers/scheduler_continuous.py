import math
import torch

class WanContinuousFlowMatchScheduler:

    def __init__(self, num_train_timesteps: int=1000, shift: float=5.0, eps: float=1e-10):
        if num_train_timesteps <= 0:
            raise ValueError(f'`num_train_timesteps` must be positive, got {num_train_timesteps}')
        if shift <= 0:
            raise ValueError(f'`shift` must be positive, got {shift}')
        self.num_train_timesteps = int(num_train_timesteps)
        self.shift = float(shift)
        self.eps = float(eps)
        self._y_min, self._weight_norm_const = self._precompute_training_weight_stats()
        self.solver = 'euler'
        self._ms_prev_x0: torch.Tensor | None = None
        self._ms_prev_h: float | None = None

    def set_solver(self, solver: str) -> None:
        if solver not in ('euler', 'dpmpp_2m'):
            raise ValueError(f"`solver` must be 'euler' or 'dpmpp_2m', got {solver!r}")
        self.solver = solver

    @staticmethod
    def _phi(u: torch.Tensor, shift: float) -> torch.Tensor:
        return shift * u / (1.0 + (shift - 1.0) * u)

    def _precompute_training_weight_stats(self) -> tuple[float, float]:
        steps = self.num_train_timesteps
        u_grid = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)[:-1]
        t_grid = self._phi(u_grid, self.shift) * float(steps)
        y_grid = torch.exp(-2.0 * ((t_grid - steps / 2.0) / steps) ** 2)
        y_min = float(y_grid.min().item())
        y_shifted_grid = y_grid - y_min
        norm_const = float(y_shifted_grid.mean().item())
        return (y_min, norm_const)

    def sample_training_t(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f'`batch_size` must be positive, got {batch_size}')
        u = torch.rand((batch_size,), device=device, dtype=torch.float32)
        sigma = self._phi(u, self.shift)
        timestep = sigma * float(self.num_train_timesteps)
        return timestep.to(dtype=dtype)

    def training_weight(self, timestep: torch.Tensor) -> torch.Tensor:
        t = timestep.to(dtype=torch.float32)
        steps = float(self.num_train_timesteps)
        y = torch.exp(-2.0 * ((t - steps / 2.0) / steps) ** 2)
        y_shifted = y - self._y_min
        weight = y_shifted / (self._weight_norm_const + self.eps)
        if weight.numel() == 1:
            return weight.reshape(())
        return weight

    def add_noise(self, original_samples: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        sigma = (timestep / float(self.num_train_timesteps)).to(original_samples.device, dtype=original_samples.dtype)
        if sigma.ndim == 0:
            return (1 - sigma) * original_samples + sigma * noise
        sigma = sigma.view(-1, *[1] * (original_samples.ndim - 1))
        return (1 - sigma) * original_samples + sigma * noise

    @staticmethod
    def training_target(sample: torch.Tensor, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        del timestep
        return noise - sample

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
        self._ms_prev_x0 = None
        self._ms_prev_h = None
        return (timesteps.to(dtype=dtype), deltas.to(dtype=dtype))

    @staticmethod
    def _lam(sigma: float) -> float:
        sigma = min(max(sigma, 0.0001), 1.0 - 0.0001)
        return math.log((1.0 - sigma) / sigma)

    def step(self, model_output: torch.Tensor, delta: torch.Tensor, sample: torch.Tensor, timestep: torch.Tensor | None=None) -> torch.Tensor:
        delta = delta.to(sample.device, dtype=sample.dtype)
        if self.solver == 'euler' or timestep is None:
            if delta.ndim == 0:
                return sample + model_output * delta
            delta = delta.view(-1, *[1] * (sample.ndim - 1))
            return sample + model_output * delta
        s = float(timestep.reshape(-1)[0].item()) / float(self.num_train_timesteps)
        d = float(delta.reshape(-1)[0].item()) if delta.ndim else float(delta.item())
        t = s + d
        s = max(s, 1e-08)
        x0 = sample - s * model_output
        h = self._lam(t) - self._lam(s)
        if self._ms_prev_x0 is None or self._ms_prev_h is None or t <= 1e-06:
            D = x0
        else:
            r = self._ms_prev_h / h if h != 0.0 else 0.0
            D = (1.0 + 0.5 / r) * x0 - 0.5 / r * self._ms_prev_x0 if r > 0.0 else x0
        x_next = t / s * sample - d / s * D
        self._ms_prev_x0 = x0
        self._ms_prev_h = h
        return x_next

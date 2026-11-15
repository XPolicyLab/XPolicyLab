"""ODE solvers for flow matching denoising.

Unified interface: solver.step(x, vel, dt) → x_next.
"""

from abc import ABC
from abc import abstractmethod
import torch


class BaseSolver(ABC):
    """Base class for ODE solvers."""

    uses_renoising = False

    @abstractmethod
    def step(
        self,
        x: torch.Tensor,
        vel: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        """Advance one solver step."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Clear trajectory-local solver state."""
        raise NotImplementedError


class EulerSolver(BaseSolver):
    """First-order Euler: x_{t+1} = x_t + v * dt."""

    def step(
        self,
        x: torch.Tensor,
        vel: torch.Tensor,
        dt: torch.Tensor | float,
    ) -> torch.Tensor:
        return x + vel * dt

    def reset(self) -> None:
        pass


class CMSolver(BaseSolver):
    """Consistency-model step: predict x0, then re-noise at the next level."""

    uses_renoising = True

    def step(
        self,
        x: torch.Tensor,
        vel: torch.Tensor,
        dt: torch.Tensor | float | None = None,
        *,
        t: torch.Tensor | float,
        s: torch.Tensor | float,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        del dt
        x0 = x - vel * t
        return (1 - s) * x0 + s * noise

    def reset(self) -> None:
        pass

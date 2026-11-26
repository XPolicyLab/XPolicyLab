import torch
import torch.nn as nn


class BaseVaeModule(nn.Module):
    """Base class for all VAE modules."""

    @property
    def latent_dim(self) -> int:
        raise NotImplementedError

    @property
    def latent_strides(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def __repr__(self) -> str:
        repr_str = f"{self.__class__.__name__}("
        repr_str += f"latent_dim={self.latent_dim}, "
        repr_str += f"latent_strides[T, H, W]={self.latent_strides})"
        return repr_str

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

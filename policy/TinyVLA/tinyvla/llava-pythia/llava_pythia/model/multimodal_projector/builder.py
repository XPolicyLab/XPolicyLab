import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config):
    """
    Constructs a vision projector based on the specified configuration.

   Args:
    - config: An object containing configuration attributes. It should have
      'mm_projector_type' to specify the type of projector and 'mm_hidden_size'
      and 'hidden_size' for the dimensions of the layers.

    Returns:
    - A PyTorch module that acts as the vision projector. The type of module
      returned depends on the 'mm_projector_type' attribute in the config:
      - 'linear': Returns a linear layer mapping from mm_hidden_size to hidden_size.
      - 'mlp{n}x_gelu': Returns a sequential model with n layers, each consisting
        of a GELU activation followed by a linear layer.
      - 'identity': Returns an IdentityMap, which simply returns the input as is.

    Raises:
    - ValueError: If the 'mm_projector_type' is not recognized.
    """
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

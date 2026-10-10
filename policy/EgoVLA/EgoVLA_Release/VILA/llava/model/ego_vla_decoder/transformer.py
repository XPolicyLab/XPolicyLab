import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(
               output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerSplitActV2(nn.Module):
  def __init__(
    self,
    hidden_size,
    proprio_size,
    out_dim,
    use_proprio,
    # use_hand_input,
    sep_proprio,
    **kwargs
  ):
    super().__init__()

    self.use_proprio = use_proprio
    self.sep_proprio = sep_proprio

    self.proprio_size = proprio_size

    self.proprio_projection = nn.Sequential(
      nn.Linear(self.proprio_size, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )

    self.proprio_projection_3d = nn.Sequential(
      nn.Linear(3, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )
    self.proprio_projection_rot = nn.Sequential(
      nn.Linear(3, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )
    self.proprio_projection_hand = nn.Sequential(
      nn.Linear(5 * 3, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, hidden_size)
    )

    self.first_norm = nn.LayerNorm(hidden_size)
    self.out_dim = out_dim
    if self.out_dim <= 0 or self.out_dim % 2:
      raise ValueError(
        f"split action decoder requires a positive even out_dim, got {self.out_dim}"
      )
    self.branch_out_dim = self.out_dim // 2

    # 3D Translation * 2
    self.output_projection_left = nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, self.branch_out_dim)
    )

    # 3D Rotation: 6 * 2
    self.output_projection_right =  nn.Sequential(
      nn.Linear(hidden_size, hidden_size),
      nn.ELU(),
      nn.Linear(hidden_size, self.branch_out_dim)
    )

    num_transformer_encoder_layers = 6
    self.layers = nn.ModuleList()
    for _ in range(num_transformer_encoder_layers):
      encoder_layer = nn.TransformerEncoderLayer(
        d_model=hidden_size, nhead=1, batch_first=True,
        # dropout=0
        activation=torch.nn.functional.elu,
      )
      self.layers.append(encoder_layer)

    self.layers.train()

    # Transformer
    for param in self.layers.parameters():
      param.requires_grad = True
    # Proprio Projection
    # for param in self.proprio_projection_2d.parameters():
    #   param.requires_grad = True
    for param in self.proprio_projection_3d.parameters():
      param.requires_grad = True
    for param in self.proprio_projection_rot.parameters():
      param.requires_grad = True
    for param in self.proprio_projection_hand.parameters():
      param.requires_grad = True

    # Output Projection
    for param in self.output_projection_left.parameters():
      param.requires_grad = True
    for param in self.output_projection_right.parameters():
      param.requires_grad = True

  def forward(
    self, latent, input_dict, memory, memory_mask,
  ):
    proprio_input = input_dict["proprio"]
    proprio_input = self.proprio_projection(proprio_input)
    proprio_input = proprio_input.unsqueeze(1)
    # Proprio input shape: (B, 1, D)
    latent = latent.reshape(
      proprio_input.shape[0],
      latent.shape[0] // proprio_input.shape[0],
      latent.shape[1]
    )

    if self.use_proprio and self.sep_proprio:
      proprio_input_3d = input_dict["proprio_3d"].reshape(-1, 2, 3)
      proprio_input_3d = self.proprio_projection_3d(proprio_input_3d)
      # proprio_input_3d = proprio_input_3d.unsqueeze(1)

      proprio_input_rot = input_dict["proprio_rot"].reshape(-1, 2, 3)
      proprio_input_rot = self.proprio_projection_rot(proprio_input_rot)
      # proprio_input_rot = proprio_input_rot.unsqueeze(1)

      proprio_input_hand = input_dict["proprio_hand_finger_tip"].reshape(-1, 2, 5 * 3)
      proprio_input_hand = self.proprio_projection_hand(proprio_input_hand)
      # proprio_input_handdof = proprio_input_handdof.unsqueeze(1)

    if self.use_proprio:
      if self.sep_proprio:
        latent = torch.cat([
          # proprio_input_2d,
          proprio_input_3d,
          proprio_input_rot,
          proprio_input_hand,
          latent
        ], dim=1)
      else:
        latent = torch.cat([
          proprio_input, latent
        ], dim=1)


    batch_size, latent_len, _ = latent.shape

    mask = None
    pos = None
    # output = src
    memory_mask = ~memory_mask
    # memory = memory.detach()
    memory_mask = memory_mask.detach()

    src_key_padding_mask = torch.zeros(
      latent.shape[0],
      latent.shape[1]
    ).bool().to(memory_mask.device)
    src_key_padding_mask = torch.concat([
      memory_mask, src_key_padding_mask
    ], dim=1)

    latent = self.first_norm(
      latent
    )

    memory = self.first_norm(
      memory
    )

    for layer in self.layers:
        input_latent = torch.concat([
          memory, latent
        ], dim = 1)
        latent = layer(
          # latent,
          input_latent,
          src_key_padding_mask=src_key_padding_mask,
          # pos=pos
        )
        latent = latent[:, -latent_len:]

    if self.use_proprio:
      if self.sep_proprio:
        latent = latent[:, 6:, :]
      else:
        latent = latent[:, 1:, :]
 
    out_left = self.output_projection_left(
      latent[:, ::2]
    ).reshape(
      -1, 1, self.branch_out_dim
    )
    out_right = self.output_projection_right(
      latent[:, 1::2]
    ).reshape(
      -1, 1, self.branch_out_dim
    )

    output = torch.cat([
      out_left, out_right
    ], dim=1).reshape(-1, self.out_dim)


    return {
      "pred": output
    }

  def inference(
    self, latent, input_dict, memory, memory_mask,
    x=None, return_kl=False
  ):
    return self.forward(
      latent, input_dict, memory, memory_mask
    )

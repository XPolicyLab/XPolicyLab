import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from .transformer import TransformerSplitActV2

class TrajDecoderConfig(PretrainedConfig):
    model_type = "traj_decoder"

    def __init__(
      self, 
      traj_decoder_type: str=None, 
      **kwargs
    ):
      super().__init__()
      self.traj_decoder_type = traj_decoder_type

class TrajDecoder(PreTrainedModel):
  config_class =  TrajDecoderConfig
  def __init__(
    self, 
    decoder_cfg: TrajDecoderConfig, config: PretrainedConfig,
    **kwargs
  ):
    super().__init__(decoder_cfg)
    self.decoder_type = config.traj_decoder_type
    self.hidden_size = config.hidden_size
    self.out_dim = config.action_output_dim
    self.proprio_size = config.proprio_size
    self.use_proprio = config.use_proprio
    self.sep_proprio = config.sep_proprio
    self.config = config

    if config.traj_decoder_type == "transformer_split_action_v2":
      self.decoder = TransformerSplitActV2(
        self.hidden_size, 
        self.proprio_size,
        self.out_dim,
        self.use_proprio,
        self.sep_proprio
      )

  def forward(
    self,
    latent,
    input_dict=None,
    memory=None,
    memory_mask=None,
  ):
    if self.config.traj_decoder_type == "transformer_split_action_v2":
      return self.decoder(
        latent, input_dict, 
        memory=memory,
        memory_mask=memory_mask,
      )
    else:
      return self.decoder(
        latent, input_dict,
      )

  def inference(
    self,
    latent,
    input_dict=None,
    memory=None,
    memory_mask=None,
    return_kl=False
  ):
    if self.config.traj_decoder_type == "transformer_split_action_v2":
      return self.decoder.inference(
        latent, input_dict, 
        memory=memory,
        memory_mask=memory_mask,
        return_kl=return_kl
      )
    else: 
      return self.decoder.inference(
        latent, 
        input_dict,
        return_kl
      )

AutoConfig.register("traj_decoder", TrajDecoderConfig)
AutoModel.register(TrajDecoderConfig, TrajDecoder)
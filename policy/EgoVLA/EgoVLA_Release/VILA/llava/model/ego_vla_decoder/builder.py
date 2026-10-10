import os
from .traj_decoder import TrajDecoder, TrajDecoderConfig
import torch

def build_traj_decoder(
  model_type_or_path: str, config
):
  if model_type_or_path is None:
      return None

  ## load from pretrained model
  if config.resume_path and os.path.exists(
    model_type_or_path
  ):
      # assert os.path.exists(
      #     model_type_or_path
      # ), f"Resume traj decoder path {model_type_or_path} does not exist!"

    print("Resuming traj decoder from: ", model_type_or_path)
    return TrajDecoder.from_pretrained(
        model_type_or_path, config, torch_dtype=eval(config.model_dtype)
    )
  ## build from scratch
  else:
      print("Build traj decoder from scratch.")
      traj_decoder_cfg = TrajDecoderConfig(model_type_or_path)
      traj_decoder = TrajDecoder(
         traj_decoder_cfg, config
      ).to(eval(config.model_dtype))
      return traj_decoder

import torch
from safetensors.torch import load_file
from crwm.assets import asset_path
from .state_dict_converters import wan_video_vae_state_dict_converter
from ..wan_video_vae import WanVideoVAE38

def load_vae(device,dtype):
    state=wan_video_vae_state_dict_converter(load_file(str(asset_path("vae"))))
    model=WanVideoVAE38()
    expected=model.state_dict()
    state={k:v for k,v in state.items() if k in expected}
    model.load_state_dict(state,strict=True)
    return model.to(device=device,dtype=dtype).eval().requires_grad_(False)

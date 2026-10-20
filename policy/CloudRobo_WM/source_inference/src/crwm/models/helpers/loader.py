from dataclasses import dataclass
import torch
from .io import load_state_dict
from crwm.assets import asset_path
from .state_dict_converters import wan_video_vae_state_dict_converter
from ..wan_video_dit import WanVideoDiT
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38

@dataclass
class Wan22LoadedComponents:
    dit: WanVideoDiT
    vae: WanVideoVAE38
    text_encoder: WanTextEncoder | None
    tokenizer: HuggingfaceTokenizer | None
    dit_path: str
    vae_path: str
    text_encoder_path: str | None
    tokenizer_path: str | None

def load_text_encoder_to_device(device: str='cpu', torch_dtype: torch.dtype=torch.bfloat16, tokenizer_max_len: int=128):
    text_encoder = _load_registered_model(str(asset_path('text_encoder')), 'wan_video_text_encoder', torch_dtype=torch_dtype, device=device)
    tokenizer = HuggingfaceTokenizer(name=str(asset_path('tokenizer')), seq_len=int(tokenizer_max_len), clean='whitespace', local_files_only=True)
    return (text_encoder, tokenizer)

def _load_registered_model(path, model_name, torch_dtype, device):
    if model_name == 'wan_video_vae':
        model = WanVideoVAE38()
        state = wan_video_vae_state_dict_converter(load_state_dict(path, torch_dtype=torch_dtype))
        expected = model.state_dict()
        state = {k: v for k, v in state.items() if k in expected}
    elif model_name == 'wan_video_text_encoder':
        model = WanTextEncoder()
        state = load_state_dict(path, torch_dtype=torch_dtype)
    else:
        raise ValueError(f'Unsupported inference asset: {model_name}')
    model.load_state_dict(state, strict=True)
    return model.to(device=device, dtype=torch_dtype).eval().requires_grad_(False)

def load_wan22_ti2v_5b_components(device='cuda', torch_dtype=torch.bfloat16, tokenizer_max_len=128, dit_config=None):
    vae = _load_registered_model(str(asset_path('vae')), 'wan_video_vae', torch_dtype, device)
    text_encoder, tokenizer = load_text_encoder_to_device(device, torch_dtype, tokenizer_max_len)
    return Wan22LoadedComponents(dit=WanVideoDiT(**dit_config), vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, dit_path='CHECKPOINT', vae_path=str(asset_path('vae')), text_encoder_path=str(asset_path('text_encoder')), tokenizer_path=str(asset_path('tokenizer')))

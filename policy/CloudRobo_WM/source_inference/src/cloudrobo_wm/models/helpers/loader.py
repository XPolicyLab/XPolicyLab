import torch
from safetensors.torch import load_file
from ...assets import asset_path
from ..wan_video_text_encoder import HuggingfaceTokenizer, WanTextEncoder
from ..wan_video_vae import WanVideoVAE38

def load_encoders(dtype=torch.bfloat16,tokenizer_max_len=128):
    vae=WanVideoVAE38()
    raw=load_file(str(asset_path('vae')),device='cpu')
    state={'model.'+name:value for name,value in raw.items()}
    expected=set(vae.state_dict())
    vae.load_state_dict({name:value for name,value in state.items() if name in expected},strict=True)
    del raw,state
    vae=vae.to(dtype=dtype).eval().requires_grad_(False)
    text=WanTextEncoder()
    state=load_file(str(asset_path('text_encoder')),device='cpu')
    text.load_state_dict(state,strict=True)
    del state
    text=text.to(dtype=dtype).eval().requires_grad_(False)
    tokenizer=HuggingfaceTokenizer(name=str(asset_path('tokenizer')),seq_len=tokenizer_max_len,
                                 clean='whitespace',local_files_only=True)
    return vae,text,tokenizer

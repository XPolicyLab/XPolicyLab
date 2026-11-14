from pathlib import Path
import gc
import json
import types
import torch
from .factory import create_model,load_tensors
from .processing import JointEefNormalizer
from .processing.timing import inference_time_grids

CONFIG_ROOT=Path(__file__).resolve().parents[2]/'configs'

def install_zero_pointmap(model):
    encode=model._encode_input_image_latents_tensor
    def rgb_anchor(*args,**kwargs):
        result=encode(*args,**kwargs)
        model._zero_anchor_shape=result
        return result
    model._encode_input_image_latents_tensor=rgb_anchor
    def pointmap_anchor(*args,**kwargs):
        result=torch.zeros_like(model._zero_anchor_shape)
        if result.shape[2]!=1:
            raise ValueError('Expected a single observed latent frame')
        return result
    model._encode_first_frame_pointmap_raw=pointmap_anchor
    context=model._set_multisource_context
    def masked_context(sample):
        context(sample)
        mask=model._ms_masks['pointmap']
        mask[:,0].zero_()
    model._set_multisource_context=masked_context

def load_runtime(checkpoint_path,*,model_config_path=None,normalization_path=None,
                 asset_root=None,device='cuda'):
    if torch.device(device).type=='cuda':
        from .runtime_flags import configure_policy_runtime
        configure_policy_runtime()
    checkpoint=Path(checkpoint_path).expanduser().resolve()
    if checkpoint.suffix!='.safetensors':
        raise ValueError('This runtime accepts the tensor-only release format')
    cfg_path=Path(model_config_path or CONFIG_ROOT/'model.json')
    config=json.loads(cfg_path.read_text())
    metadata=json.loads((cfg_path.parent/'inference_contract.json').read_text())
    contract=metadata['multisource_contract']
    normalizer=JointEefNormalizer(normalization_path or cfg_path.parent/'normalization.json')
    condition=normalizer.validate_checkpoint(contract)
    root=Path(asset_root) if asset_root is not None else checkpoint.parent/'third_party'
    model=create_model(config,root)
    model.bind_multisource_contract(contract)
    load_tensors(model,checkpoint)
    target=torch.device(device)
    if target.type=='cuda' and target.index is None:
        target=torch.device('cuda',torch.cuda.current_device())
    model.to(device=target,dtype=torch.bfloat16)
    model.device=target
    model.torch_dtype=torch.bfloat16
    install_zero_pointmap(model)
    model.eval().requires_grad_(False)
    model.prepare_for_inference()
    protocol=next(e['source']['time_protocol'] for e in contract['sources']
                  if e['source']['stats_key']==normalizer.stats_key)
    grids,valid=inference_time_grids(protocol)
    kwargs=dict(time_grid={k:torch.from_numpy(v)[None].to(target) for k,v in grids.items()},
                time_valid={k:torch.from_numpy(v)[None].to(target) for k,v in valid.items()},
                joint_video=True,joint_dino=True,joint_pointmap=True,joint_flow=False,
                text_cfg_scale=1.0)
    gc.collect()
    return model,normalizer,condition,kwargs

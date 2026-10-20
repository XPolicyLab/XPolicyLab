"""CPU-only validation of public assets, index and one real RGB-D window."""
from __future__ import annotations
import argparse,json,os,platform
from pathlib import Path

def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('data-root','text-cache-root','sidecar-root','assets-root','initial-weights'):p.add_argument('--'+name,required=True)
    a=p.parse_args();os.environ['CRWM_ASSETS_ROOT']=a.assets_root
    from .assets import asset_path
    from .builder import build_dataset
    from .checkpoint import CHECKPOINT_MODULES, read_tensor_checkpoint
    from .gradient_norm_order import REFERENCE_GRADIENT_NORM_ORDER
    from .tensor_layout import REFERENCE_TENSOR_SHAPES
    import torch
    for key in ('vae','dino'):asset_path(key)
    dataset=build_dataset(a.data_root,a.text_cache_root,a.sidecar_root)
    if len(dataset)!=560900 or len(dataset.manifests[0]['episodes'])!=2500:raise ValueError('Clean population changed')
    x=dataset[(0,0)]
    assert x['state_valid'].sum()==32 and x['action_valid'][0].sum()==26
    assert x['modality_valid'][3] and x['camera_intrinsics'].shape==(3,3,3)
    payload=read_tensor_checkpoint(a.initial_weights,REFERENCE_TENSOR_SHAPES)
    assert len(payload)==1674, 'Expected all 1674 model tensors'
    assert {key.split('.',1)[0] for key in payload}==set(CHECKPOINT_MODULES)
    assert len(REFERENCE_GRADIENT_NORM_ORDER)==1679
    assert len(set(REFERENCE_GRADIENT_NORM_ORDER))==1679
    print(json.dumps(dict(verified=True,python=platform.python_version(),episodes=2500,windows=len(dataset),state_slots=32,action_slots=26,initialization_tensors=len(payload),tensor_only=True),indent=2))

if __name__=='__main__':main()

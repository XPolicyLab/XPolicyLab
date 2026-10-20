from __future__ import annotations
import json,os
from pathlib import Path
import torch
from .models.crwm import CRWM
from .models.action_dit import ActionDiT
from .models.wan_video_dit import WanVideoDiT
from .models.mot import MoT
from .models.dino_encoder import DinoEncoder
from .models.pointmap_contraction import create_pointmap_encoder
from .models.helpers.loader import load_vae
from .models.helpers.modality_dropout import ModalityDropoutConfig

ROOT=Path(__file__).resolve().parents[2]
def read_config(name):return json.loads((ROOT/'configs'/f'{name}.json').read_text())

def build_model(device='cpu',dtype=torch.bfloat16):
    cfg=read_config('model')
    video=WanVideoDiT(**cfg['video_dit_config']).to(device=device,dtype=dtype)
    action=ActionDiT(**cfg['action_dit_config']).to(device=device,dtype=dtype)
    h=cfg['hbridge']
    mot=MoT({'video':video,'action':action},mot_checkpoint_mixed_attn=cfg['mot_checkpoint_mixed_attn'],hbridge_enabled=h['enabled'],hbridge_bottom_ratio=h['bottom_ratio'],hbridge_top_ratio=h['top_ratio'])
    vae=load_vae(device,dtype)
    dino=DinoEncoder(model_name=cfg['dino_model_name'],device=device,torch_dtype=dtype)
    point=create_pointmap_encoder(pointmap_contraction=cfg['pointmap_contraction'],norm_bounds=cfg['pointmap_norm_bounds'],max_depth_m=cfg['pointmap_max_depth_m'],device=device,torch_dtype=dtype)
    args={k:cfg[k] for k in ('proprio_dim','composite_layout','dino_dim','dino_cam_patches','dino_cam_regions','dino_temporal_stride','dino_stride_keep_far','dino_pred_x0','dino_pool_mode','freeze_dino_encoder','dino_pixel_unshuffle','pointmap_depth_vis_mode','joint_video','joint_dino','joint_pointmap','enable_pointmap')}
    for branch in ('video','action','dino','pointmap'):
        sc=cfg[branch+'_scheduler']
        for key in ('train_shift','infer_shift','num_train_timesteps'):args[branch+'_'+key]=sc[key]
    for branch in ('video','action','dino','pointmap'):args['loss_lambda_'+branch]=cfg['loss'].get('lambda_'+branch,1.)
    model=CRWM(video_expert=video,action_expert=action,mot=mot,vae=vae,dino_encoder=dino,pointmap_encoder=point,text_encoder=None,tokenizer=None,text_dim=cfg['video_dit_config']['text_dim'],device=device,torch_dtype=dtype,modality_dropout=ModalityDropoutConfig(**cfg['modality_dropout']),**args)
    model.init_multisource(cfg['condition_count'],cfg['field_weights'],cfg['horizon_weights'])
    return model

def build_dataset(data_root,text_cache_root,sidecar_root):
    from .data.image_augmentation import AugmentedRequiredModalitiesDataset
    manifest=json.loads((ROOT/'data/clean_index.json').read_text())
    manifest['root']=str(Path(data_root).expanduser().resolve())
    manifest['source']['text_cache']=str(Path(text_cache_root).expanduser().resolve())
    manifest['source']['sidecar_root']=str(Path(sidecar_root).expanduser().resolve())
    return AugmentedRequiredModalitiesDataset(manifests=[manifest],stats_path=str(ROOT/'data/train_stats.json'),**read_config('dataset'))

from pathlib import Path
import os,json
import torch
from safetensors import safe_open
from .models.wan_video_dit import WanVideoDiT
from .models.wan_video_vae import WanVideoVAE38
from .models.action_dit import ActionDiT
from .models.mot import MoT
from .models.cloudrobo_wm import CloudRoboWM
from .models.flow_cloudrobo_wm import FlowCloudRoboWM,FLOW_CACHE_SCHEMA
from .models.multisource_cloudrobo_wm import MultiSourceCloudRoboWM
from .models.dino_encoder import DinoEncoder
from .models.pointmap_contraction import create_pointmap_encoder
from .models.helpers.joint_modalities import JointModalityConfig
from .text_conditioning import text_contract

def create_model(config,asset_root,*,device='cpu',dtype=torch.bfloat16,teachers=None):
    video=WanVideoDiT(**config['video_dit_config']).to(dtype=dtype)
    action=ActionDiT(**config['action_dit_config']).to(dtype=dtype)
    h=config['hbridge']
    mot=MoT({'video':video,'action':action},mot_checkpoint_mixed_attn=config['mot_checkpoint_mixed_attn'],attn_mode='sdpa',hbridge_enabled=h['enabled'],hbridge_bottom_ratio=h['bottom_ratio'],hbridge_top_ratio=h['top_ratio'])
    if teachers is None:
        assets=Path(asset_root)
        dino_path=assets/'dino.safetensors';vae_path=assets/'vae.safetensors'
        if not dino_path.is_file() or not vae_path.is_file():raise FileNotFoundError('Complete frozen encoder tensor assets are required')
        os.environ['CLOUDROBO_WM_DINO_WEIGHTS']=str(dino_path)
        vae=WanVideoVAE38().to(dtype=dtype)
        with safe_open(str(vae_path),framework='pt',device='cpu') as f:
            if set(f.keys())!=set(vae.state_dict()):raise ValueError('Frozen VAE parameter set differs')
            vae.load_state_dict({k:f.get_tensor(k) for k in f.keys()},strict=True)
        dino=DinoEncoder(model_name=config['dino_model_name'],device='cpu',torch_dtype=dtype)
    else:
        vae,dino=teachers
    pointmap=create_pointmap_encoder(pointmap_contraction=config['pointmap_contraction'],norm_bounds=config['pointmap_norm_bounds'],max_depth_m=config['pointmap_max_depth_m'],device='cpu',torch_dtype=dtype)
    options=dict(video_expert=video,action_expert=action,mot=mot,vae=vae,dino_encoder=dino,pointmap_encoder=pointmap,text_encoder=None,tokenizer=None,text_dim=config['video_dit_config']['text_dim'],proprio_dim=config['proprio_dim'],device='cpu',torch_dtype=dtype,enable_pointmap=True,joint_video=True,joint_dino=True,joint_pointmap=True,joint_modalities=JointModalityConfig(**config['joint_modalities']),composite_layout=config['composite_layout'],dino_dim=config['dino_dim'],dino_cam_patches=config['dino_cam_patches'],dino_cam_regions=config['dino_cam_regions'],dino_temporal_stride=config['dino_temporal_stride'],dino_stride_keep_far=config['dino_stride_keep_far'],dino_pred_x0=config['dino_pred_x0'],dino_pool_mode=config['dino_pool_mode'],freeze_dino_encoder=True,dino_pixel_unshuffle=config['dino_pixel_unshuffle'],pointmap_depth_vis_mode=config['pointmap_depth_vis_mode'],loss_lambda_action=config['loss']['lambda_action'],loss_lambda_dino=config['loss']['lambda_dino'],loss_lambda_pointmap=config['loss']['lambda_pointmap'])
    for branch in ['video','action','dino','pointmap']:
        sched=config[branch+'_scheduler']
        options[branch+'_train_shift']=sched['train_shift'];options[branch+'_infer_shift']=sched['infer_shift'];options[branch+'_num_train_timesteps']=sched['num_train_timesteps']
    model=CloudRoboWM(**options)
    model.__class__=FlowCloudRoboWM
    model._init_flow_branch(flow_loss_lambda=config['lambda_flow'],p_joint_flow=config['p_joint_flow'],cross_modal_predict_flow=config['cross_modal_predict_flow'],flow_layout_version=1,flow_anchor_mode='omitted',flow_cache_schema=FLOW_CACHE_SCHEMA)
    model.__class__=MultiSourceCloudRoboWM
    model.init_multisource(condition_count=config['condition_count'],field_weights=config['field_weights'],horizon_weights=config['horizon_weights'],skip_unlabeled_flow_tokens=config['skip_unlabeled_flow_tokens'],text_config=text_contract(config['tokenizer_max_len']),perception={'enabled':False},dino_inference_batch_size=config['dino_inference_batch_size'],perception_checkpoint_layers=False,perception_checkpoint_every=config['perception_checkpoint_every'])
    model.enable_dual_cooperative_action()
    from .models.history_multisource import enable_history
    enable_history(model, config['history'])
    model.mot.checkpoint_full_layers=True
    model.mot.ffn_chunk_size=0
    model=model.to(device=device,dtype=dtype)
    model.device=torch.device(device)
    return model

def trainable_mode(model):
    model.eval();model.requires_grad_(False)
    for name,module in model.named_children():
        if name not in model.FROZEN_MODULES:module.train();module.requires_grad_(True)
    for parameter in model.parameters(recurse=False):parameter.requires_grad_(True)
    return model

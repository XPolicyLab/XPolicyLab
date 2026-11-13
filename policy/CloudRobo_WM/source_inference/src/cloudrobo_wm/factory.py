import json
from pathlib import Path
import torch
from safetensors import safe_open
from .assets import configure_assets
from .models.action_dit import ActionDiT
from .models.dino_encoder import DinoEncoder
from .models.wan_video_dit import WanVideoDiT
from .models.mot import MoT
from .models.cloudrobo_wm import CloudRoboWM
from .models.flow_cloudrobo_wm import FlowCloudRoboWM
from .models.multisource_cloudrobo_wm import MultiSourceCloudRoboWM
from .models.pointmap_contraction import create_pointmap_encoder
from .models.helpers.flex_joint import FlexJointConfig
from .models.helpers.loader import load_encoders
from .text_conditioning import text_contract

def create_model(config,asset_root):
    configure_assets(asset_root)
    dtype=torch.bfloat16
    video=WanVideoDiT(**config['video_dit_config']).to(dtype=dtype)
    action=ActionDiT(**config['action_dit_config']).to(dtype=dtype)
    bridge=config['hbridge']
    mot=MoT({'video':video,'action':action},mot_checkpoint_mixed_attn=False,
            hbridge_enabled=bridge['enabled'],hbridge_bottom_ratio=bridge['bottom_ratio'],
            hbridge_top_ratio=bridge['top_ratio'])
    vae,text,tokenizer=load_encoders(dtype,config['tokenizer_max_len'])
    dino=DinoEncoder(model_name=config['dino_model_name'],device='cpu',torch_dtype=dtype)
    pointmap=create_pointmap_encoder(pointmap_contraction=config['pointmap_contraction'],
             norm_bounds=config['pointmap_norm_bounds'],max_depth_m=config['pointmap_max_depth_m'],
             device='cpu',torch_dtype=dtype)
    sched=lambda name:config[name+'_scheduler']
    model=CloudRoboWM(video_expert=video,action_expert=action,mot=mot,vae=vae,
        dino_encoder=dino,pointmap_encoder=pointmap,text_encoder=text,tokenizer=tokenizer,
        text_dim=config['video_dit_config']['text_dim'],proprio_dim=config['proprio_dim'],
        device='cpu',torch_dtype=dtype,joint_video=True,joint_dino=True,joint_pointmap=True,
        flex_joint=FlexJointConfig(**config['flex_joint']),enable_pointmap=True,
        composite_layout=config['composite_layout'],dino_dim=config['dino_dim'],
        dino_cam_patches=config.get('dino_cam_patches'),dino_cam_regions=config.get('dino_cam_regions'),
        dino_temporal_stride=config['dino_temporal_stride'],dino_stride_keep_far=config['dino_stride_keep_far'],
        dino_pred_x0=config['dino_pred_x0'],dino_pool_mode=config['dino_pool_mode'],
        freeze_dino_encoder=True,dino_pixel_unshuffle=config['dino_pixel_unshuffle'],
        pointmap_depth_vis_mode=config['pointmap_depth_vis_mode'],
        video_infer_shift=sched('video')['infer_shift'],action_infer_shift=sched('action')['infer_shift'],
        dino_infer_shift=sched('dino')['infer_shift'],pointmap_infer_shift=sched('pointmap')['infer_shift'],
        video_num_train_timesteps=sched('video')['num_train_timesteps'],
        action_num_train_timesteps=sched('action')['num_train_timesteps'],
        dino_num_train_timesteps=sched('dino')['num_train_timesteps'],
        pointmap_num_train_timesteps=sched('pointmap')['num_train_timesteps'])
    model.__class__=FlowCloudRoboWM
    model._init_flow_branch(flow_loss_lambda=0.5,p_joint_flow=0.5,cross_modal_predict_flow=True,
                           flow_layout_version=1,flow_anchor_mode='omitted',
                           flow_cache_schema='cloudrobo_wm.flow.fp16_x264rgb.final_tile.stride4.v1')
    model.__class__=MultiSourceCloudRoboWM
    model.init_multisource(condition_count=config['condition_count'],text_config=text_contract(128),
        perception={'enabled':False},dino_inference_batch_size=config['dino_inference_batch_size'],
        perception_checkpoint_layers=False,perception_checkpoint_every=2)
    model.enable_dual_cooperative_action()
    from .models.history_multisource import enable_history
    return enable_history(model,config['history'])

def load_tensors(model,path):
    keys=tuple(dict.fromkeys(('mot','proprio_encoder',*model._dino_ckpt_keys(),*model._mode_ckpt_keys())))
    counts={}
    with safe_open(str(path),framework='pt',device='cpu') as saved:
        available=set(saved.keys())
        expected={root+'.'+name for root in keys for name in getattr(model,root).state_dict()}
        if available!=expected:
            raise ValueError('The tensor bundle differs from the complete inference parameter set')
        for root in keys:
            module=getattr(model,root)
            state={name:saved.get_tensor(root+'.'+name) for name in module.state_dict()}
            for name,value in state.items():
                target=module.state_dict()[name]
                if value.shape!=target.shape or value.dtype!=target.dtype:
                    raise ValueError('Tensor shape or dtype mismatch: '+root+'.'+name)
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError('Nonfinite inference tensor')
            module.load_state_dict(state,strict=True)
            for name,value in module.state_dict().items():
                if not torch.equal(value.cpu(),state[name]):
                    raise ValueError('Loaded tensor value mismatch: '+root+'.'+name)
            counts[root]=len(state)
    model.parameter_acceptance={'verified':True,'tensors':sum(counts.values()),'modules':counts}
    return model.parameter_acceptance

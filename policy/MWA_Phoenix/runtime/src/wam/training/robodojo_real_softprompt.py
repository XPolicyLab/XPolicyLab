"""Opt-in per-sample real ARX5 / Piper / Piper-X prompt transfer."""
from omegaconf import OmegaConf
from wam.datasets.robodojo_real_joint96 import ROBOT_IDS


def validate_robodojo_real_prompt(cfg):
    get=lambda k,d=None:OmegaConf.select(cfg,k,default=d)
    if (not get('joint96_posttrain.enabled') or get('local_robot_joint96.enabled')
            or get('resume') is not None or get('local_robot_softprompt.init_checkpoint') is not None
            or get('joint96_posttrain.cached_visuals')):
        raise ValueError('Real RoboDojo requires fresh online Joint96 weights-only initialization')
    if any(get(f'data.{s}.domain') != 'robodojo' for s in ('train','val')):
        raise ValueError('Real RoboDojo-only prompt recipe')
    if (get('model.action_dit_config.num_embodiments'),get('model.action_dit_config.soft_prompt_tokens')) != (28,8):
        raise ValueError('Preserve complete pretrained 28-slot / 8-token prompt table')
    from wam.model.modules.conditioning.frame_attention import JOINT96_ACTION_VIDEO_ATTENTION
    if get('joint96_posttrain.action_video_attention') != JOINT96_ACTION_VIDEO_ATTENTION:
        raise ValueError('Use matching current/noisy-future Joint attention')
    for s in ('train','val'):
        if get(f'data.{s}.visual_cache') is not None or get(f'data.{s}.val_set_proportion',0) != 0:
            raise ValueError('All real episodes; validation is a non-held-out loss panel')


def wrap_robodojo_real_prompt(model, train, val, cfg, registry):
    if val is None or train is val:
        raise ValueError('Need distinct train/validation readers')
    for robot, eid in ROBOT_IDS.items():
        expected={'piper':'piper100','piper_x':'piperx'}.get(robot,robot)
        if registry.content['embodiments'][eid] != expected or eid not in registry.assigned_ids:
            raise ValueError(f'Wrong pretrained mechanical-family prompt slot: {robot}')
    # The common trainer gathers this fingerprint to reject rank-wise ID drift.
    train.embodiment_registry = registry
    val.embodiment_registry = registry
    model.soft_prompt_training_contract = dict(domain='robodojo_real',
        embodiment_ids=ROBOT_IDS, action_space='native_joint14_to_canonical80',
        normalization='frame_weighted_per_mechanical_family_joint14',
        cached_visuals=False, native_fps=[25,30])
    return train, val

"""Fail closed at public entrypoints; never silently fall back to another checkout."""
import importlib.abc
from pathlib import Path
import sys


class DenyExternalFastWAM(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'fastwam' or fullname.startswith('fastwam.'):
            raise ImportError('Standalone posttrain must use its bundled encoder, not external FastWAM')


def install_import_guard():
    import wam
    root = Path(__file__).resolve().parents[3]
    if not Path(wam.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Conflicting wam package; use this checkout in an isolated process')
    if any(n == 'fastwam' or n.startswith('fastwam.') for n in sys.modules):
        raise RuntimeError('External FastWAM was already imported')
    sys.meta_path.insert(0, DenyExternalFastWAM())


def validate_scope(cfg):
    from omegaconf import OmegaConf
    select = lambda name, default=None: OmegaConf.select(cfg, name, default=default)
    if not select('joint96_posttrain.enabled') or select('local_robot_joint96.enabled', False):
        raise ValueError('This repository exposes post-training only, not pretraining')
    if select('local_robot_multiview.enabled', False):
        raise ValueError('Pretraining multiview data mixing is not supported')
    if select('resume') is not None:
        raise ValueError('This release accepts fresh weights-only initialization; full-state resume is not certified')
    robocasa = select('joint96_posttrain.robocasa.enabled', False)
    robotwin = select('joint96_posttrain.robotwin_softprompt.enabled', False)
    if robotwin:
        from wam.training.robotwin_robot37_softprompt import validate_robotwin_prompt
        validate_robotwin_prompt(cfg)
        if robocasa or not select('joint96_posttrain.online_robotwin') or select('joint96_posttrain.cached_visuals'):
            raise ValueError('RoboTwin requires explicit online RGB posttraining')
    if robocasa:
        from wam.datasets.robocasa_delta80_slot18 import CONTRACT
        from wam.training.robocasa_posttrain import validate_robocasa_config
        validate_robocasa_config(cfg)
        if select('joint96_posttrain.robocasa.frame_contract') != CONTRACT['version']:
            raise ValueError('Only RoboCasa Delta80 slot18 is supported')
        if select('data.train.allow_smoke', False):
            if select('max_steps') not in (1, 2, 3) or select('batch_size') != 1:
                raise ValueError('RoboCasa smoke is bounded to B1 and at most 3 updates')
    domain = 'robotwin' if robotwin else ('robocasa' if robocasa else 'libero')
    reader = ('wam.datasets.robotwin_joint96_robot37.RobotwinJoint96OnlineDataset' if robotwin
              else 'wam.datasets.robocasa_posttrain.RoboCasaPosttrainDataset' if robocasa
              else 'wam.datasets.libero_joint96_cache.LiberoJoint96CachedDataset')
    for split in ('train', 'val'):
        if select(f'data.{split}.domain') != domain or select(f'data.{split}._target_') != reader:
            raise ValueError('Use the explicit benchmark reader and domain')
    if (not robotwin and not select('joint96_posttrain.cached_visuals')) or select('video_latent_cache', 'off') != 'off':
        raise ValueError('Native visual cache generation is not part of this release')
    if select('model.joint_visual_codec.source_root') != 'bundled://dino-stvae-v1':
        raise ValueError('External encoder source roots are not allowed')

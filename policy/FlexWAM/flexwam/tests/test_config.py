"""Config regression coverage, including the intentional RoboTwin normalization fix."""
import hashlib
import importlib
import json
from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict

from flexwam.config import compose_task, resolve_config
from flexwam.models.architectures.view_scoring_head import arches_compatible, normalize_head_arch
from flexwam.models.architectures.critic import CameraCritic
from flexwam.models.architectures.router import CameraRouter
from flexwam.training.trainer import FlexWAMTrainer
from flexwam.utils.config_resolvers import register_default_resolvers

ROOT = Path(__file__).resolve().parents[1]


def fingerprint(cfg):
    value = OmegaConf.to_container(cfg, resolve=True)
    value['model']['_target_'] = value['model']['_target_'].replace('models.architectures.builder.', 'models.builder.')
    value.pop('output_dir', None)  # Runtime timestamp, not model/data behavior.
    for key in ['model_path', 'action_dit_pretrained_path']:
        value['model'].pop(key, None)  # Same weights can live anywhere.
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class ConfigTests(unittest.TestCase):
    def public(self, task='penbox_v2', overrides=()):
        register_default_resolvers()
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
            return compose(config_name='train', overrides=[f'task={task}', *overrides])

    def test_existing_tasks_match_baseline_except_intentional_changes(self):
        expected = json.loads((ROOT/'tests/fixtures/task_config_hashes.json').read_text())
        for task, digest in expected.items():
            with self.subTest(task=task):
                cfg = compose_task(task)
                critic = OmegaConf.to_container(cfg.model.action_critic, resolve=True)
                self.assertNotIn('same_router', critic)
                self.assertEqual(critic, OmegaConf.to_container(cfg.model.action_router, resolve=True))
                self.assertEqual(cfg.model.action_router.arch, 'spatial_query_v1')
                with open_dict(cfg.model):
                    del cfg.model.action_critic
                    del cfg.model.action_router.arch
                    # Optional state paths default off and must not change legacy tasks.
                    self.assertFalse(cfg.model.video_dit_config.state_conditioning.enabled)
                    self.assertFalse(cfg.model.action_router.state_conditioning.enabled)
                    self.assertEqual(cfg.model.state_conditioning.drop_prob, 0.0)
                    self.assertEqual(cfg.model.action_attention.mode, 'additive')
                    self.assertEqual(cfg.model.action_attention.bias_strength, 1.0)
                    del cfg.model.video_dit_config.state_conditioning
                    del cfg.model.action_router.state_conditioning
                    del cfg.model.state_conditioning
                    del cfg.model.action_attention
                self.assertFalse(cfg.migrate_state_conditioning)
                del cfg.migrate_state_conditioning
                # The shared default now explicitly selects router_zero.
                self.assertEqual(cfg.model.router_cf.intervention, 'router_zero')
                with open_dict(cfg.model.router_cf):
                    del cfg.model.router_cf.intervention
                    if task == 'penbox_v2':
                        cfg.model.router_cf.probe = 'router_zero'
                if task == 'robotwin':
                    # This intentional fix is checked independently below.
                    cfg.data.val.processor.norm_default_mode = 'z-score'
                # Cross-view fusion became cross-view attention layers at the same interval.
                video = cfg.model.video_dit_config
                self.assertIsNone(video.cross_view_attention_layers)
                with open_dict(video):
                    video.cross_view_fusion_interval = video.cross_view_attention_interval
                    video.cross_view_fusion_mode = 'full_frame'
                    del video.cross_view_attention_interval
                    del video.cross_view_attention_layers
                self.assertEqual(fingerprint(cfg), digest)

    def test_new_dataset_reuses_robot_contract(self):
        cfg = resolve_config(self.public(overrides=['data.roots=[/new/dataset]', 'train.batch_size=3']))
        self.assertEqual(list(cfg.data.train.dataset_dirs), ['/new/dataset'])
        self.assertEqual(list(cfg.data.val.dataset_dirs), ['/new/dataset'])
        self.assertEqual(cfg.batch_size, 3)
        self.assertEqual(cfg.data.train.processor.num_output_cameras, 3)
        self.assertEqual(list(cfg.data.train.video_size), [192,224])
        self.assertEqual(list(cfg.data.train.processor.train_transforms[0].size), [192,224])

    def test_state_switches_are_independent_and_validated(self):
        cfg = resolve_config(self.public(overrides=[
            'model.video.state_conditioning.enabled=true',
            'model.critic.state_conditioning.enabled=true',
            'model.state_conditioning.drop_prob=0.2',
        ]))
        self.assertTrue(cfg.model.video_dit_config.state_conditioning.enabled)
        self.assertTrue(cfg.model.action_critic.state_conditioning.enabled)
        self.assertFalse(cfg.model.action_router.state_conditioning.enabled)
        self.assertEqual(cfg.model.state_conditioning.drop_prob, 0.2)
        for override in ('model.state_conditioning.drop_prob=1.1',
                         'model.video.state_conditioning.num_tokens=2',
                         'model.router.state_conditioning.hidden_dim=0',
                         'model.critic.state_conditioning.mode=prefix'):
            with self.subTest(override=override), self.assertRaises(ValueError):
                resolve_config(self.public(overrides=[override]))

    def test_two_camera_task_and_critic_variant(self):
        public = self.public('screw')
        del public.data.observation.cameras[1]
        cfg = resolve_config(public)
        self.assertEqual(cfg.model.video_dit_config.num_views, 2)
        self.assertEqual(cfg.model.action_router.num_views, 2)
        self.assertEqual(cfg.model.action_critic.num_views, 2)
        self.assertNotIn('same_router', cfg.model.action_critic)
        self.assertEqual(cfg.model.action_critic.feature_layer_index, cfg.model.action_router.feature_layer_index)
        self.assertEqual(compose_task('penbox_v2').model.router_cf.intervention, 'router_zero')
        overridden = compose_task('penbox_v2', overrides=('train.counterfactual.intervention=pixel',))
        self.assertEqual(overridden.model.router_cf.intervention, 'pixel')
        self.assertNotIn('probe', overridden.model.router_cf)

    def test_critic_reads_its_own_config(self):
        cfg = compose_task('penbox_v2')
        self.assertEqual(cfg.model.action_critic.arch, 'spatial_query_v1')
        self.assertEqual(cfg.model.action_critic.feature_dim, cfg.model.action_router.feature_dim)
        overridden = compose_task('penbox_v2', overrides=(
            'model.critic.feature_layer_index=20',
            'model.critic.feature_dim=256',
        ))
        self.assertEqual(overridden.model.action_critic.feature_layer_index, 20)
        self.assertEqual(overridden.model.action_critic.feature_dim, 256)
        self.assertEqual(overridden.model.action_router.feature_layer_index, 18)
        self.assertEqual(overridden.model.action_router.feature_dim, 512)

    def test_missing_critic_copies_router_network(self):
        public = self.public('penbox_v2')
        del public.model.critic
        cfg = resolve_config(public)
        self.assertNotIn('same_router', cfg.model.action_critic)
        self.assertEqual(cfg.model.action_critic.arch, 'spatial_query_v1')
        self.assertEqual(cfg.model.action_critic.feature_layer_index, cfg.model.action_router.feature_layer_index)
        self.assertEqual(cfg.model.action_critic.feature_dim, cfg.model.action_router.feature_dim)

    def test_legacy_head_arch_alias_and_independent_modules(self):
        cfg = compose_task('penbox_v2', overrides=('model.router.arch=l18_spatial_query_v1',))
        self.assertEqual(cfg.model.action_router.arch, 'spatial_query_v1')
        self.assertEqual(cfg.model.action_critic.arch, 'spatial_query_v1')
        self.assertTrue(arches_compatible('l18_spatial_query_v1', 'spatial_query_v1'))
        self.assertEqual(normalize_head_arch('l18'), 'spatial_query_v1')
        kwargs = dict(
            token_dim=8, num_views=2, feature_dim=8, pool_grid=[2, 2],
            num_queries=2, num_heads=2, spatial_hidden_dim=16,
            branch_output_dim=8, head_hidden_dim=16,
        )
        router, critic = CameraRouter(**kwargs), CameraCritic(**kwargs)
        self.assertIsNot(router, critic)
        self.assertTrue(all(id(left) != id(right) for left, right in zip(router.parameters(), critic.parameters())))

    def test_router_overrides_do_not_change_critic(self):
        cfg = compose_task('penbox_v2', overrides=(
            'model.router.feature_layer_index=20',
            'model.router.feature_dim=256',
        ))
        self.assertEqual(cfg.model.action_router.feature_layer_index, 20)
        self.assertEqual(cfg.model.action_router.feature_dim, 256)
        self.assertEqual(cfg.model.action_critic.feature_layer_index, 18)
        self.assertEqual(cfg.model.action_critic.feature_dim, 512)

    def test_bad_camera_or_layer_config_fails_early(self):
        for overrides in [
            ['model.video.num_views=2'],
            ['model.router.feature_layer_index=24'],
            ['model.critic.feature_layer_index=24'],
            ['data.sampling.video_stride=3'],
        ]:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                resolve_config(self.public(overrides=overrides))

    def test_model_factory_and_processor_targets_resolve(self):
        cfg = compose_task('penbox_v2')
        for target in [cfg.model._target_,cfg.data.train.processor._target_]:
            module,name=target.rsplit('.',1)
            self.assertTrue(callable(getattr(importlib.import_module(module), name)))

    def test_robotwin_validation_uses_training_normalization(self):
        for mode in ['q01/q99', 'z-score']:
            with self.subTest(mode=mode):
                cfg = resolve_config(self.public('robotwin', [f'data.normalization.mode={mode}']))
                self.assertEqual(cfg.data.train.processor.norm_default_mode, mode)
                self.assertEqual(cfg.data.val.processor.norm_default_mode, mode)

    def test_runtime_projection_is_idempotent(self):
        cfg=compose_task('penbox_v2')
        self.assertIs(resolve_config(cfg),cfg)

    def test_no_router_ablation_keeps_heads_off_under_default_stages(self):
        for task in ('ablation/L_no_router', 'ablation/L_wan_no_router', 'ablation/penbox_wan_no_router'):
            with self.subTest(task=task):
                cfg = compose_task(task)
                self.assertFalse(cfg.model.action_router.enabled)
                self.assertFalse(cfg.model.action_critic.enabled)
                stages = cfg.training_stages
                self.assertTrue(stages.enabled)
                self.assertEqual(float(stages.stage1_ratio), 1.0)
                self.assertEqual(float(stages.stage2_ratio), 0.0)
                self.assertEqual(float(stages.stage3_ratio), 0.0)
                self.assertEqual(float(stages.stage4_ratio), 0.0)
                bounds = FlexWAMTrainer._stage_step_bounds(
                    FlexWAMTrainer._stage_fractions(stages), 13470
                )
                self.assertEqual(bounds, [0, 13470, 13470, 13470, 13470, 13470])

    def test_router_no_cf_ablation_trains_router_without_critic_stages(self):
        for task in ('ablation/L_router_no_cf', 'ablation/L_wan_router_no_cf', 'ablation/penbox_wan_router_no_cf'):
            with self.subTest(task=task):
                cfg = compose_task(task)
                self.assertTrue(cfg.model.action_router.enabled)
                self.assertFalse(cfg.model.action_critic.enabled)
                stages = cfg.training_stages
                self.assertTrue(stages.enabled)
                self.assertEqual(float(stages.stage1_ratio), 0.1)
                self.assertEqual(float(stages.stage2_ratio), 0.9)
                self.assertEqual(float(stages.stage3_ratio), 0.0)
                self.assertEqual(float(stages.stage4_ratio), 0.0)
                bounds = FlexWAMTrainer._stage_step_bounds(
                    FlexWAMTrainer._stage_fractions(stages), 10000
                )
                self.assertEqual(bounds, [0, 1000, 10000, 10000, 10000, 10000])

    def test_direct_cf_ablation_skips_critic_and_uses_router_zero(self):
        cfg = compose_task('ablation/penbox_attention_bias_cross2_cf_direct')
        self.assertTrue(cfg.model.action_router.enabled)
        self.assertFalse(cfg.model.action_critic.enabled)
        self.assertEqual(cfg.model.router_cf.teacher, 'cf')
        self.assertEqual(cfg.model.router_cf.intervention, 'router_zero')
        self.assertEqual(cfg.model.action_attention.mode, 'logit_bias')
        self.assertEqual(int(cfg.model.video_dit_config.cross_view_attention_interval), 2)
        stages = cfg.training_stages
        self.assertEqual(float(stages.stage1_ratio), 0.1)
        self.assertEqual(float(stages.stage2_ratio), 0.0)
        self.assertEqual(float(stages.stage3_ratio), 0.0)
        self.assertEqual(float(stages.stage4_ratio), 0.9)
        self.assertEqual(float(stages.stage5_ratio), 0.0)
        bounds = FlexWAMTrainer._stage_step_bounds(
            FlexWAMTrainer._stage_fractions(stages), 10000
        )
        self.assertEqual(bounds, [0, 1000, 1000, 1000, 10000, 10000])


if __name__ == '__main__':
    unittest.main()

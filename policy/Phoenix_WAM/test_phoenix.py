"""CPU regression tests. Synthetic payloads are NOT posttraining weights."""
import copy
import os
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from .joint96_assets import POLICY_DIR, read_source, sha256
from .model import Model, _activate_runtime
from .phoenix_contract import (ACTIVE, MODE, ROBOTS, PROMPT, CachedText, current_rgb,
                               install_repeat9_joint, preprocess_state, preprocessing_contract,
                               validate_training)


def fixture(directory, family='arx_x5'):
    """Build only metadata/small tensors, with distinct action/state parameters."""
    from wam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
    source = POLICY_DIR / 'joint96_source'
    old = OmegaConf.load(source / 'inference.json')
    proc = OmegaConf.to_container(old.observation.processor, resolve=True)
    proc.update(norm_default_mode='q01/q99', norm_exception_mode=None,
                normalization_passthrough_dims=None, use_stepwise_action_norm=False)
    cfg = OmegaConf.create(dict(model=OmegaConf.to_container(old.model, resolve=True),
        joint96_posttrain=dict(enabled=True, cached_visuals=False, robodojo_openwam_joint14=True,
                              expected_pretrain_step=256013),
        data={'train': dict(_target_='wam.datasets.robodojo_openwam_joint14.RoboDojoOpenWAMJoint14Dataset',
            embodiment=family, gripper_preprocessing='openwam_real_clip01_v1', visual_cache=None, processor=proc)}))
    stats = directory / f'stats_slot{ROBOTS[family][0]}.json'
    stats.write_bytes((source / 'normalization' / stats.name).read_bytes())
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(load_dataset_stats_from_json(stats))
    textfile = directory / 'text.pt'
    torch.save(dict(prompt_template=PROMPT, rows={PROMPT.format(task='test'): dict(
        context=torch.ones(128, 4096), mask=torch.arange(128) < 3)}), textfile)
    index = dict(prompt_template=PROMPT, package_sha256={stats.name: sha256(stats), 'text.pt': sha256(textfile)},
                 joint14_preprocessing={family: preprocessing_contract(family)})
    (directory / 'index.json').write_text(json.dumps(index))
    fields = {}
    for name in ('action', 'state'):
        n = processor.normalizer.normalizers[name]['default']
        fields[name] = dict(scale=n.scale.tolist(), offset=n.offset.tolist(),
                            q01=n.get_stats()['q01'].tolist(), q99=n.get_stats()['q99'].tolist())
    contract = dict(domain='robodojo_real', embodiment_ids={family: ROBOTS[family][0]},
        action_space='native_joint14_to_canonical80', cached_visuals=False, cached_text=True,
        dataloader='openwam_raw_hdf5_joint14', normalization='frame_weighted_per_family_q01_q99_clip5',
        action_offsets=list(range(32)), video_offsets=list(range(0, 33, 4)),
        rgb_geometry='aspect_bicubic_center_crop_256_per_view', rgb_decode='robodojo_cv2_no_channel_swap',
        loss_weights=dict(vae=.5, stvae=.5, action=1.), raw_index_sha256=sha256(directory/'index.json'),
        normalization_contract=dict(version='robodojo_joint14_q01_q99_clip5_v2', mode='q01/q99',
            forward_clip=[-5., 5.], inverse_clip=None, family=family, canonical_active_dims=ACTIVE,
            stats_filename=stats.name, stats_sha256=sha256(stats),
            joint14_preprocessing=preprocessing_contract(family), fields=fields))
    from .joint96_assets import read_release
    geometry = 'direct_bicubic_resize_256_per_view_horizontal_concat'
    cfg.joint96_posttrain.rgb_geometry = geometry
    cfg.data.train._target_ = 'wam.datasets.robodojo_h5_resize256.RoboDojoH5Resize256Dataset'
    contract.update(dataloader='openwam_raw_hdf5_joint14_direct_resize256',
        rgb_geometry=geometry, rgb_camera_order=['cam_head','cam_left_wrist','cam_right_wrist'],
        rgb_view_size=[256,256], rgb_crop=None,
        rgb_interpolation='PIL_torchvision_BICUBIC_antialias_true',
        visual_encoding='three_view_boundaries_preserved_latent_horizontal',
        geometry_source_sha256=read_release()['resize_training_source_sha256'])
    identity = json.loads((source/'cache_metadata/manifest.json').read_text())['contract']['encoder_identity']
    registry = json.loads((source/'registry.json').read_text())
    from wam.embodiment_registry import EmbodimentRegistry
    encoder = {k:identity[k] for k in ('source_sha256','posterior','per_camera_input_hw','temporal_compression','extra_latent_scaling')}
    encoder['stvae'] = {k:identity[k] for k in ('stvae_checkpoint_sha256','teacher_sha256','stvae_step',
        'teacher_checkpoint_state','teacher_output_source','teacher_preprocessing','dinov3_source_commit')}
    metadata = {k:identity[k] for k in ('channel_order','camera_order','action_dim','proprio_dim','action_video_attention')}
    metadata.update(schema='internw0.local_joint96.weights.v1',
        model_paths=dict(joint_visual_encoder=encoder, posttrain_initialization={'step':256013}),
        soft_prompt=dict(schema='wam.action_soft_prompt.v1', num_embodiments=28, num_tokens=8,
            context_dim=1024, registry=registry, registry_sha256=EmbodimentRegistry.content_hash(registry),
            training_contract=contract))
    payload = dict(step=5000, joint96_metadata=metadata,
        mot={'mixtures.action.embodiment_prompt.embedding.weight':torch.ones(28,8192)})
    return cfg, payload, processor


class PhoenixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _, assets = read_source()
        _activate_runtime(assets['runtime_root'])
        torch.set_num_threads(1)
        from . import phoenix_backend
        cls.backend = phoenix_backend

    def test_public_release_is_latest_and_self_contained(self):
        from .joint96_assets import read_release
        release = read_release()
        self.assertEqual(release['runtime_mode'], 'joint96_real_phoenix')
        package = POLICY_DIR / 'phoenix_source'
        manifest = json.loads((package/'manifest.json').read_text())
        for name, digest in manifest['sha256'].items():
            self.assertEqual(sha256(package/name), digest)
        for family, spec in release['checkpoints'].items():
            self.assertEqual(spec['pretrain_step'], 256013)
            self.assertEqual(spec['step'], 40000)
            self.assertEqual(spec['filename'], f"phoenix-wam-{family.replace('_','-')}-h5-trim-cached-resize256-nocrop-joint14-init256013-q01q99-clip5-step_040000-run20261019_01.pt")
            self.assertEqual(spec['path'], spec['filename'])
            cfg = (package/(family+'.json')).read_text()
            self.assertNotIn('/mnt/', cfg)

    def test_online_t5_preserves_prompt_and_padding_contract(self):
        with tempfile.TemporaryDirectory() as d:
            _, _, processor = fixture(Path(d))
            calls = []
            def encode(prompt):
                calls.append(prompt)
                return torch.ones(1,128,4096), torch.ones(1,128,dtype=torch.bool)
            runtime = SimpleNamespace(model=SimpleNamespace(encode_prompt=encode), text=None,
                processor=processor, device='cpu', robot_type='arx_x5',
                action_dim_is_pad=self.backend._static_dimension_is_pad(processor,'action'))
            session = self.backend.PhoenixSession(runtime,SimpleNamespace(num_inference_steps=10,replan_steps=32,seed=0,rand_device='cpu'))
            session._text_condition('test')
            session._text_condition('test')
            self.assertEqual(calls, [PROMPT.format(task='test')])

    def test_cached_visual_checkpoint_is_rejected(self):
        for family in ROBOTS:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as d:
                cfg, payload, _ = fixture(Path(d), family)
                contract = payload['joint96_metadata']['soft_prompt']['training_contract']
                contract.update(cached_visuals=True, dataloader='openwam_joint14_with_legacy_clean_visual_cache')
                with self.assertRaisesRegex(ValueError, 'direct-resize'):
                    validate_training(cfg, payload, Path(d), family, 5000)

    def test_contract_and_all_robot_slots(self):
        for family in ROBOTS:
            with self.subTest(family=family), tempfile.TemporaryDirectory() as d:
                cfg, payload, _ = fixture(Path(d), family)
                result = validate_training(cfg, payload, Path(d), family, 5000)
                self.assertTrue(result['support']['posttrained'])
                self.assertEqual(result['support']['embodiment_id'], ROBOTS[family][0])

    def test_reject_legacy_or_wrong_normalization(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, payload, _ = fixture(Path(d))
            for field, value in [('cached_visuals', True), ('embodiment_ids', {'piper':4}),
                                 ('normalization', 'frame_weighted_per_mechanical_family_joint14')]:
                p = copy.deepcopy(payload)
                p['joint96_metadata']['soft_prompt']['training_contract'][field] = value
                with self.assertRaises(ValueError):validate_training(cfg,p,Path(d),'arx_x5',5000)
            for field, value in [('forward_clip',[-1.,1.]), ('family','piper'), ('stats_sha256','bad')]:
                p = copy.deepcopy(payload)
                p['joint96_metadata']['soft_prompt']['training_contract']['normalization_contract'][field] = value
                with self.assertRaises(ValueError):validate_training(cfg,p,Path(d),'arx_x5',5000)
            p = copy.deepcopy(payload)
            p['joint96_metadata']['soft_prompt']['training_contract']['normalization_contract']['fields']['state']['scale'][0] += .1
            with self.assertRaises(ValueError):validate_training(cfg,p,Path(d),'arx_x5',5000)

    def test_reject_wrong_step_lineage_config_mapping(self):
        with tempfile.TemporaryDirectory() as d:
            cfg, payload, _ = fixture(Path(d))
            for step in (256013, 40001, 10000):
                with self.assertRaises(ValueError):validate_training(cfg,payload,Path(d),'arx_x5',step)
            bad = copy.deepcopy(payload)
            bad['joint96_metadata']['model_paths']['posttrain_initialization']['step'] = 245000
            with self.assertRaises(ValueError):validate_training(cfg,bad,Path(d),'arx_x5',5000)
            for key,value in [('data.train.processor.norm_default_mode','z-score'),
                              ('model.memory.enabled',True)]:
                badcfg=copy.deepcopy(cfg);OmegaConf.update(badcfg,key,value)
                with self.assertRaises(ValueError):validate_training(badcfg,payload,Path(d),'arx_x5',5000)
            cfg.data.train.processor.action_state_merger.state_target_slices[0].source_slice = [7,13]
            with self.assertRaises(ValueError):validate_training(cfg,payload,Path(d),'arx_x5',5000)

    def test_gripper_preprocessing_and_finite(self):
        for family in ROBOTS:
            x=np.zeros(14,np.float32);x[[6,13]]=[-.03,1.03]
            original=x.copy();actual=preprocess_state(x,family)
            np.testing.assert_array_equal(x,original)
            np.testing.assert_array_equal(actual[[6,13]],original[[6,13]] if family=='arx_x5' else [0,1])
            for bad in (np.nan,np.inf):
                x[0]=bad
                with self.assertRaises(ValueError):preprocess_state(x,family)
        for value in (-.051,1.051):
            x=np.zeros(14,np.float32);x[6]=value
            with self.assertRaises(ValueError):preprocess_state(x,'piper')

    def test_state_action_normalization_and_inverse(self):
        with tempfile.TemporaryDirectory() as d:
            cfg,payload,processor=fixture(Path(d),'piper')
            runtime=SimpleNamespace(model=None,processor=processor,device='cpu',robot_type='piper',
                action_dim_is_pad=self.backend._static_dimension_is_pad(processor,'action'))
            session=self.backend.PhoenixSession(runtime,SimpleNamespace(num_inference_steps=10,replan_steps=32,seed=0,rand_device='cpu'))
            x=np.arange(14,dtype=np.float32)/10;x[[6,13]]=[-.03,1.03]
            actual=session._normalize_state(x)
            n=processor.normalizer.normalizers['state']['default']
            expected=(torch.from_numpy(preprocess_state(x,'piper'))*n.scale+n.offset).clamp(-5,5)
            torch.testing.assert_close(actual[:,ACTIVE],expected[None],rtol=0,atol=0)
            self.assertEqual(torch.count_nonzero(actual[:,runtime.action_dim_is_pad]),0)
            z=torch.zeros(32,80);z[:,ACTIVE]=torch.linspace(-2,2,14)
            a=processor.normalizer.normalizers['action']['default']
            expected=(z[:,ACTIVE]-a.offset)/a.scale
            np.testing.assert_array_equal(session._denormalize_action(z)[0],expected.numpy())
            z[:,7]=1
            with self.assertRaises(ValueError):session._denormalize_action(z)

    def test_prepared_rgb_colors_float32(self):
        from torchvision.transforms.functional import to_tensor
        rng=np.random.default_rng(42)
        for h,w in [(256,256)]:
            obs={'observation':{k:{'rgb':rng.integers(0,256,(h,w,3),dtype=np.uint8)}
                 for k in ('head_camera','left_camera','right_camera')}}
            actual=current_rgb(obs)
            self.assertEqual(actual.dtype,torch.float32)
            for i,k in enumerate(obs['observation']):
                image=obs['observation'][k]['rgb']
                torch.testing.assert_close(actual[0,i],to_tensor(image)*2-1,rtol=0,atol=0)
            obs['observation']['head_camera']['rgb']=np.zeros((h,w,3),np.float32)
            with self.assertRaises(ValueError):current_rgb(obs)

    def test_repeat9_wraps_full_encoder_not_only_stvae(self):
        calls=[]
        def encode(video, *, camera_is_pad):
            calls.append((video.clone(),camera_is_pad.clone()))
            b=video.shape[0];t=(video.shape[3]-1)//4+1
            result=torch.ones(b,96,t,16,48)*video.shape[3]
            return result
        visual=SimpleNamespace(current_conditioning='legacy_clip_prefix',stvae_encoder=object(),encode=encode)
        evidence=install_repeat9_joint(visual)
        x=torch.randn(2,3,3,1,256,256);mask=torch.zeros(2,3,dtype=torch.bool)
        result=visual.encode(x,camera_is_pad=mask)
        self.assertEqual(tuple(result.shape),(2,96,1,16,48))
        self.assertTrue(torch.all(result==9))
        torch.testing.assert_close(calls[0][0],x.repeat(1,1,1,9,1,1),rtol=0,atol=0)
        self.assertEqual(evidence['encoded_frames'],9)
        visual.encode(x.repeat(1,1,1,9,1,1),camera_is_pad=mask)
        self.assertEqual(evidence['calls'],1)
        with self.assertRaises(RuntimeError):install_repeat9_joint(visual)

    def test_text_zero_padding_all_visible_and_unknown_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            fixture(Path(d));p=Path(d)/'text.pt';text=CachedText(p,sha256(p))
            context,mask=text.get('test',device='cpu',dtype=torch.bfloat16)
            self.assertEqual(tuple(context.shape),(1,128,4096));self.assertTrue(mask.all())
            self.assertEqual(torch.count_nonzero(context[:,3:]),0)
            self.assertTrue(torch.all(context[:,:3]==1))
            with self.assertRaises(ValueError):text.get('unknown',device='cpu',dtype=torch.float32)
            with self.assertRaises(ValueError):CachedText(p,'bad')

    def test_real_joint_encoder_calls_both_branches_with_nine_frames(self):
        from wam.model.modules.codecs.joint_visual_encoder import JointVisualEncoder
        calls=[]
        class Wan(torch.nn.Module):
            def __init__(self):
                super().__init__();self.weight=torch.nn.Parameter(torch.zeros(1))
                self.model=SimpleNamespace(clear_cache=lambda:None)
            def encode(self,videos,**kwargs):
                calls.append(('wan',videos[0].shape[1]))
                return torch.ones(1,48,3,16,16,dtype=torch.bfloat16)
        class STVAE(torch.nn.Module):
            def encode_raw_video_latents(self,video,*,num_views):
                calls.append(('dino_stvae',video.shape[2]))
                return torch.ones(video.shape[0],48,3,16,48,dtype=torch.bfloat16)*2
        wan=Wan();visual=JointVisualEncoder(wan,STVAE(),allow_cpu_for_tests=True)
        install_repeat9_joint(visual)
        z=visual.encode(torch.zeros(1,3,3,1,256,256),camera_is_pad=torch.zeros(1,3,dtype=torch.bool))
        self.assertEqual(calls,[('wan',9)]*3+[('dino_stvae',9)])
        self.assertTrue(torch.all(z[:,:48]==1));self.assertTrue(torch.all(z[:,48:]==2))

    def test_device_limit_is_separate_and_logged(self):
        with tempfile.TemporaryDirectory() as d:
            raw=np.full((32,14),2.,np.float32);raw[:,6]=-.2;raw[:,13]=1.2
            session=SimpleNamespace(pending_model_actions=[],get_action=lambda request:raw,
                get_timing_rollout=lambda:{'replan_calls':1},
                runtime=SimpleNamespace(audit_dir=Path(d),visual_evidence={'encoded_frames':9}))
            adapter=SimpleNamespace(replan_steps=32,env_cfg_type='piper',action_type='joint',
                                    robot_action_dim_info={})
            with patch('XPolicyLab.utils.process_data.unpack_robot_state',side_effect=lambda x,*a,**kw:x):
                result=self.backend.predict_phoenix(adapter,session,{},'test')
            np.testing.assert_array_equal(result[:,0],raw[:,0])
            np.testing.assert_array_equal(result[:,6],0);np.testing.assert_array_equal(result[:,13],1)
            self.assertEqual(float(raw[0,0]),2.)
            record=json.loads((Path(d)/'device_adapter.jsonl').read_text())
            self.assertEqual(record['changed_values'],64)

    def test_options_and_new_mode_route(self):
        for family,(_,_,hz) in ROBOTS.items():
            opts=self.backend.validate_options(dict(env_cfg_type=family))
            self.assertEqual(opts['hz'],hz);self.assertEqual(opts['replan'],32)
        for opts in (dict(current_visual_mode='repeat9_full_stvae'),dict(sigma_shift=6),dict(action_horizon=16),dict(replan_steps=33)):
            with self.assertRaises(ValueError):self.backend.validate_options(dict(env_cfg_type='arx_x5',**opts))
        fake=SimpleNamespace(model_cfg={'runtime_mode':'joint96_real_phoenix'},_load_phoenix_policy=lambda:'new')
        with patch.object(fake, '_load_phoenix_policy') as load:
            Model._load_real_policy(fake)
            load.assert_called_once_with()

    def test_phoenix_deploy_selects_new_route(self):
        config = OmegaConf.load(POLICY_DIR / 'deploy.yml')
        self.assertEqual(config.policy_name, 'Phoenix_WAM')
        self.assertEqual(config.runtime_mode, 'joint96_real_phoenix')
        self.assertEqual(config.audit_dir, 'logs/phoenix')
        config.env_cfg_type = 'arx_x5'
        config.action_type = 'joint'
        self.assertEqual(self.backend.validate_options(config)['replan'], 32)
        fake = SimpleNamespace(model_cfg=config, _load_phoenix_policy=lambda: 'new')
        Model._load_real_policy(fake)
        fake = SimpleNamespace(model_cfg={'runtime_mode': 'joint96_real'})
        with self.assertRaisesRegex(ValueError, 'H5'):
            Model._load_real_policy(fake)

    def test_sampler_bridge_uses_cached_text_masks_shapes_and_reset(self):
        with tempfile.TemporaryDirectory() as d:
            _,_,processor=fixture(Path(d),'piper')
            runtime=SimpleNamespace(model=object(),processor=processor,device='cpu',robot_type='piper',
                model_dtype=torch.bfloat16,embodiment_id=4,posttrain_step=5000,pretrain_step=256013,
                action_dim_is_pad=self.backend._static_dimension_is_pad(processor,'action'),
                text=CachedText(Path(d)/'text.pt',sha256(Path(d)/'text.pt')),
                first_action_recorded=False,audit_dir=Path(d),current_visual_mode=MODE)
            session=self.backend.PhoenixSession(runtime,SimpleNamespace(num_inference_steps=10,replan_steps=32,seed=0,rand_device='cpu'))
            observation={'observation':{k:{'rgb':np.zeros((480,640,3),np.uint8)}
                for k in ('head_camera','left_camera','right_camera')},
                'joint_action':{'vector':np.zeros(14,np.float32)}}
            with patch('XPolicyLab.policy.Phoenix_WAM.joint96_backend.infer_frame_joint_window',
                       return_value={'action':torch.zeros(32,80)}) as call:
                action=session.get_action({'observation':observation,'instruction':'test'})
                self.assertEqual(action.shape,(32,14))
                args,kw=call.call_args
                self.assertIsNone(args[1]);self.assertEqual(args[2].shape,(1,3,3,256,256))
                self.assertEqual(args[2].dtype,torch.float32)
                self.assertEqual(kw['proprio'].shape,(1,80));self.assertEqual(kw['num_frames'],9)
                self.assertEqual(kw['action_horizon'],32);self.assertEqual(kw['num_inference_steps'],10)
                self.assertEqual(kw['embodiment_ids'].tolist(),[4])
                self.assertEqual(kw['context'].shape,(1,128,4096));self.assertTrue(kw['context_mask'].all())
                for key in ('action_dim_is_pad','proprio_dim_is_pad'):
                    self.assertEqual(kw[key].shape,(1,80));self.assertEqual(kw[key].dtype,torch.bool)
                    self.assertFalse(kw[key][0,ACTIVE].any());self.assertEqual(kw[key].sum(),66)
                with self.assertRaises(RuntimeError):session.get_action({'observation':observation,'instruction':'test'})
                session.reset_model();self.assertFalse(session.pending_model_actions)
                session.get_action({'observation':observation,'instruction':'test'})
                self.assertEqual(call.call_count,2)


if __name__ == '__main__':
    unittest.main(verbosity=2)

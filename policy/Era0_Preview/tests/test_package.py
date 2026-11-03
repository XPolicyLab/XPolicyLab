"""CPU checks for portable assets, protocol integrity, and the full driver lifecycle."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

POLICY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(POLICY.parents[2]))
from XPolicyLab.policy.Era0_Preview.checkpoint import resolve_assets
from XPolicyLab.policy.Era0_Preview import evaluate


def assets(root):
    model=root/'model';model.mkdir(parents=True)
    cfg=dict(vlm_backend='qwen3_5',architecture_version='vlm_fast_action_bridge',
        qwen_temporal_embedding='vlm_history_current_v1',
        qwen_image_attention_mode='bidirectional_image_blocks_current_only',
        action_context_mode='prefix_kv',num_actions=25,action_loss_mode='flow_matching',
        vlm_policy_contract='qwen35_224_letterbox_premerge196_semantic_pyramid_v4',
        qwen_history_time_rope=dict(dim=16,max_phase=0.5,completed_steps=3,
            start_step=1,ramp_steps=1,reference_history_slots=7,frame_interval=40))
    (model/'config.json').write_text(json.dumps(cfg))
    (model/'hy_xvla_adapter.json').write_text(json.dumps(dict(domain_id=16,
        action_representation='relative_chunk_ee_RT',action_normalization='zscore',action_loss_mode='flow_matching')))
    for n in ['tokenizer_config.json','processor_config.json']:(model/n).write_text('{}')
    (model/'model.safetensors').touch();(root/'norm_stats.pkl').write_bytes(b'fixture: not loaded')
    return model


class PackageTests(unittest.TestCase):
    def test_bundle_and_original_step_layout(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);model=assets(root/'run'/'release')
            self.assertEqual(resolve_assets(dict(ckpt_name=str(model)))['norm'],model.parent/'norm_stats.pkl')
            cfg=json.loads((model/'config.json').read_text());cfg['qwen_image_attention_mode']='causal'
            (model/'config.json').write_text(json.dumps(cfg))
            with self.assertRaisesRegex(ValueError,'qwen_image_attention_mode'):
                resolve_assets(dict(ckpt_name=str(model.parent)))

    def test_missing_seed_and_invalid_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            r=Path(d)/'eval_result/RoboDojo/stack_bowls/Era0_Preview/arx_x5/0_ckpt_name=era0-preview,action_type=ee/test';r.mkdir(parents=True)
            (r/'_result.json').write_text(json.dumps(dict(eval_time=25,details={'0':dict(success=True,score=1)})))
            self.assertFalse(evaluate.result_status(d,'stack_bowls',0,25,'era0-preview','test')['valid'])
            self.assertFalse(evaluate.result_status(d,'stack_bowls',1,25,'era0-preview','test')['valid'])

    def test_plan_and_full_three_seed_driver_without_gpu(self):
        # Real subprocess/queue/report path, replacing only expensive policy/simulator launch.
        with tempfile.TemporaryDirectory() as d:
            r=Path(d);policy=r/'Era0_Preview';policy.mkdir()
            sdk=r/'SDK';configs=sdk/'task/RoboDojo/config';configs.mkdir(parents=True)
            tasks=['stack_bowls','press_by_number']
            (configs/'_task.yml').write_text('common:\n  eval_nums: 2\ntasks: {}\n')
            for name in tasks:(configs/(name+'.yml')).write_text('{}')
            inventory=sdk/'scripts/internal/task_inventory.py'
            inventory.parent.mkdir(parents=True)
            inventory.write_text('print("stack_bowls\\npress_by_number")\n')
            shutil.copy2(POLICY/'simulator_entry.py',policy/'simulator_entry.py')
            shutil.copy2(POLICY/'obs_lag.py',policy/'obs_lag.py')
            shutil.copy2(POLICY/'sdk_compat.py',policy/'sdk_compat.py')
            shutil.copytree(POLICY/'sdk_sources',policy/'sdk_sources')
            for source in (POLICY/'sdk_sources/base').rglob('*.py'):
                target=sdk/source.relative_to(POLICY/'sdk_sources/base')
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
            (sdk/'scripts/internal/summarize_result.py').write_text(
                "from pathlib import Path\np=Path(__file__).parents[2]/'eval_result/RoboDojo/_summary.md'\np.write_text('# SDK summary fixture')\n")
            (policy/'eval.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
python - "$2" "$6" <<'CODE'
import json,os,sys
from pathlib import Path
assert os.environ['EVAL_NUM']=='native'
assert os.environ['HY_OBS_LAG']=='1'
assert os.environ['ROBODOJO_EVAL_LEGACY_RENDER']=='0'
assert 'ROBODOJO_SAMPLE_RATIO' not in os.environ
assert 'ROBODOJO_EVAL_OUTPUT_DIR' not in os.environ
assert Path(os.environ['ERA0_CHECKPOINT_PATH']).is_dir()
p=Path(os.environ['ROBODOJO_ROOT'])/'eval_result/RoboDojo'/sys.argv[1]/'Era0_Preview'/'arx_x5'/(sys.argv[2]+'_ckpt_name=era0-preview,action_type=ee')/os.environ['ROBODOJO_RUN_ID']
p.mkdir(parents=True)
(p/'_result.json').write_text(json.dumps(dict(eval_time=2,details={str(i):dict(success=True,score=1) for i in range(2)})))
CODE
''')
            model=assets(r/'assets');out=r/'out'
            argv=['evaluate','--checkpoint',str(model),'--output',str(out),'--robodojo-root',str(sdk),
                  '--gpu-ids','0,1']
            import threading
            class FastEvent(threading.Event):
                def wait(self,timeout=None):return super().wait(min(timeout,0.02) if timeout is not None else timeout)
            # Patching Event globally would change ThreadPoolExecutor internals; substitute module binding only.
            from types import SimpleNamespace
            th=SimpleNamespace(Event=FastEvent,Lock=threading.Lock)
            with patch.object(evaluate,'POLICY',policy),patch.object(evaluate,'threading',th),patch.object(sys,'argv',argv), patch.dict(os.environ, {}, clear=True):
                evaluate.main();self.assertFalse(out.exists())
                with patch.object(sys,'argv',argv+['--run']),patch.dict(os.environ,{'ROBODOJO_SAMPLE_RATIO':'0.1','ROBODOJO_EVAL_OUTPUT_DIR':'/ignored/custom/videos'}):
                    evaluate.main()
            report=json.loads((out/'evaluation_summary.json').read_text())
            self.assertTrue(report['complete']);self.assertEqual(report['valid_episodes'],12)
            contract=json.loads((out/'evaluation_contract.json').read_text())
            self.assertEqual(contract['policy_image_lag_updates'],1)
            self.assertTrue(contract['full_suite'])
            self.assertEqual({x['seed'] for x in report['cells']},{0,1,2})
            self.assertIn('SDK summary fixture',(sdk/'eval_result/RoboDojo/_summary.md').read_text())
            self.assertFalse((out/'aggregated_summary.md').exists())
            self.assertFalse((out/'RoboDojo').exists())

    def test_packaged_observation_and_pose_contract(self):
        import numpy as np
        from XPolicyLab.policy.Era0_Preview.runtime.robodojo_eval.io import encode_obs,action_chunk_to_robodojo_dicts
        img=np.zeros((9,13,3),dtype=np.uint8);img[:]=[200,20,5]
        pose=np.array([.1,.2,.3,1,0,0,0],dtype=np.float32)
        obs=dict(vision={k:dict(color=img.copy()) for k in ['cam_head','cam_left_wrist','cam_right_wrist']},
                 state=dict(left_ee_pose=pose,right_ee_pose=pose,left_ee_joint_state=[.2],right_ee_joint_state=[.4]),instruction='cover blocks')
        encoded=encode_obs(obs)
        self.assertEqual(encoded['observation.state'].shape,(1,32))
        np.testing.assert_array_equal(encoded['raw_images.top_head'],img)
        action=np.concatenate([pose,[.2],pose,[.4]])[None]
        decoded=action_chunk_to_robodojo_dicts(action)[0]
        np.testing.assert_allclose(decoded['left_ee_pose'],pose)
        self.assertEqual(decoded['right_ee_joint_state'].shape,(1,))


if __name__=='__main__':unittest.main()

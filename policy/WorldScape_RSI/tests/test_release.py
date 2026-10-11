import unittest,tempfile,json,hashlib,sys,types,os
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from ..jev.network import BinaryJEV,EpisodeRouter,features,INPUT_DIM,FEATURE_VERSION,CLASSES
ROOT=Path(__file__).resolve().parents[1]
def obs():return {"instruction":"pick up the cup","vision":{"cam_head":{"color":np.zeros((16,16,3),dtype=np.uint8)}},"state":{a+"_ee_pose":[0,0,1,1,0,0,0] for a in ("left","right")}}
class RouterTests(unittest.TestCase):
    def test_features_whitelist(self):
        a=obs();b=dict(a,task_name="secret",seed=999,layout=99,score=100,success=True)
        self.assertTrue(np.array_equal(features(a),features(b)));self.assertEqual(features(a).shape,(INPUT_DIM,))
    def test_missing_and_nan(self):
        with self.assertRaises(ValueError):features({"instruction":"hello"})
        a=obs();a["state"]={"left_arm_joint_state":[float("nan")]*6}
        with self.assertRaises(ValueError):features(a)
    def test_untrained_rejected(self):
        with self.assertRaises(ValueError):EpisodeRouter()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"test.pt";torch.save({"trained":False},p)
            with self.assertRaises(ValueError):EpisodeRouter(checkpoint=p)
    def test_fixed_episode(self):
        r=EpisodeRouter("wam");a=r.select(obs());r.mode="astra"
        self.assertEqual(r.select({})["executor"],"wam");r.reset();self.assertEqual(r.select({})["executor"],"astra")
    def test_checkpoint_contract(self):
        # Synthetic weights only exercise loader; never distributed as trained assets.
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"test.pt";n=BinaryJEV()
            for x in n.parameters():x.data.zero_()
            n.layers[-1].bias.data[1]=1
            torch.save(dict(trained=True,training_manifest_sha256="synthetic-test-only",feature_version=FEATURE_VERSION,classes=CLASSES,state_dict=n.state_dict()),p)
            r=EpisodeRouter(checkpoint=p);self.assertEqual(r.select(obs())["executor"],"astra")
            self.assertAlmostEqual(sum(r.select(obs())["probabilities"]),1,places=6)
    def test_full_icl(self):
        from ..icl.loader import load_messages
        tasks=json.loads((ROOT/"configs/tasks.json").read_text())
        self.assertEqual(len(tasks),42)
        for row in tasks:
            task=row["task"];m=json.loads((ROOT/"icl"/task/"manifest.json").read_text())
            msgs=load_messages(task)
            images=sum(1 for message in msgs for item in message.get("content",[]) if isinstance(item,dict) and item.get("type")=="image_url")
            self.assertEqual(images,m["images"],task)
    def test_no_foreign_primitives(self):
        self.assertFalse((ROOT/"harness/peer_primitives.py").exists())
        self.assertFalse((ROOT/"harness/merged_guidance.py").exists())
    def test_runtime_guard(self):
        from ..runtime import configure
        with patch.dict(os.environ,{"RDA_ATTACK_MODE":"1"}):
            with self.assertRaises(RuntimeError):configure()
    def test_model_lifecycle_without_services(self):
        # Mock only API/dependencies; test actual public facade route/reset/report contract.
        pkg=types.ModuleType("XPolicyLab");mod=types.ModuleType("XPolicyLab.model_template");mod.ModelTemplate=object
        with patch.dict(sys.modules,{"XPolicyLab":pkg,"XPolicyLab.model_template":mod}):
            from ..model import Model
        with patch.dict(os.environ,{"WRSI_ROUTE_MODE":"wam"},clear=True):
            m=Model({"task_name":"general_pickup"});m.reset()
            m.report_execution({"phase":"start","score":100,"attack_feedback":{}})
            self.assertNotIn("score",m._reports[0]);self.assertIsNone(m._astra)
            self.assertEqual(len(m.release_metadata()["release_sha256"]),64)
            from ..deploy import eval_one_episode
            bad_client=types.SimpleNamespace(call=lambda **kw:{"release_sha256":"wrong","task":"general_pickup"})
            env=types.SimpleNamespace(task_name="general_pickup",deploy_cfg={"task_name":"general_pickup"})
            with self.assertRaises(RuntimeError):eval_one_episode(env,bad_client)
            self.assertEqual(m.select_executor(obs())["executor"],"wam")
            with self.assertRaises(RuntimeError):m.get_action()
            m.reset();self.assertIsNone(m.router.decision)
    def test_all_agent_imports_and_icl_hook(self):
        from ..harness import open_restore
        # Catch hidden absolute imports/dependencies without API calls or simulation.
        for task in json.loads((ROOT/"configs/tasks.json").read_text()):
            module=open_restore.module(task["task"],"agent")
            self.assertTrue(callable(module.VLMAgent),task["task"])

    def test_astra_initialization_and_reset(self):
        pkg=types.ModuleType("XPolicyLab");mod=types.ModuleType("XPolicyLab.model_template");mod.ModelTemplate=object
        with patch.dict(sys.modules,{"XPolicyLab":pkg,"XPolicyLab.model_template":mod}):
            from ..model import Model
            from ..harness import model as original
        class FakeClient:
            model="offline-test"
            def __init__(self,*a,**kw):pass
        with patch.dict(sys.modules,{"XPolicyLab":pkg,"XPolicyLab.model_template":mod, original.__name__:original}),tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{"WRSI_ROUTE_MODE":"astra"},clear=True),patch.object(original,"load_vlm_profile",return_value={"name":"offline-test"}),patch.object(original,"VLMClient",FakeClient):
            for task in ("general_pickup","stack_bowls","classify_objects_by_language"):
                m=Model({"task_name":task,"vlm_agent":{"log_dir":tmp}});m.reset()
                self.assertEqual(m.select_executor(obs())["executor"],"astra")
                m._astra.agent._ensure_icl();self.assertTrue(m._astra.agent.icl_msgs)
                m.update_obs(obs());m.reset();self.assertIsNone(m.router.decision)
                self.assertIsNone(m._astra.agent.obs)

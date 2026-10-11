import unittest
from types import SimpleNamespace
from unittest.mock import patch
from ..wam_loop import execute
from ..wam_backend import build_config


class Env:
    def __init__(self, limit=5, fail=False):
        self.step = 0
        self.limit = limit
        self.fail = fail
    def is_episode_end(self):return self.step >= self.limit
    def get_obs(self):return {"step": self.step}
    def take_action(self, action):
        if self.fail:raise RuntimeError("execution error")
        self.step += 1


class Client:
    def __init__(self, empty=False):self.calls=[];self.empty=empty
    def call(self, func_name, obs=None):
        self.calls.append((func_name, obs))
        if func_name == "wam_get_action":return [] if self.empty else [{}, {}, {}]


class WamIntegrationTests(unittest.TestCase):
    def test_full_then_terminal_partial(self):
        env=Env();client=Client()
        result=execute(env,client,{},SimpleNamespace(),None)
        commits=[v for k,v in client.calls if k=="wam_commit_action"]
        self.assertEqual(commits,[{"executed_steps":{0:3},"finished_env_idx_list":[]},
                                  {"executed_steps":{0:2},"finished_env_idx_list":[0]}])
        self.assertEqual([v["step"] for k,v in client.calls if k=="wam_update_obs"],[0,1,2,3,4])
        self.assertEqual(result["wam_steps_executed"],5)
        self.assertEqual(result["decisions"],2)
    def test_uncertain_execution_discards(self):
        client=Client()
        with self.assertRaisesRegex(RuntimeError,"execution error"):
            execute(Env(fail=True),client,{},SimpleNamespace(),None)
        self.assertEqual(client.calls[-1],("wam_discard_action",{"execution_started":True}))
        self.assertFalse(any(k=="wam_commit_action" for k,v in client.calls))
    def test_empty_prediction_discards_without_execution(self):
        client=Client(empty=True)
        with self.assertRaises(ValueError):execute(Env(),client,{},SimpleNamespace(),None)
        self.assertEqual(client.calls[-1],("wam_discard_action",{"execution_started":False}))
    def test_budget_ends_before_prediction(self):
        client=Client()
        result=execute(Env(),client,{"episode_budget_s":0},SimpleNamespace(),None)
        self.assertEqual(result["termination_reason"],"wall_budget")
        self.assertEqual(client.calls,[])
    def test_robodojo_configuration(self):
        with patch.dict("os.environ",{},clear=True):
            cfg=build_config({"task_name":"general_pickup","env_cfg_type":"arx_x5","seed":2,
                              "wam":{"checkpoint_path":"/test/checkpoint"}})
        self.assertEqual(cfg["runtime_profile"],"robodojo")
        self.assertEqual(cfg["checkpoint_path"],"/test/checkpoint")
        self.assertEqual(cfg["diffusion_concat_view_order"],[0,2,1])
        self.assertEqual(cfg["seed"],2)
        self.assertFalse(cfg["eval_batch"])
        self.assertFalse(cfg["multi_gpu"]["enabled"])
    def test_reject_unreviewed_override(self):
        with self.assertRaises(ValueError):build_config({"wam":{"action_horizon":1}})
    def test_facade_lazy_lifecycle(self):
        import sys,types,os
        stub=types.ModuleType("XPolicyLab.model_template");stub.ModelTemplate=object
        with patch.dict(sys.modules,{"XPolicyLab.model_template":stub}):
            from ..model import Model
        from unittest.mock import Mock
        backend=Mock()
        with patch.dict(os.environ,{"WRSI_ROUTE_MODE":"wam"},clear=True),patch(
                __package__.rsplit(".",1)[0]+".wam_backend.create",return_value=backend) as factory:
            model=Model({"task_name":"general_pickup"})
            with self.assertRaises(RuntimeError):model.wam_get_action()
            factory.assert_not_called()
            model.select_executor({})
            model.wam_update_obs({"instruction":"pick"})
            model.wam_get_action()
            model.wam_commit_action({"executed_steps":{0:3}})
            factory.assert_called_once()
            backend.commit_action.assert_called_once_with({"executed_steps":{0:3}})
            self.assertIsNone(model._astra)
            model.reset();backend.reset.assert_called_once()
            with self.assertRaises(RuntimeError):model.wam_get_action()

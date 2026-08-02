from __future__ import annotations

import importlib
import os
import pathlib
import unittest
from unittest import mock

from XPolicyLab.model_template import ModelTemplate


class _FakeProxy:
    def __init__(self, route_result, model_cfg):
        self.route_result = route_result
        self.model_cfg = model_cfg
        self.closed = False

    def reset(self):
        return "reset"

    def get_action(self, obs):
        return {"echo": obs}

    def get_action_batch(self, obs_list):
        return [{"echo": obs} for obs in obs_list]

    def prepare_case(self, case_meta):
        return case_meta

    def trial_end(self, result):
        return result

    def close(self):
        self.closed = True


class ModelContractTest(unittest.TestCase):
    def test_standard_model_contract_and_router(self):
        old = os.environ.get("CSU_SKIP_RUNTIME_CHECKS")
        os.environ["CSU_SKIP_RUNTIME_CHECKS"] = "1"
        try:
            module = importlib.import_module("XPolicyLab.policy.CSU-AI-0.model")
            original = module.ExpertProcessProxy
            module.ExpertProcessProxy = _FakeProxy
            try:
                model = module.Model(
                    {"task_name": "stack_bowls", "env_cfg_type": "arx_x5"}
                )
                self.assertIsInstance(model, ModelTemplate)
                self.assertEqual(model.route_result["selected_expert"], "xiaomi")
                model.update_obs({"value": 1})
                self.assertEqual(model.get_action(), {"echo": {"value": 1}})
                model.update_obs_batch([{"value": 2}, {"value": 3}])
                self.assertEqual(len(model.get_action_batch()), 2)
                self.assertEqual(model.reset(), "reset")
                model.close()
                self.assertTrue(model.proxy.closed)
            finally:
                module.ExpertProcessProxy = original
        finally:
            if old is None:
                os.environ.pop("CSU_SKIP_RUNTIME_CHECKS", None)
            else:
                os.environ["CSU_SKIP_RUNTIME_CHECKS"] = old

    def test_isolated_expert_runtime_paths(self):
        proxy_module = importlib.import_module(
            "XPolicyLab.policy.CSU-AI-0.expert_proxy"
        )

        spatial = object.__new__(proxy_module.ExpertProcessProxy)
        spatial.route_result = {
            "selected_expert": "spatial_forcing",
            "checkpoint_path": "/models/spatial",
            "policy_env": "/envs/spatial",
        }
        spatial.runtime_dir = pathlib.Path("/tmp/csu-spatial-runtime")
        with mock.patch.dict(os.environ, {}, clear=True):
            spatial_env = spatial._expert_environment()
        self.assertEqual(
            spatial_env["OPENPI_DATA_HOME"],
            "/tmp/csu-spatial-runtime/openpi-cache",
        )

        hunyuan = object.__new__(proxy_module.ExpertProcessProxy)
        hunyuan.route_result = {
            "selected_expert": "hunyuan",
            "checkpoint_path": "/models/hunyuan",
            "policy_env": "/envs/hunyuan",
        }
        hunyuan.runtime_dir = pathlib.Path("/tmp/csu-hunyuan-runtime")
        with mock.patch.dict(os.environ, {}, clear=True):
            hunyuan_env = hunyuan._expert_environment()
        self.assertEqual(hunyuan_env["HY_VLA_CKPT_PATH"], "/models/hunyuan")
        self.assertEqual(hunyuan_env["HY_VLA_ROOT"], "/envs/hunyuan")


if __name__ == "__main__":
    unittest.main()

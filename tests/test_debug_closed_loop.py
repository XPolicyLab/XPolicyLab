"""Real WebSocket demo-policy loops, using explicit synthetic benchmark metadata.

Run from the checkout with:
    python -m unittest discover -s tests -p 'test_debug_closed_loop.py' -v

No weights, simulator, Conda, or production env_cfg are needed. Only metadata
location and test image sizes are substituted; the server, codec, synchronous
client, demo Model, deployment loops, and action validator are real. This does
not exercise the Conda-based eval.sh launchers or establish model task success.
"""
from __future__ import annotations

import asyncio
from contextlib import redirect_stdout
import io
import itertools
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from uuid import uuid4

import numpy as np

# Allow both unittest discovery and direct execution in an uninstalled checkout.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from client_server.ws.model_server import PolicyServer, PolicyServerConfig
from XPolicyLab.policy.demo_policy.model import Model
from XPolicyLab.utils import debug_env_client, process_data


class DebugClosedLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_single_dual_joint_ee_plain_encoded_single_batch(self):
        with tempfile.TemporaryDirectory(prefix="xpl-debug-") as tmp:
            workspace = Path(tmp)
            metadata = workspace / "env_cfg"
            (metadata / "robot").mkdir(parents=True)
            (metadata / "sim").mkdir()
            robots = {
                "single": {"arm_dim": [6], "ee_dim": [1]},
                "dual": {"arm_dim": [6, 6], "ee_dim": [1, 1]},
            }
            (metadata / "robot" / "_robot_info.json").write_text(json.dumps(robots))
            (metadata / "sim" / "debug.yml").write_text("scene:\n  num_envs: 2\n")
            for robot in robots:
                (metadata / f"{robot}.yml").write_text(
                    f"config:\n  robot: {robot}\n  sim: debug\n"
                )
            # Exercise the actual metadata readers against an isolated parent
            # workspace; never create or alter the user's sibling env_cfg tree.
            fixture_module = workspace / "XPolicyLab" / "utils" / "process_data.py"
            fixture_module.parent.mkdir(parents=True)
            with patch.object(process_data, "__file__", str(fixture_module)), \
                 patch.object(debug_env_client, "Batch_Size", 2):
                for robot, action_type, encoded, batch in itertools.product(
                    robots, ("joint", "ee"), (False, True), (False, True)
                ):
                    with self.subTest(robot=robot, action_type=action_type,
                                      encoded=encoded, batch=batch):
                        await self._run_loop(robot, action_type, encoded, batch)

    async def _run_loop(self, robot, action_type, encoded, batch):
        cfg = {
            "policy_name": "demo_policy", "env_cfg_type": robot,
            "action_type": action_type, "protocol": "ws",
            "evaluation_id": f"test-{uuid4().hex}", "trial_id": "debug-trial",
            "action_case_id": None, "repeat_index": None,
            "obs_encoded": encoded, "max_connect_attempts": 1,
            "connect_timeout_s": 3, "handshake_timeout_s": 3,
            "request_timeout_s": 5, "close_timeout_s": 2,
        }
        output = io.StringIO()
        with redirect_stdout(output):
            model = Model(cfg)
            observations_seen = []
            original_update = model.update_obs
            original_batch_update = model.update_obs_batch

            def check_observation(obs):
                packed = process_data.pack_robot_state(
                    obs, action_type, model.robot_action_dim_info
                )
                self.assertEqual(packed.shape, (model.action_dim,))
                for camera in obs["vision"].values():
                    color = camera["color"]
                    self.assertIsInstance(color, np.ndarray)
                    self.assertEqual(color.shape, (12, 16, 3))
                    self.assertEqual(color.dtype, np.uint8)
                    # A red/blue asymmetric stimulus detects accidental swaps;
                    # JPEG loss is allowed, but the channel order is not.
                    np.testing.assert_allclose(
                        color[0, 0], [220, 30, 10], atol=3
                    )
                observations_seen.append(obs["env_idx"])

            def update(obs):
                check_observation(obs)
                return original_update(obs)

            def update_batch(observations):
                for obs in observations:
                    check_observation(obs)
                return original_batch_update(observations)

            # Observe what the real server hands the real model; retain the
            # model's own implementation, including generated action chunks.
            model.update_obs = update
            model.update_obs_batch = update_batch
            server = PolicyServer(model, PolicyServerConfig(host="127.0.0.1", port=0))
            await server.start()
            cfg["policy_server_url"] = server.url

            def run_environment():
                env = debug_env_client.TestEnv(cfg)
                original_get_obs = env.get_obs

                def get_obs(env_idx=0):
                    # Keep the real observation schema and encoder, but use
                    # small distinct RGB fixtures to keep the matrix inexpensive.
                    env.obs_encoded = False
                    try:
                        obs = original_get_obs(env_idx)
                    finally:
                        env.obs_encoded = encoded
                    for camera in obs["vision"].values():
                        camera["color"] = np.full((12, 16, 3), [220, 30, 10], np.uint8)
                        camera["depth"] = np.zeros((12, 16, 3), np.uint8)
                        camera["shape"] = (12, 16)
                    if encoded:
                        env._encode_obs_colors(obs)
                    return obs

                env.get_obs = get_obs
                try:
                    env.reset()
                    if batch:
                        env.eval_one_episode_batch()
                    else:
                        env.eval_one_episode()
                    self.assertEqual(env.episode_step, env.episode_step_limit)
                    self.assertEqual(env.episode_step, 20)
                    env.finish_episode()
                finally:
                    env.model_client.close()

            try:
                await asyncio.wait_for(asyncio.to_thread(run_environment), timeout=30)
            finally:
                await server.stop()
            self.assertEqual(len(observations_seen), 40 if batch else 20)
            self.assertEqual(set(observations_seen), {0, 1} if batch else {0})
            self.assertIn("[TestEnv] Episode finished", output.getvalue())


if __name__ == "__main__":
    unittest.main()

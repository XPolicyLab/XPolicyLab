"""Long rule planning through real SDK transport without extra control steps."""
import asyncio
from copy import deepcopy
import importlib
from pathlib import Path
import threading
import time
import unittest

import numpy as np

from skills.shared.orchestration.public_rule_policy_interface.implementation import BlockingRulePolicy
from tests.contracts.test_official_rule_policy_interface import observation, action

from rsi.deployment import locations
SDK = Path(locations()['official_root'])


class EmptyPlanningChunkTests(unittest.TestCase):
    def test_empty_poll_is_not_an_action_or_ack_and_first_real_target_stays_exact(self):
        ready = threading.Event()
        acknowledgements = []
        def run(initial, exchange):
            ready.wait(2)
            acknowledgements.append(exchange(action(.25)))
        policy = BlockingRulePolicy(run, planning_poll_seconds=.01)
        try:
            policy.update_obs(observation(.2))
            self.assertEqual(policy.get_action(), [])
            policy.update_obs(observation(.3))
            self.assertEqual(policy.get_action(), [])
            self.assertFalse(acknowledgements)
            self.assertEqual(policy.diagnostics()['actions_returned'], 0)
            ready.set()
            for _ in range(100):
                policy.update_obs(observation(.3)); chunk = policy.get_action()
                if chunk: break
            self.assertEqual(chunk, [action(.25)])
            self.assertFalse(acknowledgements)
            policy.update_obs(observation(.9))
            policy.get_action()
            self.assertEqual(len(acknowledgements), 1)
            self.assertAlmostEqual(float(acknowledgements[0]['state']['left_arm_joint_state'][0]), .9, places=6)
        finally:
            ready.set(); policy.close()

    def test_real_official_deploy_client_and_server_skip_empty_chunks_without_physics(self):
        from client_server.ws.model_client import WsModelClient
        from client_server.ws.model_server import PolicyServer, PolicyServerConfig
        deploy = importlib.import_module('XPolicyLab.policy.demo_policy.deploy')
        self.assertEqual(Path(deploy.__file__).resolve(), SDK/'XPolicyLab/policy/demo_policy/deploy.py')
        def run(initial, exchange):
            # This deliberately exceeds the client's request timeout. Empty
            # polls must keep the actual official transport usable instead.
            time.sleep(.15)
            exchange(action(.1))
            time.sleep(.15)
            exchange(action(.2))
        policy = BlockingRulePolicy(run, planning_poll_seconds=.005)
        loop = asyncio.new_event_loop()
        worker = threading.Thread(target=loop.run_forever, daemon=True); worker.start()
        server = PolicyServer(policy, PolicyServerConfig(host='127.0.0.1', port=0))
        client = None
        try:
            asyncio.run_coroutine_threadsafe(server.start(), loop).result(5)
            client = WsModelClient(url=server.url, evaluation_id='r40-cpu', trial_id='long-plan',
                request_timeout_s=.1, close_timeout_s=2, max_connect_attempts=1)
            class Environment:
                def __init__(self): self.actions=[]; self.captures=0
                def is_episode_end(self): return len(self.actions)==3
                def get_obs(self): self.captures+=1; return observation(.4)
                def take_action(self, target): self.actions.append(deepcopy(target))
            env = Environment()
            deploy.eval_one_episode(TASK_ENV=env, model_client=client)
            self.assertEqual(env.actions, [action(.1), action(.2), action(.2)])
            self.assertGreater(env.captures, 3)
            self.assertGreater(policy.diagnostics()['empty_planning_chunks'], 0)
            self.assertEqual(policy.diagnostics()['actions_returned'], 3)
        finally:
            policy.close()
            if client is not None: client.close()
            asyncio.run_coroutine_threadsafe(server.stop(),loop).result(5)
            loop.call_soon_threadsafe(loop.stop);worker.join(5);loop.close()

    def test_reset_cancels_a_worker_waiting_for_ack_without_replaying_its_action(self):
        count = []
        def run(initial, exchange):
            count.append(float(initial['state']['left_arm_joint_state'][0]))
            exchange(action(count[-1]))
        policy = BlockingRulePolicy(run, planning_poll_seconds=.01)
        try:
            policy.update_obs(observation(.2)); self.assertEqual(policy.get_action(),[action(float(np.float32(.2)))])
            policy.reset()
            policy.update_obs(observation(.7)); self.assertEqual(policy.get_action(),[action(float(np.float32(.7)))])
            self.assertEqual(len(count),2)
            self.assertIsNone(policy.diagnostics()['workflow_error'])
        finally:
            policy.close()


if __name__=='__main__':unittest.main()

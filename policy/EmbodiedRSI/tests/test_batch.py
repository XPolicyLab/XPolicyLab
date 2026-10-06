"""Batch isolation, action barrier and shrinking native environment indices."""
import unittest
from unittest.mock import Mock

from XPolicyLab.policy.EmbodiedRSI.model import Model
from XPolicyLab.policy.EmbodiedRSI.deploy import eval_one_episode_batch


class BatchTests(unittest.TestCase):
    def test_ready_action_is_not_requeried_while_another_agent_thinks(self):
        model = Model.__new__(Model)
        first = Mock()
        second = Mock()
        first.get_action.return_value = [{"target": 2}]
        second.get_action.side_effect = [[], [], [{"target": 5}]]
        model.contexts = {2: first, 5: second}
        model.ready = {}
        self.assertEqual(model.get_action_batch([2, 5]), [[], []])
        self.assertEqual(model.get_action_batch([2, 5]), [[], []])
        self.assertEqual(first.get_action.call_count, 1)
        self.assertEqual(model.get_action_batch([2, 5]), [[{"target": 2}], [{"target": 5}]])
        self.assertEqual(first.get_action.call_count, 1)
        self.assertEqual(model.ready, {})
        model.update_obs_batch([{"env_idx": 5, "value": "five"}, {"env_idx": 2, "value": "two"}])
        first.update_obs.assert_called_once_with({"env_idx": 2, "value": "two"})
        second.update_obs.assert_called_once_with({"env_idx": 5, "value": "five"})
        model.on_trial_end({"results": [{"env_idx": 2, "truncated": True}]})
        first.on_trial_end.assert_called_once()
        second.on_trial_end.assert_not_called()

    def test_native_active_indices_shrink_without_shifting_agent_identity(self):
        class Environment:
            step_lim = 3
            env_seeds = [0, 0, 17, 0, 0, 19]
            end_flag = [True, True, False, True, True, False]
            success = [False] * 6
            take_action_cnt = [0] * 6

            def get_running_env_idx_list(self):
                return [i for i in (2, 5) if not self.end_flag[i]]

            def is_episode_end(self):
                return not self.get_running_env_idx_list()

            def get_obs_batch(self, indices, last_frame=False):
                return [{"env_idx": i, "last": last_frame} for i in indices]

            def take_action_batch(self, actions, indices):
                assert [a["target"] for a in actions] == indices
                for i in indices:
                    self.take_action_cnt[i] += 1
                    if self.take_action_cnt[i] >= (1 if i == 2 else 3):
                        self.end_flag[i] = True

        class Client:
            def __init__(self):
                self.calls = []

            def call(self, **kwargs):
                self.calls.append(kwargs)
                if kwargs["func_name"] == "get_action_batch":
                    return [[{"target": i}] for i in kwargs["obs"]]

        client = Client()
        eval_one_episode_batch(Environment(), client)
        actions = [c["obs"] for c in client.calls if c["func_name"] == "get_action_batch"]
        self.assertEqual(actions, [[2, 5], [5], [5]])
        ended = [c["obs"]["results"] for c in client.calls if c["func_name"] == "trial_end"]
        self.assertEqual([[r["env_idx"] for r in rows] for rows in ended], [[2], [5]])
        prepared = next(c["obs"] for c in client.calls if c["func_name"] == "prepare_case")
        self.assertEqual(prepared["environments"], [{"env_idx": 2, "layout_id": 17}, {"env_idx": 5, "layout_id": 19}])

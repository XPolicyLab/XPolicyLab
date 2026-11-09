from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from benches.robodojo.codec import EpisodeCursor
from benches.robodojo.runtime import RoboDojoRuntimeEnvironment

REASON = "RoboDojo marked task=make_kong layout=1 unstable after step 58; the episode is not scorable"


def worker(unscorable=None):
    """The runtime worker's episode bookkeeping only; nothing here starts the simulator."""
    env = RoboDojoRuntimeEnvironment.__new__(RoboDojoRuntimeEnvironment)
    env._cursor = EpisodeCursor()
    env._unscorable = unscorable
    return env


class UnscorableEpisodeTests(unittest.TestCase):
    def test_calls_for_a_dropped_episode_repeat_why(self):
        env = worker(("episode-1", REASON))
        for call in (lambda: env._check_episode("episode-1"),
                     lambda: env._call_tick("episode-1", 58, command=[0.0] * 14)):
            with self.assertRaises(RuntimeError) as raised:
                call()
            self.assertEqual(str(raised.exception), REASON)

    def test_other_episodes_are_still_stale(self):
        env = worker(("episode-1", REASON))
        with self.assertRaisesRegex(ValueError, "Stale or missing episode"):
            env._check_episode("episode-0")


class UnscorableOutcomeTests(unittest.TestCase):
    def test_the_outcome_says_why_the_episode_does_not_count(self):
        import types

        env = worker(("episode-1", REASON))
        env._outcome_token, env._record_video, env._ground_truth_log = "token", False, None
        env._env = types.SimpleNamespace(reward_manager=types.SimpleNamespace(_gated_score_lst=[40.0]),
                                         take_action_cnt=[58], end_flag=[False], success=[False], step_lim=600)
        outcome = env._call_outcome("token")
        self.assertEqual(outcome["unscorable"], REASON)
        self.assertEqual((outcome["steps"], outcome["process_score"]), (58, 40.0))
        env._unscorable = None
        self.assertNotIn("unscorable", env._call_outcome("token"))


if __name__ == "__main__":
    unittest.main()

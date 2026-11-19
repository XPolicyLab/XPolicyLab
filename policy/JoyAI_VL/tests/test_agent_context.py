"""Exercise agent context assembly and the shared registry with synthetic observations."""
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import configure
from roboshell import docs, task_card
from roboshell.server import tools


def observation(instruction='Follow the evaluator instruction.\nReturn to the supplied origin.'):
    return SimpleNamespace(
        instruction=instruction, step=0, step_limit=1100,
        arms={name: SimpleNamespace(flange=np.eye(4), gripper=1.0) for name in ('left', 'right')},
        images={name: np.zeros((8, 8, 3), dtype=np.uint8)
                for name in ('head', 'left_wrist', 'right_wrist')})


class AgentContext(unittest.TestCase):
    def test_task_card_keeps_runtime_instruction_and_ignores_legacy_switches(self):
        first = observation()
        expected = task_card.task_card(first)
        with patch.dict(os.environ, HEXASHELL_SKILLS='1', ROBOSHELL_RECIPE='1',
                        ROBOSHELL_NOTES='1', ROBOSHELL_PROFILE='default'):
            self.assertEqual(task_card.task_card(first), expected)
        self.assertIn(first.instruction, expected)
        self.assertIn('1100', expected)
        for old_section in ('TASK RECIPE', 'TASK NOTES', 'Notes for this task',
                            'How this task is scored', '0.85', '18 degrees'):
            self.assertNotIn(old_section, expected)
        changed = observation('An unseen evaluator instruction.')
        self.assertEqual(task_card.task_card(changed),
                         expected.replace(first.instruction, changed.instruction))

    def test_shared_tools_for_known_random_and_unseen_tasks(self):
        with patch.dict(os.environ, ROBOSHELL_PROFILE='default', ROBOSHELL_TOOLS='task_specific_tool'):
            baseline = tools.load_tools(None)
            expected = {k: (v['path'], v['spec']) for k, v in baseline.items()}
            self.assertTrue(expected)
            for task in ('stack_bowls', 'stack_bowls_random', 'solve_equation', 'unseen_task'):
                self.assertTrue(tools.select_profile(task))
                loaded = tools.load_tools(task)
                self.assertEqual({k: (v['path'], v['spec']) for k, v in loaded.items()}, expected)
                self.assertTrue(all('/tasks/' not in v['path'] for v in loaded.values()))

    def test_manual_is_shared_and_builds_without_task_files(self):
        one = Path(docs.build('stack_bowls'))
        expected = one.read_bytes()
        two = Path(docs.build('unseen_task'))
        self.assertEqual(one, two)
        self.assertEqual(two.read_bytes(), expected)
        self.assertIn(b'robot.goto', expected)

    def test_fixed_context_configuration(self):
        config = json.loads((ROOT / 'configs/model_api.example.json').read_text())
        self.assertFalse(config['skills'])
        config['skills'] = True
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'skills=false'):
                configure.load_settings(path)

    def test_bridge_assembles_runtime_context(self):
        from roboshell.server.bridge import Model as Bridge

        first = observation()
        fake_tools = SimpleNamespace(host=SimpleNamespace(snap=first))
        with tempfile.TemporaryDirectory() as temporary:
            bridge = Bridge.__new__(Bridge)
            bridge.cfg = {'task_name': 'unseen_task', 'seed': 19321}
            bridge.episode_index = 0
            bridge.run_root = temporary
            bridge.executor = SimpleNamespace(obs={})
            bridge.budget = 0
            bridge.registry = {}
            launched = []
            bridge.launch_agent = lambda: launched.append(bridge.episode.prompt)
            episode = SimpleNamespace(refresh_obs=lambda: None)
            fake_module = SimpleNamespace(RobotTools=lambda episode: fake_tools)
            with patch('roboshell.server.bridge.Episode', return_value=episode), \
                    patch.dict(sys.modules, {'roboshell.server.robot15': fake_module}):
                bridge.start_episode()
            self.assertEqual(launched, [task_card.task_card(first)])
            contexts = list(Path(temporary).rglob('context.json'))
            self.assertEqual(len(contexts), 1)
            self.assertEqual(json.loads(contexts[0].read_text()), task_card.info())
            self.assertFalse(list(Path(temporary).rglob('recipe.json')))


if __name__ == '__main__':
    unittest.main()

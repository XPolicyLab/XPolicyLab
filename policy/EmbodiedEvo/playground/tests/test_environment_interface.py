import unittest

from gymnasium import spaces

from environments import Environment, execute


class _Native:
    observation_space = spaces.Dict({"position": spaces.Box(-1.0, 1.0, (1,))})

    def reset(self, *, seed=None, options=None):
        return {"position": [0.0]}, {"seed": seed, "options": options}

    def step(self, code):
        return (
            {"position": [1.0]},
            0.5,
            False,
            False,
            {
                "ok": True,
                "stdout": "done",
                "stderr": "",
                "task_completed": False,
                "frames": {"front": ["frame"]},
            },
        )

    def render(self):
        return "image"


class _Environment(Environment):
    def __init__(self):
        super().__init__()
        self.env = _Native()
        self.configure(self.env)

    def reset(self, seed=None, *, options=None):
        self.begin_episode(seed)
        return self.env.reset(seed=seed, options=options)

    def _step(self, code):
        obs, reward, terminated, truncated, info = self.env.step(code)
        info["step_index"] = self.next_step_index()
        return obs, reward, terminated, truncated, info


class EnvironmentInterfaceTests(unittest.TestCase):
    def test_code_is_the_only_action_and_reset_step_are_gymnasium_shapes(self):
        environment = _Environment()
        self.assertIsInstance(environment, Environment)
        self.assertIsInstance(environment.action_space, spaces.Text)
        self.assertTrue(environment.action_space.contains("move()"))
        self.assertFalse(environment.action_space.contains(""))

        observation, info = environment.reset(seed=7)
        self.assertEqual(observation, {"position": [0.0]})
        self.assertEqual(info["seed"], 7)
        transition = environment.step("move()")
        self.assertEqual(len(transition), 5)
        self.assertEqual(transition[1:4], (0.5, False, False))

    def test_host_conversion_keeps_execution_artifacts_out_of_the_transition(self):
        result = execute(_Environment(), "move()", index=3)
        self.assertEqual(result.index, 3)
        self.assertTrue(result.ok)
        self.assertEqual(result.frames, {"front": ["frame"]})
        self.assertEqual(result.stdout, "done")

    def test_invalid_code_is_rejected_at_the_environment_boundary(self):
        environment = _Environment()
        environment.reset()
        with self.assertRaises(ValueError):
            environment.step("")
        with self.assertRaises(TypeError):
            environment.step(123)


if __name__ == "__main__":
    unittest.main()

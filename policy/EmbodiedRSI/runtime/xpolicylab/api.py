"""RoboDojo observation/action primitives; reset reports the Test restriction."""

from XPolicyLab.policy.EmbodiedRSI.runtime.execution.primitives import PrimitiveApi


class RoboDojoLowLevelApi(PrimitiveApi):
    def functions(self):
        return {
            "get_instruction": self.get_instruction,
            "get_observation": self.get_observation,
            "step": self.step,
            "reset": self.reset,
        }

    def get_instruction(self):
        return self._env.task_prompt()

    def get_observation(self):
        return self._env.public_observation()

    def step(self, action):
        return self._env.control_step(action)

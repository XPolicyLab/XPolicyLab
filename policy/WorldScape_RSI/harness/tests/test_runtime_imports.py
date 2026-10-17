"""Check private runtime imports in fresh interpreters without a GPU or weights."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

from test_harness import framework


@unittest.skipUnless(framework, "Requires the adjacent XPolicyLab checkout")
class RuntimeImportTests(unittest.TestCase):
    def run_script(self, code):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                   PYTHONPATH=os.pathsep.join([str(Path(framework).parent), framework]))
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_curobo_resolves_from_the_migrated_policy(self):
        self.run_script('''
from pathlib import Path
from XPolicyLab.policy.WorldScape_RSI.harness.backends import register_vendor
from XPolicyLab.policy.WorldScape_RSI.harness.release import ROOT
register_vendor()
import curobo
assert Path(curobo.__file__).resolve() == ROOT / "vendor/runtime_support/curobo/curobo/__init__.py"
''')

    def test_native_planner_does_not_import_top_level_utils_or_env(self):
        self.run_script('''
import sys, types
# GPU libraries are import fixtures only; native transforms and planner definitions are real.
from XPolicyLab.policy.WorldScape_RSI.harness.backends import register_vendor
register_vendor()
import curobo
for name, exports in {
    "batch_motion_planner": ["BatchMotionPlanner", "MotionPlannerCfg"],
    "inverse_kinematics": ["InverseKinematics", "InverseKinematicsCfg"],
    "motion_planner": ["MotionPlanner"],
    "runtime": [],
    "types": ["DeviceCfg", "GoalToolPose", "JointState", "Pose", "ToolPoseCriteria"],
}.items():
    mod = types.ModuleType("curobo." + name)
    for symbol in exports: setattr(mod, symbol, type(symbol, (), {}))
    sys.modules[mod.__name__] = mod
    setattr(curobo, name, mod)
# Poison the ambiguous names to ensure only package-relative imports succeed.
sys.modules["utils"] = types.ModuleType("utils")
sys.modules["env"] = types.ModuleType("env")
from XPolicyLab.policy.RoboRSI_RoboProbe.robot_planner import PublicCuroboPlanner
from XPolicyLab.policy.WorldScape_RSI.harness.vendor.runtime_support.native_runtime.env.planner_manager import curobo_planner
assert PublicCuroboPlanner.__bases__ == (curobo_planner.CuroboPlanner,)
assert curobo_planner.calculate_target_pose.__module__.startswith(
    "XPolicyLab.policy.WorldScape_RSI.harness.vendor.runtime_support.")
''')


if __name__ == "__main__":
    unittest.main()

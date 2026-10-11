"""VLM policy adapter for the RoboDojo benchmark.

The package is mounted into RoboDojo as an XPolicyLab policy
(``RoboDojo/XPolicyLab/policy/vlm_agent`` is a symlink to this directory,
created by ``scripts/robodojo/setup_vlm_policy.sh``), which lets a VLM drive
the standard RoboDojo eval loop without forking either submodule.

Two processes import from here and they must not pull each other's
dependencies, so nothing heavy is imported at package level:

* the **policy server** imports :mod:`model` (VLM calls, prompt building);
* the **simulation client** imports :mod:`deploy` (episode loop, IK checks).
"""

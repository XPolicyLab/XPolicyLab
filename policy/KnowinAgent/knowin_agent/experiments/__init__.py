"""M5 experiment infrastructure: switches and profiles, scene splits, run manifests, results store, tables."""

from knowin_agent.experiments.manifest import collect, harness_digest, read_manifest, write_manifest
from knowin_agent.experiments.splits import assert_not_test_dev, scenes_for, selection_scenes
from knowin_agent.experiments.switches import (REGISTRY, Profile, Switch, SwitchError, apply, compliance_report, defaults,
                                       diff, load_profile, resolve, validate)

__all__ = ["REGISTRY", "Profile", "Switch", "SwitchError", "apply", "assert_not_test_dev", "collect",
           "compliance_report", "defaults", "diff", "harness_digest", "load_profile", "read_manifest", "resolve",
           "scenes_for", "selection_scenes", "validate", "write_manifest"]

"""Use the pinned RoboTwin WebSocket client with a fixed-input rollout."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys


def main():
    root = Path(os.environ["CLOUDROBO_ROBOTWIN_ROOT"]).resolve()
    binding = json.loads((Path(__file__).parent / "SDK_BINDING.json").read_text())
    for relative, key in [("scripts/eval_policy_xpolicylab.py", "official_client_sha256"),
                          ("envs/_base_task.py", "base_task_sha256")]:
        if hashlib.sha256((root / relative).read_bytes()).hexdigest() != binding[key]:
            raise RuntimeError("RoboTwin source differs from the supported version: " + relative)
    for path in [root, root / "scripts", root / "description/utils", root / "XPolicyLab"]:
        sys.path.insert(0, str(path))
    spec = importlib.util.spec_from_file_location("_cached_official_client", root / "scripts/eval_policy_xpolicylab.py")
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    from fixed_runtime import load_cases, run_fixed_task
    parsed = official.parse_args()
    if official.parse_bool(parsed.get("eval_batch", False)) or official.parse_bool(parsed.get("expert_check", True)):
        raise ValueError("Use eval_batch=false and expert_check=false")
    if parsed.get("action_type") != "joint" or parsed.get("env_cfg_type") != "arx_x5":
        raise ValueError("This adapter requires arx_x5 joint actions")
    profile = {"seen": "clean_seen", "unseen": "random_unseen"}.get(parsed.get("instruction_type"))
    cases = load_cases(profile, parsed["task_name"], os.environ.get("CLOUDROBO_FIXED_CACHE_ROOT"))
    if parsed["task_config"] != cases[0]["task_config"] or int(parsed.get("test_num", 100)) != 100:
        raise ValueError("Require the cached task configuration and 100 trials")
    parsed["num_workers"] = 1
    original_load = official.load_task_args
    def load_args(values):
        args, embodiment = original_load(values)
        args["data_type"]["depth"] = False
        args["data_type"]["pointcloud"] = False
        return args, embodiment
    official.load_task_args = load_args
    output = Path(os.environ["CLOUDROBO_FIXED_OUTPUT"]).resolve()
    official.build_eval_save_dir = lambda *args, **kwargs: output
    official.eval_remote_policy = lambda *args, **kwargs: run_fixed_task(official, *args, **kwargs)
    from test_render import Sapien_TEST
    Sapien_TEST()
    official.main(parsed)


if __name__ == "__main__":
    main()

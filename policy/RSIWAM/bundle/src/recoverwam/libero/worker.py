"""One isolated simulator package and one resident policy per process."""

import argparse
from pathlib import Path
from importlib.metadata import version
import re
import numpy as np
from omegaconf import OmegaConf
from .protocols import REVISIONS


def check_runtime(cfg, tasks):
    import mujoco

    if mujoco.__version__ != "3.3.2":
        raise RuntimeError(f"Reference protocol requires MuJoCo 3.3.2; got {mujoco.__version__}")
    root = Path(cfg.release_benchmark_root).resolve()
    if cfg.release_benchmark == "pro":
        for package, expected in [
            ("rpent-liberopro", "0.2.0"),
            ("rpent-libero", "0.2.0"),
            ("robosuite", "1.5.2"),
        ]:
            if version(package) != expected:
                raise ValueError(f"{package} must be {expected}")
        for name in ("AUTHORITATIVE_INIT_REVISION", "AUTHORITATIVE_BDDL_REVISION"):
            if (root / name).read_text().strip() != REVISIONS["pro"]:
                raise ValueError(f"Wrong authoritative PRO revision marker: {name}")
        from .pro_alias import route_liberopro_as_libero
        from .pro_runtime import configure_harnessvla_liberopro_runtime

        route_liberopro_as_libero()
        configure_harnessvla_liberopro_runtime()
    from libero.libero import benchmark, get_libero_path

    if not Path(benchmark.__file__).resolve().is_relative_to(root):
        raise RuntimeError("Imported the wrong benchmark package; use an isolated environment")
    from .eval_helpers import load_trusted_task_init_states

    registry = benchmark.get_benchmark_dict()
    for suite_name, task_id in tasks:
        suite = registry[suite_name]()
        if suite.n_tasks != 10:
            raise ValueError(f"Expected 10 tasks in {suite_name}")
        if cfg.release_benchmark == "ood":
            continue
        states = load_trusted_task_init_states(suite, task_id)
        if len(states) != 50:
            raise ValueError(f"Expected 50 packaged states: {suite_name},{task_id}")
        if cfg.release_benchmark != "pro":
            continue
        task = suite.get_task(task_id)
        bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        text = bddl.read_text()
        match = re.search(r"\(:language\s+([^)]*)\)", text, re.S)
        if not match or " ".join(str(task.language).split()).strip('"') != " ".join(
            match.group(1).split()
        ).strip('"'):
            raise ValueError(f"Task language differs from BDDL: {bddl}")
        if (
            suite_name == "libero_spatial_task"
            and task_id == 0
            and "not between" not in task.language.lower()
        ):
            raise ValueError("PRO task0 must use the perturbed not-between instruction")
        from libero.libero.envs import OffScreenRenderEnv

        env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
        try:
            env.seed(1)
            env.reset()
            obs = env.set_init_state(states[1])
            obs, _, _, _ = env.step(np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                if np.asarray(obs[key]).shape != (256, 256, 3):
                    raise ValueError(f"Invalid camera {key}")
        finally:
            env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = OmegaConf.load(args.config)
    # Parse manifest before importing any simulator to keep runtime routing explicit.
    tasks = [
        (line.split(",")[0], int(line.split(",")[1]))
        for line in Path(cfg.EVALUATION.task_manifest).read_text().splitlines()
        if line.strip()
    ]
    check_runtime(cfg, tasks)
    from .eval_libero_task_manifest import eval_task_manifest

    eval_task_manifest(cfg)


if __name__ == "__main__":
    main()

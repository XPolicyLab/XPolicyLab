"""Environment-side reset evidence without policy execution or task scoring."""

from pathlib import Path

import numpy as np

from PhysicalRSI_core.infra.storage import atomic_json, file_digest


class InitializationFinished(BaseException):
    """Exit native main after one requested reset batch, without its retry loop."""


class ResetOnlyClient:
    def __init__(self, *args, **kwargs):
        pass

    def call(self, *, func_name, **kwargs):
        if func_name != "reset" or kwargs:
            raise RuntimeError("Initialization must not invoke a policy")

    def close(self):
        pass


class InitializationObserver:
    initialization = True

    def __init__(
        self, output, count, *, camera_calibration=False, robot_calibration=False
    ):
        self.output = Path(output)
        self.count = count
        self.camera_calibration = camera_calibration
        self.robot_calibration = robot_calibration
        self.env = self.app = None
        self.records = []
        self.failed = False

    def attach(self, env, *, app):
        self.env, self.app = env, app

        def run_eval():
            try:
                self.capture(env)
            except BaseException:
                self.failed = True
                raise
            raise InitializationFinished()

        env.run_eval = run_eval
        return env

    def capture(self, env):
        indices = list(env.get_running_env_idx_list())
        if (
            self.records
            or len(indices) != self.count
            or len(set(indices)) != self.count
            or env.abandoned_seeds
            or env.unstable_envs
            or env.unstable_nums
        ):
            raise ValueError("Initialization did not retain the complete stable cohort")
        observations = env.get_obs_batch(env_idx_list=indices)
        if env.abandoned_seeds or env.unstable_envs or env.unstable_nums:
            raise ValueError("Initialization became unstable while observing")
        if len(observations) != len(indices):
            raise ValueError("Initialization observation coverage mismatch")
        rows = []
        for index, obs in zip(indices, observations):
            seed = int(env.env_seeds[index])
            if (
                not env.scene_manager.layout_manager.layout_valid[index]
                or env.end_flag[index]
                or int(env.take_action_cnt[index]) != 0
                or obs["env_idx"] != index
            ):
                raise ValueError(
                    "Initialization is invalid, terminated or already acted"
                )
            folder = self.output / "observations" / str(seed)
            folder.mkdir(parents=True, exist_ok=False)
            state = {}
            for name, value in obs["state"].items():
                array = np.asarray(value)
                if (
                    array.ndim != 1
                    or not array.size
                    or array.dtype.kind not in "ifu"
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("Invalid initialized robot state")
                state[name] = array.tolist()
            if not state or not obs.get("vision"):
                raise ValueError(
                    "Initialization requires proprioception and RGB evidence"
                )
            images = {}
            for number, (name, camera) in enumerate(obs["vision"].items()):
                rgb = np.asarray(camera["color"])
                if (
                    rgb.dtype != np.uint8
                    or rgb.ndim != 3
                    or rgb.shape[2] != 3
                    or rgb.size > 4096 * 4096 * 3
                ):
                    raise ValueError("Invalid initialized RGB evidence")
                path = folder / f"{number}.npy"
                np.save(path, rgb, allow_pickle=False)
                images[name] = dict(
                    file=str(path.relative_to(self.output)),
                    sha256=file_digest(path),
                    shape=list(rgb.shape),
                )
            atomic_json(folder / "state.json", state)
            rows.append(
                dict(
                    layout_id=seed,
                    state="native_reset_checked",
                    policy_actions=0,
                    state_file=str((folder / "state.json").relative_to(self.output)),
                    state_sha256=file_digest(folder / "state.json"),
                    images=images,
                )
            )
        if sorted(row["layout_id"] for row in rows) != list(range(self.count)):
            raise ValueError("Initialization layout identity mismatch")
        if self.camera_calibration:
            from .native_camera import capture

            path = self.output / "camera-diagnostics.json"
            atomic_json(path, capture(env, indices))
            for row in rows:
                row["camera_diagnostics"] = dict(
                    file=path.name, sha256=file_digest(path)
                )
        if self.robot_calibration:
            from .native_robot import capture

            path = self.output / "robot-diagnostics.json"
            atomic_json(path, capture(env, indices))
            for row in rows:
                row["robot_diagnostics"] = dict(
                    file=path.name, sha256=file_digest(path)
                )
        self.records = rows

    def close(self):
        errors = []
        if self.env is not None:
            for close in (self.env.model_client.close, self.env.close):
                try:
                    close()
                except Exception as error:
                    errors.append(error)
        if self.app is not None:
            try:
                self.app.close()
            except Exception as error:
                errors.append(error)
        if errors:
            self.failed = True
            raise RuntimeError("Native initialization cleanup failed") from errors[0]

    def finish(self):
        if self.failed or len(self.records) != self.count:
            raise ValueError("Native initialization incomplete")
        return self.records


def finalize(root, request):
    """Validate process, cohort and recorded reset observations; never score tasks."""
    from .native_execution import _inside
    from PhysicalRSI_core.infra.storage import read_json, digest

    root = Path(root).resolve()
    process = read_json(_inside(root, "processes/simulator/process.json"))
    if (
        process.get("state") != "completed"
        or type(process.get("returncode")) is not int
        or process["returncode"] != 0
    ):
        raise ValueError("Initialization process did not complete cleanly")
    if (
        read_json(_inside(root, "request.json")) != request
        or read_json(_inside(root, "run/request.json")) != request
        or (root / "run/failed.json").exists()
    ):
        raise ValueError("Initialization request changed or worker failed")
    result = read_json(_inside(root, "run/initialized.json"))
    rows = result["initializations"]
    if result["state"] != "completed" or sorted(
        row["layout_id"] for row in rows
    ) != list(range(len(request["layouts"]))):
        raise ValueError("Initialization cohort coverage mismatch")
    for row in rows:
        if (
            row["state"] != "native_reset_checked"
            or row["policy_actions"] != 0
            or not row["images"]
        ):
            raise ValueError("Invalid native reset record")
        if (
            file_digest(_inside(root, "run/" + row["state_file"]))
            != row["state_sha256"]
        ):
            raise ValueError("Initialization state evidence changed")
        for image in row["images"].values():
            if file_digest(_inside(root, "run/" + image["file"])) != image["sha256"]:
                raise ValueError("Initialization RGB evidence changed")
        case = request["layouts"][row["layout_id"]]
        layout_root = Path(request["layout_root"]).resolve()
        if (
            digest(read_json(_inside(layout_root, case["file"])))
            != case["layout_sha256"]
        ):
            raise ValueError("Initialized layout changed")
        row["layout_sha256"] = case["layout_sha256"]
    if request.get("camera_calibration"):
        diagnostic = _inside(root, "run/camera-diagnostics.json")
        for row in rows:
            if row.get("camera_diagnostics") != dict(
                file="camera-diagnostics.json", sha256=file_digest(diagnostic)
            ):
                raise ValueError(
                    "Requested camera diagnostic evidence missing or changed"
                )
    if request.get("robot_calibration"):
        diagnostic = _inside(root, "run/robot-diagnostics.json")
        for row in rows:
            if row.get("robot_diagnostics") != dict(
                file="robot-diagnostics.json", sha256=file_digest(diagnostic)
            ):
                raise ValueError(
                    "Requested robot diagnostic evidence missing or changed"
                )
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Initialization artifacts must not be symlinks")
        if path.is_file():
            artifacts[str(path.relative_to(root))] = file_digest(path)
    receipt = root / "initialization.json"
    atomic_json(
        receipt,
        dict(
            schema="physicalrsi.robodojo.initialization/v1",
            task=request["task"],
            split=request["split"],
            initializations=rows,
            artifacts=artifacts,
            physical_qualification=False,
        ),
    )
    return receipt

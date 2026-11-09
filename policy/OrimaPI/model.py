"""Use OrimaPI's existing XPolicyLab model with the formal code-skill profile."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np
import orimapi
import yaml
from XPolicyLab.utils.process_data import get_robot_action_dim_info
from orimapi.xpolicylab.model import Model as OrimaModel


def _checkout_config(source, benchmark):
    """Resolve paths in the bundled OrimaPI configuration."""
    config_path = source / "configs/robodojo.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["policies"] = {"code-policy": config["policies"]["code-policy"]}
    config["fixed_policy"] = "code-policy"
    config["runtime"].update(
        mode="direct", exec_horizon=1, record_frames=False, supervisor_timeout_s=120
    )
    config["xpolicylab"] = {}
    worker_python = Path(
        os.getenv("ORIMAPI_NATIVE_PYTHON", str(source / ".venv-robodojo/bin/python"))
    ).expanduser().absolute()
    # Native workers set CUDA_VISIBLE_DEVICES themselves. Their device uses the
    # physical index, while in-process vision uses the logical cuda:0.
    gpu = os.getenv("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not gpu.isdecimal():
        raise ValueError("Use a numeric CUDA device for OrimaPI native workers")
    for skill in config["policies"]["code-policy"]["code_skills"].values():
        native = skill.get("code_parameters", {}).get("native_ik")
        if native is not None:
            native.update(
                python=str(worker_python), simulator_root=str(benchmark), device=f"cuda:{gpu}"
            )

    def rebase(value):
        if isinstance(value, dict):
            return {key: rebase(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rebase(item) for item in value]
        if isinstance(value, str):
            for prefix, root in (
                ("../third_party/RoboDojo", benchmark),
                ("../checkpoint", source / "checkpoint"),
            ):
                if value == prefix or value.startswith(prefix + "/"):
                    return str(root / value[len(prefix):].lstrip("/"))
            if value.startswith(("../", "./")):
                return str((config_path.parent / value).resolve())
        return value

    return rebase(config)


class Model(OrimaModel):
    def __init__(self, model_cfg):
        if (
            model_cfg.get("bench_name"),
            model_cfg.get("env_cfg_type"),
            model_cfg.get("action_type"),
        ) != ("RoboDojo", "arx_x5", "joint"):
            raise ValueError("OrimaPI requires RoboDojo/arx_x5/joint/code-skills")
        self.robot_action_dim_info = get_robot_action_dim_info(model_cfg["env_cfg_type"])
        if (
            self.robot_action_dim_info.get("arm_dim") != [6, 6]
            or self.robot_action_dim_info.get("ee_dim") != [1, 1]
        ):
            raise ValueError("Expected dual ARX X5 6-DoF arms and scalar grippers")
        fast_chunking = os.getenv("ORIMAPI_FAST_TRAJECTORY_CHUNKING", "").strip().lower()
        if model_cfg.get("eval_batch") or fast_chunking not in {"", "0", "false", "no", "off"}:
            raise ValueError("OrimaPI uses one environment and a one-action horizon")
        self._debug = os.getenv("EVAL_ENV_TYPE") == "debug"
        if self._debug:
            self._debug_state = None
            self._debug_fresh = False
            self._closed = False
            return
        if model_cfg.get("ckpt_name") != "code-skills":
            raise ValueError("OrimaPI requires RoboDojo/arx_x5/joint/code-skills")
        source = Path(__file__).resolve().parent / "OrimaPI"
        if Path(orimapi.__file__).resolve().parent.parent != source:
            raise ValueError("Run policy/OrimaPI/install.sh in the active policy environment")
        xpl_root = Path(__file__).resolve().parents[2]
        benchmark = Path(os.getenv("ROBODOJO_ROOT", str(xpl_root.parent))).resolve()
        result = benchmark / "eval_result/OrimaPI_traces" / uuid4().hex
        with TemporaryDirectory(prefix="orimapi-config-") as temp:
            config_path = Path(temp) / "robodojo.yaml"
            config_path.write_text(
                yaml.safe_dump(_checkout_config(source, benchmark)),
                encoding="utf-8",
            )
            super().__init__(
                {
                    **model_cfg,
                    "orimapi_config": str(config_path),
                    "result_dir": str(result),
                    "raw_images": True,
                },
            )
        try:
            # Match the official OrimaPI suite's fixed policy seed, separate from layouts.
            self.prepare_job(0)
        except BaseException:
            self.close()
            raise

    def record_case(self, case_meta=None):
        if self._debug:
            return
        return super().record_case(case_meta)

    def update_obs(self, obs):
        if not self._debug:
            return super().update_obs(obs)
        for camera in obs["vision"].values():
            image = np.asarray(camera["color"])
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError("Expected decoded uint8 RGB images")
        state = {}
        for idx, side in enumerate(("left", "right")):
            for part, dim in (
                ("arm", self.robot_action_dim_info["arm_dim"][idx]),
                ("ee", self.robot_action_dim_info["ee_dim"][idx]),
            ):
                key = f"{side}_{part}_joint_state"
                val = np.asarray(obs["state"][key], dtype=np.float32).reshape(-1)
                if val.size != dim or not np.isfinite(val).all():
                    raise ValueError(f"Invalid robot state field: {key}")
                state[key] = val.copy()
        self._debug_state = state
        self._debug_fresh = True

    def get_action(self):
        if not self._debug:
            return super().get_action()
        if self._debug_state is None or not self._debug_fresh:
            raise ValueError("Fresh observations are required before requesting actions")
        self._debug_fresh = False
        return [{key: val.copy() for key, val in self._debug_state.items()}]

    def on_trial_end(self, result=None):
        if self._debug:
            return self.reset()
        return super().on_trial_end(result)

    def reset(self):
        if self._debug:
            self._debug_state = None
            self._debug_fresh = False
            return
        return super().reset()

    def close(self):
        if self._debug:
            self._closed = True
            return
        return super().close()

    def update_obs_batch(self, obs_list):
        if len(obs_list) != 1:
            raise ValueError("OrimaPI runs one active environment per server")
        return self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and list(env_idx_list) != [0]:
            raise ValueError("OrimaPI runs one active environment per server")
        return [self.get_action()]

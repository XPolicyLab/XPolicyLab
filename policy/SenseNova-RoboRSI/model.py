"""Official XPolicyLab boundary for the original RoboDojo hybrid Agent."""

from __future__ import annotations

import atexit
import copy
import json
import os
import queue
import sys
import tempfile
import time
import uuid
import weakref
from pathlib import Path
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.utils.checkpoint_resolver import resolve_checkpoint_root
from XPolicyLab.utils.process_data import get_robot_action_dim_info

from .icl_context import prepare_icl_context
from .runtime_config import POLICY_DIR, load_agent_config, submission
from .tool_runtime import resolve_tool_python

_LIVE_MODELS = weakref.WeakSet()
ACTION_POLL_SECONDS = 30.0


def close_models() -> None:
    """Stop all episode and VLA owners before Python joins background threads."""
    for model in list(_LIVE_MODELS):
        model.shutdown()


class Model(ModelTemplate):
    """One original conversation per episode; one acknowledged joint step per call."""

    def __init__(self, model_cfg: dict[str, Any]):
        super().__init__()
        if not os.environ.get("CODEX_API_KEY", "").strip():
            raise ValueError(
                "CODEX_API_KEY is required; set it in the runtime environment"
            )
        if model_cfg.get("bench_name") != "RoboDojo":
            raise ValueError("SenseNova-RoboRSI supports only bench_name=RoboDojo")
        if model_cfg.get("env_cfg_type") != "arx_x5":
            raise ValueError("The original hybrid supports only env_cfg_type=arx_x5")
        if model_cfg.get("action_type") != "joint":
            raise ValueError("The original hybrid FK/DLS emits action_type=joint")
        if model_cfg.get("eval_batch", False):
            raise ValueError("SenseNova-RoboRSI requires eval_batch=false")
        # Validate the independent original tool payload before materializing
        # ICL or starting the expensive VLA service. Never substitute the host.
        self._tool_python = resolve_tool_python(POLICY_DIR)
        self.model_cfg = dict(model_cfg)
        self.model_cfg.setdefault(
            "checkpoint_path", "checkpoints/pi05-arx-x5-seed0-step59999"
        )
        self.action_type = "joint"
        self.robot_action_dim_info = get_robot_action_dim_info("arx_x5")
        dimensions = self.robot_action_dim_info
        if dimensions["arm_dim"] != [6, 6] or dimensions["ee_dim"] != [1, 1]:
            raise ValueError(
                "The pinned ARX-X5 hybrid metadata requires six joints and one gripper per arm"
            )
        source = POLICY_DIR / "runtime/RoboRSI"
        if not source.is_dir():
            raise ValueError("Pinned RoboRSI source is missing; run bash install.sh")
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from robots.robodojo.eef_types import ActionSpace

        from .pi05_source import load_openpi_client

        load_openpi_client(POLICY_DIR)

        channels = []
        for index, arm in enumerate(("left", "right")):
            channels.extend(
                (
                    (f"{arm}_arm_joint_state", dimensions["arm_dim"][index]),
                    (f"{arm}_ee_joint_state", dimensions["ee_dim"][index]),
                )
            )
        self._action_space = ActionSpace(channels)
        self._agent_config = load_agent_config(self.model_cfg)
        self._temporary = self._connector = self._icl = self._obs = self._pending = None
        self._vla = self._vla_temporary = None
        self._step = 0
        self._awaiting_feedback = self._ended = False
        # The server constructs Model before opening its websocket port. Load
        # VLA weights here so cold start does not consume a native action RPC.
        self._ensure_vla()
        _LIVE_MODELS.add(self)
        atexit.register(self.shutdown)

    def reset(self) -> None:
        """End the old conversation; retain only the owned immutable VLA service."""
        self.close()
        self._temporary = tempfile.TemporaryDirectory(prefix="roborsi-xpl-")
        self._directory = Path(self._temporary.name)
        self._episode_id = uuid.uuid4().hex
        self._agent_config = load_agent_config(
            self.model_cfg, episode_dir=self._directory, trajectory_id=self._episode_id
        )
        self._step = 0
        self._obs = self._pending = None
        self._awaiting_feedback = self._ended = False

    def update_obs(self, obs: dict[str, Any]) -> None:
        """Accept the standard server's decoded RGB; acknowledge only at get_action."""
        if self._temporary is None or self._ended:
            raise ValueError("Call reset before updating an episode observation")
        if not isinstance(obs, dict) or not isinstance(obs.get("vision"), dict):
            raise ValueError("Observation requires a vision mapping")
        instruction = obs.get("instruction") or obs.get("instructions")
        if (
            isinstance(instruction, (list, tuple))
            and instruction
            and all(isinstance(value, str) and value.strip() for value in instruction)
        ):
            instruction = "\n".join(instruction)
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Observation requires instruction (or instructions)")
        if not isinstance(obs.get("state"), dict):
            raise ValueError("Observation requires robot state")
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            color = obs["vision"].get(name, {}).get("color")
            if not isinstance(color, np.ndarray) or color.dtype != np.uint8:
                raise ValueError(
                    "The policy server must supply three decoded uint8 RGB colors"
                )
            if color.ndim != 3 or color.shape[-1] != 3:
                raise ValueError("Camera colors must have shape H,W,3")
        self._obs = {**obs, "instruction": instruction}

    def _ensure_vla(self) -> int:
        if self._vla is None:
            from .pi05_runtime import Pi05Server

            checkpoint = resolve_checkpoint_root(
                self.model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR
            )
            self._vla_temporary = tempfile.TemporaryDirectory(prefix="roborsi-vla-")
            server = Pi05Server(
                checkpoint_dir=checkpoint,
                data_home=POLICY_DIR / "runtime/pi05-data",
                output_dir=Path(self._vla_temporary.name),
                gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                policy_dir=POLICY_DIR,
            )
            try:
                self._vla = server.start()
            except BaseException:
                server.close()
                self._vla_temporary.cleanup()
                self._vla_temporary = None
                raise
        return self._vla.port

    def _start(self) -> None:
        from .hybrid_agent import HybridController, HybridRunContext
        from .hybrid_backend import HybridBackend, load_reference_source

        pins = submission()
        profile = pins["hybrid"]
        reference = load_reference_source(
            POLICY_DIR / profile["source_path"],
            POLICY_DIR / profile["manifest_path"],
            profile["manifest_sha256"],
        )
        self._icl = prepare_icl_context(
            POLICY_DIR, self.model_cfg["task_name"], self._directory
        )
        base_task = self._icl.source_identity["mapping"]["base_task"]
        config_path = (
            POLICY_DIR
            / profile["profile_path"]
            / "resources/fresh-configs"
            / f"{base_task}-s0.json"
        )
        cfg = json.loads(config_path.read_text())
        cfg = copy.deepcopy(cfg)
        cfg.pop("continue_from_run", None)
        cfg["options"]["task"] = self.model_cfg["task_name"]
        cfg["options"]["gpu"] = cfg["options"]["physical_gpu"] = 0
        cfg["options"]["vla_port"] = self._ensure_vla()
        episode = cfg["test_episodes"][0]
        episode.update(
            task=self.model_cfg["task_name"],
            eval_seed=int(self.model_cfg.get("seed") or 0),
        )
        # Internal records describe the requested case. The official evaluator's
        # result, outside this policy, is authoritative for the actual layout.
        cfg["episode_identity_scope"] = (
            "requested identity; actual native layout is not exposed to the policy"
        )
        case = self._directory / "agent"
        case.mkdir(exist_ok=True)
        self._agent_config = load_agent_config(
            self.model_cfg, episode_dir=case, trajectory_id=self._episode_id
        )
        (case / "icl-source.json").write_text(
            json.dumps(self._icl.source_identity, indent=2) + "\n"
        )
        backend = HybridBackend(
            reference,
            task_name=self.model_cfg["task_name"],
            episode_dir=case,
            initial_observation=self._obs,
            step_limit=cfg["options"]["review_window_steps"],
            metadata_path=POLICY_DIR / profile["robot_metadata_path"],
            expected_metadata_sha256=profile["robot_metadata_sha256"],
            resource_root=POLICY_DIR / "hybrid",
            vla_port=cfg["options"]["vla_port"],
        )
        agent = self._agent_config.create_agent(case)
        context = HybridRunContext(
            reference=reference,
            backend=backend,
            cfg=cfg,
            episode_dir=case,
            prepared_workspace=self._icl.workspace,
            training_set=self._icl.workspace,
            public_instructions=POLICY_DIR
            / profile["profile_path"]
            / "resources/interfaces"
            / f"{base_task}-s0.md",
            agent_python=self._tool_python,
            bubblewrap=POLICY_DIR / "runtime/codex/codex-resources/bwrap",
            bubblewrap_sha256=pins["cli"]["companion_sha256"]["codex-resources/bwrap"],
        )
        self._connector = HybridController(agent, context).start()

    def get_action(self) -> list[dict[str, np.ndarray]]:
        """Yield one original joint update; wait for its actual executed observation."""
        if self._obs is None or self._ended:
            raise ValueError("A fresh update_obs is required before get_action")
        try:
            if self._connector is None:
                self._start()
            elif self._pending is not None:
                self._connector.acknowledge(self._pending.request_id, self._obs)
                self._pending = None
                self._step += 1
                self._awaiting_feedback = False
            try:
                pending = self._connector.next_action(timeout=ACTION_POLL_SECONDS)
            except queue.Empty:
                # An empty standard chunk means the original Agent is still
                # thinking. Keep the same observation/session; the deployment
                # loop retries without a synthetic action or simulator step.
                return []
            if pending is None:
                raise RuntimeError(
                    "Hybrid Agent ended before the official environment terminated"
                )
            values = [
                number
                for name in self._action_space.names
                for number in pending.action[name]
            ]
            decoded = dict(self._action_space.decode(values).data)
            self._pending = pending
            self._obs = None
            self._awaiting_feedback = True
            return [decoded]
        except BaseException:
            self.close()
            raise

    def end_episode(self, obs: dict[str, Any]) -> None:
        """Deliver native terminal observation without reading or inventing its score."""
        if self._connector is None or self._pending is None:
            return
        try:
            self.update_obs(obs)
            self._connector.acknowledge(
                self._pending.request_id, self._obs, terminated=True
            )
            self._pending = None
            self._step += 1
            self._awaiting_feedback = False
            self._ended = True
            deadline = time.monotonic() + 16
            while not self._connector.done and time.monotonic() < deadline:
                time.sleep(0.05)
        finally:
            self.close()

    def update_obs_batch(self, obs_list) -> None:
        if len(obs_list) != 1:
            raise ValueError("SenseNova-RoboRSI supports one environment per server")
        self.update_obs(obs_list[0])

    def get_action_batch(self, env_idx_list=None):
        if env_idx_list is not None and len(env_idx_list) != 1:
            raise ValueError("SenseNova-RoboRSI supports one environment per server")
        return [self.get_action()]

    def close(self) -> None:
        """Release episode state before cleanup; VLA weights stay owned by this Model."""
        try:
            if self._connector is not None:
                self._connector.close()
        finally:
            self._connector = self._icl = self._obs = self._pending = None
            self._awaiting_feedback = False
            if self._temporary is not None:
                self._temporary.cleanup()
                self._temporary = None

    def shutdown(self) -> None:
        """Stop the owned CLI and persistent VLA service at server shutdown."""
        try:
            self.close()
        finally:
            if self._vla is not None:
                self._vla.close()
                self._vla = None
            if self._vla_temporary is not None:
                self._vla_temporary.cleanup()
                self._vla_temporary = None

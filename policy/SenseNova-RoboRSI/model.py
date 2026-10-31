"""Official XPolicyLab boundary for the original RoboDojo hybrid Agent."""

from __future__ import annotations

import atexit
import copy
import fcntl
import json
import logging
import os
import queue
import re
import socket
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
from .runtime_config import (
    POLICY_DIR, load_agent_config, resolve_adaptive_steps, resolve_base_url, resolve_model, submission,
)
from .tool_runtime import resolve_tool_python

_LIVE_MODELS = weakref.WeakSet()
ACTION_POLL_SECONDS = 30.0
logger = logging.getLogger(__name__)


class _SharedVLA:
    """Non-owning handle for a node-local shared pi0.5 websocket service."""

    def __init__(self, port: int):
        self.port = port

    def close(self) -> None:
        return None


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
        resolve_base_url()
        resolve_model()
        self._adaptive_steps = resolve_adaptive_steps()
        # Validate the independent original tool payload before materializing
        # ICL or starting the expensive VLA service. Never substitute the host.
        self._tool_python = resolve_tool_python(POLICY_DIR)
        self.model_cfg = dict(model_cfg)
        self.model_cfg.setdefault("vla_backend", "pi05")
        if self.model_cfg["vla_backend"] not in ("pi05", "simate_beta"):
            raise ValueError("vla_backend must be pi05 or simate_beta")
        self.model_cfg.setdefault(
            "checkpoint_path", "checkpoints/simate-beta-robodojo-eval"
            if self.model_cfg["vla_backend"] == "simate_beta"
            else "checkpoints/pi05-arx-x5-seed0-step59999"
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
        self._trajectory_directory = None
        self._trajectory_id = None
        self._vla = None
        self._simate_observer = None
        self._step = 0
        self._resume = None
        self._unfinished_native = None
        self._resume_request_path = None
        self._resume_start_lock = None
        self._deferred_controller = None
        self._cleanup_issue = None
        self._awaiting_feedback = self._ended = False
        # The server constructs Model before opening its websocket port. Load
        # VLA weights here so cold start does not consume a native action RPC.
        self._ensure_vla()
        _LIVE_MODELS.add(self)
        atexit.register(self.shutdown)

    def reset(self) -> None:
        """End the old conversation; retain only the owned immutable VLA service."""
        if getattr(self, "_deferred_controller", None) is not None:
            raise RuntimeError(
                "Previous episode cleanup is deferred; restart the policy server before reset"
            )
        self.close()
        if os.environ.get("ROBORSI_RESUME_MANIFEST"):
            raise ValueError(
                "Legacy restore-receipt manifests do not execute physical replay; "
                "prepare a verified ROBORSI_RESUME_REQUEST instead"
            )
        task = self.model_cfg.get("task_name")
        if not isinstance(task, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", task):
            raise ValueError("An exact RoboDojo task name is required")
        from .hybrid_resume import (
            declared_identity,
            load_resume,
            process_identity,
            write_json,
        )

        identity = declared_identity(
            self.model_cfg, os.environ.get("ROBODOJO_RUN_ID", "")
        )
        self._resume = None
        self._resume_request_path = None
        resume_path = os.environ.get("ROBORSI_RESUME_REQUEST")
        if resume_path:
            request_path = Path(resume_path).absolute()
            from .hybrid_resume import digest, read_json

            request_sha = os.environ.get("ROBORSI_RESUME_REQUEST_SHA256")
            if request_sha and digest(request_path) != request_sha:
                raise ValueError("Resume request changed after launcher validation")
            requested = read_json(request_path).get("identity", {})
            for key, expected in identity.items():
                if expected is not None and requested.get(key) != expected:
                    raise ValueError(f"Resume request differs from deployment: {key}")
            identity = requested
            self._resume = load_resume(request_path, identity)
            self._resume_request_path = request_path
            if request_path.with_suffix(request_path.suffix + ".claimed").exists():
                raise FileExistsError("Resume request already started a controller")
        # Each attempt keeps a separate output directory. The wire identity
        # belongs to the logical episode and survives process/session changes.
        self._episode_id = uuid.uuid4().hex
        self._trajectory_id = (
            self._resume["trajectory_id"] if self._resume else self._episode_id
        )
        root = self.model_cfg.get("trajectory_dir")
        root = Path(root).expanduser() if root else POLICY_DIR / "outputs/trajectories"
        if not root.is_absolute():
            root = POLICY_DIR / root
        seed = int(self.model_cfg.get("seed") or 0)
        self._trajectory_directory = root / task / f"seed-{seed}" / self._episode_id
        self._trajectory_directory.mkdir(parents=True, mode=0o700)
        write_json(self._trajectory_directory / "trajectory.json", {
            "schema_version": 1, "task": task, "seed": seed,
            "episode_id": self._episode_id, "trajectory_id": self._trajectory_id,
        })
        self._temporary = tempfile.TemporaryDirectory(
            prefix="runtime-", dir=self._trajectory_directory
        )
        # Preserve interrupted episodes even when close() cannot complete.
        self._temporary._finalizer.detach()
        self._directory = Path(self._temporary.name)
        self._agent_config = load_agent_config(
            self.model_cfg, episode_dir=self._directory, trajectory_id=self._trajectory_id
        )
        self._step = 0
        owner = {
            "format": "xpolicylab_episode_owner_v1",
            "identity": identity,
            "episode_id": self._episode_id,
            "trajectory_id": self._trajectory_id,
            "pid": os.getpid(),
            "process_start": process_identity(os.getpid()),
            "started_at": time.time(),
            "closed": False,
        }
        if self._resume:
            owner.update(
                source=self._resume["source"],
                recovery_state=self._resume["recovery_state"],
                recovery_count=self._resume["recovery_count"],
            )
        write_json(self._directory / "episode-owner.json", owner)
        self._obs = self._pending = None
        self._awaiting_feedback = self._ended = False

    def _reserve_resume(self) -> None:
        """Exclude other starters without consuming a request during initialization."""
        if self._resume is None or getattr(self, "_resume_start_lock", None) is not None:
            return
        request_path = self._resume_request_path
        # Keep this inode: unlinking a lock file permits competing inode locks.
        descriptor = os.open(
            request_path.with_suffix(request_path.suffix + ".startup.lock"),
            os.O_CREAT | os.O_RDWR | os.O_CLOEXEC,
            0o600,
        )
        try:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise FileExistsError("Resume request has an active controller starter") from error
            if request_path.with_suffix(request_path.suffix + ".claimed").exists():
                raise FileExistsError("Resume request already started a controller")
        except BaseException:
            os.close(descriptor)
            raise
        self._resume_start_lock = descriptor

    def _claim_resume(self) -> None:
        """Consume only after controller initialization, just before execution."""
        if self._resume is None:
            return
        self._reserve_resume()
        from .hybrid_resume import digest, process_identity

        request_path = self._resume_request_path
        expected = os.environ.get("ROBORSI_RESUME_REQUEST_SHA256")
        if expected and digest(request_path) != expected:
            raise ValueError("Resume request changed before controller startup")
        with request_path.with_suffix(request_path.suffix + ".claimed").open("x") as claim:
            json.dump(
                {"pid": os.getpid(), "process_start": process_identity(os.getpid()),
                 "attempt": str(self._directory), "controller_start_requested": True},
                claim,
            )

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
        from .hybrid_resume import read_json, validate_native_identity, write_json

        owner_path = self._directory / "episode-owner.json"
        owner = read_json(owner_path)
        native = obs.get("roborsi_native_identity")
        if native is not None or self._resume or os.environ.get("ROBORSI_REQUIRE_NATIVE_IDENTITY") == "1":
            expected_step = self._step + int(
                self._pending is not None and getattr(self._pending, "kind", "joint") == "joint"
            )
            source_native = self._resume["native_identity"] if self._resume else None
            native = validate_native_identity(
                native, owner["identity"], step=expected_step,
                step_limit=(source_native or owner.get("native_identity", {})).get("step_limit"),
                scene_sha256=(source_native or owner.get("native_identity", {})).get("scene_sha256"),
            )
            unfinished = getattr(self, "_unfinished_native", None)
            if not self._resume and unfinished and all(
                native.get(key) == unfinished.get(key)
                for key in ("task", "eval_seed", "layout", "run_id", "scene_sha256")
            ):
                raise ValueError(
                    "Interrupted native episode requires a prepared resume request; "
                    "refusing a fresh workspace and renewed budget for the same scene"
                )
            self._unfinished_native = None
            if owner["identity"].get("layout") is None:
                owner["identity"]["layout"] = native["layout"]
            if owner["identity"].get("run_id") is None:
                owner["identity"]["run_id"] = native["run_id"]
            owner["native_identity"] = native
            write_json(owner_path, owner)
        self._obs = {**obs, "instruction": instruction}
        if self.model_cfg.get("vla_backend") == "simate_beta":
            from .simate_bridge import SimateObserver

            observed_step = self._step + int(
                self._pending is not None and getattr(self._pending, "kind", "joint") == "joint"
            )
            if self._simate_observer is None:
                self._simate_observer = SimateObserver(self._ensure_vla(), self._episode_id)
            self._simate_observer.observe(self._obs, observed_step)

    def _ensure_vla(self) -> int:
        if self._vla is None:
            shared_port = os.environ.get("ROBORSI_SHARED_VLA_PORT")
            if shared_port:
                try:
                    port = int(shared_port)
                except ValueError as error:
                    raise ValueError(
                        "ROBORSI_SHARED_VLA_PORT must be an integer"
                    ) from error
                if not 1 <= port <= 65535:
                    raise ValueError(
                        "ROBORSI_SHARED_VLA_PORT is outside the TCP port range"
                    )
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=2):
                        pass
                except OSError as error:
                    raise RuntimeError(
                        f"shared pi0.5 service is not reachable on localhost:{port}"
                    ) from error
                self._vla = _SharedVLA(port)
                return port
            checkpoint = resolve_checkpoint_root(
                self.model_cfg, POLICY_DIR / "checkpoints", policy_dir=POLICY_DIR
            )
            # Keep each owned service's diagnostics after failure or shutdown.
            # mkdtemp creates a private directory without a deleting finalizer.
            log_root = POLICY_DIR / "outputs/vla"
            log_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=log_root)).resolve()
            options = dict(
                checkpoint_dir=checkpoint, output_dir=output_dir,
                gpu_id=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
                port=0, policy_dir=POLICY_DIR,
            )
            if self.model_cfg.get("vla_backend") == "simate_beta":
                from .simate_runtime import SimateServer

                server = SimateServer(seed=int(self.model_cfg.get("seed") or 0), **options)
            else:
                from .pi05_runtime import Pi05Server

                server = Pi05Server(data_home=POLICY_DIR / "runtime/pi05-data", **options)
            try:
                self._vla = server.start()
            except BaseException:
                server.close()
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
        cfg["model"] = self._agent_config.document["agent"]["model"]
        cfg.pop("continue_from_run", None)
        cfg["options"]["task"] = self.model_cfg["task_name"]
        cfg["options"].update(self._adaptive_steps)
        cfg["options"]["gpu"] = cfg["options"]["physical_gpu"] = 0
        cfg["options"]["vla_port"] = self._ensure_vla()
        episode = cfg["test_episodes"][0]
        episode.update(
            task=self.model_cfg["task_name"],
            eval_seed=int(self.model_cfg.get("seed") or 0),
        )
        native_identity = self._obs.get("roborsi_native_identity")
        if native_identity is not None:
            from .hybrid_resume import validate_native_identity

            validate_native_identity(
                native_identity,
                {"task": self.model_cfg["task_name"], "eval_seed": int(self.model_cfg.get("seed") or 0)},
                step=0, step_limit=cfg["options"]["review_window_steps"],
            )
            episode["layout_id"] = native_identity["layout"]
            episode["id"] = (
                f"{episode['task']}-s{episode['seed']}-eval{episode['eval_seed']}"
                f"-layout{episode['layout_id']}"
            )
        if self._resume:
            from .hybrid_resume import validate_policy

            validate_policy(self._resume["config"], cfg)
            cfg = copy.deepcopy(self._resume["config"])
            cfg["options"]["gpu"] = cfg["options"]["physical_gpu"] = 0
            cfg["options"]["vla_port"] = self._ensure_vla()
        # Internal records describe the requested case. The official evaluator's
        # result, outside this policy, is authoritative for the actual layout.
        cfg["episode_identity_scope"] = (
            "verified official reset and saved-scene sha256" if native_identity
            else "unverified debug observation; not eligible for native resume"
        )
        case = self._directory / "agent"
        case.mkdir(exist_ok=True)
        self._agent_config = load_agent_config(
            self.model_cfg, episode_dir=case, trajectory_id=self._trajectory_id
        )
        if self._agent_config.document["agent"]["model"] != cfg["model"]:
            raise ValueError("ROBORSI_CODEX_MODEL changed while preparing the episode; reset before retrying")
        (case / "icl-source.json").write_text(
            json.dumps(self._icl.source_identity, indent=2) + "\n"
        )
        self._reserve_resume()
        backend = HybridBackend(
            reference,
            task_name=self.model_cfg["task_name"],
            episode_dir=self._trajectory_directory,
            initial_observation=self._obs,
            step_limit=cfg["options"]["review_window_steps"],
            metadata_path=POLICY_DIR / profile["robot_metadata_path"],
            expected_metadata_sha256=profile["robot_metadata_sha256"],
            resource_root=POLICY_DIR / "hybrid",
            vla_port=cfg["options"]["vla_port"],
            replay=cfg["options"].get("replay"),
            vla_session_id=self._episode_id
            if self.model_cfg.get("vla_backend") == "simate_beta" else None,
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
            / "BENCHMARK.md",
            agent_python=self._tool_python,
            bubblewrap=POLICY_DIR / "runtime/codex/codex-resources/bwrap",
            bubblewrap_sha256=pins["cli"]["companion_sha256"]["codex-resources/bwrap"],
            resume=self._resume,
            trajectory_dir=self._trajectory_directory,
            before_execution=self._claim_resume,
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
                self._step += int(getattr(self._pending, "kind", "joint") == "joint")
                self._pending = None
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
            if getattr(pending, "kind", "joint") == "finish":
                self._pending = pending
                self._obs = None
                self._awaiting_feedback = True
                return [{"_roborsi_control": "finish"}]
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
        controller = self._connector
        terminal_accepted = False
        try:
            self.update_obs(obs)
            self._connector.acknowledge(
                self._pending.request_id, self._obs, terminated=True
            )
            self._step += int(getattr(self._pending, "kind", "joint") == "joint")
            self._pending = None
            self._awaiting_feedback = False
            self._ended = True
            deadline = time.monotonic() + 16
            while not self._connector.done and time.monotonic() < deadline:
                time.sleep(0.05)
            if self._resume:
                from .hybrid_resume import read_json

                validation_path = (
                    self._directory / "agent/controller/resume-validation.json"
                )
                if (
                    not validation_path.is_file()
                    or read_json(validation_path).get("matched") is not True
                    or self._connector.error is not None
                ):
                    raise RuntimeError(
                        "Resumed episode terminated before validated continuation"
                    )
            terminal_accepted = True
        finally:
            execution_failed = sys.exc_info()[0] is not None
            try:
                self.close()
            except Exception as error:
                # Native run_eval persists its result only after this RPC returns.
                # A cleanup failure must not discard an accepted terminal ACK, or
                # hide an earlier observation/replay validation failure.
                self._deferred_controller = controller
                self._cleanup_issue = {
                    "format": "roborsi_native_cleanup_issue_v1",
                    "error_type": type(error).__name__,
                    "reason": "controller_cleanup_failed",
                    "native_terminal_accepted": terminal_accepted,
                    "acknowledged_steps": self._step,
                    "supervisor_cleanup_required": True,
                }
                try:
                    from .hybrid_resume import write_json

                    write_json(self._directory / "cleanup-error.json", self._cleanup_issue)
                except OSError:
                    logger.warning("Could not persist native episode cleanup receipt")
                logger.warning(
                    "Native episode controller cleanup deferred to supervisor (%s)",
                    type(error).__name__,
                )
                if not terminal_accepted and not execution_failed:
                    raise
        return copy.deepcopy(getattr(self, "_cleanup_issue", None))

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
        if getattr(self, "_deferred_controller", None) is not None:
            # Keep the uncancelled owner quarantined until the case supervisor
            # reaps this server. reset() must never reuse it for another trial.
            return copy.deepcopy(self._cleanup_issue)
        try:
            if self._connector is not None:
                self._connector.close()
            observer = getattr(self, "_simate_observer", None)
            if observer is not None:
                observer.close()
                self._simate_observer = None
            if self._temporary is not None:
                from .hybrid_resume import read_json, write_json

                owner_path = self._directory / "episode-owner.json"
                if owner_path.is_file():
                    owner = read_json(owner_path)
                    if self._ended:
                        self._unfinished_native = None
                    elif self._connector is not None and owner.get("native_identity"):
                        self._unfinished_native = owner["native_identity"]
                    owner.update(
                        closed=True,
                        closed_at=time.time(),
                        acknowledged_steps=self._step,
                    )
                    write_json(owner_path, owner)
                try:
                    from .trajectory_archive import archive_episode

                    archive_episode(
                        self._directory,
                        task=str(self.model_cfg.get("task_name", "unknown")),
                        episode_id=self._episode_id,
                    )
                except Exception as error:
                    (self._directory / "trajectory_archive_error.txt").write_text(
                        f"{type(error).__name__}: {error}\n"
                    )
        except BaseException:
            # A failed close must never permit a later reset to create another
            # episode while its old controller may still own writers/processes.
            if self._connector is not None:
                self._deferred_controller = self._connector
            raise
        finally:
            # A failed controller cleanup keeps its reservation: its thread may
            # still execute. Otherwise close (or process exit) releases startup
            # failures without consuming the request or changing trajectory ID.
            if getattr(self, "_deferred_controller", None) is None:
                descriptor = getattr(self, "_resume_start_lock", None)
                self._resume_start_lock = None
                if descriptor is not None:
                    os.close(descriptor)
            self._connector = self._icl = self._obs = self._pending = None
            self._awaiting_feedback = False
            if self._temporary is not None:
                # Keep the source episode directory for post-mortem and full
                # trajectory recovery.  archive_episode copies it when the
                # episode closes; never delete the source after that copy.
                finalizer = getattr(self._temporary, "_finalizer", None)
                if finalizer is not None:
                    finalizer.detach()
                self._temporary = None

    def shutdown(self) -> None:
        """Stop the owned CLI and persistent VLA service at server shutdown."""
        try:
            self.close()
        finally:
            if self._vla is not None:
                self._vla.close()
                self._vla = None

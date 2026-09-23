"""EchoPolicy model adapter with the guoke2 VLM orchestration path."""
from __future__ import annotations

import asyncio
import logging
import os
import threading
from pathlib import Path
from copy import deepcopy
from typing import Any

import numpy as np

from XPolicyLab.model_template import ModelTemplate
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.base import Backend, BackendConnection, FrontendSession
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.robodojo import RoboDojoObservationAdapter
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.protocols.xpolicylab_ws import _canonical_to_xpolicylab
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.proxy import OrchestratorProxy, ProxyConfig
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.base import StrategyContext
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.strategies.subgoal import SubgoalConfig, SubgoalStrategy
from XPolicyLab.policy.EchoPolicy.vlm_orchestrator.vlm import GoogleVLM, PassthroughVLM

logger = logging.getLogger(__name__)


def _pad_value(value: Any, size: int) -> Any:
    if isinstance(value, np.ndarray):
        if value.ndim == 0 or value.shape[0] >= size:
            return value
        return np.concatenate([value, np.repeat(value[-1:], size - value.shape[0], axis=0)], axis=0)
    if isinstance(value, dict):
        return {key: _pad_value(item, size) for key, item in value.items()}
    if isinstance(value, list):
        return value + [deepcopy(value[-1]) for _ in range(size - len(value))]
    return value


def _action_count(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return 1 if value.ndim == 1 else int(value.shape[0])
    return len(value) if isinstance(value, (list, tuple)) else 0


class _QueueSession(FrontendSession):
    def __init__(self):
        self.observations: asyncio.Queue[list[dict] | None] = asyncio.Queue()
        self.responses: asyncio.Queue[tuple[list[dict], list[int]]] = asyncio.Queue()

    async def send_metadata(self, metadata: dict) -> None:
        return None

    async def recv_obs(self) -> dict | None:
        batch = await self.observations.get()
        return None if batch is None else batch[0]

    async def recv_batch(self) -> list[dict] | None:
        return await self.observations.get()

    async def send_action(self, canonical_action: dict) -> int | None:
        count = _action_count(canonical_action.get("actions"))
        await self.responses.put(([canonical_action], [count]))
        return count

    async def send_action_batch(self, canonical_actions: list[dict]) -> list[int | None]:
        counts = [_action_count(item.get("actions")) for item in canonical_actions]
        await self.responses.put((canonical_actions, counts))
        return counts


class _LocalPi05BackendConnection(BackendConnection):
    def __init__(self, owner: "Model"):
        self.owner = owner

    async def recv_metadata(self) -> dict:
        return {"action_type": self.owner.action_type, "batch_size": self.owner.batch_size}

    async def infer(self, canonical_obs: dict) -> dict:
        return (await self.infer_batch([canonical_obs]))[0]

    async def infer_batch(self, canonical_obs: list[dict]) -> list[dict]:
        return await asyncio.to_thread(self.owner._infer_candidates, canonical_obs)

    async def close(self) -> None:
        return None


class _LocalPi05Backend(Backend):
    def __init__(self, owner: "Model"):
        self.owner = owner

    async def connect(self) -> BackendConnection:
        return _LocalPi05BackendConnection(self.owner)


class Model(ModelTemplate):
    """Guoke2-compatible EchoPolicy adapter for XPolicyLab's standard server."""

    def __init__(self, model_cfg: dict[str, Any]):
        self._model_cfg = dict(model_cfg)
        self.action_type = str(model_cfg.get("action_type", "joint"))
        self.batch_size = int(model_cfg.get("pi05_batch_size", os.environ.get("ECHO_PI05_BATCH_SIZE", "160")))
        self._adapter = RoboDojoObservationAdapter()
        self._latest_layouts: dict[int, Any] = {}
        self._latest_steps: dict[int, int] = {}
        self._build_planner()

        # Keep OpenPI imports lazy so protocol tests can run on CPU-only hosts.
        from XPolicyLab.policy.Pi_05.model import Model as Pi05Model
        pi_cfg = dict(model_cfg)
        checkpoint = os.environ.get("ECHO_POLICY_CHECKPOINT_PATH")
        if not checkpoint:
            candidate = Path(__file__).resolve().parent / "checkpoints" / f"pi05_robodojo_{model_cfg.get('checkpoint_num', 59999)}"
            if candidate.exists():
                checkpoint = str(candidate)
        if checkpoint and not pi_cfg.get("model_path") and not pi_cfg.get("checkpoint_path"):
            pi_cfg["model_path"] = checkpoint
        self._pi05 = Pi05Model(pi_cfg)
        self.model = self._pi05.model

        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._session: _QueueSession | None = None
        self._worker: asyncio.Future | None = None
        self._thread = threading.Thread(target=self._loop_main, name="echopolicy-orchestrator", daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)
        self._new_session()

    def _loop_main(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _build_planner(self) -> None:
        api_key = os.environ.get("VLM_API_KEY")
        if api_key:
            vlm = GoogleVLM(
                model=os.environ.get("VLM_MODEL", "gemini-3.8-flash"),
                base_url=os.environ.get("VLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta"),
                api_key=api_key,
                thinking_level=os.environ.get("VLM_THINKING_LEVEL", "low"),
                proxy_url=os.environ.get("VLM_PROXY_URL"),
            )
        elif os.environ.get("ECHO_ALLOW_PASSTHROUGH") == "1":
            vlm = PassthroughVLM()
            logger.warning("VLM_API_KEY is unset; passthrough mode is explicitly enabled")
        else:
            raise RuntimeError("EchoPolicy requires VLM_API_KEY at runtime")
        self._strategy = SubgoalStrategy(
            StrategyContext(
                vlm=vlm,
                image_key="observation/exterior_image_1_left",
                prompt_key="prompt",
                extra_image_keys=["observation/wrist_image_left", "observation/wrist_image_right"],
            ),
            SubgoalConfig(progress_interval=int(os.environ.get("ECHO_PROGRESS_INTERVAL", "10"))),
        )

    def _new_session(self) -> None:
        self._session = _QueueSession()
        config = ProxyConfig(
            strategy=self._strategy,
            cfg_enabled=False,
            action_chunk_size=int(os.environ.get("ECHO_ACTION_CHUNK_SIZE", "10")),
            vla_candidates=int(os.environ.get("ECHO_VLA_CANDIDATES", "16")),
            vla_image_noise_std=float(os.environ.get("ECHO_VLA_IMAGE_NOISE_STD", "5.0")),
            vla_image_augmentation=os.environ.get("ECHO_VLA_IMAGE_AUGMENTATION", "1") != "0",
            vla_joint_state_noise_std=float(os.environ.get("ECHO_VLA_JOINT_NOISE_STD", "0.05")),
            batch_vlm_concurrency=0,
            global_vlm_concurrency=None,
            frontend=self._session,
            backend=_LocalPi05Backend(self),
        )
        proxy = OrchestratorProxy(config)
        self._worker = asyncio.run_coroutine_threadsafe(proxy._handle_session(self._session), self._loop)

    def _submit(self, awaitable):
        return asyncio.run_coroutine_threadsafe(awaitable, self._loop).result(timeout=1800)

    def update_obs(self, obs: dict[str, Any]) -> None:
        self.update_obs_batch([obs])

    def update_obs_batch(self, obs_list: list[dict[str, Any]]) -> None:
        canonical = []
        for index, obs in enumerate(obs_list):
            env_id = int(obs.get("env_idx", index))
            step = self._latest_steps.get(env_id, -1) + 1
            self._latest_steps[env_id] = step
            item, layout = self._adapter.to_canonical(obs, step=step, episode_marker=obs.get("episode_id", obs.get("__episode_id")))
            item["env_idx"] = env_id
            self._latest_layouts[env_id] = layout
            canonical.append(item)
        if self._session is None:
            raise RuntimeError("EchoPolicy session is closed")
        self._submit(self._session.observations.put(canonical))

    def get_action(self, **kwargs):
        return self.get_action_batch(env_idx_list=[0], **kwargs)[0]

    def get_action_batch(self, env_idx_list=None, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported get_action_batch arguments: {sorted(kwargs)}")
        if self._session is None:
            raise RuntimeError("EchoPolicy session is closed")
        ids = list(env_idx_list or sorted(self._latest_steps))
        responses, _counts = self._submit(self._session.responses.get())
        if len(responses) != len(ids):
            raise RuntimeError(f"orchestrator returned {len(responses)} actions for {len(ids)} environments")
        return [self._adapter.to_robodojo_action(response, self._latest_layouts.get(int(env_id))) for response, env_id in zip(responses, ids, strict=True)]

    def _infer_candidates(self, canonical_obs: list[dict]) -> list[dict]:
        """Run one fixed-size Pi05 batch, matching guoke2's batch=160 path."""
        from XPolicyLab.policy.Pi_05.model import encode_obs, stack_obs
        from XPolicyLab.utils.process_data import get_robot_action_dim_info, unpack_robot_state
        dim_info = get_robot_action_dim_info(self._model_cfg["env_cfg_type"])
        encoded = [encode_obs(_canonical_to_xpolicylab(obs), self.action_type, dim_info) for obs in canonical_obs]
        stacked = _pad_value(stack_obs(encoded), self.batch_size)
        raw = np.asarray(self._pi05.policy.infer(stacked)["actions"])
        if raw.shape[0] < len(encoded) or not np.isfinite(raw).all():
            raise RuntimeError(f"Pi05 returned invalid batched actions with shape {raw.shape}")
        return [{"actions": unpack_robot_state(raw[index], self.action_type, dim_info, source_type="obs")} for index in range(len(encoded))]

    def reset(self) -> None:
        if self._session is not None:
            self._submit(self._session.observations.put(None))
        if self._worker is not None:
            try:
                self._worker.result(timeout=30)
            except Exception:
                logger.exception("EchoPolicy orchestrator session failed during reset")
        self._session = None
        self._latest_layouts.clear()
        self._latest_steps.clear()
        self._new_session()
        self._pi05.reset()

    def close(self) -> None:
        if self._session is not None:
            try:
                self._submit(self._session.observations.put(None))
            except Exception:
                pass
        if self._worker is not None:
            try:
                self._worker.result(timeout=10)
            except Exception:
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        if not self._loop.is_closed():
            self._loop.close()

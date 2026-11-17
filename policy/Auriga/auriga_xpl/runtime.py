"""Long-lived Auriga runtime inside the XPolicyLab policy server.

One :class:`AurigaRuntime` per server process; :meth:`AurigaRuntime.play` runs one RoboDojo episode
end to end with Auriga's own run owners, without the web layer:

    first observation -> Workbench (Store, Harness, Codex session, benchmark prompt) over
    RemoteRobotBackend -> actor turns and benchmark continuations (wait_benchmark) -> joint rows
    through the Bridge -> RoboDojo ends the episode, or the actor stops -> the client's final check
    -> verdict recorded through trial_end -> episode summary -> every owned resource released

This mirrors ``auriga.apps.gateway.server.serve`` for ``--benchmark`` runs; the differences are the
robot backend (in-process over the XPolicyLab observation/action interface), no HTTP viewer, no
PTY terminal, and the evaluation record, which comes from the RoboDojo eval client's verdict instead
of a native evaluator call (the policy has no evaluator access).
"""

from __future__ import annotations

import asyncio
import atexit
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ADAPTER_ROOT = Path(__file__).resolve().parents[1]
SIM_DT = 0.004  # RoboDojo env_cfg/sim/sim_config.yml (the planner is built with the same dt)
API_ENV = ("AURIGA_API_BASE_URL", "AURIGA_API_KEY", "AURIGA_API_KEY_FILE")
PROVIDER_ID = "auriga_eval"


def _resolve(path: str | os.PathLike | None, default: str) -> Path:
    value = Path(path or default).expanduser()
    return value if value.is_absolute() else ADAPTER_ROOT / value


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class AurigaRuntime:
    """Process-wide state: configuration, model provider home, planners, calibration."""

    def __init__(self, cfg: dict[str, Any], *, planners: dict | None = None, calibration: dict | None = None,
                 planner_reset=None, robodojo_root: str | os.PathLike | None = None):
        self.cfg = dict(cfg or {})
        self.src = _resolve(os.environ.get("AURIGA_SRC") or None, self.cfg.get("auriga_src") or "third_party/Auriga")
        if not (self.src / "auriga" / "__init__.py").is_file():
            raise RuntimeError(f"Auriga checkout not found at {self.src}; run install.sh")
        if str(self.src) not in sys.path:
            sys.path.insert(0, str(self.src))
        # The model relay's retry policy is RoboProbe's vendored module, imported under its own
        # XPolicyLab package name; load it now, before the server threads start importing.
        from auriga.apps.reference.reference_api import upstream
        upstream()
        self._load_api_key()
        if not os.environ.get("AURIGA_API_BASE_URL") or not os.environ.get("AURIGA_API_KEY"):
            # Never print the values; only report that they are missing.
            raise RuntimeError("AURIGA_API_BASE_URL and AURIGA_API_KEY (or AURIGA_API_KEY_FILE) must be set; "
                               "the Auriga team provides them with the evaluation application")
        self.result_dir = _resolve(self.cfg.get("result_dir"), "results")
        self.result_dir.mkdir(parents=True, exist_ok=True)
        self.robodojo_root = Path(robodojo_root).resolve() if robodojo_root else None
        self.poll_s = float(self.cfg.get("poll_s") or 20.0)
        self.verdict_wait_s = float(self.cfg.get("verdict_wait_s") or 120.0)
        budget = self.cfg.get("episode_budget_s")
        self.episode_budget_s = float(budget) if budget not in (None, "", 0, "null") else None
        self.codex_source = self._write_codex_source()
        self.config = self._load_config()
        self._lock = threading.Lock()
        self.episodes = 0
        self.last_summary: dict | None = None
        # Perception services (VGGT-Omega + SAM): started here when the perception module is present,
        # and their URLs injected into the frozen configuration; otherwise the configured endpoint
        # is used as is (an already running service).
        self.perception_handle = self._start_perception()
        self.perception = self._check_perception()
        # Planners and calibration are loaded once per server, before the first episode, so the
        # cuRobo warm-up is not spent inside a trial.
        self._planners = None if planners is None else self._planner_map(planners)
        self._planner_reset = planner_reset
        self._calibration = calibration
        self.planners, self.calibration  # noqa: B018 - load now

    # -- process-wide setup ----------------------------------------------------------------------
    @staticmethod
    def _load_api_key() -> None:
        path = os.environ.get("AURIGA_API_KEY_FILE")
        if path and not os.environ.get("AURIGA_API_KEY"):
            value = Path(path).expanduser().read_text(encoding="utf-8").strip()
            if value:
                os.environ["AURIGA_API_KEY"] = value  # read by the model relay; never logged

    def _write_codex_source(self) -> Path:
        """Private Codex provider home: Responses endpoint from the environment, key by env_key only.
        One per server process: servers started in parallel from the same adapter directory share
        result_dir, and each one removes its own provider home in close()."""
        home = Path(tempfile.mkdtemp(prefix=".codex-provider-", dir=self.result_dir))  # mode 0700
        base_url = os.environ["AURIGA_API_BASE_URL"].rstrip("/")
        text = "\n".join([
            f'model_provider = {json.dumps(PROVIDER_ID)}',
            f'[model_providers.{PROVIDER_ID}]',
            f'name = {json.dumps(PROVIDER_ID)}',
            f'base_url = {json.dumps(base_url)}',
            'wire_api = "responses"',
            'env_key = "AURIGA_API_KEY"',
        ]) + "\n"
        path = home / "config.toml"
        path.write_text(text, encoding="utf-8")
        path.chmod(0o600)
        (home / "auth.json").unlink(missing_ok=True)
        return home

    def _load_config(self):
        from auriga.core.config import Config
        eval_config = _resolve(self.cfg.get("auriga_config"), "auriga_eval.json")
        model = os.environ.get("AURIGA_MODEL") or self.cfg.get("model") or None
        effort = os.environ.get("AURIGA_REASONING_EFFORT") or self.cfg.get("reasoning_effort") or None
        overrides: dict = {"model": {"codex_home": str(self.codex_source)},
                           "run": {"benchmark": True, "prompt": None, "max_actions": None, "wall_budget_s": None}}
        if model:
            overrides["model"]["name"] = model
        if effort:
            overrides["model"]["effort"] = effort
        # Layers: Auriga defaults <- configs/auriga.json of the pinned checkout <- auriga_eval.json.
        # The host environment (AURIGA_CONFIG, legacy variables) is ignored on purpose.
        config = Config.load(str(eval_config), environ={}, overrides=overrides)
        features = config.features
        invariants = {"pi05": False, "record_video": False, "export_video": False, "terminal": False}
        bad = [name for name, value in invariants.items() if getattr(features, name) != value]
        if bad or config.analysis.scene_truth_log:
            raise RuntimeError("auriga_eval.json must disable " + ",".join(bad + (["analysis.scene_truth_log"]
                               if config.analysis.scene_truth_log else [])))
        if features.motion_execution_mode != "resampled":
            raise RuntimeError("benchmark requires resampled motion execution")
        return config

    def _start_perception(self):
        if not self.config.features.external_perception:
            return None
        try:
            from auriga_xpl import perception as perception_module
        except ImportError:
            return None
        start = getattr(perception_module, "start_services", None)
        if start is None:
            return None
        handle = start(self.cfg, self.cfg.get("gpu_id"))
        atexit.register(handle.stop)
        urls = dict(getattr(handle, "urls", None) or {})
        if urls.get("vggt"):
            self.config = self.config.with_overrides({"features": {"vggt_endpoint": urls["vggt"]}})
        return handle

    def _check_perception(self) -> dict:
        features = self.config.features
        if not features.external_perception:
            return {"enabled": False}
        endpoint = str(features.vggt_endpoint).rstrip("/")
        try:
            with urlopen(endpoint + "/health", timeout=5) as response:
                healthy = 200 <= response.status < 300
        except Exception as error:  # noqa: BLE001 - recorded, not fatal
            return {"enabled": True, "endpoint": endpoint, "healthy": False, "error": type(error).__name__}
        return {"enabled": True, "endpoint": endpoint, "healthy": healthy}

    @property
    def planners(self) -> dict:
        if self._planners is None:
            from auriga_xpl import planner as planner_module
            self._planners = self._planner_map(planner_module.load_planners(SIM_DT))
            # cuRobo's seed samplers advance with every call; the self-test built a fresh planner per
            # episode, so every episode starts from the post-construction sampler state.
            self._planner_reset = getattr(planner_module, "reset_episode_state", None)
        return self._planners

    @staticmethod
    def _planner_map(planners: dict) -> dict:
        """RobotManager keys planners by robot_name; both X5 arms share one planner ("x5")."""
        if "x5" in planners:
            return {"x5": planners["x5"]}
        left, right = planners.get("left"), planners.get("right")
        if left is None or (right is not None and right is not left):
            raise RuntimeError("load_planners must return one shared planner for both X5 arms "
                               "({'x5': planner} or the same object under 'left' and 'right')")
        return {"x5": left}

    @property
    def calibration(self) -> dict:
        if self._calibration is None:
            from auriga_xpl.calibration import load_calibration
            self._calibration = load_calibration()
        return self._calibration

    # -- one episode -----------------------------------------------------------------------------
    def play(self, bridge, *, task_name: str | None, seed: Any = None) -> dict:
        """Run one episode until RoboDojo ends it or the actor stops; returns the run summary."""
        if not task_name:
            raise RuntimeError("task_name is required (deploy.yml task_name, set by eval.sh)")
        with self._lock:  # one episode at a time; the next one starts after full cleanup
            self.episodes += 1
            seq, first_obs = bridge.latest()
            if first_obs is None:
                raise RuntimeError("episode started without an observation")
            run_id = f"{_utc_stamp()}-{uuid.uuid4().hex[:8]}"
            root = self.result_dir / "runs" / str(task_name) / run_id
            memory = self.result_dir / "grasp-memory" / str(task_name) / run_id
            memory.mkdir(parents=True, exist_ok=False)  # per-episode empty shared-memory store
            try:
                seed_value = int(seed) if seed not in (None, "") else 0
            except (TypeError, ValueError):
                seed_value = 0
            config = self.config.with_overrides({
                "features": {"grasp_memory_dir": str(memory)},
                "run": {"task": str(task_name), "seed": seed_value, "layout_id": 0, "run_dir": str(root)}})
            if self._planner_reset is not None:
                self._planner_reset(self.planners)
            summary = asyncio.run(self._episode(bridge, config, root, str(task_name), seed_value))
            self.last_summary = summary
            return summary

    async def _episode(self, bridge, config, root: Path, task: str, seed: int) -> dict:
        from auriga.apps.evaluation.episode import episode_summary, record_evaluation, wait_benchmark
        from auriga.apps.gateway.workbench import DEFAULT_PROMPT, Workbench
        from auriga.apps.library.experiments import finish_run, register_run
        from auriga.apps.library.scoring import classify_episode_result
        from auriga.runtime.supervision import (INACTIVITY_REASON, model_budget_exhausted, robot_gpu_oom,
                                                 start_with_shutdown)
        from .remote_backend import RemoteRobotBackend

        started_at = time.monotonic()
        register_run(root, category="xpolicylab", task=task, title=f"{task} seed {seed}", tags=["xpolicylab"],
                     kind="workbench")
        versions = {"auriga_source": str(self.src), "xpolicylab_adapter": "policy/Auriga"}
        backend = RemoteRobotBackend(bridge, planners=self.planners, calibration=self.calibration, config=config,
                                     run_dir=root / "private" / "robot", task=task, seed=seed,
                                     robodojo_root=self.robodojo_root, versions=versions)
        workbench = Workbench(root, task=task, robot=backend, config=config)
        (root / "xpolicylab.json").write_text(json.dumps({
            "perception": self.perception, "poll_s": self.poll_s, "episode_budget_s": self.episode_budget_s,
            "episode_index_in_server": self.episodes, "planner_state_reset": self._planner_reset is not None,
            "model": {"provider": PROVIDER_ID, "wire_api": "responses", "name": config.model.name,
                      "effort": config.model.effort}}, indent=2) + "\n")
        done = asyncio.Event()
        state = {"stop_reason": None}
        final_status, infrastructure_error = "stopped", None

        async def watch():
            # Host supervision only: end the actor's wait when the client has left the trial or the
            # optional wall budget is spent. A trial that ends during an action is reported by the
            # action receipt itself (native terminal), like the worker path.
            over_since = None
            while not done.is_set():
                harness = getattr(workbench, "harness", None)
                if bridge.over and harness is not None and not harness.episode_done.is_set():
                    over_since = over_since or time.monotonic()
                    if not harness.active_action and time.monotonic() - over_since > 1.0:
                        state["stop_reason"] = "client_trial_over"
                        done.set()
                if self.episode_budget_s is not None and time.monotonic() - started_at > self.episode_budget_s:
                    state["stop_reason"] = "episode_wall_budget"
                    done.set()
                await asyncio.sleep(0.2)

        watcher = asyncio.create_task(watch())
        try:
            try:
                await start_with_shutdown(workbench, done, prompt=DEFAULT_PROMPT, benchmark=True,
                                          task_notes=None, task_notes_source=None, wall_budget_s=None)
            except asyncio.TimeoutError:
                await workbench.stop("official_startup_timeout")
                raise RuntimeError("official_startup_timeout") from None
            finished = await wait_benchmark(workbench, done)
            if not finished or workbench.has_active_programs():
                reason = ("actor_finished_with_active_programs" if finished else
                          state["stop_reason"] or "shutdown-signal" if done.is_set() else
                          getattr(workbench.harness, "wait_stop_reason", None) or "wall-budget")
                await workbench.stop(reason)
            if getattr(workbench.harness, "wait_stop_reason", None) == INACTIVITY_REASON:
                infrastructure_error = INACTIVITY_REASON
            elif robot_gpu_oom(workbench.harness):
                infrastructure_error = workbench.harness.robot.failure
            native_terminal = getattr(workbench.harness, "episode_terminal", None)
            unresolved = (workbench.current_unknown_actions() or workbench.harness.active_action
                          or workbench.has_active_programs()
                          or (not native_terminal and (workbench.session.state == "running" or workbench.status == "unknown")))
            final_status = "unknown" if unresolved else "completed" if native_terminal else "stopped"
            if unresolved:
                workbench.store.event("observer.finalization_skipped", {"reason": "motion_or_turn_stop_unconfirmed"})
            # Hand the scene to RoboDojo's own final check (empty answer) and record its verdict.
            bridge.finish()
            verdict = await asyncio.to_thread(bridge.wait_verdict, self.verdict_wait_s)
            if verdict is not None and not verdict.get("abandoned"):
                evaluation = record_evaluation(workbench.store, workbench.identity, self._evaluation(verdict))
                (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
        except Exception as error:
            final_status = "failed"
            infrastructure_error = f"{type(error).__name__}: {error}"
            traceback.print_exc()
            try:
                workbench.store.event("run.infrastructure_failed", {"reason": infrastructure_error})
            except Exception:  # noqa: BLE001
                pass
        finally:
            bridge.finish()
            done.set()
            watcher.cancel()
            await asyncio.gather(watcher, return_exceptions=True)
            summary = {"task_success": None, "normalized_score": None}
            try:
                native_turn = workbench.session.turn_result or {}
                if native_turn.get("status") == "failed":
                    expected_limit_stop = model_budget_exhausted(workbench.session)
                    if expected_limit_stop:
                        workbench.harness.wait_stop_reason = "model_request_limit"
                        workbench.store.event("codex.expected_budget_stop", {"turn_error": native_turn.get("error")})
                    elif final_status != "unknown" and not getattr(workbench.harness, "episode_terminal", None):
                        final_status = "failed"
                    if infrastructure_error is None and not expected_limit_stop:
                        error = native_turn.get("error") or {}
                        detail = error.get("message") if isinstance(error, dict) else str(error)
                        infrastructure_error = "codex_turn_failed: " + str(detail or json.dumps(error))[:4096]
                if workbench.current_unknown_actions() or workbench.harness.active_action or workbench.has_active_programs():
                    final_status = "unknown"
                summary = episode_summary(workbench.store, workbench.identity, workbench.session, workbench.harness,
                                          benchmark=True, final_status=final_status, started_at=started_at,
                                          infrastructure_error=infrastructure_error)
            except Exception as error:  # noqa: BLE001 - the cleanup below must still run
                summary["summary_error"] = f"{type(error).__name__}: {error}"
            finally:
                cleanup = await workbench.close()
                if isinstance(cleanup, dict):
                    summary["cleanup"] = cleanup
                    summary["cleanup_status"] = "failed" if cleanup.get("errors") else "completed"
                    if cleanup.get("errors"):
                        final_status = "unknown"
                summary["xpolicylab"] = {"verdict": bridge.verdict, "facade_warnings": list(
                    getattr(getattr(backend, "facade", None), "warnings", []) or []),
                    "model_api": self._api_outcome(summary)}
                summary = classify_episode_result(summary, final_status)
                summary = finish_run(root, status=final_status, summary=summary)
                summary["run_dir"] = str(root)
        return summary

    @staticmethod
    def _api_outcome(summary: dict) -> dict:
        """Classify how the model endpoint behaved in this episode (run record; no values printed)."""
        requests = summary.get("model_requests") or {}
        error = str(summary.get("infrastructure_error") or "")
        transport = requests.get("transport_error") or {}
        text = (error + " " + json.dumps(transport, default=str)).lower()
        if summary.get("model_budget_exhausted"):
            kind = "model_request_limit"
        elif any(code in text for code in ("401", "403", "unauthorized", "forbidden", "invalid api key")):
            kind = "auth_rejected"
        elif "429" in text or "rate limit" in text or "rate_limit" in text:
            kind = "rate_limited"
        elif requests.get("transport_blocked") or "model_transport_terminal_error" in text:
            kind = "transport_terminal"
        elif error.startswith("codex_turn_failed"):
            kind = "model_turn_failed"
        elif error:
            kind = "infrastructure_other"
        else:
            kind = "ok"
        return {"classification": kind, "completed_calls": requests.get("completed_calls"),
                "http_attempts": requests.get("http_attempt_count"), "limit": requests.get("limit"),
                "transport_recovery": requests.get("transport_recovery")}

    @staticmethod
    def _evaluation(verdict: dict) -> dict:
        success = verdict.get("success")
        return {"task_success": success, "normalized_score": 1.0 if success else None,
                "native_step": verdict.get("native_step"), "audience": "observer_only",
                "authority": "RoboDojo eval client final check (XPolicyLab trial_end)",
                "success_authority": "xpolicylab_trial_end"}

    def close(self) -> None:
        """Stop owned perception services and drop the provider home; planners die with the process."""
        if self.perception_handle is not None:
            self.perception_handle.stop()
        shutil.rmtree(self.codex_source, ignore_errors=True)

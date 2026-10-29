"""What an exported XPolicyLab policy runs: policy/<Name>/model.py's Model delegates here.

The package folder (written by `harness export xpolicylab`) holds:

    agent.json       the frozen agent: what it was resolved from (agent and model
                     trees), the embodiment, the services it uses, the lock and its
                     identity, the code fingerprint
    deploy.toml      where its weights are (checkpoints/, download_checkpoints.sh), and
                     optionally how a service runs here ([services.<name>] as in a
                     deploy file: an external URL, another python)
    the-harness/     this repository at the exported commit (install.sh)

At start the agent is resolved again, strictly, against a deploy of the GPUs the
policy server sees, and must have the identity it was exported with: whatever GPUs the
evaluator has, the policy that runs is the one that was submitted. Every policy server
is an instance of its own (its own ports, state and traces), so evaluators may run
several tasks at once on one machine. Services that do not all fit the free memory of
the visible GPUs run on demand in one swap group, one model loaded at a time:
policies and world models first, then every model but the kinematics (the simulator
waits while a model loads; only wall time changes). Services the instance started
stop with it, also when it is killed.
"""

import atexit
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

from ..embodiments import load_embodiment
from .xpolicylab import Bridge


# When the services do not all fit the visible GPUs at once, the kinds that load on
# demand (one swap group, one model loaded at a time), level by level: first the
# learned policies and world models, then every model but the kinematics, which
# every tick of a motion needs.
SWAP_LEVELS = ((), ("policy", "world_model"), ("policy", "world_model", "pointer", "segmenter", "vlm", "llm"))


def relative_tree(value, root: Path):
    """Absolute paths inside the repository written as "@HARNESS@/<relative>" (agent.json
    is the same on every machine)."""
    if isinstance(value, str) and value.startswith(str(root)):
        return "@HARNESS@/" + Path(value).relative_to(root).as_posix()
    if isinstance(value, list):
        return [relative_tree(item, root) for item in value]
    if isinstance(value, dict):
        return {key: relative_tree(item, root) for key, item in value.items()}
    return value


def relocate(value, root: Path):
    """The inverse of relative_tree for a checkout at `root`."""
    if isinstance(value, str) and value.startswith("@HARNESS@/"):
        return str(Path(root) / value[len("@HARNESS@/"):])
    if isinstance(value, list):
        return [relocate(item, root) for item in value]
    if isinstance(value, dict):
        return {key: relocate(item, root) for key, item in value.items()}
    return value


def visible_gpus() -> list[int]:
    """GPUs of this process (CUDA_VISIBLE_DEVICES, as XPolicyLab's setup script sets it
    from policy_gpu_id), else every GPU nvidia-smi lists."""
    value = os.environ.get("CUDA_VISIBLE_DEVICES")
    if value is not None and value.strip():
        return [int(item) for item in value.split(",") if item.strip().isdigit()]
    from ..services.tools import gpus

    return [gpu["index"] for gpu in gpus() or []]


def free_port() -> int:
    """A TCP port on 127.0.0.1 that is free now (the system picks it)."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _toml(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{json.dumps(key)} = {_toml(item)}" for key, item in value.items()) + " }"
    raise TypeError(f"no TOML for {value!r}")


def package_deploy(package: Path, services: list[str], instance: Path, log=print, gpu_rows=None):
    """The deploy of this instance, written to instance/deploy.toml: the services the
    agent uses, on the visible GPUs (by their free memory now), on ports of their own;
    all resident when they fit, else on demand in one swap group by SWAP_LEVELS. The
    package's deploy.toml gives the weights and per-service settings (a service it sets
    the mode of keeps it)."""
    from ..config import read_toml
    from ..services.deploy import LOCAL, load_deploy
    from ..services.registry import load_services
    from ..services.tools import gpus

    template = read_toml(package / "deploy.toml")
    unknown = set(template) - {"weights", "robodojo", "services"}
    if unknown:
        raise ValueError(f"{package / 'deploy.toml'}: unknown keys {sorted(unknown)} (weights, robodojo, services)")
    registry = load_services()
    entries = {}
    for name in services:
        entry = dict(template.get("services", {}).get(name, {}))
        entry.setdefault("mode", "resident")
        if entry["mode"] in LOCAL:
            entry.setdefault("gpu", "auto")
            entry.setdefault("port", free_port())
        entries[name] = entry
    devices = visible_gpus()
    needs_gpu = [name for name, entry in entries.items() if entry["mode"] in LOCAL and registry[name].vram_gb > 0]
    if needs_gpu and not devices:
        raise RuntimeError(f"No GPU is visible to the policy server, and {needs_gpu} need one")
    rows = {row["index"]: row for row in (gpu_rows if gpu_rows is not None else gpus()) or []}
    free = [rows[d]["total_gb"] - rows[d]["used_gb"] for d in devices if d in rows]
    memory = round(min(free), 1) if free else 24.0
    root = package.parents[2]
    head = {"weights": str((package / template.get("weights", "checkpoints")).resolve()),
            "state": str(instance / "services"), "robodojo": str((package / template["robodojo"]).resolve())
            if "robodojo" in template else str(root), "xpolicylab": str(package.parents[1]),
            "gpus": devices, "gpu_memory_gb": memory, "broker_port": free_port()}
    path = instance / "deploy.toml"
    instance.mkdir(parents=True, exist_ok=True)

    def write(kinds):
        lines = [f"# Written by harness_policy.connectors.package for {package.name} (free GPU memory at start)\n"]
        lines += [f"{key} = {_toml(value)}\n" for key, value in head.items()]
        for name, entry in entries.items():
            entry = dict(entry)
            if entry["mode"] == "resident" and registry[name].kind in kinds and \
                    "mode" not in template.get("services", {}).get(name, {}):
                entry["mode"], entry["swap"] = "on_demand", "models"
            lines.append(f"\n[services.{json.dumps(name)}]\n")
            lines += [f"{key} = {_toml(value)}\n" for key, value in entry.items()]
        path.write_text("".join(lines))
        return load_deploy(path)

    for level, kinds in enumerate(SWAP_LEVELS):
        try:
            deploy = write(kinds)
        except ValueError as error:
            if level + 1 == len(SWAP_LEVELS):
                raise ValueError(f"The services {services} do not fit GPUs {devices} ({memory:g} GB free each) even "
                                 f"one model at a time: {error}") from None
            continue
        if kinds:
            log(f"harness: the services do not all fit GPUs {devices} ({memory:g} GB free each) at once; "
                f"{', '.join(kinds)} services load on demand, one at a time")
        return deploy


def reap(parent: int, deploy_path: str):
    """Stop an instance's services once its policy server is gone (also when it was
    killed without a chance to stop them)."""
    from ..services import supervisor
    from ..services.deploy import load_deploy

    while True:
        try:
            os.kill(parent, 0)
        except ProcessLookupError:
            break
        except PermissionError:
            pass
        time.sleep(2.0)
    supervisor.down(load_deploy(deploy_path), log=lambda *a: None)


class _Slot:
    """One environment: its bridge, the worker server the episode's kernel calls, and
    which reset wave has already started a driver."""

    def __init__(self, bridge, server, endpoint):
        self.bridge = bridge
        self.server = server
        self.endpoint = endpoint
        self.started_wave = 0


class HarnessModel:
    """The XPolicyLab Model of an exported agent."""

    def __init__(self, model_cfg: dict, package_dir, log=print):
        from ..config.agent import code_fingerprint, resolve_agent
        from ..services import supervisor
        from ..services.deploy import REPO, client_config

        self.log = log
        self.package = Path(package_dir).resolve()
        self.cfg = dict(model_cfg)
        document = json.loads((self.package / "agent.json").read_text())
        # The task name only labels traces: the policy runs the same for every task (the
        # official client does not tell it the task).
        task = str(self.cfg.get("task_name") or "")
        self.task = None if task in ("", "any", "None", "null") else task
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.output = Path(self.cfg.get("result_dir") or self.package / "results") / \
            f"{self.task or 'task'}-{stamp}-{os.getpid()}"
        self.output.mkdir(parents=True, exist_ok=True)
        # The agent first: a policy that is not the submitted one starts nothing.
        sources = relocate(document["sources"], REPO)
        services = list(document["services"])
        self.deploy = package_deploy(self.package, services, self.output, log=log)
        self.agent = resolve_agent(sources["agent"], model=sources["model"], deploy=self.deploy, strict=True,
                                   name=document["name"], runtime_overrides=sources["runtime_overrides"])
        if self.agent.identity != document["identity"]:
            raise RuntimeError(f"This machine resolves the agent to identity {self.agent.identity}, not the exported "
                               f"{document['identity']}: refusing to run a different configuration")
        hosted = [config for config in (self.agent.main, self.agent.runtime.get("vlm"),
                                         self.agent.runtime.get("pointer"))
                  if isinstance(config, dict) and config.get("backend", "openrouter") == "openrouter"]
        missing = sorted({config.get("api_key_env") or "OPENROUTER_API_KEY" for config in hosted}
                         - {key for key, value in os.environ.items() if value})
        if missing:
            raise RuntimeError(f"Set {', '.join(missing)} in the policy server's environment: the agent's hosted "
                               "models need it")
        code = code_fingerprint()
        if code["sha256"] != document.get("code", {}).get("sha256"):
            log(f"harness: WARNING the code differs from the exported commit {document.get('commit', '?')[:12]} "
                f"({code['files']} files, sha256 {code['sha256'][:12]}, exported {document['code']['sha256'][:12]})")
        (self.output / "agent.lock.json").write_text(json.dumps({
            "identity": self.agent.identity, "lock": self.agent.lock, "code": code,
            "commit": document.get("commit"), "deploy": str(self.deploy.path)}, indent=1))
        self._closed = False
        self._started = [name for name in services if self.deploy.placement(name).mode == "resident"]
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, self._terminate)
        atexit.register(self.close)
        self.reaper = subprocess.Popen([sys.executable, "-c", "import sys; from harness_policy.connectors.package import reap; "
                          "reap(int(sys.argv[1]), sys.argv[2])", str(os.getpid()), str(self.deploy.path)],
                         cwd=str(REPO), start_new_session=True, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO),
                                                                           os.environ.get("PYTHONPATH", "")])})
        supervisor.up(self.deploy, self._started, log=log)
        embodiment = load_embodiment(document["embodiment"])
        kinematics = client_config(self.deploy, "kinematics")
        self._embodiment = embodiment
        self._kinematics_url = kinematics["url"] if kinematics else None
        self.poll_seconds = float(self.cfg.get("poll_seconds") or 90)
        self.slots = {}
        self.wave = 0
        slot = self._make_slot()
        slot.started_wave = 0
        self.slots[0] = slot
        # The single-environment calls use slot 0. These names are what close() stops.
        self.bridge = slot.bridge
        self.server = slot.server
        self.endpoint = slot.endpoint
        self.script = None
        if os.environ.get("HARNESS_SCRIPT"):
            self.script = json.loads(Path(os.environ["HARNESS_SCRIPT"]).read_text())
        self.episodes = 0
        self._pending_obs = None
        self.threads = []
        log(f"harness: agent {self.agent.name} {self.agent.identity} ready for {self.task or 'any task'}; "
            f"traces in {self.output}")

    # -- XPolicyLab's Model interface ---------------------------------------------

    def reset(self):
        """A new wave of episodes. Each slot's program ends; the next observation starts
        the next one. One reset covers every environment in a batch. Returns at once."""
        self.wave += 1
        self._pending_obs = None
        for slot in self.slots.values():
            slot.bridge.host_reset()
            slot.started_wave = self.wave - 1

    def update_obs(self, obs):
        """Hand the client's observation to environment 0; the first one after a reset
        starts that episode's driver."""
        self._deliver(0, obs)

    def update_obs_batch(self, obs_list):
        """Remember one observation per running environment. The following
        ``get_action_batch`` says which environment each one belongs to."""
        self._pending_obs = list(obs_list)

    def get_action(self):
        """A chunk of at most one action: the one a tick asked for, a hold after the program
        ended, or [] when none is ready within poll_seconds (Bridge.host_get_action)."""
        return self.bridge.host_get_action()

    def get_action_batch(self, env_idx_list=None):
        """One chunk per environment, each of length 0 or 1.

        The official client's batch loop takes ``len(actions[0])`` as the chunk length
        for every environment, so a short chunk would index off the others. While any
        environment is still computing, every chunk is empty and the simulator does not
        step. A ready tick is left waiting for the next call.
        """
        obs_list = self._pending_obs
        if obs_list is None:
            raise RuntimeError("get_action_batch without update_obs_batch")
        indices = list(range(len(obs_list))) if env_idx_list is None else [int(i) for i in env_idx_list]
        if len(obs_list) != len(indices):
            raise RuntimeError(f"update_obs_batch got {len(obs_list)} observations for {len(indices)} environments")
        slots = [self._slot(index) for index in indices]
        # An action already waiting was computed from the previous observation. Take it
        # before publishing this one, so the tick sees this observation as the next one.
        pending = []
        for slot, obs in zip(slots, obs_list):
            if slot.bridge.host_status() == "action":
                pending.append((slot, obs))
            else:
                self._deliver_slot(slot, obs)
                pending.append((slot, None))
        self._pending_obs = None
        deadline = time.monotonic() + self.poll_seconds
        while any(slot.bridge.host_status() == "wait" for slot, _ in pending):
            if time.monotonic() >= deadline:
                return [[] for _ in slots]
            time.sleep(0.005)
        actions = [slot.bridge.host_get_action() for slot, _ in pending]
        for slot, obs in pending:
            if obs is not None:
                slot.bridge.host_update_obs(obs)
        return actions

    def close(self):
        """Shut the instance down once: close the bridge, let running episodes write their
        traces (up to 60 s each), then stop the worker server, the instance's services and
        the reaper."""
        if self._closed:
            return
        self._closed = True
        from ..services import supervisor

        for slot in self.slots.values():
            slot.bridge.close()
        # Their ticks now report the episode over: let the drivers write their traces
        # and summaries before the worker endpoint and the services go.
        for thread in self.threads:
            thread.join(timeout=60)
        for slot in self.slots.values():
            slot.server.shutdown()
        supervisor.down(self.deploy, log=self.log)
        self.reaper.terminate()
        try:
            self.reaper.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.reaper.kill()

    def _terminate(self, signum, frame):
        self.close()
        raise SystemExit(128 + signum)

    # -- one episode ------------------------------------------------------------------

    def _slot(self, index):
        index = int(index)
        slot = self.slots.get(index)
        if slot is None:
            slot = self._make_slot()
            slot.started_wave = self.wave - 1
            self.slots[index] = slot
        return slot

    def _make_slot(self):
        from ..core.worker import create_server
        from .xpolicylab import Bridge

        bridge = Bridge(self._embodiment, kinematics_url=self._kinematics_url, poll_seconds=self.poll_seconds)
        server = create_server(bridge, "127.0.0.1", 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        return _Slot(bridge, server, f"http://127.0.0.1:{server.server_port}")

    def _deliver(self, index, obs):
        self._deliver_slot(self._slot(index), obs)

    def _deliver_slot(self, slot, obs):
        slot.bridge.host_update_obs(obs)
        if slot.started_wave != self.wave:
            slot.started_wave = self.wave
            self.episodes += 1
            host_episode = slot.bridge.episode
            thread = threading.Thread(target=self._episode, args=(self.episodes, slot, host_episode), daemon=True)
            self.threads = [item for item in self.threads if item.is_alive()] + [thread]
            thread.start()

    def _episode(self, number, slot, host_episode):
        from ..runtime.chat import scripted_models
        from ..runtime.driver import TaskConfig, run_program_episode

        agent = self.agent
        task = self.task or "task"
        try:
            config = TaskConfig(
                path=self.package / "agent.json", environment={"endpoint": slot.endpoint, "timeout_seconds": 600},
                main=dict(agent.main), runtime=dict(agent.runtime), system_prompt=agent.system_prompt, task=task,
                seeds=[number], max_model_calls=agent.max_model_calls,
                max_wall_seconds=float(self.cfg.get("max_wall_seconds") or 5400), verifier=dict(agent.verifier),
                base_prompt=agent.base_prompt, task_description="",
                lock=agent.lock, agent_id=agent.identity, placement={"deploy": "xpolicylab", **agent.placement})
            scripted = {} if self.script is None else scripted_models(self.script)
            summary = run_program_episode(config, number, self.output / f"episode-{number:03d}", **scripted)
            self.log(f"harness: episode {number} {summary['status']} calls={summary['model_calls']} "
                     f"steps={slot.bridge.step} cost=${summary['cost_usd']:.3f}"
                     + (f" ({summary['error']})" if summary.get("error") else ""))
        except Exception as error:  # the host keeps running; the arms hold
            self.log(f"harness: episode {number} failed: {type(error).__name__}: {error}")
        finally:
            slot.bridge.program_over(host_episode)

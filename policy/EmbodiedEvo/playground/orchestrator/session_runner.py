"""Isolated workspace sessions: one simulator and one agent per trial."""

from __future__ import annotations

import json
import hashlib
import os
import select
import shutil
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from orchestrator import launch_agent
from orchestrator.agent_recovery import recovery_reason, resume_argv, rollout_state
from orchestrator.config import CODE_ROOT, REPO, SESSION_PROTOCOL, config_dict
from orchestrator.files import (
    codex_token_usage,
    hashes,
    run_lock,
    test_seeds,
    shared_test_seeds,
    test_scene_records,
    valid_test_seeds,
    write_json,
)
from orchestrator.replay import run_replay
from orchestrator.resources import GpuSlots, gpu_lease, sim_env
from orchestrator.scoring import aggregate as aggregate_scores
from orchestrator.scoring import result_score


def validate(cfg):
    if cfg.protocol != SESSION_PROTOCOL or cfg.simulator.backend not in {
        "robodojo",
    }:
        raise ValueError("Workspace sessions require a supported native simulator task")
    if cfg.agent.isolation != "docker":
        raise ValueError("Workspace sessions require Docker isolation")
    if cfg.baseline != (cfg.playground is None):
        raise ValueError("Only baseline tasks may omit Playground")
    if not cfg.expose_success_feedback:
        raise ValueError("Session instructions and native primitives expose official success")
    limits = [cfg.test.budget, cfg.test.scenes, cfg.test.submission_timeout_sec, cfg.feedback_fps]
    if cfg.playground is not None:
        if not cfg.playground.expose_success_feedback:
            raise ValueError("Session instructions and native primitives expose official success")
        limits += [cfg.playground.budget, cfg.playground.submission_timeout_sec]
    if min(limits) <= 0:
        raise ValueError("Budgets, scene count and feedback FPS must be positive")


def definition(cfg):
    # Never open the sealed Test seed list here.
    return {"config": config_dict(cfg)}


def prepare_inputs(cfg, run):
    """Pin the handwritten workspace and optionally append its task recipe."""
    from simulators.primitives import PROGRAM_PRIMITIVES, load_surface

    inputs = run / "inputs"
    source = cfg.workspace
    if not source.is_dir():
        raise ValueError(f"Missing formal workspace source: {source}")
    hashes(source)  # Reject links before copying any part of the workspace.
    if cfg.baseline:
        for component in ("skills", "lessons", "playground.md", "test.md"):
            if (source / component).exists():
                raise ValueError(f"Baseline workspace cannot contain {component}")
    instructions = ("instruction.md",) if cfg.baseline else ("playground.md", "test.md")
    for name in instructions:
        template = source / name
        if not template.is_file():
            raise ValueError(f"Missing workspace instruction template: {template}")
    if not (source / "submission/solution.py").is_file():
        raise ValueError(f"Missing workspace submission: {source / 'submission/solution.py'}")
    names = set(load_surface(cfg.simulator.backend)(None).functions())
    if names != PROGRAM_PRIMITIVES:
        raise ValueError(f"Program primitives must be {sorted(PROGRAM_PRIMITIVES)}")
    if not cfg.api_doc.is_file():
        raise ValueError(f"Missing primitives manual: {cfg.api_doc}")
    manual = cfg.api_doc.read_text()
    # Baseline manuals describe the primitives available during Test.
    documented = PROGRAM_PRIMITIVES - {"reset"} if cfg.baseline else PROGRAM_PRIMITIVES
    missing = [name for name in sorted(documented) if f"{name}(" not in manual]
    if missing:
        raise ValueError(f"primitives.md must document {missing}")
    for component in ("primitives", "scripts"):
        source_component = source / component
        if not source_component.is_dir():
            raise ValueError(f"Missing workspace component: {source_component}")
    shutil.copytree(source, inputs)
    if cfg.simulator.backend == "robodojo":
        from orchestrator.recipes import append_recipe

        write_json(run / "recipe.json", append_recipe(inputs, cfg.simulator.env_task))
    write_json(run / "resolved.json", config_dict(cfg))
    return hashes(inputs)


def create_workspace(cfg, run, inputs, *, phase, harness=None, submission=None):
    """Copy pinned source, selecting the phase instruction and frozen knowledge."""
    if phase not in {"playground", "test"}:
        raise ValueError(f"Unknown workspace phase: {phase}")
    if cfg.baseline and (phase != "test" or harness is not None or submission is not None):
        raise ValueError(
            "Baseline requires fresh Test workspaces without inherited code or knowledge"
        )
    if harness is not None and phase != "test":
        raise ValueError("Frozen knowledge can only be applied in Test")
    workspace = run / "workspace"
    instruction = inputs / ("instruction.md" if cfg.baseline else f"{phase}.md")
    for required in (instruction, submission or inputs / "submission/solution.py"):
        if not Path(required).is_file():
            raise ValueError(f"Missing workspace source file: {required}; use a new --run-dir")
    # Phase templates stay in the host snapshot. The agent sees only its selected
    # instruction.md; helpers with similar names in subdirectories stay intact.
    templates = {"playground.md", "test.md", "instruction.md"}

    def ignore_templates(directory, names):
        return templates.intersection(names) if Path(directory) == inputs else set()

    shutil.copytree(inputs, workspace, ignore=ignore_templates)
    # Empty runtime directories need not be present in a handwritten source tree.
    components = ("scratch", "observations")
    if not cfg.baseline:
        components += ("skills", "lessons")
    for component in components:
        (workspace / component).mkdir(exist_ok=True)
    if harness is not None:
        for component in ("skills", "lessons"):
            shutil.rmtree(workspace / component)
            shutil.copytree(harness / component, workspace / component)
    if submission is not None:
        shutil.copyfile(submission, workspace / "submission/solution.py")
    shutil.copyfile(instruction, workspace / "instruction.md")
    return workspace


def agent_command(cfg, run, socket_dir, phase):
    launch_agent.require_image(cfg.agent.image)
    if cfg.agent.network != "none":
        launch_agent.ensure_network()
    name = f"embodiedevo-session-{os.getpid()}-{time.time_ns()}"
    argv = launch_agent.docker_argv(
        run / "workspace",
        # Keep the CLI's cache/database HOME outside the run. Only the mirrored
        # rollout JSONL belongs in the reviewable output directory.
        launch_agent.stage_home(socket_dir / "agent-home"),
        launch_agent.adapt_for_docker(
            launch_agent.agent_argv(
                cfg.agent.preset,
                "Read instruction.md. "
                "Work autonomously toward the phase goal. "
                "Interact using scripts/env.py; finish the session when done. "
                "Camera frames in observations/ are native-resolution PNGs. Use "
                "the current_cam_*.png files; never open raw_cam_* files or "
                "create copies in scratch, and inspect no more than two current "
                "frames per iteration to keep model requests focused.",
                model=cfg.agent.model,
                effort=cfg.agent.reasoning_effort,
            ),
            preset=cfg.agent.preset,
        ),
        image=cfg.agent.image,
        network=cfg.agent.network,
        cpus=cfg.agent.cpus,
        memory_mb=cfg.agent.memory_mb,
        container_name=name,
    )
    mounts = [
        "--env",
        "EMBODIEDEVO_SOCKET=/run/embodiedevo/session.sock",
        "--mount",
        f"type=bind,src={socket_dir},dst=/run/embodiedevo,readonly",
    ]
    fixed = ["instruction.md", "primitives", "scripts", "observations"]
    if phase == "test" and not cfg.baseline:
        fixed += ["skills", "lessons"]
    if (run / "program-records.json").exists():
        from orchestrator.program_records import validate_bundle

        if phase != "test" or cfg.baseline:
            raise ValueError("Program records belong only to condition C Test")
        reference = json.loads((run / "program-records.json").read_text())
        bundle = validate_bundle(reference, full=True)
        (run / "workspace/program_records").mkdir(exist_ok=True)
        mounts += ["--mount", f"type=bind,src={bundle},dst=/workspace/program_records,readonly"]
    for component in fixed:
        mounts += [
            "--mount",
            f"type=bind,src={run / 'workspace' / component},dst=/workspace/{component},readonly",
        ]
    argv[argv.index(cfg.agent.image) : argv.index(cfg.agent.image)] = mounts
    return name, argv


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run_episode(
    cfg,
    run,
    experiment,
    *,
    phase,
    seed,
    seconds,
    gpu=None,
    harness=None,
    submission=None,
):
    """Monitor processes and pipe events, never workspace semaphore files."""
    protocol_path = experiment / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    profile_path = os.environ.get("EMBODIEDEVO_ROBODOJO_RUNTIME_PROFILE")
    profile = (json.loads(Path(profile_path).read_text())
               if profile_path and cfg.simulator.backend == "robodojo" else {})
    if protocol.get("execution_backend") == "xpolicylab" and (
        cfg.simulator.backend != "robodojo" or not profile.get("sif")
    ):
        raise ValueError("External Test requires the XPolicyLab adapter runtime; no direct simulator fallback")
    if cfg.simulator.backend == "robodojo" and profile_path:
        if profile.get("sif"):
            from integrations.robodojo_policy.execution import run_episode as official_episode

            return official_episode(cfg, run, experiment, phase=phase, seed=seed, seconds=seconds,
                                    gpu=gpu, harness=harness, submission=submission)
    create_workspace(
        cfg,
        run,
        experiment / "inputs",
        phase=phase,
        harness=harness,
        submission=submission,
    )
    budget = cfg.playground.budget if phase == "playground" else cfg.test.budget
    execution_seconds = (
        cfg.playground.submission_timeout_sec
        if phase == "playground"
        else cfg.test.submission_timeout_sec
    )
    agent = simulator = None
    container = None
    agent_rc = 1
    failure = None
    stage_started_wall = time.time()
    stage_started_mono = time.monotonic()
    simulator_started_wall = simulator_started_mono = None
    simulator_ready_wall = simulator_ready_mono = None
    simulator_finished_wall = simulator_finished_mono = None
    agent_started_wall = agent_started_mono = None
    agent_finished_wall = agent_finished_mono = None
    token_usage = {}
    recovery_policy = json.loads(os.environ.get("EMBODIEDEVO_AGENT_RECOVERY", "{}"))
    recovery_events = []
    recovery_due = None
    recovery_id = None
    last_agent_progress = time.monotonic()
    runtime_path = run / "runtime.json"
    # A campaign may evaluate multiple settings with the same frozen knowledge.
    # Pin the effective episode configuration for both execution and archival replay.
    write_json(run / "episode-config.json", config_dict(cfg))
    runtime = {
        "run_id": experiment.name,
        "episode_id": str(run.relative_to(experiment)),
        "phase": phase,
        "backend": cfg.simulator.backend,
        "env_task": cfg.simulator.env_task,
        "gpu": str(gpu) if gpu is not None else os.environ.get("CUDA_VISIBLE_DEVICES"),
        "supervisor_pid": os.getpid(),
    }

    def save_runtime(**updates):
        """Publish non-sensitive host process metadata for the monitor."""
        runtime.update(updates)
        write_json(runtime_path, runtime)

    save_runtime(status="starting", started_at=stage_started_wall)
    with tempfile.TemporaryDirectory(prefix="embodiedevo-session-") as temporary, gpu_lease(gpu):
        socket_dir = Path(temporary)
        # Coverage workers may place temporary agent homes on the shared
        # filesystem so a full host /tmp cannot stop a campaign.  Docker's
        # daemon is root-squashed on that NFS mount and cannot traverse a
        # default 0700 TemporaryDirectory; make only the parent mount point
        # traversable while staged auth/config files remain 0600.
        socket_dir.chmod(0o755)
        agent_home = socket_dir / "agent-home"
        agent_session = run / "agent/session.jsonl"
        session_source = None
        session_offset = 0

        def mirror_agent_session():
            """Expose the single rollout JSONL while the container is running."""
            nonlocal session_source, session_offset, last_agent_progress
            candidates = sorted((agent_home / ".codex" / "sessions").rglob("*.jsonl"))
            if not candidates:
                return
            source = candidates[0]
            if source != session_source:
                session_source = source
                session_offset = 0
                agent_session.unlink(missing_ok=True)
            try:
                with source.open("rb") as stream:
                    stream.seek(session_offset)
                    chunk = stream.read()
                if chunk:
                    agent_session.parent.mkdir(parents=True, exist_ok=True)
                    with agent_session.open("ab") as destination:
                        destination.write(chunk)
                    session_offset += len(chunk)
                    last_agent_progress = time.monotonic()
            except OSError:
                # The CLI can rotate/create the rollout while it starts.
                return

        read_fd, write_fd = os.pipe()
        os.set_blocking(read_fd, False)
        events = bytearray()
        ready = False
        action_deadline = None
        startup_deadline = time.monotonic() + float(
            os.environ.get("EMBODIEDEVO_SIMULATOR_STARTUP_TIMEOUT", "1200"))
        agent_deadline = None
        command = [
            str(cfg.python),
            "-m",
            "orchestrator.session",
            "--config",
            str(run / "episode-config.json"),
            "--run-dir",
            str(run),
            "--socket",
            str(socket_dir / "session.sock"),
            "--phase",
            phase,
            "--seed",
            str(seed),
            "--budget",
            str(budget),
        ]
        if seconds is not None:
            command += ["--seconds", str(seconds)]
        command += [
            "--execution-seconds",
            str(execution_seconds),
            "--ready-fd",
            str(write_fd),
        ]
        try:
            container, argv = agent_command(cfg, run, socket_dir, phase)
            simulator = subprocess.Popen(
                command,
                cwd=CODE_ROOT,
                env=sim_env(gpu, cfg.simulator.backend),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                pass_fds=(write_fd,),
                start_new_session=True,
            )
            simulator_started_wall = time.time()
            simulator_started_mono = time.monotonic()
            save_runtime(status="simulator_starting", simulator_pid=simulator.pid)
            os.close(write_fd)
            write_fd = None
            while True:
                mirror_agent_session()
                readable, _, _ = select.select([read_fd], [], [], 0.2)
                if readable:
                    chunk = os.read(read_fd, 65536)
                    events.extend(chunk)
                    while b"\n" in events:
                        line, _, rest = events.partition(b"\n")
                        events = bytearray(rest)
                        event = json.loads(line)
                        if event["event"] == "ready":
                            ready = True
                            simulator_ready_wall = time.time()
                            simulator_ready_mono = time.monotonic()
                            agent_deadline = None if seconds is None else time.monotonic() + seconds
                            agent = subprocess.Popen(
                                argv,
                                cwd=REPO,
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                                start_new_session=True,
                            )
                            agent_started_wall = time.time()
                            agent_started_mono = time.monotonic()
                            last_agent_progress = agent_started_mono
                            save_runtime(
                                status="running",
                                simulator_pid=simulator.pid,
                                agent_pid=agent.pid,
                                container=container,
                                ready_at=simulator_ready_wall,
                            )
                        elif event["event"] == "executing":
                            # Also interrupts native calls that cannot deliver a Python alarm.
                            action_deadline = time.monotonic() + event["timeout"] + 10
                        elif event["event"] == "idle":
                            action_deadline = None
                            last_agent_progress = time.monotonic()
                    if not chunk:
                        time.sleep(0.1)
                now = time.monotonic()
                if not ready and (simulator.poll() is not None or now >= startup_deadline):
                    failure = "simulator_initialization_failed"
                    break
                if ready and simulator.poll() is not None:
                    terminal_path = run / "session-result.json"
                    terminal = (
                        json.loads(terminal_path.read_text()) if terminal_path.exists() else {}
                    )
                    if (
                        simulator.returncode != 0
                        or not terminal.get("closed")
                        or not terminal.get("valid")
                    ):
                        failure = "simulator_failed"
                        break
                if action_deadline and now >= action_deadline:
                    failure = "execution_timeout"
                    break
                if agent_deadline and now >= agent_deadline:
                    agent_rc = 124
                    if action_deadline is not None:
                        failure = "time_limit_during_execution"
                    break
                if recovery_due is not None:
                    # A submission can finish while the old CLI is stopping.
                    # Wait for the host's idle event before reconnecting.
                    if now >= recovery_due and action_deadline is None:
                        mirror_agent_session()
                        agent = subprocess.Popen(
                            resume_argv(argv, recovery_id), cwd=REPO,
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True,
                        )
                        last_agent_progress = now
                        recovery_due = None
                        save_runtime(status="running", agent_pid=agent.pid,
                                     agent_recoveries=len(recovery_events))
                    continue
                if (
                    recovery_policy and cfg.agent.preset == "codex"
                    and agent is not None and action_deadline is None
                    and simulator.poll() is None
                    and (agent.poll() not in (None, 0) or now - last_agent_progress >= float(
                        recovery_policy.get("idle_seconds", 900)
                    ))
                    and len(recovery_events) < int(recovery_policy.get("max_restarts", 3))
                ):
                    # The process can exit while select() waits; include its
                    # terminal transport error before deciding to finalize.
                    mirror_agent_session()
                    session_path = run / "session-result.json"
                    state = json.loads(session_path.read_text()) if session_path.exists() else {}
                    session_id, transport_error = rollout_state(agent_session)
                    reason = recovery_reason(
                        state=state, returncode=agent.poll(),
                        idle_seconds=now - last_agent_progress,
                        idle_limit=float(recovery_policy.get("idle_seconds", 900)),
                        error=transport_error,
                    )
                    if session_id and reason:
                        subprocess.run(
                            ["docker", "rm", "-f", container],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=30, check=False,
                        )
                        stop_process(agent)
                        mirror_agent_session()
                        recovery_id = session_id
                        recovery_due = time.monotonic() + float(
                            recovery_policy.get("backoff_seconds", 30)
                        )
                        recovery_events.append({
                            "at": time.time(), "reason": reason,
                            "session_id": session_id,
                            "executions_used": state.get("executions_used"),
                        })
                        write_json(run / "agent/recovery.json", recovery_events)
                        save_runtime(
                            status="agent_reconnecting", agent_recoveries=len(recovery_events)
                        )
                        continue
                if agent is not None and agent.poll() is not None:
                    agent_rc = agent.returncode
                    if agent_finished_mono is None:
                        agent_finished_wall = time.time()
                        agent_finished_mono = time.monotonic()
                    # Wait for a submitted action even if the CLI disconnects/exits.
                    if action_deadline is None:
                        break
                mirror_agent_session()
        except Exception as exc:
            failure = f"launch_failed: {exc}"
        finally:
            if container:
                try:
                    subprocess.run(
                        ["docker", "rm", "-f", container],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=30,
                        check=False,
                    )
                except (OSError, subprocess.TimeoutExpired) as exc:
                    failure = failure or f"container_cleanup_failed: {exc}"
            stop_process(agent)
            stop_process(simulator)
            if agent_started_mono is not None and agent_finished_mono is None:
                agent_finished_wall = time.time()
                agent_finished_mono = time.monotonic()
            if simulator_started_mono is not None and simulator_finished_mono is None:
                simulator_finished_wall = time.time()
                simulator_finished_mono = time.monotonic()
            os.close(read_fd)
            if write_fd is not None:
                os.close(write_fd)
            mirror_agent_session()
            token_usage = codex_token_usage(agent_home)
    stage_finished_wall = time.time()
    stage_finished_mono = time.monotonic()

    def elapsed(start, end):
        return round(end - start, 6) if start is not None and end is not None else None

    timing = {
        "phase": phase,
        "stage_started_at": stage_started_wall,
        "stage_finished_at": stage_finished_wall,
        "stage_wall_sec": elapsed(stage_started_mono, stage_finished_mono),
        "simulator_started_at": simulator_started_wall,
        "simulator_ready_at": simulator_ready_wall,
        "simulator_finished_at": simulator_finished_wall,
        "simulator_startup_sec": elapsed(simulator_started_mono, simulator_ready_mono),
        "simulator_process_sec": elapsed(simulator_started_mono, simulator_finished_mono),
        "agent_started_at": agent_started_wall,
        "agent_finished_at": agent_finished_wall,
        "agent_process_sec": elapsed(agent_started_mono, agent_finished_mono),
        "agent_exit_code": agent_rc,
        "token_usage": token_usage,
    }
    write_json(run / "timing.json", timing)

    path = run / "session-result.json"
    state = json.loads(path.read_text()) if path.exists() else {}
    valid = (
        failure is None
        and bool(state.get("valid"))
        and agent_rc in (0, 124)
        and state.get("reason") != "executing"
    )
    result = {
        **state,
        "protocol": SESSION_PROTOCOL,
        "phase": phase,
        "closed": True,
        "reason": failure
        or (
            state.get("reason")
            if state.get("closed")
            else "time_budget"
            if agent_rc == 124
            else "agent_exited"
        ),
        "valid": valid,
        "agent_rc": agent_rc,
        "failure": failure,
        "task_completed": bool(state.get("task_completed")) if valid else False,
        "timing": timing,
        **result_score(run, state, valid=valid),
    }
    # The agent and its simulator are stopped before archival replay. Its
    # outcome is diagnostic only: preserve the original result and budgets.
    write_json(run / "result.json", result)
    save_runtime(status="replaying")
    result["final_replay"] = run_replay(
        cfg,
        run,
        experiment,
        phase=phase,
        seed=seed,
        gpu=gpu,
    )
    write_json(run / "result.json", result)
    save_runtime(
        status="finished" if valid else "failed",
        finished_at=stage_finished_wall,
        simulator_pid=simulator.pid if simulator is not None else None,
        agent_pid=agent.pid if agent is not None else None,
        agent_exit_code=agent_rc,
        failure=failure,
        result_path=str((run / "result.json").relative_to(experiment)),
    )
    return result


def check_run(cfg, run):
    state = json.loads((run / "protocol.json").read_text())
    if state["definition"] != definition(cfg):
        raise ValueError("Run configuration changed; use a new --run-dir")
    if state["inputs"] != hashes(run / "inputs"):
        raise ValueError("Pinned workspace source changed")
    if json.loads((run / "resolved.json").read_text()) != config_dict(cfg):
        raise ValueError("Resolved configuration changed")
    return state


def explore(cfg, run_dir, *, gpu=None, timeout_sec=None):
    validate(cfg)
    if cfg.baseline:
        raise ValueError("Baseline has no Playground; use test or run to start Test directly")
    run = Path(run_dir).resolve()
    seconds = cfg.agent.timeout_sec if timeout_sec is None else float(timeout_sec)
    if seconds is not None and seconds <= 0:
        seconds = None
    with run_lock(run):
        if any(path.name != ".protocol.lock" for path in run.iterdir()):
            raise ValueError(
                "PlayGround requires a fresh --run-dir; live scenes are never silently restarted"
            )
        state = {
            "protocol": SESSION_PROTOCOL,
            "definition": definition(cfg),
            "phase": "playground",
            "inputs": prepare_inputs(cfg, run),
            "playground_seconds": seconds,
        }
        write_json(run / "protocol.json", state)
        result = run_episode(
            cfg,
            run / "playground",
            run,
            phase="playground",
            seed=cfg.playground.seed,
            seconds=seconds,
            gpu=gpu,
        )
        check_run(cfg, run)
        # Task success is independent of protocol validity. A normal policy
        # failure or exhausted budget can freeze; an infrastructure failure
        # must leave this Playground blocked, even if it wrote partial lessons.
        timing_path = run / "playground/timing.json"
        timing = json.loads(timing_path.read_text()) if timing_path.exists() else {}
        ran_agent = (
            timing.get("simulator_ready_at") is not None
            and timing.get("agent_started_at") is not None
        )
        if not ran_agent or result.get("valid") is not True or result.get("failure"):
            state.update(
                playground_valid=False,
                playground_reason=result.get("failure") or result.get("reason"),
            )
            write_json(run / "protocol.json", state)
            return 1
        frozen = run / "frozen_harness"
        frozen.mkdir()
        for component in ("skills", "lessons"):
            source = run / "playground/workspace" / component
            hashes(source)
            shutil.copytree(source, frozen / component)
        state.update(
            phase="frozen",
            harness=hashes(frozen),
            playground_valid=bool(result.get("valid")),
            playground_reason=result.get("reason"),
        )
        write_json(run / "protocol.json", state)
    return 0


def carried_test_seeds(seed_plan, settings):
    """Read pinned paired scenes and approved slot replacements after H freezes."""
    def read_pinned(reference):
        raw = Path(reference["path"]).read_bytes()
        if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
            raise ValueError("Paired Test source changed")
        return json.loads(raw)

    if not isinstance(seed_plan, dict):
        return json.loads(Path(seed_plan).read_text())
    carried = read_pinned(seed_plan)
    seeds = list(carried["seeds"])
    if carried.get("settings", settings) != settings:
        raise ValueError("Paired Test settings changed")
    replaced = set()
    for override in seed_plan.get("overrides", []):
        index, source_index = override["index"], override["source_index"]
        replacement = read_pinned(override)
        if (type(index) is not int or not 0 <= index < len(seeds) or index in replaced
                or type(source_index) is not int
                or not 0 <= source_index < len(replacement["seeds"])
                or replacement["settings"][source_index] != settings[index]):
            raise ValueError("Invalid approved paired Test replacement")
        seeds[index] = replacement["seeds"][source_index]
        replaced.add(index)
    return {**carried, "seeds": seeds}


def prepare_test(cfg, run, *, create_plan=True, settings=None, seed_plan=None, frozen_source=None):
    """Pin a fresh baseline or validate frozen knowledge, then open the Test plan."""
    validate(cfg)
    if frozen_source is not None:
        from orchestrator.frozen_source import prepare

        if cfg.baseline or cfg.simulator.backend != "robodojo":
            raise ValueError("External knowledge requires an adapted RoboDojo Test")
        if prepare(cfg, run, frozen_source) is None:
            return None
    if cfg.baseline and not (run / "protocol.json").exists():
        run.mkdir(parents=True, exist_ok=True)
        if any(path.name != ".protocol.lock" for path in run.iterdir()):
            raise ValueError("Baseline requires a fresh --run-dir or its existing pinned run")
        state = {
            "protocol": SESSION_PROTOCOL,
            "definition": definition(cfg),
            "phase": "test",
            "inputs": prepare_inputs(cfg, run),
            "harness": {},
        }
        write_json(run / "protocol.json", state)
    state = check_run(cfg, run)
    frozen = run / "frozen_harness"
    if cfg.baseline:
        if (
            state.get("phase") != "test" or state.get("harness") != {}
            or frozen.exists() or (run / "playground").exists()
        ):
            raise ValueError("Baseline must not inherit a Playground or frozen harness")
    elif frozen_source is None:
        result_path = run / "playground/result.json"
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        if (
            state.get("phase") != "frozen"
            or state.get("playground_valid") is not True
            or result.get("valid") is not True
            or result.get("failure")
            or state.get("harness") != hashes(frozen)
        ):
            raise ValueError("Finish PlayGround and freeze skills/lessons before Test")
    if frozen_source is not None:
        # Existing paired plans only: never call the sampling path, even
        # if a plan disappears between dependency validation and this read.
        raw = Path(cfg.test.shared_plan).read_bytes()
        shared = json.loads(raw)
        source_seeds = shared.get("seeds")
        if (hashlib.sha256(raw).hexdigest() != frozen_source["shared_plan_sha256"]
                or shared.get("env_task") != cfg.simulator.env_task
                or shared.get("settings") != settings
                or shared.get("excluded_seed") != cfg.test.excluded_seed
                or not valid_test_seeds(cfg, source_seeds, settings)):
            raise ValueError("Existing shared Test plan is incompatible with this task")
    plan_path = run / "test/plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        seeds = plan.get("seeds")
        if (
            plan.get("definition") != state["definition"]
            or plan.get("harness") != state["harness"]
            or (frozen_source is not None and (seeds != source_seeds or
                plan.get("scenes") != test_scene_records(cfg, source_seeds, settings)))
            or not valid_test_seeds(cfg, seeds, plan.get("settings", settings))
            or (plan.get("shared_plan_sha256") is not None and
                hashlib.sha256(Path(cfg.test.shared_plan).read_bytes()).hexdigest() != plan["shared_plan_sha256"])
            or (settings is not None and plan.get("settings", settings) != settings)
        ):
            raise ValueError("Test plan changed")
    elif create_plan:
        if frozen_source is not None:
            seeds = source_seeds
        elif seed_plan is not None:
            # Infrastructure migrations can repeat the same held-out scenes
            # under an explicitly changed simulator profile. Open this only
            # after the new Playground has frozen; never reuse its old H.
            carried = carried_test_seeds(seed_plan, settings)
            seeds = carried.get("seeds")
            if (
                not valid_test_seeds(cfg, seeds, carried.get("settings", settings))
                or carried.get("settings", settings) != settings
            ):
                raise ValueError("Reserved Test plan changed")
        else:
            seeds = (shared_test_seeds(cfg, settings=settings) if cfg.test.shared_plan else
                     test_seeds(cfg) if settings is None else test_seeds(cfg, settings=settings))
        plan = {"definition": state["definition"], "harness": state["harness"], "seeds": seeds}
        plan["scenes"] = test_scene_records(cfg, seeds, settings)
        if cfg.test.shared_plan:
            plan["shared_plan_source"] = cfg.test.shared_plan
            plan["shared_plan_sha256"] = hashlib.sha256(Path(cfg.test.shared_plan).read_bytes()).hexdigest()
        if seed_plan is not None:
            plan["seed_plan_source"] = seed_plan if isinstance(seed_plan, dict) else str(seed_plan)
        if settings is not None:
            plan["settings"] = settings
        write_json(plan_path, plan)
    else:
        return None
    return state, seeds


def test_report(run, scenes, results_by_index, *, scheduler=None):
    """Write completed results in stable scene order, including partial progress."""
    indices = sorted(results_by_index)
    results = [results_by_index[index] for index in indices]
    complete = len(results) == scenes and all(r["valid"] for r in results)
    successes = sum(r["valid"] and r["task_completed"] for r in results)
    report = {
        "protocol": SESSION_PROTOCOL,
        "complete": complete,
        "successes": successes,
        "scenes": scenes,
        "results": results,
        "scene_indices": indices,
        "success_rate": successes / scenes if complete else None,
        **aggregate_scores(results, complete=complete),
    }
    if scheduler is not None:
        report["scheduler"] = scheduler
    write_json(run / "test.json", report)
    return report


def scene_result(scene, identity):
    """Read an existing scene; never rerun a failed or interrupted scene."""
    if (scene / "result.json").exists():
        if json.loads((scene / "identity.json").read_text()) != identity:
            raise ValueError("Test scene identity changed")
        return json.loads((scene / "result.json").read_text())
    return {"valid": False, "task_completed": False, "failure": "interrupted_scene"}


def test(cfg, run_dir, *, gpu=None, test_gpus=None, gpu_slots=1):
    # Scheduling is host metadata, not a change to the pinned task or seed plan.
    slots = GpuSlots(test_gpus or [gpu], gpu_slots) if test_gpus is not None else None
    run = Path(run_dir).resolve()
    with run_lock(run):
        state, seeds = prepare_test(cfg, run)
        frozen = None if cfg.baseline else run / "frozen_harness"
        pending, results = [], {}
        scheduler = {"gpus": slots.gpus, "slots_per_gpu": gpu_slots} if slots else None
        for index, seed in enumerate(seeds):
            scene = run / "test" / f"scene_{index:04d}"
            identity = {"scene_index": index, "seed": seed, "definition": state["definition"]}
            if scene.exists():
                results[index] = scene_result(scene, identity)
            else:
                pending.append((index, seed, scene, identity))

        def launch_scene(item, selected_gpu):
            index, seed, scene, identity = item
            check_run(cfg, run)
            if frozen is not None and state["harness"] != hashes(frozen):
                raise ValueError("Frozen harness changed")
            # Queued scenes have no directory until a worker obtains GPU capacity.
            scene.mkdir()
            write_json(scene / "identity.json", identity)
            result = run_episode(
                cfg,
                scene,
                run,
                phase="test",
                seed=seed,
                seconds=cfg.test.generation_timeout_sec,
                gpu=selected_gpu,
                harness=frozen,
            )
            return index, result

        def launch_in_pool(item):
            with slots.lease() as selected_gpu:
                return launch_scene(item, selected_gpu)

        test_report(run, len(seeds), results, scheduler=scheduler)
        if slots is None:
            for item in pending:
                index, result = launch_scene(item, gpu)
                results[index] = result
                test_report(run, len(seeds), results)
        elif pending:
            workers = min(len(pending), len(slots.gpus) * gpu_slots)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(launch_in_pool, item) for item in pending]
                for future in as_completed(futures):
                    index, result = future.result()
                    results[index] = result
                    test_report(run, len(seeds), results, scheduler=scheduler)
        if frozen is not None and state["harness"] != hashes(frozen):
            raise ValueError("Frozen harness changed")
        report = test_report(run, len(seeds), results, scheduler=scheduler)
        return 0 if report["complete"] else 1

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import socketserver
import threading
import time
import traceback
import uuid

from . import codex
from .api import InvalidAction
from .backend import Backend
from .common import ROOT, digest, event, inventory, snapshot, stop_process, validate_paths, write_json

INSTRUCTIONS = """You are the robot policy in a PolicyLab experiment.
Public training data: /training (read-only).
Persistent workspace: /workspace.
Read the public task/data/action interface in /tools/BENCHMARK.md.
Use the shell, image viewer and your own programs; organize your own workspace and skills.
Use only supplied training material, the public interface, permitted observations and your own history.
Do not access task source, benchmark solutions, hidden state, evaluator diagnostics, credentials or future episodes, including over the network.

During exploration, study training data only; no live simulator is available.
During exploration, all robot operations except ready() are blocked.
Account for every supplied demonstration and all relevant public information: instructions, camera observations, measured robot state, actions, timing, joint ordering, units and metadata.
Compare variations and instruction mismatches.
Inspect informative frames and trajectories in depth, including grasp/contact/release transitions and several ending frames where present.
**Demonstrated ending requirements:** Compare each matching demonstration's initial state with several ending frames and measured states, accounting for every object and both arms.
Record object counts, identity, placement, orientation, support, stability, release or retained grasp, and final arm/gripper state where supported by the evidence.
Include objects that remain unused or unchanged; their demonstrated ending state is part of the comparison, not automatically outside the task.
Distinguish consistently demonstrated ending conditions from layout-dependent details, variations and uncertain requirements, citing demonstration frames and measured states.
Use contact sheets, crops, numerical analysis and offline script checks as useful.
Construct an imagined environment from demonstrated observations and the public interface.
Mentally study alternative action sequences through each subgoal: expected observations, action chunks, progress checks, likely failures, recovery and ending conditions.
Compare predictions with recorded transitions and revise unsupported assumptions.
Turn these rehearsals into observation-conditioned skills, not fixed layout-specific action scripts.
Write reusable skills/programs, a completion checklist, demonstration references, coverage and uncertainties into /workspace.
Separate observed evidence from inferred plans; imagined motions are untested, and a demonstration's ending pose does not prove a hidden evaluator rule.
Choose your own research depth within the supplied budget, reserving time to save usable artifacts.
During exploration, use `from robot import ready`.
When prepared during exploration, call ready() and end your response.

During evaluation, use `from robot import observe, act, reset, review, stop`.
observe() reads the latest observation without advancing physics.
act(action, repeat=1) advances and records native steps up to the supplied repeat limit.
For clear transit toward one observed waypoint, use up to 10 act() calls per program.
Recompute targets from measured state; stop when the waypoint is reached.
Prefer repeat=1 near grasping, contact or placement, repeat=5 for cautious travel, and repeat=10 for clear transit when permitted by the supplied limit and remaining budget.
Check returned observations and done after each call.
Pause for model image inspection when a visual change cannot be judged by the program, or on unexpected motion, uncertainty, contact or subgoal completion.
Respect remaining steps and review boundaries; reserve steps to verify completion, including demonstrated ending poses where applicable.
Never blindly replay demo actions or repeat uncertain timed-out calls.
For each action, write any required concise assessment of observed state and active subgoal, and a rationale for the next motion; an issued command does not prove progress.
Use your saved skills and relevant demo evidence rather than rereading the full corpus.
During evaluation, ready() is unavailable.
Follow the evaluation episode's reset/review/stop permissions.
Do not use stop() merely because the task looks complete while native done remains false.
Compare several ending observations with task-matched demonstrations, including object support, release and final arm poses.
**Ending-state verification:** Recheck every item in your saved completion checklist, including unused or unchanged objects, against current RGB/proprio and several demonstrated ending frames.
Mark each item met, unmet or uncertain with observed evidence; a command, apparent task progress or an earlier checklist claim is not proof that an ending requirement is met.
Label supporting evidence as directly observed, inferred or uncertain; an inference alone does not verify a requirement.
Do not dismiss a demonstrated ending difference merely because the short task instruction does not mention it; inspect the evidence and plan corrections when warranted.
**Recovery before abandonment:** An unknown failure cause or an apparently complete checklist does not establish that further execution is unproductive.
When native done remains false, challenge earlier claims, identify plausible mismatches from public evidence, and test meaningful observation-conditioned corrections when available.
Before an authorized stop, explain attempted alternatives, their observed effects, remaining uncertainty and why further recovery is unlikely to help; do not consume steps through unchanged retries or idle holds without a corrective hypothesis.
An authorized stop after review ends execution and records the native outcome; it cannot trigger or assert success.
Native done means termination, not necessarily success.
Save useful experience and end your response after done or an authorized stop; exiting earlier is recorded as incomplete.
Reassess each instance's goals and state.
"""


def public_observation(value, folder, public_prefix):
    if isinstance(value, dict):
        return {k: public_observation(v, folder, public_prefix) for k, v in value.items()}
    if isinstance(value, list):
        return [public_observation(v, folder, public_prefix) for v in value]
    if isinstance(value, str) and value.startswith("obs://"):
        relative = Path(value[6:])
        target = (folder / relative).resolve()
        if relative.is_absolute() or not target.is_relative_to(folder.resolve()) or not target.is_file():
            raise ValueError("Adapter observation file must exist inside its observation directory")
        return public_prefix + "/" + relative.as_posix()
    return value


class Experiment:
    def __init__(self, cfg, run, backend):
        self.cfg, self.run, self.backend = cfg, run, backend
        self.lock = threading.RLock()
        self.phase, self.active, self.latest = "idle", None, None
        self.count = 0
        self.request_sequence = 0
        self.segment_done = False
        self.fault = None
        self.deadline = float("inf")
        self.results = []
        options = cfg.get("options", {})
        self.review_window_steps = options.get("review_window_steps")
        self.review_max_windows = options.get("review_max_windows", 10)
        self.review_allow_reset = options.get("review_allow_reset", False)
        if self.review_window_steps is not None and (type(self.review_window_steps) is not int or self.review_window_steps <= 0
                                                     or type(self.review_max_windows) is not int or self.review_max_windows <= 0):
            raise ValueError("Review windows must be positive integers")
        if type(self.review_allow_reset) is not bool or (self.review_allow_reset and self.review_window_steps is None):
            raise ValueError("review_allow_reset requires review windows and a boolean value")
        self.reviewed_windows = 0
        self.used_steps_before_attempt = 0
        self.last_review_step = None

    def total_step(self, reply=None):
        reply = reply or self.latest
        return self.used_steps_before_attempt + int(reply["observation"]["control_step"])

    def review_status(self, reply=None):
        if self.review_window_steps is None or self.phase != "evaluation" or reply is None:
            return {}
        step = self.total_step(reply) if self.review_allow_reset else int(reply["observation"]["control_step"])
        limit = self.review_window_steps * self.review_max_windows if self.review_allow_reset else int(reply["observation"]["step_limit"])
        window = self.reviewed_windows + 1
        boundary = window * self.review_window_steps if window <= self.review_max_windows else None
        success = reply["done"] and reply["metrics"].get("success") is True
        return {"base_step_limit": self.review_window_steps, "max_step_limit": limit,
                "review_window": min(window, self.review_max_windows),
                "next_review_step": boundary, "steps_to_review": max(0, boundary - step) if boundary else 0,
                "review_due": boundary is not None and step >= boundary and not success
                and (not reply["done"] or (self.review_allow_reset and step < limit))}

    def checkpoint(self, name):
        destination = self.run / "checkpoints" / name
        files = snapshot(self.run / "workspace", destination / "workspace")
        write_json(destination / "files.json", files)
        return str(destination.relative_to(self.run))

    def finalize(self, reason):
        if self.active is None or self.active.get("recorded"):
            return
        terminal = self.latest is not None and self.latest["done"]
        result = {"episode": self.active["episode"], "phase": "evaluation_attempt" if reason == "agent_reset" else self.phase,
                  "record": self.active["name"], "reason": reason,
                  "native_done": terminal,
                  "metrics": self.latest["metrics"] if self.latest else {},
                  "success": self.latest["metrics"]["success"] if terminal else None}
        if reason == "reset_requested":
            result["native_success"] = result["success"]
            result["success"] = False
            result["failure_type"] = "reset_requested"
        result["checkpoint"] = self.checkpoint(self.active["name"])
        write_json(self.active["record_dir"] / "result.json", result)
        event(self.run / "episodes.jsonl", **result)
        self.results.append(result)
        self.active["recorded"] = True

    def accept(self, reply, terminal_reason="native_terminal"):
        self.latest = reply
        obs = public_observation(reply["observation"], self.active["observation_dir"],
                                 "/observations/" + self.active["name"])
        if self.review_allow_reset and self.phase == "evaluation":
            obs["attempt_control_step"] = obs["control_step"]
            obs["control_step"] = self.total_step(reply)
            obs["step_limit"] = self.review_window_steps * self.review_max_windows
            obs["steps_remaining"] = max(0, obs["step_limit"] - obs["control_step"])
        obs.update(self.review_status(reply))
        result = {"observation": obs, "done": reply["done"], "phase": self.phase}
        if reply["done"] and self.cfg["terminal_feedback"]:
            result["success"] = reply["metrics"]["success"]
        event(self.active["record_dir"] / "transitions.jsonl", **reply)
        write_json(self.run / "observations" / "latest.json", result)
        self.public = result
        if reply["done"]:
            self.finalize(terminal_reason)
            if self.phase == "evaluation" and not obs.get("review_due"):
                self.segment_done = True
        return result

    def end_episode(self, reason):
        if self.active is not None and not self.active.get("recorded"):
            self.accept(self.backend.request(op="finish"), reason)

    def reset_episode(self, episode, preserve_review=False):
        self.end_episode("practice_reset")
        self.count += 1
        name = f"{self.phase}-{self.count:04d}"
        record_dir = self.run / "episodes" / name
        observation_dir = self.run / "observations" / name
        record_dir.mkdir(parents=True)
        observation_dir.mkdir(parents=True)
        self.active = {"episode": episode, "name": name, "record_dir": record_dir,
                       "observation_dir": observation_dir, "recorded": False}
        self.latest = None
        if not preserve_review:
            self.reviewed_windows = 0
            self.used_steps_before_attempt = 0
        self.last_review_step = None
        write_json(record_dir / "episode.json", episode)
        reply = self.backend.request(op="reset", episode=episode,
                                     record_dir=str(record_dir / "native"), observation_dir=str(observation_dir))
        if self.review_window_steps is not None and self.phase == "evaluation":
            expected = self.review_window_steps * self.review_max_windows
            if reply["observation"].get("step_limit") != expected:
                raise ValueError(f"Review experiment requires a {expected}-step adapter horizon")
            step = int(reply["observation"]["control_step"])
            if not preserve_review:
                self.reviewed_windows = min((max(step, 1) - 1) // self.review_window_steps,
                                            self.review_max_windows - 1)
        return self.accept(reply)

    def request(self, request):
        with self.lock:
            self.request_sequence += 1
            event(self.run / "requests.jsonl", sequence=self.request_sequence, request=request)
            if not isinstance(request, dict):
                raise InvalidAction("Request must be a JSON object")
            if time.monotonic() >= self.deadline or self.segment_done:
                raise InvalidAction("This segment has ended; save your notes and exit")
            if self.fault:
                raise InvalidAction("Environment unavailable; exit and retain your notes")
            op = request.get("op")
            allowed = {"observe": {"op"}, "act": {"op", "action", "repeat"},
                       "reset": {"op"}, "ready": {"op"},
                       "review": {"op", "analysis"}, "stop": {"op", "reason"}}
            if op not in allowed or set(request) != allowed[op]:
                raise InvalidAction("Use observe, act(action, repeat), review(analysis), stop(reason), reset or ready")
            if self.phase == "exploration" and op != "ready":
                raise InvalidAction("Exploration is training-data study only; no live simulator is available. Save your research and call ready().")
            if op == "ready" and self.phase != "exploration":
                raise InvalidAction(f"{op} is only available during exploration")
            if op == "reset" and self.phase != "exploration" and self.review_allow_reset and not (
                self.phase == "evaluation" and self.review_allow_reset and self.active is not None
                and not self.latest["done"] and self.last_review_step == self.total_step()
                and int(self.latest["observation"]["control_step"]) > 0
                and self.total_step() < self.review_window_steps * self.review_max_windows
            ):
                raise InvalidAction("Evaluation reset requires a review, actions in this attempt, and remaining steps")
            try:
                if op == "reset" and self.phase == "evaluation" and not self.review_allow_reset:
                    if self.active is None or self.latest["done"]:
                        raise InvalidAction("Evaluation has no active attempt to stop")
                    event(self.active["record_dir"] / "stop.jsonl", control_step=self.total_step(),
                          reason="Evaluation reset requested; terminated under the no-reset policy.")
                    result = self.accept(self.backend.request(op="finish"), "reset_requested")
                    self.segment_done = True
                    return result
                if op == "reset":
                    if self.phase == "evaluation":
                        step = self.total_step()
                        reply = self.backend.request(op="finish")
                        result = self.accept(reply, "native_terminal" if reply["metrics"].get("success") else "agent_reset")
                        if reply["metrics"].get("success"):
                            return result
                        self.used_steps_before_attempt = step
                        self.segment_done = False
                        return self.reset_episode(self.active["episode"], preserve_review=True)
                if op == "ready":
                    self.end_episode(op)
                    self.segment_done = True
                    return {"status": op, "message": "End your response now."}
                if self.active is None:
                    raise InvalidAction("No active evaluation episode")
                if op == "stop":
                    if self.phase != "evaluation" or self.latest["done"] or self.last_review_step != self.total_step():
                        raise InvalidAction("Stop requires an evaluation review at the current step")
                    minimum = self.cfg.get("options", {}).get("stop_min_steps", 0)
                    if type(minimum) is not int or minimum < 0:
                        raise InvalidAction("stop_min_steps must be a nonnegative integer")
                    if self.total_step() < minimum:
                        raise InvalidAction(f"Stop is unavailable before step {minimum}; continue and review the demonstrated ending requirements")
                    reason = request["reason"]
                    if not isinstance(reason, str) or not reason.strip() or len(reason) > 2000:
                        raise InvalidAction("Stop reason must be a nonempty string of at most 2000 characters")
                    event(self.active["record_dir"] / "stop.jsonl", control_step=self.total_step(), reason=reason)
                    result = self.accept(self.backend.request(op="finish"), "agent_stop")
                    self.segment_done = True
                    return result
                if op == "observe":
                    return self.public
                if op == "review":
                    if self.phase != "evaluation" or self.review_window_steps is None or (self.latest["done"] and not self.review_status(self.latest).get("review_due")):
                        raise InvalidAction("No evaluation review is available")
                    analysis = request["analysis"]
                    fields = {"observed_state", "demo_comparison", "why_not_done", "next_plan", "ending_check"}
                    if not isinstance(analysis, dict) or set(analysis) != fields or any(
                        not isinstance(analysis[key], str) or not analysis[key].strip() for key in fields
                    ):
                        raise InvalidAction("Review needs observed_state, demo_comparison, why_not_done, next_plan and ending_check")
                    scheduled = self.review_status(self.latest).get("review_due", False)
                    if scheduled:
                        self.reviewed_windows += 1
                    self.last_review_step = self.total_step()
                    event(self.active["record_dir"] / "reviews.jsonl", window=self.reviewed_windows,
                          kind="checkpoint" if scheduled else "early",
                          control_step=self.total_step() if self.review_allow_reset else self.latest["observation"]["control_step"], analysis=analysis)
                    self.public["observation"].update(self.review_status(self.latest))
                    write_json(self.run / "observations" / "latest.json", self.public)
                    if self.latest["done"]:
                        self.segment_done = True
                    return self.public
                if self.latest["done"]:
                    raise InvalidAction("Episode ended; end your response")
                if self.review_status(self.latest).get("review_due"):
                    raise InvalidAction("Checkpoint review due; inspect observations and call review(analysis)")
                repeat = request["repeat"]
                maximum = self.cfg.get("options", {}).get("max_repeat", 10)
                if type(repeat) is not int or not 1 <= repeat <= maximum:
                    raise InvalidAction(f"repeat must be an integer from 1 to {maximum}")
                self.last_review_step = None
                for step_index in range(repeat):
                    if time.monotonic() >= self.deadline:
                        break
                    record = {"request_sequence": self.request_sequence, "step_index": step_index,
                              "requested_repeat": repeat, "action": request["action"]}
                    event(self.active["record_dir"] / "actions.jsonl", status="requested", **record)
                    reply = self.backend.request(op="step", action=request["action"])
                    result = self.accept(reply)
                    if self.review_allow_reset and self.phase == "evaluation" and not reply["done"] and self.total_step() >= self.review_window_steps * self.review_max_windows:
                        result = self.accept(self.backend.request(op="finish"))
                    event(self.active["record_dir"] / "actions.jsonl", status="completed",
                          applied_action=reply.get("applied_action"),
                          control_step=reply["observation"].get("control_step"), **record)
                    if result["done"] or result["observation"].get("review_due"):
                        break
                return self.public
            except InvalidAction:
                raise
            except Exception:
                self.fault = traceback.format_exc()
                (self.run / "adapter_error.txt").write_text(self.fault)
                self.finalize("adapter_error")
                raise


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        self.connection.settimeout(30)
        sequence = None
        try:
            line = self.rfile.readline(1024 * 1024 + 1)
            if len(line) > 1024 * 1024 or not line.endswith(b"\n"):
                raise InvalidAction("Request too large or incomplete")
            request = json.loads(line)
            sequence = self.server.experiment.request_sequence + 1
            result = self.server.experiment.request(request)
        except (InvalidAction, json.JSONDecodeError) as exc:
            result = {"error": str(exc)}
        except Exception:
            result = {"error": "Environment error; do not repeat the action. Exit and retain your notes."}
        event(self.server.experiment.run / "responses.jsonl", sequence=sequence, response=result)
        try:
            self.wfile.write((json.dumps(result, allow_nan=False) + "\n").encode())
        except (BrokenPipeError, ConnectionResetError):
            pass


def run_segment(experiment, cfg, run, private_home, name, prompt, seconds, session, launch=codex.launch):
    segment = run / "segments" / name
    segment.mkdir(parents=True)
    with experiment.lock:
        experiment.segment_done = (experiment.phase == "evaluation" and experiment.latest is not None
                                   and experiment.latest["done"] and not experiment.review_status(experiment.latest).get("review_due"))
        experiment.deadline = time.monotonic() + seconds
    process = launch(cfg, run, private_home, segment, prompt, session)
    started, stop_requested = time.monotonic(), None
    reason = "agent_exited"
    current_segment, refreshes = segment, []
    try:
        while True:
            if process.poll() is not None:
                if (cfg.get("options", {}).get("live_context_recovery", False)
                        and experiment.phase == "evaluation" and not experiment.segment_done
                        and not experiment.fault and time.monotonic() < experiment.deadline
                        and len(refreshes) < 64):
                    messages = []
                    log = current_segment / "model.jsonl"
                    if log.exists():
                        for line in log.read_text().splitlines():
                            try:
                                item = json.loads(line)
                            except ValueError:
                                continue
                            if item.get("type") == "turn.failed":
                                messages.append(item.get("error", {}).get("message", ""))
                    failed = messages[-1] if messages else ""
                    image_limit = "maximum number of images" in failed
                    rate_limit = "429" in failed
                    if image_limit or rate_limit:
                        found, _ = codex.events_summary(log)
                        entry = {"source": str(current_segment.relative_to(run)), "error": failed,
                                 "new_session": image_limit, "control_step": experiment.latest["observation"].get("control_step", experiment.latest["observation"].get("step")),
                                 "native_reset": False}
                        refreshes.append(entry)
                        event(segment / "live_context_recovery.jsonl", **entry)
                        if rate_limit:
                            until = min(experiment.deadline, time.monotonic() + 60)
                            while time.monotonic() < until:
                                time.sleep(.2)
                        current_segment = segment / "rounds" / f"refresh-{len(refreshes):04d}"
                        current_segment.mkdir(parents=True)
                        recovery_prompt = prompt + "\nContinue this SAME live episode from its current measured state; the simulator has NOT reset and the native action budget is unchanged. Model context was refreshed only for gateway capacity limits. Do not explore again. Read your own saved notes, call observe() and inspect current images. If observe returns pending_vla_proposal, review that same unmoved proposal before requesting another."
                        process = launch(cfg, run, private_home, current_segment, recovery_prompt,
                                         None if image_limit else found)
                        continue
                break
            if experiment.fault:
                reason = "adapter_error"
                break
            now = time.monotonic()
            if now >= experiment.deadline:
                reason = "budget_exhausted"
                break
            if experiment.segment_done:
                stop_requested = stop_requested or now
                if now - stop_requested >= 15:
                    reason = "submitted"
                    break
            time.sleep(.05)
    finally:
        stop_process(process)
    with experiment.lock:
        if not experiment.fault:
            if experiment.phase == "evaluation":
                experiment.finalize(reason)
            else:
                experiment.end_episode(reason)
        if experiment.segment_done:
            reason = "ready" if experiment.phase == "exploration" else "finished"
        experiment.deadline = 0
        checkpoint = experiment.checkpoint("segment-" + name)
    found_session, usage = codex.events_summary(current_segment / "model.jsonl")
    summary = {"status": reason, "returncode": process.returncode, "wall_seconds": time.monotonic() - started,
               "session_id": found_session or session, "usage": usage, "checkpoint": checkpoint,
               "live_context_recoveries": refreshes, "last_model_log": str(current_segment.relative_to(run))}
    write_json(segment / "result.json", summary)
    return summary


def run_experiment(cfg, output, launch=codex.launch):
    options = cfg.get("options", {})
    independent = options.get("evaluation_independent", False)
    if type(independent) is not bool:
        raise ValueError("evaluation_independent must be boolean")
    if type(options.get("replay_fresh_session", False)) is not bool:
        raise ValueError("replay_fresh_session must be boolean")
    continuation = Path(cfg["continue_from_run"]).resolve() if cfg.get("continue_from_run") else None
    previous = None
    prepared_source = None
    study_session = None
    if continuation:
        previous = json.loads((continuation / "config.json").read_text())
        if not independent or any(previous[key] != cfg[key] for key in ("training_set", "model", "reasoning_effort", "environment_revision")):
            raise ValueError("Continuation requires the same independent experiment and training/model versions")
        prepared_source = continuation / "prepared_workspace"
        if not prepared_source.is_dir():
            if any((continuation / "segments").glob("evaluation-*")):
                raise ValueError("Missing preparation for evaluation continuation")
            prepared_source = None
            study_session, _ = codex.events_summary(continuation / "segments/exploration/model.jsonl")
            if not study_session:
                raise ValueError("Interrupted study has no saved session")
    provenance = validate_paths(cfg)
    output = Path(output).resolve()
    for protected in (cfg["training_set"], cfg["environment_dir"], cfg["agent_workspace"]):
        if output.is_relative_to(Path(protected).resolve()):
            raise ValueError("Run output must be outside the three supplied directories")
    run = output / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    for name in ("observations", "bridge", "tools", "checkpoints", "episodes", "segments"):
        (run / name).mkdir(parents=True)
    workspace = Path(cfg["agent_workspace"]).resolve()
    if continuation:
        if workspace.exists():
            raise ValueError("Continuation needs a new workspace directory")
        source = prepared_source or Path(previous["agent_workspace"])
        shutil.copytree(source, workspace, symlinks=True)
    workspace.mkdir(parents=True, exist_ok=True)
    (run / "workspace").symlink_to(workspace, target_is_directory=True)
    write_json(run / "config.json", cfg)
    write_json(run / "training_files.json", inventory(cfg["training_set"]))
    shutil.copy2(ROOT / "policylab/robot.py", run / "tools/robot.py")
    public = Path(cfg["public_instructions"]).read_text() if cfg["public_instructions"] else "Read the observation for the benchmark's action interface."
    (run / "tools/BENCHMARK.md").write_text(public)
    provenance["public_instructions_sha256"] = digest(run / "tools/BENCHMARK.md")
    provenance["exploration_mode"] = "training_only"
    provenance["framework_files_sha256"] = {
        path.name: digest(path) for path in sorted((ROOT / "policylab").glob("*.py"))
    }
    if launch is codex.launch:
        provenance["codex_binary_sha256"] = digest(codex.native_binary(cfg["codex_binary"]))
    write_json(run / "provenance.json", provenance)
    write_json(run / "workspace_initial.json", inventory(workspace))
    private_home = ROOT / ".runtime/codex" / run.name
    if study_session:
        private_home = ROOT / ".runtime/codex" / continuation.name
    backend, experiment = None, None
    server = None
    segments, error = [], None
    print(f"Run: {run}", flush=True)
    try:
        experiment = Experiment(cfg, run, None)
        address = str(run / "bridge/robot.sock")
        if len(address.encode()) >= 104:
            raise ValueError("Run path too long for a Unix socket; choose a shorter --output directory")
        server = socketserver.UnixStreamServer(address, Handler)
        server.experiment = experiment
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)
        thread.start()
        experiment.phase = "exploration"
        repeat_limit = cfg.get("options", {}).get("max_repeat", 10)
        sharing = ("Preparation will be copied into an isolated workspace and fresh session for each evaluation; evaluation experience is not shared.\n"
                   if independent else "Your Codex session and workspace persist across phases and evaluation episodes.\n")
        prompt = (INSTRUCTIONS + sharing + f"\nStudy the training data now. Your exploration budget is {cfg['explore_seconds']} wall seconds.\n")
        if prepared_source:
            segments.append({"status": "preparation_reused", "source": str(prepared_source), "session_id": None,
                             "checkpoint": experiment.checkpoint("preparation-reused")})
        else:
            segments.append(run_segment(experiment, cfg, run, private_home, "exploration", prompt,
                                        cfg["explore_seconds"], study_session, launch))
        if experiment.fault:
            raise RuntimeError("Adapter failed during exploration")
        session = segments[-1]["session_id"]
        if not prepared_source and (not session or segments[-1]["status"] not in ("ready", "budget_exhausted")):
            raise RuntimeError("Codex did not complete a usable exploration session; see model.jsonl/model.stderr")
        if independent:
            snapshot(workspace, run / "prepared_workspace")
        for index, episode in enumerate(cfg["test_episodes"]):
            resume_session = None
            resume_home = None
            elapsed_seconds = 0
            resume_seconds = cfg["episode_seconds"]
            replay = options.get("replay")
            if continuation and replay and replay["episode_id"] == episode["id"] and not options.get("replay_fresh_session", False):
                source_record = Path(replay["source_record"]).resolve()
                if source_record.parent.parent != continuation or json.loads((source_record / "episode.json").read_text()) != episode:
                    raise ValueError("Continuation replay must match this episode in the source run")
                source_index = int(source_record.name.rsplit("-", 1)[1])
                source_segment = continuation / "segments" / f"evaluation-{source_index:04d}"
                source_result = json.loads((source_segment / "result.json").read_text())
                resume_session = source_result["session_id"]
                if not resume_session:
                    raise ValueError("Interrupted evaluation has no saved session")
                resume_home = Path(source_result.get("codex_home", ROOT / ".runtime/codex" / continuation.name / source_segment.name))
                elapsed_seconds = source_result.get("wall_seconds_total", source_result["wall_seconds"])
                resume_seconds = cfg["episode_seconds"] - elapsed_seconds
                if resume_seconds <= 0:
                    raise ValueError("Evaluation wall-time budget exhausted")
            if independent:
                isolated = run / "evaluation_workspaces" / f"evaluation-{index+1:04d}"
                if continuation and replay and replay["episode_id"] == episode["id"]:
                    # Fresh sessions still inherit only THIS interrupted episode's public workspace.
                    source_workspace = continuation / "evaluation_workspaces" / Path(replay["source_record"]).name
                else:
                    source_workspace = continuation / "evaluation_workspaces" / source_segment.name if resume_session else run / "prepared_workspace"
                files = snapshot(source_workspace, isolated)
                (run / "workspace").unlink()
                (run / "workspace").symlink_to(isolated, target_is_directory=True)
                event(run / "evaluation_workspaces.jsonl", episode=episode["id"], workspace=str(isolated), initial_files=files)
            if backend:
                backend.close()
            backend = None
            log_name = f"simulator-evaluation-{index+1:04d}.log"
            backend = Backend(cfg, run, log_name=log_name)
            backend.request(op="initialize")
            event(run / "backend_events.jsonl", phase="evaluation", episode=episode["id"], log=log_name)
            with experiment.lock:
                experiment.backend = backend
                experiment.phase = "evaluation"
                experiment.active = None
                experiment.reset_episode(episode)
            review_guidance = ("At each review_due checkpoint, inspect current RGB/proprio and your own trace, compare with the supplied public demonstrations, "
                               "then call review() with observed_state, demo_comparison, why_not_done, next_plan and ending_check. "
                               + ("If the current physical state is clearly unrecoverable before a checkpoint, call review() early with evidence from RGB/proprio and demonstrations, then reset() immediately. "
                                  "After any review, choose to continue or reset the same layout and seeds; all attempts share one action budget. "
                                  if experiment.review_allow_reset else "Continue the same episode afterward without resetting. "
                                  "If further execution appears unrecoverable between checkpoints, submit an evidence review early, then call stop(reason). "
                                  "An early review does not move the next scheduled checkpoint. Requesting reset ends evaluation as a protocol failure. ")
                               + "A visually complete arrangement with native done=false is not by itself a reason to stop. "
                               "**Ending requirements:** Compare all objects, including unused or unchanged ones, and both arms with task-matched demonstration endings. "
                               "Check object counts, placement, orientation, release or retained grasp, support, stability and final arm/gripper state where demonstrated. "
                               "In ending_check, mark each requirement met, unmet or uncertain with current observations and demonstration references; revisit earlier claims and plan corrections for observed mismatches. "
                               "Do not dismiss a demonstrated ending difference solely because the short task instruction omits it. "
                               "Distinguish directly observed, inferred and uncertain evidence; an unknown failure cause does not establish that recovery is impossible. "
                               "Before judging further execution unproductive, challenge earlier claims and test meaningful corrections when available, then explain attempted alternatives, their observed effects and remaining uncertainty without repeating unchanged actions merely to consume steps. "
                               "After a review, if further execution is not worthwhile and the minimum stop step has been reached, call stop(reason) to end execution and record the native outcome; stopping cannot trigger success. "
                               if experiment.review_window_steps is not None else "")
            if options.get("stop_min_steps", 0):
                review_guidance += (f"stop() is unavailable before control step {options['stop_min_steps']}. "
                                    "Continue through the first two base-budget windows, review at their boundaries, then enter the third window before considering a reviewed stop. "
                                    "Native termination or a technical/wall-time limit still ends execution immediately. ")
            if continuation and replay and replay["episode_id"] == episode["id"] and options.get("replay_fresh_session", False):
                review_guidance += "This is the SAME interrupted episode restored by private recorded-action replay. Your copied workspace contains only its own history. Observe the current cameras and measured state before acting. Discard any old pending VLA proposal; request a new proposal at the current step. "
            prompt = (f"Evaluation instance {index + 1}. Continue using your workspace and experience. "
                      f"Call observe() for the current instruction and observations. "
                      f"Each act() call may repeat 1..{repeat_limit} native steps. "
                      f"Adaptive approved_steps must be an integer from {options.get('approved_steps_min', 5)} to {options.get('approved_steps_max', 50)}; "
                      f"if omitted, use {options.get('approved_steps_default', max(5, options.get('approved_steps_min', 5)))}. "
                      "These recorded episode limits override historical skill examples; every VLA proposal also reports them. "
                      "Track control_step and any public episode step limit; never exceed known remaining steps, "
                      "and reserve steps for completion checks. "
                      f"You have {resume_seconds} wall seconds. No practice reset is available. "
                      f"{review_guidance}Save what you learn, then end your response after native done.")
            evaluation_cfg, evaluation_home = cfg, private_home
            if independent:
                prompt = INSTRUCTIONS + "\nThis is an independent evaluation with a copy of your task's preparation skills; no other evaluation history is available.\n" + prompt
                evaluation_cfg = {**cfg, "_observation_episode": experiment.active["name"]}
                evaluation_home = ROOT / ".runtime/codex" / run.name / f"evaluation-{index+1:04d}"
                session = None
                if resume_session:
                    session, evaluation_home = resume_session, resume_home
                    prompt += "\nContinue only this same interrupted episode from its replayed checkpoint; observe current RGB/proprio before issuing another action.\n"
            segments.append(run_segment(experiment, evaluation_cfg, run, evaluation_home, f"evaluation-{index+1:04d}",
                                        prompt, resume_seconds, session, launch))
            segments[-1].update(codex_home=str(evaluation_home),
                                wall_seconds_total=elapsed_seconds + segments[-1]["wall_seconds"])
            write_json(run / "segments" / f"evaluation-{index+1:04d}" / "result.json", segments[-1])
            session = segments[-1]["session_id"]
            if experiment.review_window_steps is not None and segments[-1]["status"] != "finished":
                raise RuntimeError("Continuous review episode interrupted before native termination")
            if experiment.fault or (segments[-1]["returncode"] != 0 and segments[-1]["status"] not in ("finished", "budget_exhausted")):
                raise RuntimeError("Evaluation interrupted by an adapter or agent error")
    except BaseException as exc:
        error = type(exc).__name__ + ": " + str(exc)
        (run / "error.txt").write_text(traceback.format_exc())
        if experiment is not None:
            with experiment.lock:
                experiment.finalize("interrupted" if isinstance(exc, KeyboardInterrupt) else "error")
    finally:
        if server:
            server.shutdown()
            server.server_close()
        if backend:
            backend.close()
    evaluated = [row for row in experiment.results if row["phase"] == "evaluation"] if experiment else []
    attempted = {row["episode"]["id"] for row in evaluated}
    summary = {"status": "error" if error else "completed", "error": error,
               "test_total": len(cfg["test_episodes"]), "test_attempted": len(evaluated),
               "test_successes": sum(row["success"] is True for row in evaluated),
               "test_native_failures": sum(row["success"] is False and row["reason"] != "reset_requested" for row in evaluated),
               "test_protocol_failures": sum(row["reason"] == "reset_requested" for row in evaluated),
               "test_technical_errors": sum(row["success"] is None for row in evaluated),
               "test_not_attempted": [e["id"] for e in cfg["test_episodes"] if e["id"] not in attempted],
               "segments": segments}
    summary["success_rate_all_scheduled"] = summary["test_successes"] / summary["test_total"]
    write_json(run / "summary.json", summary)
    return run, summary

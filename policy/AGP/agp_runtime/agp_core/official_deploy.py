"""One AGP episode inside the official RoboDojo evaluator (XPolicyLab/policy/AGP/deploy.py calls this).

eval_one_episode(TASK_ENV, model_client, policy_dir=...) does, in the simulator process:

  1. Wrap TASK_ENV in agp_core.official_env.TaskEnvWhitelist and read the episode bookkeeping once (task, seed,
     layout id = env_seeds[0]); the episode index is this process's own count of episodes, from 0 (the evaluator's
     success_nums / fail_nums are not read). The official step limit is the public task code's value for that task
     (agp_core/task_step_limits.json; TASK_ENV.step_lim is not read; a task missing from the table is an infrastructure
     failure). model_client is not used: the policy server is a stub (XPolicyLab/policy/AGP/model.py) and the agent
     runs in child processes.
  2. Create the session AGP_HOME/runs/official/<ROBODOJO_RUN_ID>/layout-<layout>-a<attempt> (AGP_HOME: agp_core/paths.py;
     episode.json records the bookkeeping, the frozen system (the built-in switch set and deploy.yml's model
     settings), the run controls and where the benchmark was imported from).
  3. Build the worker (agp_core/official_worker.py) on OfficialEnv with the policy-side planner (agp_core/policy_planner.py)
     and the RGB-only observer (agp_core/rgb_observe.py); the official step limit is never touched.
  4. Start the policy in a new process group with a clean environment (no Isaac/Kit variables, no PYTHONPATH,
     PATH=/usr/local/bin:/usr/bin:/bin, the frozen OURS_* switches and model settings): agp_core/run_agent.py (MCP server
     + codex) or, with AGP_SCRIPTED=<name>, the LLM-free driver agp_core/official_scripted.py over the same MCP tools.
  5. Serve the worker bridge until the policy writes policy_finished.json, the policy process exits, the episode
     has ended and a grace period for the final observe/assess has passed, or the wall-clock limit is reached.
     With OURS_HIDE_END on, codex never sees the official end flag; at the official end the bridge answers nothing more
     and the policy is stopped at once (no grace), as the official deploy loop stops calling its policy. A scripted test
     policy (AGP_SCRIPTED) keeps the flag and the grace: it must finish on its own.
  6. finally: stop the policy's process group; if the official episode has not ended, set success False and call
     is_episode_end() (honest failure: EvalEnv.reset starts with success True); write episode_metrics.json and the
     TASK_ENV access log (task_env_access.json).

Frozen system: the switch set is built in (agp_core/features.py SYSTEM; deploy.yml has no switches entry, load_config
refuses one) and deploy.yml `agp:` holds the model settings (model, effort, service_tier). The policy gets exactly
these; OURS_* values of the caller's environment never reach it. Caller values that differ from the frozen ones are
logged ("[AGP] ignoring caller setting ...") and listed in episode.json (`caller_settings_ignored`); the frozen values
are logged once per episode and stored in episode.json (`frozen`).

Infrastructure failures (agent launcher, MCP server, codex authentication/provider errors, worker start-up, a
whitelist violation, an unexpected adapter error) raise agp_core.official_env.AgentInfrastructureFailure, a
BaseException, so main.py cannot count the layout as failed or skip it. A PhysX error from an official call is
re-raised unchanged so main.py's own recovery handles it.

External kills: a policy process ended by a signal that our own code did not send (codex, the MCP server,
agp_core/run_agent.py or the scripted driver; return code -N, or 128+N where the process reports a signal that way; the
kernel OOM killer included) before the official episode ended is an infrastructure failure of kind `external_kill`: no
result is written, the launcher re-runs the layout with a new attempt number and stops the layout after two external
kills in a row. Signals our code sends keep their meaning: the episode watchdog (the wall-clock limit here,
agp_core/run_agent.py's policy time limit), the end-on-assess termination and the clean-up after the episode ended end the
episode as a policy outcome (honest failure unless the evaluator already decided). An external kill after the official
end does not change the evaluator's verdict; it is recorded in episode_metrics.json
(`external_kill_after_official_end`).

Crashes of our own policy code: the MCP server (agp_core/server.py) or the agent launcher agp_core/run_agent.py dying
mid-episode without an external signal -- it exits on its own, raises, ends without policy_finished.json, or ends by a
fault signal of its own (FAULT_SIGNALS: SIGSEGV, SIGBUS, SIGFPE, SIGILL, SIGABRT, SIGTRAP, SIGSYS, which no other
process sends it) -- is our own code failing. The episode ends honestly as a policy failure of kind `mcp_crash`: no
infrastructure failure is raised, so unless the evaluator had already decided, success is set False (finalize) and
main.py writes and counts the result; the session records it apart in mcp_crash.json and in episode_metrics.json
(`policy_end`, kind mcp_crash, counted true); the launcher lists it apart and never re-runs it (a valid result).
Mid-episode means after codex started (agp_core/run_agent.py writes agent.pid once codex runs); an end of the launcher
before that (the MCP server cannot start, preload or configuration fails) stays an infrastructure abort, and external
kills (any other signal) keep the classification above. A crash after the official end leaves the evaluator's verdict
standing (`after_official_end`). codex's own signals are unchanged (external).

Freeze check: at the start of every episode (each layout is a new evaluator process), right after the session is created
and before the worker or the policy starts, the freeze manifest (agp_core/paths.py FREEZE_MANIFEST, relative to AGP_HOME, or
the test-only override AGP_FREEZE_MANIFEST) is verified against the current state with agp_core/freeze.py: the prompt and
its rendering, the switches and model settings, the pinned codex release, every runtime-loaded file, the public
constants files, the checkout's git state and the interpreters. Any difference raises AgentInfrastructureFailure of
kind `freeze_mismatch` (infrastructure_failure.json lists the differing entries): nothing runs, and agp_core/run_official.py
stops the batch instead of retrying. A frozen measurement (AGP_FROZEN_MEASUREMENT=1, set by agp_core/run_official.py
--frozen-measurement together with AGP_FREEZE_SEAL, the seal of the manifest the launcher verified) also refuses every
test-only run control (AGP_SCRIPTED, AGP_FAULT, AGP_FREEZE_MANIFEST), a manifest with another seal, and a pinned run
control (OURS_ROOT, AGP_EPISODE_TIMEOUT_S, AGP_END_GRACE_S) whose value differs from the manifest. episode.json records
the verdict (`freeze`). The policy gets its pins in the session's frozen_policy.json (the pinned codex release binary
and its sha256, the sha256 of the prompt rendering), which agp_core/run_agent.py re-checks right before codex starts. At the
end, the implementation files this process loaded are compared with the manifest (episode_metrics.json `freeze_audit`).

Development mode and A/B arms (episode.json freeze.mode). The check above is a frozen measurement's
(`frozen_measurement`: unchanged) or a test's (`test_override`: AGP_FREEZE_MANIFEST). Every other episode either
  * belongs to an A/B arm (`ab_arm`: AGP_ARM_MANIFEST=<the arm's manifest>, AGP_ARM_SEAL=<its seal>, set by
    agp_core/run_official.py --arm-manifest for agp_core/evaluation/ab.py): the arm's manifest (agp_core/freeze.py write_arm, sealed
    at the start of the A/B batch: the whole state above plus the arm's overlay of the frozen configuration) is
    verified
    against the current state with that overlay, exactly as strictly (any difference, another seal, a manifest without
    an arm section, AGP_FREEZE_MANIFEST next to it: freeze_mismatch, nothing runs); the episode then runs with
    the frozen configuration plus the overlay (deploy.yml's model, effort, service_tier replaced, the built-in switch
    set updated; episode.json frozen.arm_overlay).
    A frozen measurement refuses AGP_ARM_MANIFEST and AGP_ARM_SEAL (and XPolicyLab/policy/AGP/deploy.py refuses them
    first), and refuses a manifest with an arm section;
  * or is a development run (`development`): no manifest is verified, so changing the implementation does not block
    it. The state is collected anyway: its digest (state_digest), whether it still equals the frozen system's manifest
    (agp_core/paths.py FREEZE_MANIFEST; record only), and the policy's pins (codex binary, prompt rendering) come from it.

Run controls (environment of the simulator process; recorded in episode.json):
    AGP_SCRIPTED=<smoke|observe|idle>   scripted policy instead of codex (no LLM)
    AGP_FAULT=<fault>                   test-only fault injection (all but worker_init need AGP_SCRIPTED):
        mcp_start     the MCP server cannot start
        agent_exit    the policy launcher exits at once
        worker_init   the worker start-up fails
        stall         the policy never finishes (a stand-in that only sleeps): the episode watchdog ends it as a
                      policy failure, exactly as for codex
        mcp_crash     the scripted driver runs the MCP server over HTTP as agp_core/run_agent.py does
                      (agp_core/mcp_crash_server.py); mid-episode, after a paired motion, the server exits on its own
                      (code 70, no signal) while answering a tool call; the driver records it in policy_finished.json
                      as agp_core/run_agent.py would (mcp_server exited_during_episode): a counted policy failure.
                      Development tree only: the packaged runtime leaves agp_core/mcp_crash_server.py out
                      (scripts/package_policy.py), so this module refuses mcp_crash when it runs from the package
                      (agp_core/paths.py PACKAGED; DEVELOPMENT_ONLY_FAULTS)
        agent_crash   the scripted driver stands in for agp_core/run_agent.py: once its MCP session is up it writes
                      agent.pid (as run_agent.py does when codex starts), makes a paired motion and then dies on its
                      own (os._exit(70), no policy_finished.json): a counted policy failure
    AGP_EPISODE_TIMEOUT_S, AGP_END_GRACE_S   override deploy.yml agp.episode_timeout_s / agp.end_grace_s (a frozen
                                        measurement accepts only the manifest's values)
    AGP_FREEZE_MANIFEST=<path>          test only: verify this manifest instead of agp_core/paths.py FREEZE_MANIFEST
    AGP_FROZEN_MEASUREMENT=1, AGP_FREEZE_SEAL=<seal>   set by agp_core/run_official.py --frozen-measurement
    AGP_ARM_MANIFEST=<path>, AGP_ARM_SEAL=<seal>       an A/B arm (agp_core/run_official.py --arm-manifest)

Packaging (scripts/package_policy.py): this module runs from the packaged copy
XPolicyLab/policy/AGP/agp_runtime that deploy.yml agp.runtime_root names (load_config refuses a runtime_root outside the
policy folder, and AGP_HOME inside the checkout); the policy's interpreter, its TMPDIR, the sessions and codex's homes
are in AGP_HOME (agp_core/paths.py), which the policy process gets explicitly. Nothing is written into the policy folder.

Account pool (agp_core/accounts.py, leased by agp_core/run_agent.py): a launcher that finds no usable slot exits before codex
starts with policy_finished.json `account_error` (an infrastructure failure naming the accounts' use and reset times);
codex's own refresh / sign-in, credit and spend-cap failures count as provider errors (PROVIDER_ERROR).
episode_metrics.json `account` copies launch.json `account` and policy_finished.json `account_end` (labels, hashes, use,
write-back; never a credential); episode.json freeze.not_frozen `account_pool` is the pool summary (credentials stay
unfrozen).

The adapter touches TASK_ENV only through the whitelist of agp_core/official_env.py: no test hook reads the evaluator's
reward_manager or marks the environment unstable.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import traceback

import yaml

from agp_core import features
from agp_core import official_env as oe
from agp_core import paths

CODE = Path(__file__).resolve().parents[1]      # the implementation running (the packaged runtime)
W = CODE                                        # (older name)
AGP_HOME = paths.AGP_HOME                       # interpreters, sessions, TMPDIR, codex homes (never the checkout)
RUNS = AGP_HOME / "runs" / "official"
AGENT_PYTHON = AGP_HOME / ".venv-agent" / "bin" / "python"
SCRIPTED = ("smoke", "observe", "idle")
GRACE_REASON = "episode ended; grace period for the final observation and assessment elapsed"
HIDDEN_END_REASON = "official episode end; the policy is stopped at once (OURS_HIDE_END)"
WATCHDOG_REASON = "wall-clock limit"
FAULTS = ("mcp_start", "agent_exit", "worker_init", "stall", "mcp_crash", "agent_crash")
SCRIPTED_FAULTS = ("mcp_start", "agent_exit", "stall", "mcp_crash", "agent_crash")
# test-only faults whose stand-in file exists in the development tree only: scripts/package_policy.py leaves the crash
# server out of the published package, so the packaged runtime refuses the fault instead of failing later
DEVELOPMENT_ONLY_FAULTS = {"mcp_crash": "agp_core/mcp_crash_server.py"}
# a policy crash of our own code (counted policy failure, never re-run)
CRASH_KIND = "mcp_crash"
# Signals a process raises against itself when it crashes (memory fault, abort(), illegal instruction, ...): for our
# own processes (agp_core/run_agent.py, the MCP server, the scripted driver) these are crashes, not external kills.
FAULT_SIGNALS = frozenset(int(s) for s in (signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE, signal.SIGILL,
                                           signal.SIGABRT, signal.SIGTRAP, signal.SIGSYS))
REQUIRED_ON = ("OURS_POLICY_PLANNER", "OURS_RGB_ONLY")
# worker_ext switches act on measured finger joints, which the official observation does not contain
REQUIRED_OFF = ("OURS_FAST_PRECISION", "OURS_GRIP_CONVERGE", "OURS_HOLD_SMOOTH")
# deploy.yml agp.<key> -> the variable agp_core/run_agent.py reads
MODEL_SETTINGS = {"model": "OURS_MODEL", "effort": "OURS_EFFORT", "service_tier": "OURS_SERVICE_TIER"}
PASS_THROUGH = ("HOME", "USER", "LOGNAME", "LANG", "LANGUAGE", "LC_ALL", "LC_CTYPE", "TZ", "TERM",
                "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY", "all_proxy",
                "ALL_PROXY", "SSL_CERT_FILE", "SSL_CERT_DIR")
SIGNAL_EXIT_BASE = 128           # a process that reports a signal N as its exit status uses 128 + N
# the frozen system (agp_core/freeze.py)
FREEZE_KIND = "freeze_mismatch"
FROZEN_MEASUREMENT = "AGP_FROZEN_MEASUREMENT"
# an A/B arm's sealed manifest and the seal the launcher verified (agp_core/evaluation/ab.py, agp_core/run_official.py
# --arm-manifest); refused in a frozen measurement
ARM_MANIFEST = "AGP_ARM_MANIFEST"
ARM_SEAL = "AGP_ARM_SEAL"
# deploy.yml agp.<key>: paths, absolute once resolved (agp_core/paths.py deploy_pins: ~ = home, runtime_root and codex_binary
# relative to the policy folder); the freeze manifest is not a deploy.yml entry (agp_core/paths.py FREEZE_MANIFEST)
PINS = paths.PIN_KEYS
AUTO_UPDATED_CODEX = ("/standalone/current/", "/.local/bin/codex")
PROVIDER_ERROR = re.compile(
    r"\b(401|403|429|500|502|503|504)\b|unauthori[sz]ed|authenticat|forbidden|rate.?limit|quota|usage.?limit|"
    r"api.?key|stream (?:error|disconnected)|connection (?:refused|reset|error|closed)|network|"
    r"service unavailable|overloaded|"
    # codex's login renewal failing ("Your access token could not be refreshed ... Please log out and sign in again.",
    # a revoked login), and workspace credit / spend-cap ends (account pool)
    r"could not be refreshed|sign in again|invalidated oauth|out of credits|spend cap", re.I)


def log(message):
    print(f"[AGP] {message}", flush=True)


def atomic(path, data):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1, default=str, allow_nan=False))
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- configuration
def load_config(policy_dir, overlay=None):
    """deploy.yml of the policy folder: `agp:` holds the model settings, the episode limits and the pinned paths; the
    switch set is the built-in one (agp_core/features.py SYSTEM), and a deploy.yml `agp.switches` entry is refused, as is
    `agp.freeze_manifest` (the freeze manifest is agp_core/paths.py FREEZE_MANIFEST, relative to AGP_HOME).
    `overlay` (an A/B arm's configuration, taken from its verified arm manifest) replaces model settings and updates
    switches before the same checks run; deploy_yml_sha256 stays the file's, `arm_overlay` records the overlay."""
    deploy = yaml.safe_load((Path(policy_dir) / "deploy.yml").read_text()) or {}
    agp = deploy.get("agp") or {}
    if "switches" in agp:
        raise oe.AgentInfrastructureFailure("deploy.yml agp.switches is not a setting: the switch set is built into the "
                                            "runtime (agp_core/features.py SYSTEM)")
    if "freeze_manifest" in agp:
        raise oe.AgentInfrastructureFailure("deploy.yml agp.freeze_manifest is not a setting: the freeze manifest is "
                                            "fixed in the runtime (agp_core/paths.py FREEZE_MANIFEST "
                                            f"{paths.FREEZE_MANIFEST}, relative to AGP_HOME); remove the entry")
    if overlay is not None:
        from agp_core import freeze
        try:
            overlay = freeze.validate_overlay(overlay)
        except ValueError as exc:
            raise oe.AgentInfrastructureFailure(f"A/B arm overlay refused: {exc}", kind=FREEZE_KIND) from exc
        agp = freeze.effective_agp(agp, overlay)
    switches = features.system_switches()             # the built-in set, an A/B arm's overlay applied
    switches.update({str(k): str(int(v)) if isinstance(v, bool) else str(v)
                     for k, v in (agp.get("switches") or {}).items()})
    for name in REQUIRED_ON:
        if switches.get(name) != "1":
            raise oe.AgentInfrastructureFailure(f"the switch set must set {name}: 1")
    for name in REQUIRED_OFF:
        if switches.get(name, "0") != "0":
            raise oe.AgentInfrastructureFailure(f"the switch set must keep {name}: 0 (needs measured fingers)")
    unknown = [k for k in switches if not k.startswith("OURS_")]
    if unknown:
        raise oe.AgentInfrastructureFailure(f"the switch set has non-switch entries {unknown}")
    clash = sorted(set(switches) & set(MODEL_SETTINGS.values()))
    if clash:
        raise oe.AgentInfrastructureFailure(f"the switch set must not hold the model settings {clash}; "
                                            f"set agp.model, agp.effort and agp.service_tier")
    model = {}
    for key, variable in MODEL_SETTINGS.items():
        value = agp.get(key)
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise oe.AgentInfrastructureFailure(f"deploy.yml agp.{key} must be a non-empty string (frozen "
                                                f"{variable} of the policy), got {value!r}")
        model[variable] = value
    pins = {}
    resolved = paths.deploy_pins(agp, policy_dir)
    for key in PINS:                      # the pinned paths (agp_core/paths.py PIN_KEYS)
        value = resolved.get(key)
        if not isinstance(value, str) or not os.path.isabs(value):
            raise oe.AgentInfrastructureFailure(f"deploy.yml agp.{key} must be a path (absolute, ~/..., or relative "
                                                f"as agp_core/paths.py deploy_pins resolves it) (M2 S2), got "
                                                f"{agp.get(key)!r}")
        pins[key] = value
    if not paths.inside(pins["runtime_root"], policy_dir):
        raise oe.AgentInfrastructureFailure(f"deploy.yml agp.runtime_root must be inside the policy folder {policy_dir} "
                                            f"(the packaged runtime, scripts/package_policy.py), got {pins['runtime_root']}")
    folder = Path(policy_dir).resolve()
    if folder.parent.name == "policy" and paths.inside(AGP_HOME, folder.parents[2]):     # .../XPolicyLab/policy/AGP
        raise oe.AgentInfrastructureFailure(f"AGP_HOME {AGP_HOME} lies in the checkout {folder.parents[2]}: the runtime "
                                            f"would write there")
    if any(part in pins["codex_binary"] for part in AUTO_UPDATED_CODEX):
        raise oe.AgentInfrastructureFailure(f"deploy.yml agp.codex_binary must be a pinned codex release binary, not "
                                            f"the self-updating link {pins['codex_binary']}")
    return {"policy_name": deploy.get("policy_name"), "switches": switches, "model_settings": model, **pins,
            "episode_timeout_s": float(agp.get("episode_timeout_s", 14400)),
            "end_grace_s": float(agp.get("end_grace_s", 180)),
            "agent_finish_wait_s": float(agp.get("agent_finish_wait_s", 60)),
            "deploy_yml_sha256": sha256(Path(policy_dir) / "deploy.yml"), "arm_overlay": overlay}


def frozen_settings(config):
    """Every OURS_* value the policy runs with: the switch set (built in) plus the model settings (deploy.yml)."""
    return {**config["switches"], **config["model_settings"]}


NOT_SETTINGS = ("OURS_ROOT",)    # where the implementation lives (XPolicyLab/policy/AGP/deploy.py reads it)


def caller_settings_ignored(environ, config):
    """OURS_* values of the caller's environment that the frozen system does not take: {name: caller value}.

    A value equal to the frozen one is not listed (it changes nothing); every other OURS_* entry is, including names
    the frozen system does not use (e.g. OURS_PROMPT_FILE), except NOT_SETTINGS."""
    frozen = frozen_settings(config)
    return {k: v for k, v in sorted(environ.items())
            if k.startswith("OURS_") and k not in NOT_SETTINGS and frozen.get(k) != v}


def run_controls(environ, config):
    scripted = environ.get("AGP_SCRIPTED") or None
    fault = environ.get("AGP_FAULT") or None
    if scripted is not None and scripted not in SCRIPTED:
        raise oe.AgentInfrastructureFailure(f"AGP_SCRIPTED must be one of {SCRIPTED}, got {scripted!r}")
    if fault is not None and fault not in FAULTS:
        raise oe.AgentInfrastructureFailure(f"AGP_FAULT must be one of {FAULTS}, got {fault!r}")
    if fault in SCRIPTED_FAULTS and scripted is None:
        raise oe.AgentInfrastructureFailure(f"AGP_FAULT={fault} is a test of the scripted path; set AGP_SCRIPTED")
    if fault in DEVELOPMENT_ONLY_FAULTS and paths.PACKAGED:
        raise oe.AgentInfrastructureFailure(
            f"AGP_FAULT={fault} needs {DEVELOPMENT_ONLY_FAULTS[fault]}, a test-only file of the development tree that "
            f"the packaged runtime {CODE} leaves out (scripts/package_policy.py); it runs only in the development "
            f"tree's own tests")
    return {"mode": f"scripted:{scripted}" if scripted else "codex", "scripted": scripted, "fault": fault,
            "episode_timeout_s": float(environ.get("AGP_EPISODE_TIMEOUT_S") or config["episode_timeout_s"]),
            "end_grace_s": float(environ.get("AGP_END_GRACE_S") or config["end_grace_s"]),
            "agent_finish_wait_s": config["agent_finish_wait_s"],
            "arm_manifest": environ.get(ARM_MANIFEST) or None}


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(text))[:120] or "run"


def new_session(run_id, layout_id, runs=None):
    run_dir = Path(RUNS if runs is None else runs) / safe_name(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(rf"layout-{int(layout_id)}-a(\d+)")
    attempts = [int(m.group(1)) for p in run_dir.iterdir() if (m := pattern.fullmatch(p.name))]
    attempt = max(attempts, default=0) + 1
    session = run_dir / f"layout-{int(layout_id)}-a{attempt}"
    session.mkdir()
    return session, attempt


def implementation_manifest(policy_dir):
    files = sorted(list((CODE / "agp_core").glob("*.py")) + [CODE / "agp_core" / "PROMPT.md"]
                   + list((CODE / "migration").glob("*.py")))
    out = {str(p.relative_to(CODE)): sha256(p) for p in files if p.is_file()}
    out.update({f"policy/{p.name}": sha256(p) for p in sorted(Path(policy_dir).iterdir()) if p.is_file()})
    return out


# ----------------------------------------------------------------------------- the frozen system
def _freeze_problem(entry, expected, found):
    return {"entry": entry, "path": [entry], "expected": str(expected), "found": str(found)}


def freeze_mode(environ):
    """Which system check an episode gets (see the module docstring): frozen_measurement (AGP_FROZEN_MEASUREMENT=1),
    ab_arm (AGP_ARM_MANIFEST: an A/B arm's sealed manifest), test_override (AGP_FREEZE_MANIFEST, test only) or
    development (no manifest verified; the state is recorded)."""
    if environ.get(FROZEN_MEASUREMENT) == "1":
        return "frozen_measurement"
    if environ.get(ARM_MANIFEST):
        return "ab_arm"
    if environ.get("AGP_FREEZE_MANIFEST"):
        return "test_override"
    return "development"


def freeze_check(policy_dir, config, environ):
    """Check the system about to run against its manifest (agp_core/freeze.py) and, in a frozen measurement, the run
    controls. Returns (record for episode.json, problems, the state the policy pins come from, or None); problems []
    = may run. A verification that cannot run (agp_core/freeze.py itself broken) is a mismatch too. A development
    run verifies no manifest (it records the current state's digest instead); an A/B arm verifies its own manifest."""
    mode = freeze_mode(environ)
    try:
        record, problems, state = (_development_check(policy_dir, config) if mode == "development"
                                   else _freeze_check(policy_dir, config, environ, mode))
    except Exception as exc:
        traceback.print_exc()
        problem = _freeze_problem("verification", "a completed verification", f"{type(exc).__name__}: {exc}")
        record = {"mode": mode, "manifest": environ.get(ARM_MANIFEST) or environ.get("AGP_FREEZE_MANIFEST")
                  or paths.freeze_manifest(),
                  "override": bool(environ.get("AGP_FREEZE_MANIFEST")),
                  "frozen_measurement": environ.get(FROZEN_MEASUREMENT) == "1", "seal": None, "ok": False,
                  "problem_count": 1, "problems": [problem], "entries": None, "seconds": None}
        return record, [problem], None
    if os.path.realpath(config["runtime_root"]) != os.path.realpath(CODE):
        problems.append(_freeze_problem("deploy.yml agp.runtime_root", f"{CODE} (the implementation running)",
                                        config["runtime_root"]))
        record.update(ok=False, problem_count=len(problems), problems=problems[:50])
    return record, problems, state


def _not_frozen(policy_dir):
    from agp_core import freeze
    return {"user_skills_disabled": freeze.skill_overrides_record(policy_dir),   # (codex's home: fresh per episode)
            "account_pool": freeze.account_pool_record()}


def _freeze_check(policy_dir, config, environ, mode):
    from agp_core import freeze
    frozen_measurement = mode == "frozen_measurement"
    arm = mode == "ab_arm"
    override = environ.get("AGP_FREEZE_MANIFEST") or None
    if arm:
        path, expected_seal = environ[ARM_MANIFEST], environ.get(ARM_SEAL) or None
    else:
        path, expected_seal = override or paths.freeze_manifest(), environ.get("AGP_FREEZE_SEAL") or None
    try:
        verdict = freeze.verify(path, policy_dir, expected_seal=expected_seal)
    except Exception as exc:              # a broken verification refuses like a mismatch
        traceback.print_exc()
        verdict = {"ok": False, "seal": None, "entries": 0, "seconds": None,
                   "problems": [_freeze_problem("verification", "a completed verification", f"{type(exc).__name__}: "
                                                                                           f"{exc}")]}
    problems = list(verdict["problems"])
    try:
        manifest = freeze.load_manifest(path)
        state = manifest.get("state")
    except (OSError, ValueError, AttributeError):
        manifest, state = None, None
    arm_spec = freeze.arm_of(manifest) if isinstance(manifest, dict) else None
    if frozen_measurement:
        problems.extend(freeze.run_control_problems(environ, state))
        if not expected_seal:
            problems.append(_freeze_problem("AGP_FREEZE_SEAL", "the seal of the manifest the launcher verified",
                                            "<unset>"))
    if arm:                               # the arm's manifest, sealed by the A/B runner at the batch start
        if arm_spec is None:
            problems.append(_freeze_problem(ARM_MANIFEST, "an A/B arm manifest (state section arm)",
                                            f"{path}: no arm section"))
        if not expected_seal:
            problems.append(_freeze_problem(ARM_SEAL, "the seal of the arm manifest the launcher verified", "<unset>"))
        if override:
            problems.append(_freeze_problem("AGP_FREEZE_MANIFEST", "unset (an A/B arm verifies its own manifest)",
                                            override))
    elif arm_spec is not None:
        problems.append(_freeze_problem("manifest arm section", "a frozen system's manifest (no A/B arm section)",
                                        json.dumps(arm_spec, sort_keys=True)))
    record = {"mode": mode, "manifest": str(path), "override": override is not None,
              "frozen_measurement": frozen_measurement, "expected_seal": expected_seal, "seal": verdict.get("seal"),
              "created_utc": verdict.get("created_utc"), "ok": not problems, "problem_count": len(problems),
              "problems": problems[:50], "entries": verdict.get("entries"), "seconds": verdict.get("seconds"),
              "not_frozen": _not_frozen(policy_dir)}
    if arm:
        record["arm"] = arm_spec
    return record, problems, state if isinstance(state, dict) else None


def _development_check(policy_dir, config):
    """A development run (no frozen measurement, no A/B arm, no test override) verifies no manifest. It records
    the digest of the current state and, for the record only, whether that state still equals the frozen system's
    manifest (agp_core/paths.py FREEZE_MANIFEST); the policy's pins (codex binary, prompt rendering) come from the current
    state."""
    from agp_core import freeze
    started = time.monotonic()
    current = freeze.collect(policy_dir)
    frozen = {"manifest": paths.freeze_manifest(), "matches": None, "differences": None, "first": []}
    try:
        manifest = freeze.load_manifest(frozen["manifest"])
        differences = freeze.check_manifest(manifest, current)
        frozen.update(seal=manifest.get("seal"), matches=not differences, differences=len(differences),
                      first=[p["entry"] for p in differences[:5]])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        frozen["error"] = f"{type(exc).__name__}: {exc}"
    record = {"mode": "development", "manifest": None, "override": False, "frozen_measurement": False,
              "expected_seal": None, "seal": None, "verified": False, "ok": True, "problem_count": 0, "problems": [],
              "state_digest": freeze.state_digest(current), "entries": sum(1 for _ in freeze.leaves(current)),
              "seconds": round(time.monotonic() - started, 3), "deploy_manifest_for_record": frozen,
              "not_frozen": _not_frozen(policy_dir)}
    return record, [], current


def frozen_policy_pins(config, record, state):
    """session/frozen_policy.json: what agp_core/run_agent.py re-checks right before codex starts (official conditions)."""
    codex = (state or {}).get("codex") or {}
    prompt = (state or {}).get("prompt") or {}
    return {"codex_binary": config["codex_binary"], "codex_sha256": codex.get("binary_sha256"),
            "codex_version": codex.get("version_output"), "prompt_rendered_sha256": prompt.get("rendered_sha256"),
            "manifest": record["manifest"], "seal": record["seal"], "written_by": "agp_core/official_deploy.py"}


# ----------------------------------------------------------------------------- the policy process
def agent_environment(frozen, policy_timeout_s, environ=None):
    """Clean environment for the policy: nothing of Isaac/Kit, CUDA, LD_*, PYTHONPATH or the sim venv, and of the
    caller's OURS_* values nothing at all: `frozen` (frozen_settings(config): the built-in switch set and deploy.yml's
    model settings) is the only source of OURS_* entries apart from the policy time limit. AGP_HOME is this process's
    (packaging: the policy's interpreter and codex homes are there; TMPDIR is AGP_HOME/tmp)."""
    environ = os.environ if environ is None else environ
    env = {k: environ[k] for k in PASS_THROUGH if k in environ}
    tmp = AGP_HOME / "tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    env.update(PATH="/usr/local/bin:/usr/bin:/bin", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONUNBUFFERED="1", TMPDIR=str(tmp), AGP_HOME=str(AGP_HOME),
               OURS_POLICY_TIMEOUT_S=str(int(policy_timeout_s)))
    env.update(frozen)
    return env


STALL = ("import os, sys, time; print('[AGP fault] policy stand-in that never finishes (pid %d)' % os.getpid(), "
         "flush=True); time.sleep(10 ** 6)")


def agent_command(controls, session):
    py = [str(AGENT_PYTHON), "-B"]
    if controls["fault"] == "agent_exit":
        return py + ["-c", "import sys; print('[AGP fault] agent launcher exits at once', flush=True); sys.exit(7)"]
    if controls["fault"] == "stall":
        return py + ["-c", STALL, str(session)]
    if controls["scripted"]:
        cmd = py + [str(CODE / "agp_core" / "official_scripted.py"), str(session), "--name", controls["scripted"]]
        driver_faults = ("mcp_start", "mcp_crash", "agent_crash")
        return cmd + (["--fault", controls["fault"]] if controls["fault"] in driver_faults else [])
    return py + [str(CODE / "agp_core" / "run_agent.py"), str(session)]


class PolicyProcess:
    """The policy launcher in its own process group (killed as a group)."""

    def __init__(self, cmd, env, session):
        self.cmd = cmd
        self.log_path = Path(session) / "launcher.log"
        self.stopped_by_adapter = False
        with self.log_path.open("w") as out:
            self.process = subprocess.Popen(cmd, cwd=str(CODE), env=env, stdin=subprocess.DEVNULL, stdout=out,
                                            stderr=subprocess.STDOUT, start_new_session=True)
        self.pgid = self.process.pid
        self.started = time.monotonic()

    def poll(self):
        return self.process.poll()

    def _signal_group(self, sig):
        try:
            os.killpg(self.pgid, sig)
        except ProcessLookupError:
            pass

    def stop(self, wait_s=0.0):
        """Give the launcher wait_s to exit by itself, then SIGTERM the group, then SIGKILL. Returns its code."""
        if self.process.poll() is None and wait_s > 0:
            try:
                self.process.wait(timeout=wait_s)
            except subprocess.TimeoutExpired:
                pass
        if self.process.poll() is None:
            self.stopped_by_adapter = True
            self._signal_group(signal.SIGTERM)
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                self.process.wait()
        # children that outlived the launcher (MCP server, codex) belong to the same group
        self._signal_group(signal.SIGTERM)
        return self.process.returncode


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def provider_error_text(session, tail_chars=20000, tail_events=4000):
    """Where a provider or authentication error of codex shows: its stderr, the launcher's log and codex's own error
    events in agent_events.jsonl (type `error` / `turn.failed`, or an item of type `error`). Tool results and model
    messages are left out: they hold arbitrary numbers and words (image data, coordinates, step counts), which made
    the error pattern match by chance (2 of 222 earlier codex sessions, e.g. `401` inside base64 image data)."""
    session = Path(session)
    texts = []
    for name in ("agent_stderr.log", "launcher.log"):
        try:
            texts.append((session / name).read_text(errors="replace")[-tail_chars:])
        except OSError:
            pass
    try:
        lines = (session / "agent_events.jsonl").read_text(errors="replace").splitlines()[-tail_events:]
    except OSError:
        lines = []
    for line in lines:
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if not isinstance(event, dict):
            continue
        item = event.get("item")
        if event.get("type") in ("error", "turn.failed") or (isinstance(item, dict) and item.get("type") == "error"):
            texts.append(json.dumps(event)[:4000])
    return "\n".join(texts)


def signal_of(returncode):
    """The signal behind a return code: -N (Popen status of a process killed by signal N) or 128+N (a process that
    reports a signal N that way); None for any other value."""
    if isinstance(returncode, bool) or not isinstance(returncode, int):
        return None
    if returncode < 0:
        return -returncode
    if SIGNAL_EXIT_BASE < returncode <= SIGNAL_EXIT_BASE + 64:
        return returncode - SIGNAL_EXIT_BASE
    return None


def signal_name(number):
    try:
        return signal.Signals(number).name
    except (ValueError, TypeError):
        return f"signal {number}"


def launcher_name(controls):
    if controls["fault"] in ("agent_exit", "stall"):
        return "policy stand-in (test fault)"
    if controls["fault"] == "agent_crash":
        return "agp_core/official_scripted.py (stand-in for agp_core/run_agent.py, AGP_FAULT=agent_crash)"
    return "agp_core/official_scripted.py" if controls["scripted"] else "agp_core/run_agent.py"


def _how(code, number):
    return f"exited with code {code}" if number is None else f"crashed with {signal_name(number)} (return code {code})"


def external_kill(controls, returncode, stopped_by_adapter, session):
    """The policy process that a signal our own code did not send has ended, as a dict, else None.

    The policy launcher (agp_core/run_agent.py, the scripted driver or a test stand-in) is the adapter's direct child: a
    status -N means a signal killed it, and the adapter knows whether it sent one (`stopped_by_adapter`: end grace,
    wall-clock limit, or after policy_finished.json). codex and the MCP server are children of agp_core/run_agent.py,
    which records in policy_finished.json who stopped codex (`stopped_by`: end_on_assess, policy_timeout or
    mcp_server_exited, else null) and a signal it did not send (`external_signal`: process, returncode, signal).
    For a policy_finished.json without that attribution (an older launcher) a codex status -N or 128+N counts,
    since run_agent.py reports its own stops as 0 or 124.

    A fault signal (FAULT_SIGNALS) of the launcher or of the MCP server is that process crashing by itself, not
    an external kill (policy_crash() decides); codex's signals are unchanged. The scripted driver's records
    (AGP_FAULT=mcp_crash) are read the same way as run_agent.py's."""
    session = Path(session)
    if stopped_by_adapter:
        return None
    number = signal_of(returncode)
    if number is not None:
        if number in FAULT_SIGNALS:
            return None                   # the launcher crashed by itself
        return {"process": launcher_name(controls), "returncode": returncode, "signal": number,
                "signal_name": signal_name(number), "seen_by": "adapter (exit status of the policy launcher)"}
    finished = read_json(session / "policy_finished.json")
    if not isinstance(finished, dict):
        return None
    recorded = finished.get("external_signal")
    if isinstance(recorded, dict) and signal_of(recorded.get("returncode")) is not None:
        number = signal_of(recorded["returncode"])
        process = str(recorded.get("process") or "codex")
        if not (process == "mcp_server" and number in FAULT_SIGNALS):     # else the MCP server crashed by itself
            return {"process": process, "returncode": recorded["returncode"],
                    "signal": number, "signal_name": signal_name(number),
                    "seen_by": f"{launcher_name(controls)} (policy_finished.json external_signal)"}
    server = finished.get("mcp_server")
    if isinstance(server, dict) and server.get("exited_during_episode"):
        number = signal_of(server.get("returncode"))
        if number is not None and number not in FAULT_SIGNALS:
            return {"process": "mcp_server", "returncode": server.get("returncode"), "signal": number,
                    "signal_name": signal_name(number),
                    "seen_by": f"{launcher_name(controls)} (policy_finished.json mcp_server)"}
    if controls["scripted"]:
        return None
    if "stopped_by" not in finished:
        code = finished.get("returncode")
        number = signal_of(code)
        if number is not None:
            return {"process": "codex", "returncode": code, "signal": number, "signal_name": signal_name(number),
                    "seen_by": "agp_core/run_agent.py (policy_finished.json returncode, no stop attribution)"}
    return None


def codex_started(session):
    """agp_core/run_agent.py writes agent.pid right after codex started (the AGP_FAULT=agent_crash stand-in too)."""
    return (Path(session) / "agent.pid").exists()


def policy_crash(controls, returncode, stopped_by_adapter, session):
    """Our own policy code died mid-episode by itself, as a dict, else None.

      the MCP server   policy_finished.json `mcp_server.exited_during_episode` (agp_core/run_agent.py records it while
                       codex runs; the scripted driver's AGP_FAULT=mcp_crash path records it the same way) with an exit
                       status that is not a signal, or a fault signal of its own: it crashed or exited while the policy
                       was using it;
      the launcher     agp_core/run_agent.py (codex path) or its AGP_FAULT=agent_crash stand-in, after codex started
                       (codex_started): a non-zero exit status that is not a signal, a fault signal, or an end without
                       policy_finished.json.
    Not a crash: an end the adapter caused (stopped_by_adapter), an external kill (external_kill() decides first), a
    launcher end before codex started (start-up: the MCP server cannot start, preload or configuration fails; an
    infrastructure abort) and the scripted driver's own failures (a failed check of the test sequence)."""
    session = Path(session)
    if stopped_by_adapter:
        return None
    finished = read_json(session / "policy_finished.json")
    server = finished.get("mcp_server") if isinstance(finished, dict) else None
    if isinstance(server, dict) and server.get("exited_during_episode"):
        code = server.get("returncode")
        number = signal_of(code)
        if number is None or number in FAULT_SIGNALS:
            return {"process": "mcp_server", "returncode": code, "signal": number,
                    "signal_name": signal_name(number) if number is not None else None,
                    "what": f"the MCP server {_how(code, number)} while the policy was running (see mcp_server.log)",
                    "seen_by": f"{launcher_name(controls)} (policy_finished.json mcp_server)"}
    if controls["scripted"] is None or controls["fault"] == "agent_crash":
        if not codex_started(session):
            return None                   # ended before codex started: start-up, see _other_infrastructure
        name = launcher_name(controls)
        number = signal_of(returncode)
        if returncode != 0 and (number is None or number in FAULT_SIGNALS):
            return {"process": name, "returncode": returncode, "signal": number,
                    "signal_name": signal_name(number) if number is not None else None,
                    "what": f"{name} {_how(returncode, number)} after codex started (see launcher.log)",
                    "seen_by": "adapter (exit status of the policy launcher)"}
        if returncode == 0 and not isinstance(finished, dict):
            return {"process": name, "returncode": returncode, "signal": None, "signal_name": None,
                    "what": f"{name} ended without policy_finished.json after codex started (see launcher.log)",
                    "seen_by": "adapter (exit status 0, no policy_finished.json)"}
    return None


def _other_infrastructure(controls, returncode, stopped_by_adapter, session):
    """Reason a policy end that is neither an external kill nor a policy crash is an infrastructure failure, None for
    a policy outcome."""
    session = Path(session)
    if controls["fault"] == "stall":
        # test stand-in for a policy that runs out of time: the episode watchdog's stop is a policy outcome, as for
        # codex; anything else would be a broken test
        return None if stopped_by_adapter else f"policy stand-in ended with code {returncode} before the watchdog"
    if controls["fault"] in ("agent_exit", "agent_crash") or controls["scripted"]:
        if stopped_by_adapter:
            return "scripted policy did not finish on its own"
        if returncode != 0:
            report = read_json(session / "scripted_report.json") or {}
            return f"scripted policy exited with code {returncode}: {report.get('error') or 'see launcher.log'}"
        return None
    if stopped_by_adapter:
        return None                       # end grace or wall-clock limit: an honest failure, not an outage
    finished = read_json(session / "policy_finished.json")
    if isinstance(finished, dict) and finished.get("account_error"):
        return str(finished["account_error"])     # no usable account slot (agp_core/accounts.py): before codex started
    # The two cases below are reached only before codex started (policy_crash() takes the later ones).
    if returncode != 0:
        return f"agent launcher agp_core/run_agent.py exited with code {returncode} (see launcher.log)"
    finished = read_json(session / "policy_finished.json")
    if not isinstance(finished, dict):
        return "agent launcher ended without policy_finished.json"
    code = finished.get("returncode")
    if code in (0, 124) or finished.get("stopped_by") in ("end_on_assess", "policy_timeout"):
        return None
    tool_calls = 0
    try:
        tool_calls = len((session / "tool_timing.jsonl").read_text().splitlines())
    except OSError:
        pass
    if tool_calls == 0:
        return f"codex exited with code {code} before any tool call (authentication or configuration)"
    match = PROVIDER_ERROR.search(provider_error_text(session))
    if match:
        return f"codex exited with code {code} after a provider error ({match.group(0)!r})"
    return None                           # the model stopped on its own: a policy outcome


def policy_end_verdict(controls, returncode, stopped_by_adapter, session):
    """(kind, reason, details) of the policy's end: kind None for a policy outcome (success or not, ended honestly),
    'mcp_crash' for a crash of our own policy code (a policy failure too, counted and recorded apart), or
    'external_kill' / 'infrastructure' for a failure that must not be counted.

    codex (policy_finished.json `returncode` = codex's exit status as agp_core/run_agent.py saw it):
      killed by a signal our code did not send (external_kill())                         -> external_kill
      the MCP server exited or crashed while codex was running, not by an external signal -> mcp_crash
      agp_core/run_agent.py died by itself after codex started (policy_crash())               -> mcp_crash
      agp_core/run_agent.py ended before codex started (MCP server start-up, configuration)    -> infrastructure
      0 or 124 (run_agent.py ended it after an accepted assessment / at its time limit)  -> policy outcome
      no tool call at all                                                                  -> infrastructure
      non-zero exit after a provider / authentication error (provider_error_text)          -> infrastructure
      any other non-zero exit (the model stopped on its own)                               -> policy outcome
      stopped by the adapter (end grace, wall-clock limit)                                 -> policy outcome
    The scripted test policy must finish on its own with code 0 (else infrastructure); a signal other than a fault
    signal kills it externally like codex. AGP_FAULT=stall is the test stand-in whose watchdog stop is a policy
    outcome; AGP_FAULT=mcp_crash and agent_crash produce the two kinds of mcp_crash."""
    kill = external_kill(controls, returncode, stopped_by_adapter, session)
    if kill is not None:
        return ("external_kill", f"external kill: {kill['process']} was ended by {kill['signal_name']} (return code "
                                 f"{kill['returncode']}), a signal our code did not send", kill)
    crash = policy_crash(controls, returncode, stopped_by_adapter, session)
    if crash is not None:
        return (CRASH_KIND, f"policy crash (our own code): {crash['what']}", crash)
    reason = _other_infrastructure(controls, returncode, stopped_by_adapter, session)
    return (None, None, None) if reason is None else ("infrastructure", reason, {})


def classify_policy_end(controls, returncode, stopped_by_adapter, session):
    """None for a policy outcome (a policy crash too: it is counted), else the reason it is an
    infrastructure failure (external kills included)."""
    kind, reason, _ = policy_end_verdict(controls, returncode, stopped_by_adapter, session)
    return reason if kind in ("external_kill", "infrastructure") else None


# ----------------------------------------------------------------------------- the episode
def _benchmark_root():
    from env.global_configs import ROOT_DIR
    return os.path.realpath(ROOT_DIR)


def _check_imports(root):
    provenance = oe.module_provenance()
    # the development tree's patched copy (present when AGP_HOME is the development tree; the package has none)
    problems, notes = oe.provenance_problems(provenance, root, AGP_HOME / "source" / "RoboDojo")
    if os.path.realpath(root) == os.path.realpath(AGP_HOME / "source" / "RoboDojo"):
        problems.append(f"env.global_configs.ROOT_DIR is the patched copy {root}")
    return provenance, problems, notes


def _public_camera_root(root):
    os.environ.setdefault("ROBODOJO_ROOT", root)
    from agp_core import public_camera as pc
    model_root = str(pc.default_model().root)
    if os.path.realpath(model_root) != root:
        raise oe.AgentInfrastructureFailure(f"public camera model reads {model_root}, not the benchmark {root} "
                                            f"(set ROBODOJO_ROOT)")
    return model_root


EPISODES_STARTED = [0]   # this process's own count of episodes (the episode index; agp_core/run_official.py: one per process)


def eval_one_episode(TASK_ENV, model_client=None, *, policy_dir):
    del model_client                      # the stub policy server holds no model; nothing is sent to it
    started = time.monotonic()
    proxy = oe.TaskEnvWhitelist(TASK_ENV)
    access = proxy.access_log
    infra = fault = None
    session = policy = worker = env = None
    serve_reason = None
    metrics = {}
    policy_end = {}
    freeze_state = None
    try:
        config = load_config(policy_dir)
        controls = run_controls(os.environ, config)
        meta = proxy.episode_metadata()
        meta["episode_index"] = EPISODES_STARTED[0]
        EPISODES_STARTED[0] += 1
        if meta["num_envs"] != 1:
            raise oe.AgentInfrastructureFailure(f"{meta['num_envs']} environments; AGP runs one (eval_batch false)")
        run_id = os.environ.get("ROBODOJO_RUN_ID") or "unknown-run"
        session, attempt = new_session(run_id, meta["layout_id"])
        access.path = session / "task_env_access.json"
        # the frozen system is verified before anything else runs (development runs record their state, an A/B arm
        # verifies its own sealed manifest and then runs with that arm's configuration)
        freeze_record, freeze_problems, freeze_state = freeze_check(policy_dir, config, os.environ)
        arm = freeze_record.get("arm") if freeze_record.get("mode") == "ab_arm" else None
        if arm and not freeze_problems:
            config = load_config(policy_dir, arm.get("overlay"))
        frozen = frozen_settings(config)
        ignored = caller_settings_ignored(os.environ, config)
        root = _benchmark_root()
        provenance, problems, notes = _check_imports(root)
        record = dict(meta, run_id=run_id, attempt=attempt, session=str(session), policy=config["policy_name"],
                      mode=controls["mode"], run_controls=controls, switches=config["switches"],
                      frozen={"model_settings": config["model_settings"], "switches": config["switches"],
                              "deploy_yml_sha256": config["deploy_yml_sha256"],
                              "codex_binary": config["codex_binary"], "arm_overlay": config.get("arm_overlay")},
                      caller_settings_ignored=ignored, freeze=freeze_record,
                      benchmark_root=root, cwd=os.getcwd(), pid=os.getpid(), started_wall=time.time(),
                      import_provenance=provenance, import_notes=notes,
                      implementation_sha256=implementation_manifest(policy_dir))
        atomic(session / "episode.json", record)
        log(f"episode {meta['task']} seed {meta['seed']} layout {meta['layout_id']} index {meta['episode_index']} "
            f"limit {meta['official_step_lim']} mode {controls['mode']} session {session}")
        model = config["model_settings"]
        log(f"frozen system (deploy.yml sha256 {config['deploy_yml_sha256'][:12]}): model "
            f"{model['OURS_MODEL']}, effort {model['OURS_EFFORT']}, service_tier {model['OURS_SERVICE_TIER']}; "
            f"switches " + " ".join(f"{k[5:]}={v}" for k, v in config["switches"].items())
            + (f"; A/B arm {arm.get('experiment')}/{arm.get('arm')} overlay "
               f"{json.dumps(config.get('arm_overlay'), sort_keys=True)}" if config.get("arm_overlay") else ""))
        for name, value in ignored.items():
            source = "deploy.yml" if name in config["model_settings"] else "the built-in switch set"
            log(f"ignoring caller setting {name}={value!r} (the policy gets "
                f"{frozen[name]!r} from {source})" if name in frozen else
                f"ignoring caller setting {name}={value!r} (not part of the frozen system)")
        if freeze_record.get("mode") == "development":
            dev = freeze_record.get("deploy_manifest_for_record") or {}
            log(f"development mode: no manifest verified (M3 S0); state digest {freeze_record['state_digest'][:16]}, "
                f"{freeze_record['entries']} entries, {freeze_record['seconds']} s; the frozen system's manifest "
                f"{dev.get('manifest')} " + ("still matches" if dev.get("matches") else
                                             f"differs in {dev.get('differences')} entries (record only)"))
        else:
            log(f"{'frozen system' if freeze_record.get('mode') != 'ab_arm' else 'A/B arm manifest'} "
                f"{'verified' if not freeze_problems else 'MISMATCH'}: manifest {freeze_record['manifest']}"
                f"{' (test override)' if freeze_record['override'] else ''}, seal {str(freeze_record['seal'])[:16]}, "
                f"{freeze_record['entries']} entries, {freeze_record['seconds']} s"
                + (", frozen measurement" if freeze_record["frozen_measurement"] else ""))
        if freeze_problems:
            for problem in freeze_problems[:10]:
                log(f"  differs: {problem['entry']}: expected {problem['expected']}, found {problem['found']}")
            raise oe.AgentInfrastructureFailure(
                f"frozen system mismatch, refused before the policy starts: {len(freeze_problems)} entr"
                f"{'y differs' if len(freeze_problems) == 1 else 'ies differ'} ("
                + ", ".join(p["entry"] for p in freeze_problems[:3]) + ")",
                kind=FREEZE_KIND, details={"manifest": freeze_record["manifest"], "problems": freeze_problems[:50]})
        if problems:
            raise oe.AgentInfrastructureFailure(f"benchmark code imported from outside the checkout: {problems}")
        record["public_camera_root"] = _public_camera_root(root)

        proxy.enter("policy")
        env = oe.OfficialEnv(proxy, oe.public_arms(meta["task"], root), task=meta["task"],
                             collect_freq=oe.public_collect_freq(root), step_lim=meta["official_step_lim"])
        record.update(first_observation_schema=env.first_observation_schema,
                      depth_or_calibration_fields=oe.depth_fields(env.first_observation_schema))
        atomic(session / "episode.json", record)
        if record["depth_or_calibration_fields"]:
            raise oe.AgentInfrastructureFailure(f"the observation has non-official fields "
                                                f"{record['depth_or_calibration_fields']}")
        try:
            if controls["fault"] == "worker_init":
                raise RuntimeError("injected worker start-up failure (AGP_FAULT=worker_init)")
            from agp_core import official_worker
            # OURS_HIDE_END: for codex only; a scripted test policy reads the flag and finishes on its own
            hide_end = features.enabled("hide_end", config["switches"]) and not controls["scripted"]
            worker_switches = dict(config["switches"], **({} if hide_end else {"OURS_HIDE_END": "0"}))
            worker = official_worker.build_worker(env, session, worker_switches)
        except (oe.SimulatorFault, oe.AgentInfrastructureFailure):
            raise
        except Exception as exc:
            raise oe.AgentInfrastructureFailure(f"worker start-up failed: {type(exc).__name__}: {exc}") from exc
        log(f"worker ready ({type(worker).__name__}); starting {controls['mode']}")
        atomic(session / "frozen_policy.json", frozen_policy_pins(config, freeze_record, freeze_state))

        policy = PolicyProcess(agent_command(controls, session),
                               agent_environment(frozen, max(60.0, controls["episode_timeout_s"] - 600)),
                               session)
        deadline = started + controls["episode_timeout_s"]
        ended_at = [None]

        def should_stop():
            if (session / "policy_finished.json").exists():
                return "policy finished"
            code = policy.poll()
            if code is not None:
                return f"policy process exited with code {code}"
            now = time.monotonic()
            if env.is_episode_end():
                ended_at[0] = ended_at[0] or now
                if hide_end:
                    return HIDDEN_END_REASON
                if now - ended_at[0] > controls["end_grace_s"]:
                    return GRACE_REASON
            if now > deadline:
                return WATCHDOG_REASON
            return None

        serve_reason = worker.serve(should_stop)
        log(f"bridge closed: {serve_reason}")
    except oe.SimulatorFault as exc:
        fault = exc
    except oe.AgentInfrastructureFailure as exc:
        infra = exc
    except Exception as exc:
        infra = oe.AgentInfrastructureFailure(f"unexpected adapter failure: {type(exc).__name__}: {exc}")
        infra.__cause__ = exc
        traceback.print_exc()
    finally:
        policy_code = stopped = None
        if policy is not None:
            wait = 0.0
            if serve_reason == "policy finished":
                wait = controls["agent_finish_wait_s"]
            policy_code = policy.stop(wait)
            stopped = policy.stopped_by_adapter
            if infra is None and fault is None:
                kind, reason, details = policy_end_verdict(controls, policy_code, stopped, session)
                policy_end = {"kind": kind or "policy", "reason": reason, "details": details}
                if kind == "external_kill":
                    # A signal we did not send ended the policy. Before the official end it took the episode away
                    # from the policy: not counted. After the official end the
                    # evaluator's verdict was already fixed and stands.
                    if _official_end_seen(env):
                        policy_end["external_kill_after_official_end"] = details
                        log(f"{reason}, after the official episode end: the evaluator's verdict stands")
                    else:
                        infra = oe.AgentInfrastructureFailure(reason, kind="external_kill", details=details)
                elif kind == CRASH_KIND:
                    # Our own code (MCP server, agp_core/run_agent.py) died mid-episode by itself: a policy failure that
                    # counts. No abort: finalize ends an unfinished episode as a failure
                    # and main.py records the result; recorded apart (mcp_crash.json), never re-run.
                    after = _official_end_seen(env)
                    policy_end.update(counted=True, after_official_end=after)
                    log(f"{reason}; " + ("after the official episode end: the evaluator's verdict stands" if after else
                                         "the episode ends as a counted policy failure (not re-run)"))
                elif kind:
                    infra = oe.AgentInfrastructureFailure(reason)
        metrics = _finalize(proxy, env, worker, session, fault, infra, serve_reason, policy_code, stopped, started,
                            policy_end, freeze_audit=_freeze_audit(freeze_state, policy_dir))
        if policy_end.get("kind") == CRASH_KIND and session is not None:
            _record_crash(session, policy_end, metrics)
    if fault is not None:
        log(f"simulator fault, re-raised for main.py: {fault}")
        raise fault.original
    if infra is None and access.refused:
        infra = oe.AgentInfrastructureFailure(f"TASK_ENV whitelist violated: {access.refused[:3]}")
    if infra is not None:
        kind = getattr(infra, "kind", "infrastructure")
        log(f"infrastructure failure ({kind}), aborting the evaluation run: {infra}")
        if session is not None:
            atomic(session / "infrastructure_failure.json", {"error": str(infra), "type": type(infra).__name__,
                                                             "kind": kind, "details": getattr(infra, "details", {})})
        raise infra
    keys = ("ended_by", "low_level_actions", "action_budget", "official_ended_before_finalize", "marked_failed")
    done = {k: metrics.get(k) for k in keys}
    done["policy_end"] = (metrics.get("policy_end") or {}).get("kind")
    log("episode done: " + json.dumps(done))


def _record_crash(session, policy_end, metrics):
    """mcp_crash.json: the policy crash of this episode, recorded apart from the result it leaves (never raises)."""
    try:
        atomic(Path(session) / "mcp_crash.json", {
            "kind": CRASH_KIND, "counted": True, "rerun": False,
            "rule": "user decision 2026-10-06: an MCP server or agp_core/run_agent.py that dies mid-episode without an "
                    "external signal is our own code failing; the episode ends as a counted policy failure",
            "reason": policy_end.get("reason"), "details": policy_end.get("details"),
            "after_official_end": policy_end.get("after_official_end"),
            "marked_failed": metrics.get("marked_failed"),
            "official_ended_before_finalize": metrics.get("official_ended_before_finalize"),
            "low_level_actions": metrics.get("low_level_actions"), "action_budget": metrics.get("action_budget"),
            "infrastructure_failure": metrics.get("infrastructure_failure")})
    except Exception:
        traceback.print_exc()


def _official_end_seen(env):
    """Whether the official episode had ended when the policy stopped (no action was sent since: the cached state)."""
    if env is None:
        return False
    try:
        return bool(env.is_episode_end())
    except (Exception, oe.SimulatorFault):
        return False


def _freeze_audit(state, policy_dir):
    """The implementation files this process loaded, against the verified manifest (never raises)."""
    if state is None:
        return None
    try:
        from agp_core import freeze
        return freeze.audit_loaded(state, policy_dir)
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def account_record(session):
    """episode_metrics.json `account`: launch.json `account` (the slot and its use at selection) and
    policy_finished.json `account_end` / `account_error` (agp_core/run_agent.py, agp_core/accounts.py). Never raises."""
    if session is None:
        return None
    launch = read_json(Path(session) / "launch.json")
    finished = read_json(Path(session) / "policy_finished.json")
    out = {"lease": launch.get("account") if isinstance(launch, dict) else None,
           "end": finished.get("account_end") if isinstance(finished, dict) else None,
           "error": finished.get("account_error") if isinstance(finished, dict) else None}
    return out if any(v is not None for v in out.values()) else None


def _finalize(proxy, env, worker, session, fault, infra, serve_reason, policy_code, stopped, started,
              policy_end=None, freeze_audit=None):
    """Honest end of an unfinished episode, cross-checks and records. Never raises."""
    access = proxy.access_log
    metrics = {"ended_by": serve_reason, "policy_returncode": policy_code, "policy_stopped_by_adapter": stopped,
               "simulator_fault": str(fault) if fault else None,
               "infrastructure_failure": str(infra) if infra else None,
               "infrastructure_failure_kind": getattr(infra, "kind", None) if infra else None,
               "policy_end": policy_end or None,
               "external_kill_after_official_end": (policy_end or {}).get("external_kill_after_official_end"),
               "freeze_audit": freeze_audit}
    try:
        proxy.enter("finalize")
        if fault is None:
            ended = bool(proxy.is_episode_end())
            metrics["official_ended_before_finalize"] = ended
            metrics["marked_failed"] = False
            if not ended:
                proxy.mark_failed()
                metrics["official_ended_after_finalize"] = bool(proxy.is_episode_end())
                metrics["marked_failed"] = True
            official = proxy.take_action_cnt[0]
            own = env.steps_sent if env is not None else 0
            metrics.update(official_take_action_cnt=official, low_level_actions=own, counts_match=official == own)
    except (Exception, oe.SimulatorFault) as exc:   # recorded; the outcome is then whatever the evaluator holds
        metrics["finalize_error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    if env is not None:
        metrics.update(action_budget=env.step_lim, observations_fetched=env.observations_fetched,
                       end_checks=env.end_checks, control_time_s=env.steps_sent / env.collect_freq)
    if worker is not None:
        metrics["tool_requests"] = worker.tools
    metrics["wall_duration_s"] = time.monotonic() - started
    try:
        metrics["account"] = account_record(session)
    except Exception as exc:
        metrics["account"] = {"error": f"{type(exc).__name__}: {exc}"}
    if session is not None:
        try:
            atomic(session / "episode_metrics.json", metrics)
            access.write(session / "task_env_access.json")
        except Exception:
            traceback.print_exc()
    return metrics

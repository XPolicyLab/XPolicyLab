"""Launch the Codex policy for the speed-focused variant (native MCP only, no shell).

Under official conditions (OURS_RGB_ONLY=1, set by agp_core/official_deploy.py from the built-in switch set):
  * the model settings OURS_MODEL, OURS_EFFORT and OURS_SERVICE_TIER must come from the environment the adapter
    builds from deploy.yml (no silent default);
  * codex is watched together with the MCP server (wait_official): this launcher records in policy_finished.json who
    stopped codex (`stopped_by`: end_on_assess, policy_timeout, mcp_server_exited; null when codex ended by itself)
    and a signal it did not send (`external_signal`: codex or the MCP server ended by a signal from outside, e.g. the
    OOM killer), so the adapter can tell its own stops from external kills; an MCP server that exits while codex runs
    ends codex at once (`mcp_server`). An MCP server that ends by a fault signal of its own (SIGSEGV, SIGABRT,
    ...; fault_signal) crashed by itself: recorded as `mcp_server.crash_signal`, not as an external signal (the adapter
    counts an MCP server crash, or this launcher dying after codex started, as a policy failure of kind mcp_crash).
  * Frozen system: codex is the pinned release binary that agp_core/official_deploy.py wrote into the session's
    frozen_policy.json from the verified freeze manifest (deploy.yml agp.codex_binary), never the self-updating
    ~/.local/bin/codex link. Before the MCP server starts, this launcher re-checks that binary's sha256 and the sha256
    of its prompt rendering against those pins and refuses to start on any difference (launch.json `frozen`).
With OURS_RGB_ONLY off none of this applies.
Task notes (OURS_TASK_NOTES, opt-in): the launch message gets, right after the rendered main prompt and before the
preload's episode part, the note of the episode's official task (agp_core/task_notes.py; launch.json `task_note`). Without a
readable task name in the session's episode.json the launcher refuses to start. Off: nothing of it runs.
Task hints (OURS_TASK_HINTS, opt-in): the hint for the episode's official task (agp_core/task_hints.json: a hint for 18
tasks, a layer separate from the notes) follows as its own section right after the notes section (right after the
main prompt when the notes are off), before the preload's episode part (agp_core/task_notes.py launch_hint_section;
launch.json `task_hint`). Without a readable task name the launcher refuses to start. Off: nothing of it runs.
Codex sandbox (always): every episode runs codex with a fresh CODEX_HOME of its own
(AGP_HOME/private/codex_episodes/<run>-<session>-<pid>, mode 0700: no state, memory or rollout of another episode exists
there; the login is copied in right before codex starts and deleted when codex has ended), web search off
(`web_search = "disabled"`) and the permission profile agp_episode instead of the read-only sandbox, which would let
codex read the whole file system: it reads only the system minimum (`:minimal`), the session folder and the pinned codex
release. launch.json records the home (`codex_home`).
End of the episode, in `finally`, in this order once codex has exited: the rollout is copied into the session
(policy_rollout.jsonl), the login copy is settled (account pool, below: it lives in that home), codex's logs database
(logs_*.sqlite with its -wal and -shm files, copied byte for byte as one set, never opened) is copied into the session's
codex_logs/, and then the whole CODEX_HOME (about 15 MB: state databases, rollout, caches) is deleted.
The copies of the logs database are scanned for credential-like content (a JWT, `refresh_token`, `Bearer `, an API
key, an e-mail address; counted, never printed): on any hit they are deleted (agp_core/accounts.py keep_codex_logs;
policy_finished.json `codex_logs` {"kept": false, "reason": "credential-like content (N hits)"}).
The home stays only when its login copy could not be settled (the write-back failed) or a copy into the session failed;
it then gets agp_core/accounts.py KEPT_MARK, and `agp_core/accounts.py release-stale` (or the next lease) settles the login
copy where it is, copies what the session still lacks and removes the home. policy_finished.json `codex_home` records
the copied files and whether the home was removed.
Packaging: ROOT is the implementation this file belongs to (the packaged runtime in the policy folder), read only; the
MCP server's interpreter and codex's homes are in AGP_HOME (agp_core/paths.py; the adapter passes it).
Account pool (agp_core/accounts.py): right before codex starts the launcher leases a login -- with slots in
AGP_HOME/private/accounts a login of the least used account (up to episodes_per_login episodes share one login while its
access token outlives this launcher's codex limit and deploy.yml agp.episode_timeout_s by an hour; closer to its expiry
it is leased exclusively, or first renewed with this launcher's codex when no episode holds it), else
~/.codex/auth.json -- and copies it into the episode's CODEX_HOME. No usable slot (every login leased exclusively near
its access expiry, cooling or at its usage cap): the launcher tries again every ACCOUNT_RETRY_S for up to ACCOUNT_WAIT_S,
then exits before codex starts (policy_finished.json `account_error`, an infrastructure failure the launcher re-runs). In
`finally`, once codex is gone: a login codex renewed during the episode is written back (atomically, only a valid,
newer copy of the same login and session, never over a newer slot file), the copy and the lease are deleted, the
rollout's last usage reading and a codex failure (usage limit, refresh/401, 429) are recorded for the account.
launch.json `account` (label, account / session hash, exclusive or shared, renewal, use at selection) and
policy_finished.json `account_end` (write-back, use at the end, cooling, disabled) hold no credential value.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agp_core import accounts, features, paths  # noqa: E402
from agp_core.assess_watch import accepted_assessment, end_on_assess, stop_process  # noqa: E402
MODEL = os.environ.get("OURS_MODEL", "gpt-6-astra")
EFFORT = os.environ.get("OURS_EFFORT", "medium")
TIER = os.environ.get("OURS_SERVICE_TIER", "default")
TIMEOUT_S = float(os.environ.get("OURS_POLICY_TIMEOUT_S", "14000"))
# no usable account slot: try again every ACCOUNT_RETRY_S for up to ACCOUNT_WAIT_S before failing the episode
ACCOUNT_WAIT_S = float(os.environ.get("AGP_ACCOUNT_WAIT_S", "3600"))
ACCOUNT_RETRY_S = float(os.environ.get("AGP_ACCOUNT_RETRY_S", "30"))
TOOLS = ("goal", "observe", "measure", "step", "run", "action_status", "assess")
AGENT_PYTHON = paths.AGP_HOME / ".venv-agent" / "bin" / "python"
# One HTTP server process per episode. hybrid: registered twice, "robot" for direct tool calls and
# "robot_js" for calls from exec code. code: registered once as "robot_js", reachable only from exec code.
TOOL_MODE = os.environ.get("OURS_TOOL_MODE", "hybrid")
if TOOL_MODE not in ("hybrid", "code"):
    raise SystemExit("OURS_TOOL_MODE is hybrid or code")
FLAGS = features.current()
ENDED_ON_ASSESS = {}   # filled by assess_watch.end_on_assess(): the terminated codex's raw return code
OFFICIAL = FLAGS["rgb_only"]
STOPS = {}             # official conditions: who ended codex, and signals this launcher did not send
PERMISSION_PROFILE = "agp_episode"   # codex's file-system permissions (see the docstring)
# the end of the episode's CODEX_HOME (logs copy and credential scan, removal, the kept-home mark) is agp_core/accounts.py's
# (keep_codex_logs, remove_codex_home, mark_kept_home), shared with release_kept_homes there


def toml_value(value):
    """A `-c key=value` value for codex: JSON for scalars and lists (valid TOML), an inline table for a dict."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{json.dumps(str(k))} = {toml_value(v)}" for k, v in value.items()) + "}"
    return json.dumps(value)


if OFFICIAL:
    missing = [name for name in ("OURS_MODEL", "OURS_EFFORT", "OURS_SERVICE_TIER") if not os.environ.get(name)]
    if missing:
        raise SystemExit(f"official conditions (OURS_RGB_ONLY=1): the model settings come from deploy.yml through "
                         f"agp_core/official_deploy.py; missing {missing}")


def signal_of(code):
    """Signal behind a return code: -N (killed by signal N) or 128+N (reported that way); else None."""
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    if code < 0:
        return -code
    if 128 < code <= 128 + 64:
        return code - 128
    return None


def signal_entry(process, code):
    number = signal_of(code)
    try:
        name = signal.Signals(number).name
    except (ValueError, TypeError):
        name = f"signal {number}"
    return {"process": process, "returncode": code, "signal": number, "signal_name": name, "at_wall": time.time()}


def fault_signal(code):
    """Whether a return code is a fault signal the process raised against itself (SIGSEGV, SIGBUS, SIGFPE, SIGILL,
    SIGABRT, SIGTRAP, SIGSYS): a crash of that process, not a signal from outside."""
    return signal_of(code) in {int(s) for s in (signal.SIGSEGV, signal.SIGBUS, signal.SIGFPE, signal.SIGILL,
                                                signal.SIGABRT, signal.SIGTRAP, signal.SIGSYS)}


def wait_official(process, server, session, timeout_s, stop_on_assess, record, poll_s=0.1):
    """Wait for codex (official conditions); return the code with the meaning it always had (0 after an accepted
    assessment, 124 at the time limit, else codex's own status) and fill `record`:
        stopped_by        end_on_assess | policy_timeout | mcp_server_exited (this launcher stopped codex) or None
        codex_returncode  codex's raw status
        mcp_server        {returncode, exited_during_episode} when the MCP server ended before codex did
                          (+ crash_signal when that was a fault signal of its own: it crashed)
        external_signal   codex or the MCP server ended by a signal this launcher did not send (for the MCP server
                          not a fault signal)"""
    deadline = time.monotonic() + timeout_s
    record["stopped_by"] = None
    while True:
        try:
            code = process.wait(timeout=poll_s)
            break
        except subprocess.TimeoutExpired:
            pass
        if stop_on_assess and accepted_assessment(session):
            raw = stop_process(process)
            ENDED_ON_ASSESS["terminated_returncode"] = raw
            record.update(stopped_by="end_on_assess", codex_returncode=raw)
            return 0
        if server.poll() is not None:
            break                         # the tools are gone: end codex now (below)
        if time.monotonic() > deadline:
            raw = stop_process(process)
            record.update(stopped_by="policy_timeout", codex_returncode=raw)
            return 124
    server_code = server.poll()
    if server_code is not None:
        record["mcp_server"] = {"returncode": server_code, "exited_during_episode": True, "at_wall": time.time()}
        if fault_signal(server_code):
            record["mcp_server"]["crash_signal"] = signal_entry("mcp_server", server_code)
        elif signal_of(server_code) is not None:
            record["external_signal"] = signal_entry("mcp_server", server_code)
    if process.poll() is None:
        raw = stop_process(process)
        record.update(stopped_by="mcp_server_exited", codex_returncode=raw)
        return raw
    code = process.returncode
    record["codex_returncode"] = code
    if signal_of(code) is not None and "external_signal" not in record:
        record["external_signal"] = signal_entry("codex", code)
    return code

def frozen_pins(session):
    """The session's frozen_policy.json (agp_core/official_deploy.py), checked before anything is written or started:
    exits when the pins are missing, the codex binary is not the pinned one, or the prompt rendering differs."""
    path = session / "frozen_policy.json"
    try:
        pins = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"official conditions: no frozen pins in {path} ({type(exc).__name__}); "
                         f"agp_core/official_deploy.py writes them from the verified freeze manifest")
    binary = pins.get("codex_binary")
    if not isinstance(binary, str) or os.path.islink(binary) or not os.path.isfile(binary):
        raise SystemExit(f"official conditions: the pinned codex binary {binary!r} is not a regular file")
    digest = hashlib.sha256()
    with open(binary, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    if digest.hexdigest() != pins.get("codex_sha256"):
        raise SystemExit(f"official conditions: codex binary {binary} has sha256 {digest.hexdigest()}, the frozen "
                         f"system pins {pins.get('codex_sha256')}")
    rendered = features.render(Path(os.environ.get("OURS_PROMPT_FILE", ROOT / "agp_core/PROMPT.md")).read_text(), FLAGS)
    if hashlib.sha256(rendered.encode("utf-8")).hexdigest() != pins.get("prompt_rendered_sha256"):
        raise SystemExit(f"official conditions: the prompt rendering has sha256 "
                         f"{hashlib.sha256(rendered.encode('utf-8')).hexdigest()}, the frozen system pins "
                         f"{pins.get('prompt_rendered_sha256')}")
    return pins


session = Path(sys.argv[1]).resolve()
if not (session / "worker_ready.json").exists():
    raise SystemExit("Simulator is not ready")
if (session / "agent_events.jsonl").exists():
    raise SystemExit("Agent session already used")
CODEX = str(Path.home() / ".local" / "bin" / "codex")      # development only; official conditions: the pinned release
PINS = {}
if OFFICIAL:
    PINS = frozen_pins(session)
    CODEX = PINS["codex_binary"]

CODEX_HOMES = paths.AGP_HOME / "private" / "codex_episodes"   # one fresh CODEX_HOME per episode (see the docstring)
CODEX_HOMES.mkdir(parents=True, exist_ok=True, mode=0o700)
codex_state = CODEX_HOMES / f"{session.parent.name}-{session.name}-{os.getpid()}"
codex_state.mkdir(mode=0o700)
(codex_state / "config.toml").write_text(
    f'model="{MODEL}"\nmodel_reasoning_effort="{EFFORT}"\nservice_tier="{TIER}"\n')

template = Path(os.environ.get("OURS_PROMPT_FILE", ROOT / "agp_core/PROMPT.md")).read_text()
prompt = features.render(template, FLAGS)
if OFFICIAL and hashlib.sha256(prompt.encode("utf-8")).hexdigest() != PINS.get("prompt_rendered_sha256"):
    raise SystemExit(f"official conditions: the prompt rendering has sha256 "
                     f"{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}, the frozen system pins "
                     f"{PINS.get('prompt_rendered_sha256')}")
TASK_NOTE = None
if FLAGS["task_notes"]:
    # task notes: right after the main prompt, the note of the episode's official task (agp_core/task_notes.py; the task
    # name the adapter read from TASK_ENV.task_name, in the session's episode.json)
    from agp_core import task_notes
    note_section, TASK_NOTE = task_notes.launch_section(session)
    prompt += note_section
TASK_HINT = None
if FLAGS["task_hints"]:
    # task hints: the hint for the episode's official task, a layer of its own behind its own switch, right after the
    # notes section (agp_core/task_notes.py launch_hint_section; agp_core/task_hints.json)
    from agp_core import task_notes
    hint_section, TASK_HINT = task_notes.launch_hint_section(session)
    prompt += hint_section

config = {
    "features.plugins": False,
    "features.shell_tool": False,
    "features.multi_agent": False,
    "features.apps": False,
    "features.browser_use": False,
    "features.computer_use": False,
    "features.skill_search": False,
    "project_doc_max_bytes": 0,
    "model_reasoning_effort": EFFORT,
    "service_tier": TIER,
    # no web search; reads only the system minimum, the session and the codex release
    "web_search": "disabled",
    "default_permissions": PERMISSION_PROFILE,
    f"permissions.{PERMISSION_PROFILE}.filesystem": {
        ":minimal": "read", str(session.resolve()): "read",
        str(Path(os.path.realpath(CODEX)).parent.parent): "read"},
}
server_env = dict(os.environ, PYTHONPATH=f"{ROOT}/source/agent:{ROOT}", PYTHONDONTWRITEBYTECODE="1",
                  **{variable: "1" if FLAGS[name] else "0" for name, variable in features.FLAGS.items()})
server_cmd = [str(AGENT_PYTHON), "-m", "agp_core.server", "--session", str(session)]
images = []
with socket.socket() as probe:
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
http_server = subprocess.Popen(server_cmd + ["--http", str(port)], cwd=ROOT, env=server_env,
                               stdout=(session / "mcp_server.log").open("w"), stderr=subprocess.STDOUT)
deadline = time.monotonic() + 120
while True:
    if http_server.poll() is not None:
        raise SystemExit(f"MCP server exited early, see {session / 'mcp_server.log'}")
    try:
        socket.create_connection(("127.0.0.1", port), timeout=1).close()
        break
    except OSError:
        if time.monotonic() > deadline:
            http_server.terminate()
            raise SystemExit("MCP server did not start")
        time.sleep(0.2)
if FLAGS["preload"]:
    from agp_core.preload import preload
    try:
        message, images = preload(f"http://127.0.0.1:{port}/mcp", session)
    except Exception:
        http_server.terminate()
        raise
    prompt += message
servers = {"robot": ["code_mode"], "robot_js": ["direct"]} if TOOL_MODE == "hybrid" else {"robot_js": ["direct"]}
for name, omit in servers.items():
    config.update({f"mcp_servers.{name}.url": f"http://127.0.0.1:{port}/mcp",
                   f"mcp_servers.{name}.omit_tools_from": omit,
                   f"mcp_servers.{name}.tool_timeout_sec": 900,
                   f"mcp_servers.{name}.required": True})
for server_name in servers:
    for name in TOOLS:
        if name != "run" or FLAGS["run"]:
            config[f"mcp_servers.{server_name}.tools.{name}.approval_mode"] = "approve"
cmd = [CODEX, "exec", "--skip-git-repo-check", "--cd", str(session), "--json", "-m", MODEL]
for key, value in config.items():
    cmd += ["-c", key + "=" + toml_value(value)]
cmd += ["-c", subprocess.check_output(
    [str(AGENT_PYTHON), str(ROOT / "source/agent/agp/tools/codex_experiment_config.py")],
    text=True).strip()]
for image in images:
    cmd += ["-i", str(image)]
cmd += ["-o", str(session / "agent_last_message.txt"), "-"]
(session / "PROMPT.md").write_text(prompt)

started_wall = time.time()
started = time.monotonic()
code = 1
TERMINATED = {}
LEASE = None           # the login of this episode (agp_core/accounts.py lease); settled in `finally`
ACCOUNT = {}           # account_error / account_end for policy_finished.json
PROCESS = None


def on_sigterm(signum, frame):
    """The adapter ends the policy's process group with SIGTERM (OURS_HIDE_END: at the official end; the end grace or
    the wall-clock limit): leave through `finally`, so the login copy is deleted, policy_finished.json is written and
    the rollout is kept, within the 15 s the adapter waits before SIGKILL."""
    TERMINATED["signal"] = "SIGTERM"
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, on_sigterm)
try:
    wait_until = time.monotonic() + ACCOUNT_WAIT_S
    reported = 0.0
    while LEASE is None:
        try:
            LEASE = accounts.lease(paths.AGP_HOME, codex_state / "auth.json", owner=str(session), episode_s=TIMEOUT_S,
                                   codex=CODEX)
        except accounts.NoAccount as exc:
            if time.monotonic() >= wait_until:
                ACCOUNT["account_error"] = f"no usable codex account slot: {exc}"
                raise SystemExit(f"[AGP account] {ACCOUNT['account_error']}") from None
            if time.monotonic() - reported >= 300:
                print(f"[AGP account] no usable slot yet, trying again every {ACCOUNT_RETRY_S:.0f} s: {exc}", flush=True)
                reported = time.monotonic()
            time.sleep(ACCOUNT_RETRY_S)
    if LEASE.get("source") == "pool":
        print(f"[AGP account] slot {LEASE.get('label')}, account {LEASE.get('account_hash')}, "
              f"{LEASE.get('used_percent_at_select', 0):.0f}% of its window used", flush=True)
    (session / "launch.json").write_text(json.dumps({
        "variant": "ours", "tool_mode": TOOL_MODE, "flags": FLAGS, "command": cmd, "model": MODEL, "effort": EFFORT,
        "service_tier": TIER, "policy_timeout_s": TIMEOUT_S, "started_wall": started_wall,
        "codex_home": str(codex_state), "account": accounts.public(LEASE),
        **({"frozen": {"codex_binary": CODEX, "codex_sha256": PINS.get("codex_sha256"),
                       "codex_version": PINS.get("codex_version"),
                       "prompt_rendered_sha256": PINS.get("prompt_rendered_sha256"), "seal": PINS.get("seal")}}
           if OFFICIAL else {}),
        **({"task_note": TASK_NOTE} if FLAGS["task_notes"] else {}),
        **({"task_hint": TASK_HINT} if FLAGS["task_hints"] else {})}, indent=2))
    with ((session / "PROMPT.md").open() as inp, (session / "agent_events.jsonl").open("w") as out,
          (session / "agent_stderr.log").open("w") as err):
        process = PROCESS = subprocess.Popen(cmd, stdin=inp, stdout=out, stderr=err,
                                             env=dict(os.environ, CODEX_HOME=str(codex_state)))
        (session / "agent.pid").write_text(str(process.pid))
        if OFFICIAL:
            # also watches the MCP server and records who stopped codex (see wait_official).
            code = wait_official(process, http_server, session, TIMEOUT_S, FLAGS["end_on_assess"], STOPS)
        elif not FLAGS["end_on_assess"]:
            try:
                code = process.wait(timeout=TIMEOUT_S)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                code = 124
        else:
            # OURS_END_ON_ASSESS: the episode's last useful act is an accepted assess; the closing message
            # after it only costs a turn. End codex within about 0.1 s of the runtime's assessment file.
            code = end_on_assess(process, session, TIMEOUT_S, record=ENDED_ON_ASSESS)
finally:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)   # cleanup runs once; the adapter's SIGKILL after 15 s still applies
    if PROCESS is not None and PROCESS.poll() is None:
        # SIGTERM from the adapter while codex runs (the group got it too): codex is gone before its login is read back
        try:
            PROCESS.wait(timeout=3)
        except subprocess.TimeoutExpired:
            PROCESS.kill()
            PROCESS.wait()
    if http_server.poll() is None:
        http_server.terminate()
        try:
            http_server.wait(timeout=10)      # within the adapter's 15 s before SIGKILL
        except subprocess.TimeoutExpired:
            http_server.kill()
    attribution = {}
    if OFFICIAL:
        attribution = {"stopped_by": STOPS.get("stopped_by"), "codex_returncode": STOPS.get("codex_returncode"),
                       "external_signal": STOPS.get("external_signal"), "mcp_server": STOPS.get("mcp_server"),
                       "ended_wall": time.time()}
    # Keep the full model-facing rollout next to the session for timing analysis (and the account's usage reading).
    # Episodes can run in parallel, so match the rollout by its recorded cwd, not only by time.
    try:
        rollouts = [p for p in (codex_state / "sessions").rglob("rollout-*.jsonl")
                    if p.stat().st_mtime >= started_wall - 5 and accounts.own_rollout(p, session)] \
            if (codex_state / "sessions").exists() else []
        if rollouts:
            latest = max(rollouts, key=lambda p: p.stat().st_mtime)
            shutil.copy2(latest, session / "policy_rollout.jsonl")
            (session / "policy_rollout_source.txt").write_text(str(latest) + "\n")
    except OSError as exc:
        attribution["rollout_copy_error"] = f"{type(exc).__name__}: {exc}"
    if LEASE is not None:
        # the login does not outlive the episode's codex: renewed -> written back, then the copy is deleted
        codex_failed = (PROCESS is not None and code not in (0, 124) and not TERMINATED and not ENDED_ON_ASSESS
                        and STOPS.get("stopped_by") is None)
        rollout = session / "policy_rollout.jsonl"
        ACCOUNT["account_end"] = accounts.finish(
            paths.AGP_HOME, LEASE, rollout=rollout if rollout.exists() else None,
            failure_text=accounts.codex_error_text(session) if codex_failed else None, codex_failed=codex_failed)
    # codex's logs database into the session (copies with credential-like content deleted), then the whole CODEX_HOME is
    # deleted (see the docstring); kept when its login copy is still there (not settled: release-stale needs it) or a
    # copy failed, with a mark that lets release-stale / the next lease remove it later
    home_record = {"logs": []}
    logs_record = {"kept": False, "reason": "not copied"}
    try:
        logs_record = accounts.keep_codex_logs(codex_state, session)
        home_record["logs"] = logs_record.get("files", [])
    except OSError as exc:
        home_record["logs_copy_error"] = f"{type(exc).__name__}: {exc}"
        logs_record = {"kept": False, "reason": f"copy failed ({type(exc).__name__})"}
    if (codex_state / "auth.json").exists():
        home_record["kept"] = "login copy not settled (agp_core/accounts.py release-stale settles it in place)"
    elif "logs_copy_error" in home_record or "rollout_copy_error" in attribution:
        home_record["kept"] = "a copy into the session failed"
    else:
        home_record["remove_errors"] = accounts.remove_codex_home(codex_state)
    if "kept" in home_record:
        home_record.update(accounts.mark_kept_home(
            codex_state, session, home_record["kept"], logs_done="logs_copy_error" not in home_record,
            rollout_done="rollout_copy_error" not in attribution))
    home_record["removed"] = not codex_state.exists()
    (session / "policy_finished.json").write_text(json.dumps({
        "kind": "Codex native MCP (ours)", "returncode": code, **({"terminated": TERMINATED} if TERMINATED else {}),
        "wall_duration_s": time.monotonic() - started,
        **({"ended_on_assess": True, **ENDED_ON_ASSESS} if ENDED_ON_ASSESS else {}), **attribution, **ACCOUNT,
        "codex_home": home_record, "codex_logs": logs_record}))

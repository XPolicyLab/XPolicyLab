"""Mibot-RSI submission entry: one evaluation episode inside RoboDojo's official evaluator.

RoboDojo's evaluation client imports XPolicyLab/policy/Mibot_RSI/deploy.py in its Isaac Sim process and calls
eval_one_episode(TASK_ENV, model_client) once per episode.
deploy.py hands that call to eval_one_episode here, which

  1. checks the settings (deploy.yml), this process's environment and the task's knowledge snapshot;
  2. serves the robot from this process: the adapter of every recorded episode, with the official step budget
     read and never written, steps TASK_ENV through its own take_action;
  3. runs the model session and its sandboxed Python worker in a subprocess (`python -m v2.submission session
     RUN_DIR`, AttachedEpisode below), which reaches the robot through the authenticated local robot service;
  4. when that subprocess ends, runs RoboDojo's own end check for an episode that did not end by itself
     (close_attempt: an unfinished episode is a failure, never the success RoboDojo presets) and returns, so the
     evaluator scores the episode as it stands. An infrastructure failure (InfrastructureFailure) is never handed
     to RoboDojo's main loop, which would silently take the next layout: the process leaves the way RoboDojo's own
     shell-level restart does (stop_process: resume manifest persisted, os._exit(99)), and scripts/eval_policy.sh
     starts a new process with the same ROBODOJO_RUN_ID that runs this layout again (at most
     ROBODOJO_MAX_BASH_RETRIES - 1 restarts, shared with RoboDojo's PhysX restarts); a setup failure, which a new
     process would repeat, leaves with rc 1.
     A PhysX failure of the simulator is raised again as RoboDojo's own exception, which its main loop handles
     (its monitor is checked in get_obs and take_action, as for any policy). Commands still queued when the session
     subprocess has exited are refused, never run.
     A simulator that dies inside a robot command never returns control to this thread. While an episode is served,
     a watchdog thread (v2/simwatch.py) watches the simulator's log where RoboDojo's own PhysX monitor does not run,
     the progress of the command being executed and the episode wall limit; on a finding it records the failure,
     stops the session and leaves the process the same way (rc 99), so that the layout runs again.

The model call cap (deploy.yml max_model_calls) ends the episode the way the session ends at any limit: the
session stops before the next request, nothing new is shown to the model, and the episode is scored as it stands.
The exploration budget and the extended (5x) budget of development runs do not exist on this path and their
settings are refused. The model endpoint and key come only from the environment (MIBOT_RSI_BASE_URL and the
variable named by api_key_env, MIBOT_RSI_API_KEY by default); neither is written to any file by this module.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import queue
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse

from v2.runtime import landlock
from v2.runtime.client import (DEFAULTS as CLIENT_DEFAULTS, HARNESS_ROOT, OPTIONAL_KEYS as CLIENT_OPTIONAL_KEYS,
                               REASONING_SUMMARIES)

BASE_URL_ENV = 'MIBOT_RSI_BASE_URL'
KNOWLEDGE_ENV = 'MIBOT_RSI_KNOWLEDGE_DIR'
RUNS_ROOT_ENV = 'MIBOT_RSI_RUNS_ROOT'
DEFAULT_KEY_ENV = 'MIBOT_RSI_API_KEY'
# The default model settings (as run.json model_config records them), without the endpoint. deploy.yml may
# add reasoning_summary (auto, concise or detailed): readable reasoning summaries, only logged; off by default.
MODEL_DEFAULTS = {'model': 'gpt-6-astra', 'api_key_env': DEFAULT_KEY_ENV, 'reasoning_effort': 'xhigh',
                  'request_timeout_s': 300, 'api_retries': 1799, 'api_retry_base_delay_s': 1,
                  'api_retry_max_delay_s': 2, 'api_retry_total_timeout_s': 3600, 'max_output_tokens': 16384,
                  'compact_threshold': 200000, 'prompt_cache_ttl': '30m'}
# episode_wall_limit_s: a watchdog against a stuck episode, set well above a slow legitimate one (300 calls at xhigh
# effort), since reaching it stops the evaluation. Model API waits count towards it: a refusal or quota error is
# waited out for about 1 h 50 min; a long outage (an hour of client retries before each attempt, plus the waits)
# can reach it, and the layout then runs again in a new process. The robot service checks it between commands, the
# simulator watchdog (v2/simwatch.py) also during one.
SETTINGS_DEFAULTS = {'max_model_calls': 300, 'episode_wall_limit_s': 21600, 'runs_root': 'runs',
                     'knowledge_dir': None, 'runner_python': None,
                     'model_api_waits_s': [60, 300, 900, 1800, 3600]}
# deploy.yml keys of XPolicyLab's standard set and of the policy server; read by others, ignored here.
STANDARD_KEYS = ('policy_name', 'protocol', 'host', 'port', 'bench_name', 'task_name', 'ckpt_name', 'env_cfg_type',
                 'seed', 'action_type', 'gpu_id', 'eval_batch', 'action_dim', 'policy_server_url', 'evaluation_id',
                 'trial_id', 'action_case_id', 'repeat_index', 'ws_ping_interval_s', 'ws_ping_timeout_s')
# Settings of development features that do not exist on the submission path.
REFUSED_ENV = ('ROBOT_EXTEND_MULTIPLIER', 'ROBOT_EXPLORATION_STEP_LIMIT', 'ROBOT_TEACHER_ACCESS',
               'ROBOT_OUTER_STAGE_CONFIG', 'ROBOT_CURRICULUM_MANIFEST', 'ROBOT_CUSTOM_LAYOUT_PATH', 'ROBOT_LAYOUT_ID',
               'ROBOT_DIAG_OBJECTS', 'ROBOT_ATTEMPT_RESULT')
PHYSX_ERRORS = ('PhysXBrokenError', 'PhysXFatalError')
OFFICIAL_ENV_CFG = 'arx_x5'
# Environment the session subprocess keeps (the key and endpoint are added by name); the Python worker it starts
# inherits far less (v2/runtime/worker.py child_environment) and never the key.
RUNNER_ENV = ('PATH', 'HOME', 'USER', 'LOGNAME', 'LANG', 'LC_ALL', 'TMPDIR', 'TZ', 'SSL_CERT_FILE', 'SSL_CERT_DIR',
              'REQUESTS_CA_BUNDLE', 'HTTP_PROXY', 'HTTPS_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'no_proxy',
              'V2_DEADLINE_GRACE_S', 'V2_DEADLINE_GRACE_MAX_S')
RUNNER_STOP_S = 30
PARENT_ENV = 'MIBOT_RSI_EVALUATOR_PID'  # the session subprocess ends with this evaluation process (main)
FINAL_REPLY_S = 0.5  # as in embodied_agent/simulator.py: let the session read the last reply before the service closes


RESTART_EXIT_CODE = 99  # scripts/eval_policy.sh starts a new evaluation process (same ROBODOJO_RUN_ID) on rc 99
SETUP_EXIT_CODE = 1     # a setup failure: a new process would fail the same way, so no restart


class InfrastructureFailure(BaseException):
    """Not the agent's failure: the evaluation process must leave so that the layout is run again (see above).
    `restart` False: a setup failure (settings, environment, knowledge) that a new process would repeat."""

    def __init__(self, message='', restart=True):
        super().__init__(message)
        self.restart = restart


def stop_process(task_env, failure):
    """Leave the evaluation process for an infrastructure failure as RoboDojo's shell-level restart does
    (src/eval_client/main.py _exit_for_shell_restart): persist the resume manifest, flush, os._exit. Raising
    instead leaves Isaac Sim's shutdown hanging ("Kit appears to be hanging"), and an
    Exception would make RoboDojo take the next layout. Does not return."""
    code = RESTART_EXIT_CODE if getattr(failure, 'restart', True) else SETUP_EXIT_CODE
    print(f'[Mibot-RSI] infrastructure failure, leaving the evaluation process (rc {code}): {failure}',
          file=sys.stderr, flush=True)
    persist = getattr(task_env, 'persist_resume_manifest', None)
    if callable(persist):
        try:
            persist(restart_count=int(os.environ.get('ROBODOJO_FATAL_RESTART_COUNT', '0') or 0))
        except Exception as exc:
            print(f'[Mibot-RSI] persist_resume_manifest failed: {exc}', file=sys.stderr, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def official_physx_monitor(task_env):
    """(enabled, monitor) of RoboDojo's PhysX log monitor for this task: src/eval_client/main.py starts it only for
    tasks whose config has an Articulation section and tells the environment (EvalEnv.physx_monitor_enabled); the
    monitor object is None where it cannot be reached."""
    if getattr(task_env, 'physx_monitor_enabled', False) is not True:
        return False, None
    get = getattr(task_env, '_physx_get_monitor', None)
    if not callable(get):
        get = getattr(sys.modules.get('src.eval_client.physx_warning_monitor'), 'get_monitor', None)
    try:
        return True, (get() if callable(get) else None)
    except Exception:
        return True, None


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def kit_log_path():
    from v2.simwatch import kit_log_path as path
    return path()


def start_ticks(pid='self'):
    """A running process's start time in clock ticks since boot (/proc/<pid>/stat), or None."""
    try:
        text = Path(f'/proc/{pid}/stat').read_text()
        return text[text.rfind(')') + 2:].split()[19]
    except (OSError, IndexError):
        return None


def process_identity():
    """'<boot id>/<host>/<pid namespace>/<pid>/<start ticks>' of this process, or None: the same for its whole life
    and never for another process (pid reuse), whatever the clock does. The first three parts say where its pid can
    be looked up: a reader elsewhere (another machine, boot or container's pid namespace) cannot tell from its own
    /proc whether the process still runs."""
    try:
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        namespace = os.readlink('/proc/self/ns/pid')
    except OSError:
        return None
    ticks = start_ticks()
    return None if ticks is None else f'{boot}/{socket.gethostname()}/{namespace}/{os.getpid()}/{ticks}'


def jsonl_rows(path):
    """The JSON objects of a JSON-lines file; a line that is cut or not an object is skipped."""
    rows = []
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return rows
    for line in lines:
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def recover_unfinished(runs_root, run_id, task):
    """Complete the records of an episode that an earlier evaluation process of this run left unfinished because it
    ended mid-episode without recording it (RoboDojo's PhysX watchdog os._exit(99) on the tasks its monitor watches,
    or a kill such as the OOM killer's): a `finished` row and submission_episode.json as the robot service writes
    them, and a row in runs/submission_reruns.jsonl, since RoboDojo runs that layout again. Only rows of this run and
    task whose evaluation process (pid and start time) is gone, as seen from the same machine, boot and pid
    namespace (a row of a process elsewhere is left alone: whether it still runs cannot be told from here); never
    raises."""
    if not run_id:
        return []
    index, reruns = Path(runs_root) / 'episodes.jsonl', Path(runs_root) / 'submission_reruns.jsonl'
    rows = jsonl_rows(index)
    finished = {row.get('episode_id') for row in rows if row.get('event') == 'finished'}
    rerun = None
    recovered = []
    here = (process_identity() or '').rsplit('/', 2)[0]
    for row in rows:
        if (row.get('event') != 'started' or row.get('episode_id') in finished or row.get('robodojo_run_id') != run_id
                or row.get('task') != task or not isinstance(row.get('evaluator_identity'), str)):
            continue
        parts = row['evaluator_identity'].rsplit('/', 2)
        if len(parts) != 3 or not here or parts[0] != here:
            continue  # recorded elsewhere: whether that process still runs cannot be told here
        if start_ticks(parts[1]) == parts[2]:
            continue  # that process still runs
        try:
            # A simulator failure only where the dead process's own Kit log shows a fatal PhysX line (the watchdog's
            # patterns) from this episode: RoboDojo's PhysX watchdog then force-exited it; otherwise it was killed
            # (the OOM killer, an operator) or crashed.
            from v2.simwatch import fatal_line, process_start
            evidence = None
            if row.get('evaluator_kit_log'):
                evidence = fatal_line(row['evaluator_kit_log'], process_start(ticks=parts[2]), row.get('wall_started'))
            kind = 'simulator_failure' if evidence else 'infrastructure'
            error = ('InfrastructureFailure: ' + (f'simulator_failure: the evaluation process ended during this '
                                                  f'episode after its simulator logged: {evidence}' if evidence else
                                                  'the evaluation process ended during this episode without a fatal '
                                                  'simulator line in its Kit log (killed or crashed)')
                     + '; recorded by the next process')[:2000]
            record = {key: value for key, value in row.items() if key != 'event'}
            run_dir = Path(record.get('run_dir') or '')
            own = None
            if record.get('run_dir'):
                try:
                    own = json.loads((run_dir / 'session_result.json').read_text())
                except (OSError, ValueError):
                    pass
            record.update(**{'class': kind, 'error': error, 'recovered': now(), 'evidence': evidence,
                             'wall_finished': None, 'physical_steps': None, 'end_flag': None, 'success': None,
                             'session_result': own if isinstance(own, dict) else None})
            if rerun is None:
                rerun = {item.get('episode_id') for item in jsonl_rows(reruns)}
            if row['episode_id'] not in rerun:  # else the watchdog logged it before the process ended
                AttachedService._log(reruns,
                                     {key: record.get(key) for key in ('episode_id', 'task', 'robodojo_run_id',
                                                                       'eval_seed', 'layout_id', 'official_index',
                                                                       'error')} |
                                     {'physical_steps': None, 'restart': True, 'time': now(), 'check': 'next_start',
                                      'evidence': evidence})
            if record.get('run_dir') and run_dir.is_dir() and not (run_dir / 'submission_episode.json').exists():
                AttachedService._log(run_dir / 'submission_episode.json', record, whole=True)
            AttachedService._log(index, {**record, 'event': 'finished'})
            recovered.append(row['episode_id'])
            print(f"[Mibot-RSI] {task} layout {row.get('layout_id')}: the evaluation process ended during episode "
                  f"{row['episode_id']}; recorded it as {kind} (the layout runs again)", file=sys.stderr, flush=True)
        except Exception as exc:  # recording an earlier episode never stops this one
            print(f'[Mibot-RSI] could not record unfinished episode {row.get("episode_id")}: {exc}', file=sys.stderr,
                  flush=True)
    return recovered


def write_json(path, value, mode=None):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    if mode is not None:
        temp.chmod(mode)
    os.replace(temp, path)


def harness_modules():
    """The harness modules this process has loaded (provenance: an evaluation loads no development module)."""
    root = str(HARNESS_ROOT) + os.sep
    return sorted(name for name, module in list(sys.modules.items())
                  if str(getattr(module, '__file__', '') or '').startswith(root))


def descendants(pid):
    """Process IDs below `pid` (Linux /proc; empty elsewhere). The Python worker runs in a session of its own, so
    stopping the session subprocess's process group does not reach it."""
    children = {}
    for stat_path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = stat_path.read_text()
            children.setdefault(int(text[text.rfind(')') + 2:].split()[1]), []).append(int(stat_path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    found, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), ()):
            found.append(child)
            todo.append(child)
    return found


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def append_private(path, record):
    """A line of a session record (mode 600, as v2/runtime/loop.py writes them)."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


# ------------------------------------------------------------------ settings

def load_settings(deploy_cfg, policy_dir, environ=None):
    """The agent settings of deploy.yml with their defaults; raises InfrastructureFailure on a setting this path
    does not take (an unknown key, a budget mode, an endpoint or key in the file)."""
    environ = os.environ if environ is None else environ
    deploy_cfg = dict(deploy_cfg or {})
    unknown = sorted(set(deploy_cfg) - set(STANDARD_KEYS) - set(SETTINGS_DEFAULTS) - {'model_settings'})
    if unknown:
        raise InfrastructureFailure('deploy.yml: settings the Mibot-RSI submission does not take: ' + ', '.join(unknown)
                                    + ' (the step budget is RoboDojo\'s own step_lim; no other budget mode exists)')
    settings = {key: deploy_cfg.get(key, default) if deploy_cfg.get(key) is not None else default
                for key, default in SETTINGS_DEFAULTS.items()}
    calls = settings['max_model_calls']
    if isinstance(calls, bool) or not isinstance(calls, int) or not 1 <= calls <= 10000:
        raise InfrastructureFailure('deploy.yml: max_model_calls must be a whole number from 1 to 10000')
    wall = settings['episode_wall_limit_s']
    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
        raise InfrastructureFailure('deploy.yml: episode_wall_limit_s must be a positive number of seconds')
    waits = settings['model_api_waits_s']
    if not isinstance(waits, (list, tuple)) or any(isinstance(w, bool) or not isinstance(w, (int, float)) or w < 0
                                                    for w in waits):
        raise InfrastructureFailure('deploy.yml: model_api_waits_s must be a list of seconds')
    model = deploy_cfg.get('model_settings') or {}
    if not isinstance(model, dict):
        raise InfrastructureFailure('deploy.yml: model_settings must be a mapping')
    refused = sorted(set(model) & {'api_key', 'base_url'})
    if refused:
        raise InfrastructureFailure('deploy.yml: the model endpoint and key come only from the environment '
                                    f'({BASE_URL_ENV} and the variable named by api_key_env), not ' + ', '.join(refused))
    unknown = sorted(set(model) - set(CLIENT_DEFAULTS) - CLIENT_OPTIONAL_KEYS)
    if unknown:
        raise InfrastructureFailure('deploy.yml: unknown model_settings: ' + ', '.join(unknown))
    summary = model.get('reasoning_summary')
    if summary is not None and summary not in REASONING_SUMMARIES:
        raise InfrastructureFailure('deploy.yml: model_settings.reasoning_summary must be one of '
                                    + ', '.join(REASONING_SUMMARIES) + ' or null')
    settings['model_settings'] = {**MODEL_DEFAULTS, **model}
    policy_dir = Path(policy_dir)
    settings['policy_dir'] = str(policy_dir)
    runs_root = Path(environ.get(RUNS_ROOT_ENV) or settings['runs_root']).expanduser()
    settings['runs_root'] = str(runs_root if runs_root.is_absolute() else policy_dir / runs_root)
    return settings


def check_environment(environ):
    """Refuse the settings of development-only features (the extended 5x budget, the exploration budget, privileged
    or synthetic hooks): none of them may change an official episode."""
    refused = [key for key in REFUSED_ENV if environ.get(key)]
    if environ.get('ROBOT_BUDGET_MODE', 'benchmark') != 'benchmark':
        refused.append('ROBOT_BUDGET_MODE=' + str(environ.get('ROBOT_BUDGET_MODE')))
    if refused:
        raise InfrastructureFailure('the submission runs the official step budget only; unset ' + ', '.join(refused))


def check_task_env(task_env, settings, state):
    """RoboDojo's evaluator as the submission expects it: one environment, the arx_x5 robot configuration and the
    task's own step_lim (a positive whole number that no episode of this process has changed)."""
    if int(getattr(task_env, 'num_envs', 1) or 1) != 1:
        raise InfrastructureFailure('Mibot-RSI runs one environment per process; set eval_batch: false')
    config_name = getattr(task_env, 'config_name', None)
    if config_name is not None and config_name != OFFICIAL_ENV_CFG:
        raise InfrastructureFailure(f'Mibot-RSI supports env_cfg_type {OFFICIAL_ENV_CFG}, not {config_name}')
    action_type = parse_info(getattr(task_env, 'additional_info', '')).get('action_type')
    if action_type not in (None, 'joint'):
        raise InfrastructureFailure('Mibot-RSI sends joint actions; run it with action_type joint, not ' + action_type)
    limit = task_env.step_lim
    if isinstance(limit, bool) or not isinstance(limit, numbers.Integral) or limit <= 0:
        raise InfrastructureFailure(f'the task step_lim must be a positive whole number, not {limit!r}')
    first = state.setdefault('step_lim', {}).setdefault(task_env.task_name, int(limit))
    if limit != first:
        raise InfrastructureFailure(f'step_lim changed from {first} to {limit} during this evaluation')


def check_endpoint(url):
    """The rule the session's model client applies (v2/runtime/client.py load_config), checked before the episode
    so that a malformed endpoint is a setup failure, not one found in every restarted process."""
    try:
        parts = urllib.parse.urlsplit(url)
        ok = (parts.scheme in ('http', 'https') and parts.netloc and not parts.username and not parts.password
              and not parts.query and not parts.fragment)
    except ValueError:
        ok = False
    if not ok:
        raise InfrastructureFailure(f'{BASE_URL_ENV} must be an HTTP(S) API base without credentials or query')


def parse_info(text):
    """RoboDojo's additional_info string ("ckpt_name=..,action_type=..") as a mapping."""
    out = {}
    for part in str(text or '').split(','):
        key, sep, value = part.partition('=')
        if sep:
            out[key.strip()] = value.strip()
    return out


# ------------------------------------------------------------------ knowledge

def standard_task(task):
    from v2.tasks import STANDARD_VARIANT
    return STANDARD_VARIANT.get(task, task)


def knowledge_root(task_env, settings, environ):
    """(directory, ckpt_name or None): the directory holding one knowledge snapshot per standard task.
    $MIBOT_RSI_KNOWLEDGE_DIR, else deploy.yml knowledge_dir (both: ckpt_name None), else XPolicyLab's layout
    checkpoints/<bench>-<ckpt_name>-<env_cfg>-<action>-<seed>/ with ckpt_name from the evaluator's additional_info
    (a ckpt_name with a path separator names the directory itself)."""
    policy_dir = Path(settings['policy_dir'])
    explicit = environ.get(KNOWLEDGE_ENV) or settings.get('knowledge_dir')
    if explicit:
        path = Path(explicit).expanduser()
        return (path if path.is_absolute() else policy_dir / path), None
    info = parse_info(getattr(task_env, 'additional_info', ''))
    ckpt = info.get('ckpt_name')
    if not ckpt:
        raise InfrastructureFailure('no knowledge directory: the evaluator gave no ckpt_name, and neither '
                                    f'{KNOWLEDGE_ENV} nor deploy.yml knowledge_dir is set')
    if '/' in ckpt or os.sep in ckpt:
        path = Path(ckpt).expanduser()
        return (path if path.is_absolute() else policy_dir / path), None
    name = '-'.join(['RoboDojo', ckpt, str(getattr(task_env, 'config_name', None) or OFFICIAL_ENV_CFG),
                     info.get('action_type') or 'joint', str(getattr(task_env, 'eval_seed', 0))])
    return policy_dir / 'checkpoints' / name, ckpt


def release_knowledge(policy_dir):
    """The knowledge release pinned in the policy's release.json: {ckpt_name, tasks: {standard task: pin}}."""
    path = Path(policy_dir) / 'release.json'
    if not path.is_file():
        return {'ckpt_name': None, 'tasks': {}}
    try:
        knowledge = json.loads(path.read_text()).get('knowledge') or {}
        return {'ckpt_name': knowledge.get('ckpt_name'), 'tasks': dict(knowledge.get('tasks') or {})}
    except (OSError, ValueError, AttributeError) as exc:
        raise InfrastructureFailure(f'{path} is unreadable: {exc}') from exc


def snapshot_files(folder):
    """Relative paths of the files (and links) under a snapshot folder."""
    return [path.relative_to(folder).as_posix() for path in Path(folder).rglob('*')
            if path.is_file() or path.is_symlink()]


def load_knowledge(task, root, policy_dir, ckpt=None):
    """Verify the task's frozen release snapshot and return {root, snapshot_id, task}: every file against its
    manifest, and the manifest against the snapshot release.json pins for the task (its id and the manifest's
    sha256), however the folder was found. A `_random` variant reads its standard task's snapshot. Knowledge found
    through the checkpoint layout (`ckpt`, the evaluator's ckpt_name) must also be the pinned release. A missing,
    unpinned or modified snapshot is an infrastructure failure: an episode never runs with empty, unknown or
    altered knowledge."""
    from v2.knowledge.store import KnowledgeError, sha256_file, verify_snapshot
    knowledge_task = standard_task(task)
    # Resolved: XPolicyLab's per-seed checkpoint folder links to one seed-free folder (download_checkpoint.sh), and
    # the session is shown paths inside the snapshot, which must not name the evaluation seed.
    folder = (Path(root) / knowledge_task).resolve()
    release = release_knowledge(policy_dir)
    if not release['ckpt_name']:
        raise InfrastructureFailure('release.json pins no knowledge release (see the policy README: '
                                    'download_checkpoint.sh)')
    if ckpt is not None and ckpt != release['ckpt_name']:
        raise InfrastructureFailure(f"ckpt_name {ckpt} is not the knowledge release {release['ckpt_name']} "
                                    'pinned in release.json')
    pin = release['tasks'].get(knowledge_task) or {}
    if not pin.get('snapshot_id') or not pin.get('manifest_sha256'):
        raise InfrastructureFailure(f"release.json pins no snapshot of {knowledge_task} in {release['ckpt_name']}")
    if ckpt is not None and Path(root).is_dir() and Path(root).resolve().name == Path(root).name:
        # A real per-seed folder: the session would be shown paths naming the evaluation seed.
        raise InfrastructureFailure(f'{root} must link to the seed-free knowledge folder that '
                                    'download_checkpoint.sh makes (checkpoints/mibot-rsi-knowledge-<ckpt_name>)')
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file():
        raise InfrastructureFailure(f'no knowledge snapshot for {knowledge_task} at {folder} '
                                    '(see the policy README: download_checkpoint.sh)')
    try:
        manifest = verify_snapshot(folder)
    except (KnowledgeError, OSError, ValueError, KeyError) as exc:
        raise InfrastructureFailure(f'knowledge snapshot of {knowledge_task} failed verification: {exc}') from exc
    if not manifest.get('full') or 'excluded_skills' not in manifest or 'excluded_experiences' not in manifest:
        raise InfrastructureFailure(f'{folder} is not a full release snapshot with admission applied')
    # The knowledge reader lists its folders, so a file outside the manifest would reach the model.
    extra = sorted(set(snapshot_files(folder)) - set(manifest['files']) - {'manifest.json'})
    if extra:
        raise InfrastructureFailure(f'the knowledge snapshot of {knowledge_task} holds files outside its manifest: '
                                    + ', '.join(extra[:5]))
    if pin['snapshot_id'] != manifest['id']:
        raise InfrastructureFailure(f"knowledge of {knowledge_task} is snapshot {manifest['id']}, but "
                                    f"release.json pins {pin['snapshot_id']}")
    if sha256_file(manifest_path) != pin['manifest_sha256']:
        raise InfrastructureFailure(f'the manifest of {knowledge_task} differs from the pinned release')
    return {'root': str(folder), 'snapshot_id': manifest['id'], 'task': knowledge_task}


# ------------------------------------------------------------------ runtime identity

def runtime_identity(state):
    """Where the running harness code came from, checked once per process: an exported copy (EXPORT.json, written
    by submission/export.py) must hold exactly its listed files with their hashes (a stray module in it would be
    imported by the session, which holds the key); a checkout reports its commit and source identity."""
    if 'runtime' in state:
        return state['runtime']
    export = HARNESS_ROOT / 'EXPORT.json'
    if export.is_file():
        record = json.loads(export.read_text())
        changed = [rel for rel, digest in record['files'].items()
                   if not (HARNESS_ROOT / rel).is_file() or
                   hashlib.sha256((HARNESS_ROOT / rel).read_bytes()).hexdigest() != digest]
        if changed:
            raise InfrastructureFailure('the harness runtime differs from its export: ' + ', '.join(changed[:5]))
        extra = sorted(path.relative_to(HARNESS_ROOT).as_posix() for path in HARNESS_ROOT.rglob('*')
                       if (path.is_file() or path.is_symlink()) and '__pycache__' not in path.parts
                       and path.relative_to(HARNESS_ROOT).as_posix() not in record['files'] and path != export)
        if extra:
            raise InfrastructureFailure('the harness runtime holds files outside its export: ' + ', '.join(extra[:5]))
        identity = {'root': str(HARNESS_ROOT), 'kind': 'export', 'commit': record.get('commit'),
                    'files': len(record['files'])}
    else:
        from v2.freeze import git_state, source_identity
        git = git_state(HARNESS_ROOT)
        identity = {'root': str(HARNESS_ROOT), 'kind': 'checkout', 'commit': git.get('commit'),
                    'dirty': git.get('dirty'), 'source': source_identity()}
    state['runtime'] = identity
    return identity


# ------------------------------------------------------------------ the official evaluator's process

STATE = {}  # per evaluation process: the first step_lim of each task, the runtime identity


def eval_one_episode(task_env, model_client=None, *, policy_dir, deploy_cfg, environ=None):
    """One RoboDojo evaluation episode (see the module docstring). `model_client` (the official policy-server
    connection) is not used: the policy server serves nothing, as for the organisers' own agent entry. An
    infrastructure failure leaves the process (stop_process)."""
    del model_client
    try:
        return episode(task_env, policy_dir, deploy_cfg, os.environ if environ is None else environ)
    except InfrastructureFailure as failure:
        stop_process(task_env, failure)
        raise  # reached only where stop_process is replaced (tests)


def episode(task_env, policy_dir, deploy_cfg, environ):
    if not hasattr(task_env, 'robot_manager'):
        return debug_episode(task_env)
    try:
        settings = load_settings(deploy_cfg, policy_dir, environ)
        check_environment(environ)
        check_task_env(task_env, settings, STATE)
        key_env = settings['model_settings']['api_key_env']
        missing = [name for name in (BASE_URL_ENV, key_env) if not environ.get(name)]
        if missing:
            raise InfrastructureFailure('set ' + ' and '.join(missing) + ' in the evaluation environment')
        check_endpoint(environ[BASE_URL_ENV])
        identity = runtime_identity(STATE)
        if not STATE.get('recovered'):
            STATE['recovered'] = True
            try:
                recover_unfinished(Path(settings['runs_root']), environ.get('ROBODOJO_RUN_ID'), task_env.task_name)
            except Exception as exc:  # recording an earlier episode never stops this one
                print(f'[Mibot-RSI] could not check for unfinished episodes: {type(exc).__name__}: {exc}',
                      file=sys.stderr, flush=True)
        root, ckpt = knowledge_root(task_env, settings, environ)
        knowledge = load_knowledge(task_env.task_name, root, policy_dir, ckpt)
        service = AttachedService(task_env, settings, knowledge, identity, environ)
    except InfrastructureFailure as failure:
        failure.restart = False  # setup: a new process would fail the same way
        raise
    except Exception as exc:  # an Exception would make RoboDojo skip this layout silently
        raise InfrastructureFailure(f'{type(exc).__name__}: {exc}', restart=False) from exc
    return service.run()


def eval_one_episode_batch(task_env, model_client=None, **kwargs):
    if int(getattr(task_env, 'num_envs', 1) or 1) > 1:
        failure = InfrastructureFailure('Mibot-RSI runs one conversation per environment; set eval_batch: false',
                                        restart=False)
        stop_process(task_env, failure)
        raise failure  # reached only where stop_process is replaced (tests)
    return eval_one_episode(task_env, model_client, **kwargs)


def debug_episode(task_env):
    """XPolicyLab's debug client (EVAL_ENV_TYPE=debug) has no robot or simulator: check the wiring only, holding the
    observed joints with open grippers until the episode ends. No model is called."""
    import numpy as np
    steps = 0
    while not task_env.is_episode_end():
        obs = task_env.get_obs()
        missing = [name for name in ('cam_head', 'cam_left_wrist', 'cam_right_wrist') if name not in obs['vision']]
        if missing or 'instruction' not in obs:
            raise InfrastructureFailure('debug observation lacks ' + ', '.join(missing or ['instruction']), restart=False)
        action = {}
        for arm in ('left', 'right'):
            action[f'{arm}_arm_joint_state'] = np.asarray(obs['state'][f'{arm}_arm_joint_state'], dtype=np.float32)
            action[f'{arm}_ee_joint_state'] = np.ones(1, dtype=np.float32)
        task_env.take_action(action)
        steps += 1
    print(f'[Mibot-RSI] debug wiring check: {steps} hold actions, no model call', flush=True)
    return {'class': 'debug', 'steps': steps}


def close_attempt(env):
    """RoboDojo's end-of-episode check for an episode that did not end by itself (as embodied_agent/simulator.py):
    it is scored as a failure with its partial credit, never as the success RoboDojo presets."""
    if not env.end_flag[0]:
        env.success[0] = False
        env.is_episode_end()


class AttachedService:
    """The robot side of one episode, in the evaluator's process."""

    def __init__(self, task_env, settings, knowledge, identity, environ):
        self.env, self.settings, self.knowledge, self.identity = task_env, settings, knowledge, identity
        self.environ = environ
        self.episode_id = 'ep-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + secrets.token_hex(3)
        self.runs_root = Path(settings['runs_root'])
        # The run directory names neither seed nor layout: the worker can read its own run's files.
        self.run_dir = self.runs_root / task_env.task_name / self.episode_id
        self.runner = self._runner_log = None
        self.physx = None
        self.watch = None  # the simulator watchdog of this episode (v2/simwatch.py)
        self._runner_lock = threading.Lock()  # the watchdog never misses a session started while it acts
        self._recorded = self._finished = False  # the failure, and the episode's end, are recorded once
        self._events_offset, self._waits_checked = 0, 0.0  # model API waits shown so far (_report_waits)
        seeds = list(getattr(task_env, 'env_seeds', None) or [None])
        self.record = {'episode_id': self.episode_id, 'task': task_env.task_name,
                       'knowledge_task': knowledge['task'], 'robodojo_run_id': environ.get('ROBODOJO_RUN_ID'),
                       'eval_seed': getattr(task_env, 'eval_seed', None),
                       'layout_id': seeds[0],
                       'official_index': int(getattr(task_env, 'success_nums', 0) or 0) +
                       int(getattr(task_env, 'fail_nums', 0) or 0),
                       'official_result_dir': str(getattr(task_env, 'save_dir', '') or ''),
                       'step_lim': int(task_env.step_lim), 'max_model_calls': settings['max_model_calls'],
                       'episode_wall_limit_s': settings['episode_wall_limit_s'],
                       'knowledge_snapshot': knowledge['snapshot_id'],
                       'runtime': identity, 'run_dir': str(self.run_dir), 'wall_started': now(),
                       # who serves the episode, for recover_unfinished in the next evaluation process
                       'evaluator_pid': os.getpid(), 'evaluator_identity': process_identity(),
                       'evaluator_kit_log': kit_log_path(),
                       'official_physx_monitor': official_physx_monitor(task_env)[0]}

    # -- lifecycle

    def run(self):
        """Returns after a scored episode; raises a PhysX error unchanged, anything else as InfrastructureFailure
        (an Exception would make RoboDojo skip the layout). Record writing never changes the outcome."""
        outcome = None
        try:
            append_jsonl(self.runs_root / 'episodes.jsonl', {**self.record, 'event': 'started'})
            outcome = self._run()
        except BaseException as exc:
            physx = self._physx_error(exc)
            if not self._recorded:  # else the simulator watchdog has recorded the failure
                # RoboDojo reruns the layout after a PhysXFatalError (shell or in-process restart, main.py) and
                # abandons it only after a PhysXBrokenError (abandoned seeds, refilled from the queue).
                self._record_failure(exc, 'simulator_failure' if physx else 'infrastructure',
                                     rerun=type(exc).__name__ != 'PhysXBrokenError')
            if physx or isinstance(exc, (InfrastructureFailure, KeyboardInterrupt, SystemExit)):
                raise
            raise InfrastructureFailure(f'{type(exc).__name__}: {exc}') from exc
        finally:
            self._record_finished()
        return outcome

    def _record_failure(self, exc, kind, rerun=True, extra=None):
        """The failure in the episode record and, for a layout that runs again, in runs/submission_reruns.jsonl
        (`extra`: the watchdog's check and evidence)."""
        self._recorded = True
        self.record.update(**{'class': kind, 'error': f'{type(exc).__name__}: {exc}'[:2000]})
        if rerun:
            self._log(self.runs_root / 'submission_reruns.jsonl',
                      {key: self.record.get(key) for key in ('episode_id', 'task', 'robodojo_run_id', 'eval_seed',
                                                             'layout_id', 'official_index', 'error')} |
                      {'physical_steps': self._steps(), 'restart': getattr(exc, 'restart', True), 'time': now()} |
                      (extra or {}))

    def _record_finished(self):
        """The episode's end: submission_episode.json in its run directory and the developer index
        runs/episodes.jsonl (maps episode IDs to seeds and layouts)."""
        if self._finished:
            return
        self._finished = True
        self.record.update(wall_finished=now(), physical_steps=self._steps(),
                           end_flag=self._flag('end_flag'), success=self._flag('success'),
                           harness_modules=harness_modules())
        if self.run_dir.is_dir():
            self._log(self.run_dir / 'submission_episode.json', self.record, whole=True)
        self._log(self.runs_root / 'episodes.jsonl', {**self.record, 'event': 'finished'})

    @staticmethod
    def _log(path, record, whole=False):
        """A developer record; a failure to write it is reported, never raised. A shallow copy is written: the
        watchdog's thread may write while the main thread still updates the episode record."""
        try:
            record = dict(record)
            write_json(path, record) if whole else append_jsonl(path, record)
        except Exception as exc:
            print(f'[Mibot-RSI] could not write {path}: {type(exc).__name__}: {exc}', file=sys.stderr, flush=True)

    def _run(self):
        from v2.episode import robot_settings
        self._import_adapter()
        from robodojo_adapter import RoboDojoAdapter
        from rpc_server import CommandServer
        self.run_dir.mkdir(parents=True)
        (self.run_dir / 'workspace').mkdir()
        (self.run_dir / 'tool_images').mkdir()
        adapter = server = None
        try:
            try:
                try:
                    adapter = RoboDojoAdapter(self.env, self.run_dir / 'recording',
                                              environ=robot_settings('evaluation'), submission=True)
                    self._watch_start(adapter)
                    token = secrets.token_urlsafe(32)
                    server = CommandServer(token)
                    url = server.start()
                    with self._watched('the first observation'):
                        adapter.capture()
                        description = adapter.describe()
                    write_json(self.run_dir / 'robot_description.json', description)
                    write_json(self.run_dir / 'connection.json',
                               {'url': url, 'token': token, 'episode_id': self.episode_id}, mode=0o600)
                    self._start_runner()
                    self._serve(adapter, server)
                except BaseException as exc:
                    self._stop_runner()
                    if self._physx_error(exc):
                        raise
                    if isinstance(exc, InfrastructureFailure):
                        raise
                    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                        raise
                    raise InfrastructureFailure(f'robot service failed: {type(exc).__name__}: {exc}') from exc
            finally:
                if adapter is not None:
                    adapter.close()
                if server is not None:
                    time.sleep(FINAL_REPLY_S)
                    server.close()
                if self._runner_log is not None:
                    self._runner_log.close()
            if self._halted():
                raise self._halt_failure()
            if self.physx is not None:
                # RoboDojo abandons a layout whose physics broke and takes another (src/eval_client/main.py); the
                # error reached the model only as a failed command.
                self.record['class'] = 'simulator_failure'
                raise self.physx
            result = self._read_result()
            self.record.update(session_result=result)
            kind = result.get('class')
            if kind not in ('completed', 'capability_end'):
                reason = result.get('reason') or f'session runner exited with code {self.runner.returncode}'
                raise InfrastructureFailure(f'{kind or "infrastructure"}: {reason}')
            with self._watched('the end check'):
                close_attempt(self.env)
            self.record['class'] = kind
            return {'class': kind, 'episode_id': self.episode_id, 'success': bool(self.env.success[0])}
        finally:
            self._watch_stop()

    # -- the simulator watchdog (v2/simwatch.py)

    def _watch_start(self, adapter):
        """Start watching the simulator for this episode: its log (unless RoboDojo's own PhysX monitor runs for the
        task), the progress of each robot command (RoboDojo's step counter and the adapter's physics substep
        counter) and the wall limit. Log offsets carry over between the episodes of one process."""
        self.watch = None
        try:
            from v2.simwatch import SimWatch
            watch = SimWatch(lambda: (self._steps(), getattr(adapter, 'low_step', None)), self._watchdog_failure,
                             official=official_physx_monitor(self.env),
                             offsets=STATE.setdefault('sim_log_offsets', {}),
                             wall_limit_s=self.settings['episode_wall_limit_s'])
            self.record['sim_watch'] = watch.describe()
            watch.start()
            self.watch = watch
        except Exception as exc:  # the episode runs unwatched rather than not at all
            self.record['sim_watch'] = {'error': f'{type(exc).__name__}: {exc}'[:300]}
        if not STATE.get('sim_watch_shown') or 'error' in self.record['sim_watch']:
            STATE['sim_watch_shown'] = True
            print('[Mibot-RSI] simulator watchdog: ' + json.dumps(self.record['sim_watch']), file=sys.stderr,
                  flush=True)

    @contextlib.contextmanager
    def _watched(self, name):
        """Simulator work on this thread that the watchdog checks for progress (a robot command)."""
        watch = getattr(self, 'watch', None)
        if watch is not None:
            watch.command(name)
        try:
            yield
        finally:
            if watch is not None:
                watch.command(None)

    def _halted(self):
        watch = getattr(self, 'watch', None)
        return watch is not None and watch.fired is not None

    def _halt_failure(self):
        kind, reason = self.watch.fired
        return InfrastructureFailure(f'simulator_failure: {reason}' if kind == 'simulator_failure' else reason)

    def _watch_stop(self):
        """End the watch before the episode's end is recorded. While the watchdog acts this waits until it has left
        the process; where it returned instead (tests), the episode is not scored."""
        if self.watch is None:
            return
        self.watch.final_look()
        fired = self.watch.stop()
        self._note_watch()
        if fired is not None:
            raise self._halt_failure()

    def _note_watch(self):
        """Invalid PhysX transform lines seen during the episode (counted, never acted on) and watched lines of
        another process sharing the log file (ignored), for the episode record (v2/simwatch.py)."""
        if self.watch is None:
            return
        if self.watch.broken_envs:
            self.record.setdefault('sim_watch', {})['invalid_transform_envs'] = sorted(self.watch.broken_envs)
        if self.watch.foreign_lines:
            self.record.setdefault('sim_watch', {})['foreign_lines'] = self.watch.foreign_lines

    def _watchdog_failure(self, kind, reason):
        """A finding of the simulator watchdog, in its thread: the main thread may be wedged inside a robot command
        for good. Recorded as the robot service records an infrastructure failure (a simulator failure also as the
        session's simulator_failure event and result), then the session is stopped and the process leaves as
        stop_process does (rc 99), so that RoboDojo runs this layout again."""
        failure = InfrastructureFailure(f'simulator_failure: {reason}' if kind == 'simulator_failure' else reason)
        try:
            print(f"[Mibot-RSI] {self.record['task']} layout {self.record['layout_id']}: {reason}; stopping the "
                  'session and leaving the evaluation process so that this layout runs again', file=sys.stderr,
                  flush=True)
            with self._runner_lock:  # a session the main thread is starting now is stopped too (_start_runner)
                self._stop_runner()
            finding = dict(getattr(self.watch, 'finding', None) or {})
            audit = {'check': finding.get('check'), 'evidence': self._scrub(finding.get('evidence'))}
            self.record['watchdog'] = audit
            if kind == 'simulator_failure':
                self._note_simulator_failure(reason, audit)
            self._note_watch()
            self._record_failure(failure, 'simulator_failure' if kind == 'simulator_failure' else 'infrastructure',
                                 extra=audit)
            self._record_finished()
        except BaseException as exc:
            print(f'[Mibot-RSI] could not record the failure: {type(exc).__name__}: {exc}', file=sys.stderr,
                  flush=True)
        finally:
            stop_process(self.env, failure)

    def _scrub(self, text):
        """Evidence from a log, without the model endpoint (as the session's records)."""
        if not text:
            return text
        try:
            from v2.runtime.loop import scrub_endpoint
            return scrub_endpoint(text, (getattr(self, 'environ', None) or {}).get(BASE_URL_ENV, ''))
        except Exception:
            return text

    def _note_simulator_failure(self, reason, audit=None):
        """As the session records a simulator that broke (v2/runtime/loop.py): a simulator_failure event in its
        events.jsonl (with the watchdog's check and evidence), and class simulator_failure in session_result.json,
        which keeps what the stopped session wrote (its class and reason as session_class and session_reason)."""
        if (self.run_dir / 'session').is_dir():
            try:
                append_private(self.run_dir / 'session' / 'events.jsonl',
                               {'time': now(), 'event': 'simulator_failure', 'reason': reason, **(audit or {})})
            except OSError as exc:
                print(f'[Mibot-RSI] could not write the session event: {exc}', file=sys.stderr, flush=True)
        try:
            own = json.loads((self.run_dir / 'session_result.json').read_text())
        except (OSError, ValueError):
            own = None
        own = own if isinstance(own, dict) else {}
        result = {**own, 'class': 'simulator_failure', 'reason': reason, 'session_class': own.get('class'),
                  'session_reason': own.get('reason')}
        self.record['session_result'] = result
        if self.run_dir.is_dir():
            self._log(self.run_dir / 'session_result.json', result, whole=True)

    def _import_adapter(self):
        """The robot service modules of embodied_agent/, which use flat imports. They must come from this harness:
        a same-named module already loaded from elsewhere would be used instead, so that is refused."""
        code = str(HARNESS_ROOT / 'embodied_agent')
        if code not in sys.path:
            sys.path.insert(0, code)
        names = ('robodojo_adapter', 'rpc_server', 'evidence', 'geometry', 'motion_evidence', 'motion_timing',
                 'observation_capture', 'episode_budget')
        for name in names:
            module = sys.modules.get(name)
            if module is not None and not str(getattr(module, '__file__', '')).startswith(code):
                raise InfrastructureFailure(f'module {name} is already loaded from {getattr(module, "__file__", None)}')

    # -- the session subprocess

    def _start_runner(self):
        model = dict(self.settings['model_settings'])
        write_json(self.run_dir / 'model_settings.json', model)  # public settings: no endpoint, no key
        spec = {'schema': 'mibot_rsi_session/1', 'run_id': self.episode_id, 'campaign': 'mibot-rsi',
                'task': self.env.task_name, 'knowledge': {key: self.knowledge[key] for key in
                                                          ('root', 'snapshot_id', 'task')},
                'max_model_calls': self.settings['max_model_calls'], 'redact_endpoint': True,
                'model_api_waits_s': list(self.settings['model_api_waits_s']),
                'model_api_wait_refusals': True,  # 401/403/404 and quota refusals are waited out too
                'worker_settings': {'V2_END_WITH_PARENT': '1'}}  # the worker ends with the session (worker.serve)
        write_json(self.run_dir / 'session_spec.json', spec)
        key_env = model['api_key_env']
        env = {key: self.environ[key] for key in RUNNER_ENV if self.environ.get(key)}
        env.update({key_env: self.environ[key_env], BASE_URL_ENV: self.environ[BASE_URL_ENV],
                    'PYTHONPATH': str(HARNESS_ROOT), 'PYTHONUNBUFFERED': '1', 'PYTHONNOUSERSITE': '1'})
        if threading.current_thread() is threading.main_thread():
            # The parent-death signal follows the thread that starts the session (RoboDojo's main thread, which lasts
            # as long as the evaluation); from a shorter-lived thread the session would be ended with it.
            env[PARENT_ENV] = str(os.getpid())
        python = self.settings.get('runner_python') or sys.executable
        if landlock.abi_version() < 1:  # the worker then runs on the Python guard alone (worker.serve); run.json says so
            print("[Mibot-RSI] this kernel has no Landlock: the agent's Python worker runs on the Python audit hook "
                  'alone (native-library calls are not confined by the kernel)', file=sys.stderr, flush=True)
        self._runner_log = (self.run_dir / 'runner.log').open('w')
        with self._runner_lock:
            if self._halted():  # the watchdog acted before the session started: none is started
                raise self._halt_failure()
            self.runner = subprocess.Popen([python, '-u', '-m', 'v2.submission', 'session', str(self.run_dir)],
                                           cwd=str(HARNESS_ROOT), env=env, stdout=self._runner_log,
                                           stderr=subprocess.STDOUT, start_new_session=True)
        self.record['runner_pid'] = self.runner.pid

    def _stop_runner(self):
        """SIGTERM to the session subprocess (it then closes its worker), SIGKILL if it does not stop; then any
        process it left behind, such as a worker in its own session, is killed."""
        runner = self.runner
        if runner is None or runner.poll() is not None:
            return
        tree = descendants(runner.pid)
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(runner.pid, sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                runner.wait(timeout=RUNNER_STOP_S)
                break
            except subprocess.TimeoutExpired:
                continue
        own = os.getpgrp()
        for pid in tree:
            try:
                group = os.getpgid(pid)
                if group != own:
                    os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue

    def _serve(self, adapter, server):
        """Execute the session's robot commands on this (the simulator's) thread until the session subprocess exits;
        failed commands are answered as in embodied_agent/simulator.py."""
        from evidence import EvidenceQueryError
        deadline = time.monotonic() + float(self.settings['episode_wall_limit_s'])
        while True:
            if time.monotonic() > deadline:
                self._stop_runner()
                raise InfrastructureFailure(f"episode wall limit of {self.settings['episode_wall_limit_s']} s reached")
            if self.runner.poll() is not None:
                self._report_waits(force=True)
                self._reject_leftovers(server)
                return
            try:
                record = server.next()
            except queue.Empty:
                self._report_waits()
                continue
            if self.runner.poll() is not None:  # it exited while this command was awaited
                self._report_waits(force=True)
                self._reject_leftovers(server, record)
                return
            if self._halted():  # the watchdog is stopping the session: the simulator is not touched again
                self._reject(server, record, 'the simulator has failed; this command was not run', 'simulator_failure')
                continue
            try:
                with self._watched(f"{record['id']} ({record['request'].get('method')})"):
                    result = adapter.execute(record)
                server.complete(record, result)
            except Exception as exception:
                error = traceback.format_exc()
                if self._physx_error(exception) and self.physx is None:
                    self.physx = exception
                if isinstance(exception, EvidenceQueryError) and record['request']['method'] in ('history', 'inspect'):
                    server.complete(record, {'status': 'rejected', 'error': str(exception),
                                             'error_type': 'read_only_request_validation'}, 'fault')
                    append_jsonl(self.run_dir / 'request_errors.jsonl', {
                        'command_id': record['id'], 'request': record['request'],
                        'error_type': 'read_only_request_validation', 'message': str(exception),
                        'wall_time_s': time.time()})
                else:
                    # The model sees only the exception's last line; the traceback stays in the developer log.
                    lines = [line for line in error.strip().splitlines() if line.strip()]
                    server.complete(record, {'status': 'fault', 'error': lines[-1] if lines else 'fault'}, 'fault')
                    with (self.run_dir / 'service_errors.log').open('a') as stream:
                        stream.write(error)

    def _report_waits(self, force=False):
        """Show the session's model API waits on the evaluator's console as they happen (every 5 s at most, and once
        more when the session ends): refusals and quota errors are waited out like outages, so a wrong key or model
        name, or an exhausted balance, must be visible there and not only in the session's events (which the session
        writes with the endpoint already scrubbed, redact_endpoint)."""
        now_s = time.monotonic()
        if not force and now_s < self._waits_checked + 5:
            return
        self._waits_checked = now_s
        try:
            with (self.run_dir / 'session' / 'events.jsonl').open('rb') as stream:
                stream.seek(self._events_offset)
                data = stream.read()
        except OSError:
            return
        end = data.rfind(b'\n') + 1  # whole lines only; a line still being written is read next time
        self._events_offset += end
        for line in data[:end].splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if not isinstance(event, dict) or event.get('event') != 'model_api_wait':
                continue
            failure = f"HTTP {event['status']}" if event.get('status') else 'connection failure'
            if event.get('code'):
                failure += ' ' + str(event['code'])[:80]
            print(f"[Mibot-RSI] {self.record['task']} layout {self.record['layout_id']}: model API {failure}; waiting "
                  f"{event.get('wait_s')} s ({event.get('wait')}/{event.get('of')}) before repeating the request. If "
                  'this persists, check the endpoint, key, model name and account balance.',
                  file=sys.stderr, flush=True)

    def _reject_leftovers(self, server, record=None):
        """Commands still queued once the session subprocess has exited (a stopped worker's last request), and
        `record` already taken, are answered, never run: the robot does not move after the session ends
        (embodied_agent/simulator.py likewise stops serving at the runner's stop request)."""
        while True:
            if record is None:
                try:
                    record = server.next(timeout=0)
                except queue.Empty:
                    return
            self._reject(server, record, 'the session has ended; this command was not run', 'after_session_end')
            record = None

    def _reject(self, server, record, message, error_type):
        """Answer a command without running it, and log it in request_errors.jsonl."""
        server.complete(record, {'status': 'rejected', 'error': message, 'error_type': error_type}, 'fault')
        append_jsonl(self.run_dir / 'request_errors.jsonl', {
            'command_id': record['id'], 'request': record['request'], 'error_type': error_type, 'message': message,
            'wall_time_s': time.time()})

    def _read_result(self):
        try:
            result = json.loads((self.run_dir / 'session_result.json').read_text())
            if isinstance(result, dict):
                return result
        except (OSError, ValueError):
            pass
        return {'class': 'infrastructure', 'reason': f'the session runner exited with code '
                                                      f'{self.runner.returncode} without a result; see runner.log'}

    # -- helpers

    @staticmethod
    def _physx_error(exc):
        return type(exc).__name__ in PHYSX_ERRORS

    def _steps(self):
        try:
            return int(self.env.take_action_cnt[0])
        except Exception:
            return None

    def _flag(self, name):
        try:
            return bool(getattr(self.env, name)[0])
        except Exception:
            return None


# ------------------------------------------------------------------ the session subprocess

def attached_episode_class():
    from v2.episode import DEFAULTS, Episode

    class AttachedEpisode(Episode):
        """An evaluation episode whose robot service already runs in the official evaluator's process. The
        evaluator chooses seed and layout; this session never learns them."""

        def __init__(self, spec, config, knowledge, client=None):
            self.spec = {**DEFAULTS, **spec, 'split': 'evaluation', 'seed': None, 'layout_id': None}
            self.config, self.client, self.knowledge = config, client, knowledge
            self.run_dir = Path(spec['run_dir'])
            self.workspace = self.run_dir / 'workspace'
            self.record = None
            self.history_records, self.history_roots, self.history_recordings = [], [], {}
            self.history_index = None
            self.replay_report = None
            self.benchmark_step_limit = None
            self._sim = None

        def _note(self, record):
            from v2.runtime.loop import scrub_endpoint
            super()._note(scrub_endpoint(record, self.config['base_url']))

        def run(self):
            from v2.freeze import launch_command, source_identity
            from v2.runtime.client import config_hash, public_config
            from v2.runtime.loop import endpoint_record, scrub_endpoint
            spec, run_dir = self.spec, self.run_dir
            description = json.loads((run_dir / 'robot_description.json').read_text())
            instruction, step_limit = description.get('task'), description.get('step_limit')
            self.benchmark_step_limit = description.get('benchmark_step_limit')
            if not step_limit or step_limit != self.benchmark_step_limit or description.get('budget_mode') != 'benchmark':
                raise RuntimeError('the robot service does not report the official step budget')
            self.record = record = {
                'schema': 'v2_submission_run/1', 'run_id': spec['run_id'], 'task': spec['task'],
                'split': 'evaluation', 'knowledge_snapshot': self.knowledge['snapshot_id'],
                'knowledge_task': self.knowledge['task'], 'model': self.config['model'],
                'reasoning_effort': self.config.get('reasoning_effort'), 'config_hash': config_hash(self.config),
                'model_config': {**public_config(self.config),
                                 'base_url': endpoint_record(self.config['base_url'], True)},
                'max_model_calls': spec['max_model_calls'],
                'instruction': instruction, 'step_limit': step_limit, 'source': source_identity(),
                'command': launch_command(), 'status': 'running', 'wall_started': now(),
                'isolation': landlock.isolation_summary()}
            write_json(run_dir / 'run.json', record)
            result, error = self._run_session(run_dir / 'connection.json', instruction, step_limit)
            record['session'] = result
            if error is not None:
                record.update(status='needs_attention', error={'type': type(error).__name__,
                                                               'reason': getattr(error, 'reason', None),
                                                               'message': scrub_endpoint(
                                                                   str(error), self.config['base_url'])[:2000]})
            else:
                record['status'] = result['status']
            guard_log = run_dir / 'guard.jsonl'
            record['guard_blocks'] = len(guard_log.read_text().splitlines()) if guard_log.is_file() else 0
            record['wall_finished'] = now()
            write_json(run_dir / 'run.json', record)
            return record

    return AttachedEpisode


def classify(record):
    """How the official process treats a finished session: 'completed' and 'capability_end' are scored as they stand
    (the agent finished, gave up, kept failing, the environment ended it, or the model call cap was reached);
    'simulator_failure' and 'infrastructure' are not the agent's result and leave the process for a rerun (a model
    API refusal or quota error included: a gateway can return them transiently)."""
    error = record.get('error') or {}
    session = record.get('session') or {}
    if record.get('status') == 'completed':
        ended_by = (session.get('outcome') or {}).get('ended_by')
        if ended_by == 'simulator_failure':
            return 'simulator_failure', (session.get('outcome') or {}).get('reason')
        return 'completed', ended_by
    if error.get('reason') == 'call_limit':
        return 'capability_end', 'model call cap reached'
    return 'infrastructure', error.get('message') or f"session status {record.get('status')}"


def run_session(run_dir, client=None, environ=None):
    """`python -m v2.submission session RUN_DIR`: the model session of one episode; writes session_result.json."""
    from v2.knowledge.store import Knowledge
    from v2.runtime.client import bypass_proxy, load_config
    from v2.runtime.loop import scrub_endpoint
    environ = os.environ if environ is None else environ
    run_dir = Path(run_dir)
    result = {'class': 'infrastructure', 'reason': 'the session did not start'}
    try:
        spec = json.loads((run_dir / 'session_spec.json').read_text())
        config = load_config(run_dir / 'model_settings.json', overrides={'base_url': environ.get(BASE_URL_ENV, '')})
        bypass_proxy(config)
        knowledge = dict(spec['knowledge'], writable=False,
                         text=Knowledge(spec['knowledge']['root']).prompt_text(spec['knowledge']['task']))
        episode = attached_episode_class()({**spec, 'run_dir': str(run_dir)}, config, knowledge, client=client)
        record = episode.run()
        kind, reason = classify(record)
        session = record.get('session') or {}
        result = {'class': kind, 'reason': scrub_endpoint(reason, environ.get(BASE_URL_ENV, '')),
                  'status': record.get('status'),
                  'model_calls': session.get('model_calls'), 'usage': session.get('usage'),
                  'outcome': session.get('outcome')}
    except BaseException as exc:
        result = {'class': 'infrastructure',
                  'reason': scrub_endpoint(f'{type(exc).__name__}: {exc}', environ.get(BASE_URL_ENV, ''))[:2000]}
        write_json(run_dir / 'session_result.json', result)
        if not isinstance(exc, Exception):
            raise
        print(scrub_endpoint(traceback.format_exc(), environ.get(BASE_URL_ENV, '')), file=sys.stderr, flush=True)
        return 3
    result['harness_modules'] = harness_modules()
    write_json(run_dir / 'session_result.json', result)
    return 0 if result['class'] in ('completed', 'capability_end') else 3


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2 or argv[0] != 'session':
        print('usage: python -m v2.submission session RUN_DIR', file=sys.stderr)
        return 2

    def interrupted(*_):
        # One is enough: the session then records its end and closes its worker. A parent-death signal comes again as
        # each thread of the ending evaluation process exits, and a second KeyboardInterrupt inside that cleanup would
        # lose the session's result.
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        raise KeyboardInterrupt('SIGTERM')
    signal.signal(signal.SIGTERM, interrupted)
    parent = os.environ.get(PARENT_ENV)
    if parent:
        # Ended (SIGTERM, as by _stop_runner) with the evaluation process however that ends: RoboDojo's PhysX
        # watchdog (os._exit(99)) or a kill. Without it the session would go on calling the model against a robot
        # service that is gone until its failure rules stop it.
        from v2.runtime.worker import end_with_parent
        if not end_with_parent(signal.SIGTERM, parent):
            print('the evaluation process has already ended; the session is not started', file=sys.stderr, flush=True)
            return 3
    return run_session(argv[1])


if __name__ == '__main__':
    sys.exit(main())

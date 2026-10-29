"""One v2 actor episode: simulator, robot service, Python worker and model session.

Adapted from embodied_agent/run_task.py. The
simulator serves the robot; the model session never reads simulator truth.
After the session ends, the runner asks the simulator to stop; only then does
the simulator write result.json, the external outcome, which is for the
developer and never shown to a model. A development episode's simulator also
scores the attempt when it ends (attempt_result.json); the consolidation note
tells the model that official result.

Run directory:
  workspace/            the model's files (read-write), including .context_archives/
  recording/            RGB frames and commands.jsonl from the robot service
  tool_images/          images the model displayed (crops, overlays)
  session/              checkpoint, events, private responses
  run.json, result.json, attempt_result.json, simulator.log, guard.jsonl   developer-only
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

from v2.freeze import launch_command, provenance, source_identity
from v2.runtime import context, landlock
from v2.runtime.client import DEFAULTS as CLIENT_DEFAULTS, HARNESS_ROOT, config_hash, public_config
from v2.runtime.loop import API_WAITS_S, NeedsAttention, Session, prompt_cache_key, user_message
from v2.runtime.worker import WORKER_SOURCE, WorkerProcess, child_environment
from v2.tasks import STANDARD_VARIANT, TASKS
from v2.knowledge.store import jsonl_lines

CODE_V1 = HARNESS_ROOT / 'embodied_agent'
# The only v1 file model code may read: the Robot SDK it drives. The rest of embodied_agent/
# (simulator, adapter, assessment) is harness code the actor must not read.
SDK_SOURCE = CODE_V1 / 'robot_sdk.py'
PHYSICAL_METHODS = ('move', 'gripper', 'move_to_mark', 'trajectory', 'mark_pose', 'hold')
RUN_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}')
OFFICIAL_ENV_CFG = 'arx_x5'  # the robot configuration the holdout layout files were registered with
# A simulator whose runner died (for example SIGKILL) exits after this long without requests;
# longer than any wait the harness makes between robot calls.
SIMULATOR_IDLE_TIMEOUT_S = 4 * 3600
DEFAULTS = {'repo': str(Path.home() / 'workspace' / 'RoboDojo'), 'env_cfg': 'arx_x5', 'device_id': '0',
            'service_port': 19999, 'startup_timeout_s': 240, 'lock_file': '/tmp/embodied_robot_simulator.lock',
            # On machines shared under one account, the simulator keeps Isaac Sim's caches, logs and settings in a
            # HOME of its own; unset, it uses the account's HOME.
            'simulator_home': os.environ.get('V2_SIMULATOR_HOME') or None}
# Motion speed limits of v2 episodes (with the adapter defaults, 0.2 m/s, 1 rad/s and 1.5 rad/s, large
# reorienting moves were rotation-bound).
SPEED_LIMITS = {'speed_m_s': 0.4, 'speed_rad_s': 2.0, 'mark_joint_speed_rad_s': 3.0}
# Motion profile of v2 episodes (embodied_agent/motion_timing.py): None keeps
# the original constant-speed moves; e.g. 'smooth' would ramp every Cartesian move and waypoint. Set here, not
# taken from the operator's shell, so that a frozen copy fixes it.
MOTION_PROFILE = None
DEFAULT_COMPACT_THRESHOLD = CLIENT_DEFAULTS['compact_threshold']
# How long the consolidation note waits for the simulator to score an attempt that just ended.
ATTEMPT_RESULT_WAIT_S = 30


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_json(path, value, mode=None):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + '\n')
    if mode is not None:
        temp.chmod(mode)
    os.replace(temp, path)


def validate(spec):
    """Split rules: development is seed 0 (any of its layouts) on a task's
    standard variant. Seeds 1 and 2 are the evaluation environment: an evaluation episode is one of the task's
    official holdout episodes, or any seed-1/2 episode with a snapshot whose official evaluation of this task
    has completed."""
    if not RUN_ID.fullmatch(str(spec.get('run_id', ''))):
        raise ValueError('run_id must use letters, digits, underscores or hyphens')
    if spec['task'] not in TASKS:
        raise ValueError('task must be one of the tasks in v2/tasks.py')
    seed, layout = spec['seed'], spec['layout_id']
    if type(seed) is not int or type(layout) is not int:
        raise ValueError('seed and layout_id must be integers')
    if spec['split'] == 'development':
        if seed != 0 or layout < 0:
            raise ValueError('development uses seed 0 (any of its layouts)')
        if spec['task'] in STANDARD_VARIANT:
            raise ValueError(f"development uses the standard variant {STANDARD_VARIANT[spec['task']]}; "
                             f"{spec['task']} is evaluated with its release snapshot")
    elif spec['split'] == 'evaluation':
        if seed not in (1, 2) or layout < 0:
            raise ValueError('evaluation uses seed 1 or 2')
    else:
        raise ValueError('split must be development or evaluation')
    if spec.get('official'):
        from v2.holdout import registered_holdout
        entry = next((e for e in registered_holdout(spec['task']) if (e['seed'], e['layout_id']) == (seed, layout)),
                     None) if spec['split'] == 'evaluation' else None
        if entry is None:
            raise ValueError('an official episode must be one of the preregistered holdout episodes')
        if spec.get('expected_layout_sha256') != entry['layout_sha256']:
            raise ValueError('an official episode needs the preregistered layout file hash')
        if spec.get('env_cfg', DEFAULTS['env_cfg']) != OFFICIAL_ENV_CFG:
            raise ValueError(f'official episodes use env_cfg {OFFICIAL_ENV_CFG}')
    elif spec['split'] == 'evaluation':
        # Development never runs seeds 1 and 2. Once a snapshot's evaluation of the task has completed, more
        # seed-1/2 runs of that same frozen snapshot are allowed (they cannot change it); a new snapshot is
        # protected again.
        if not official_evaluation_completed(spec.get('runs_root'), spec['task'], spec.get('snapshot')):
            raise ValueError('seeds 1 and 2 run only in the official evaluation, or with a snapshot whose '
                             'official evaluation of this task has completed')
        developing = campaigns_in_development(spec.get('runs_root'), STANDARD_VARIANT.get(spec['task'], spec['task']))
        if developing:
            raise ValueError(f"seeds 1 and 2 stay closed while a campaign of this task is in development "
                             f"in this runs root ({', '.join(developing)})")


def campaigns_in_development(runs_root, task):
    """Campaigns under runs_root still developing this task: they have episodes and have not ended
    (ended, released and evaluated campaigns no longer explore)."""
    names = []
    for path in sorted(Path(runs_root).glob('*/campaign.json')) if runs_root else ():
        try:
            state = (json.loads(path.read_text()).get('tasks') or {}).get(task)
        except (OSError, ValueError):
            continue
        if state and state.get('episodes') and state.get('status') not in ('ended', 'released', 'evaluated'):
            names.append(path.parent.name)
    return names


def official_evaluation_completed(runs_root, task, snapshot):
    """Whether runs_root holds an official evaluation of this task with this knowledge snapshot that has
    completed every registered holdout episode (base and extension, v2/holdout.py)."""
    if not runs_root or not snapshot:
        return False
    from v2.holdout import registered_holdout
    required = {(entry['seed'], entry['layout_id']) for entry in registered_holdout(task)}
    for path in Path(runs_root).glob('holdout_*/evaluation.json'):
        try:
            report = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if (report.get('official') is True and report.get('task') == task and report.get('snapshot') == snapshot
                and required <= {(r.get('seed'), r.get('layout_id')) for r in report.get('results') or []
                                 if r.get('status') == 'completed'}):
            return True
    return False


# ---------------------------------------------------------------- simulator

def simulator_command(spec, run_dir):
    if spec.get('simulator'):
        return list(spec['simulator'])
    return [sys.executable, '-u', str(CODE_V1 / 'simulator.py'), '--task_name', spec['task'],
            '--env_cfg_type', spec['env_cfg'], '--device_id', str(spec['device_id']),
            '--policy_name', 'embodied_harness', '--port', str(spec['service_port']),
            '--additional_info', run_dir.name, '--seed', str(spec['seed']), '--headless', '--enable_cameras']


def simulator_environment(spec, run_dir, layout_sha256, base=None):
    env = dict(os.environ if base is None else base)
    for key in list(env):
        # No credentials, and none of the v1 privileged or synthetic hooks.
        if 'API_KEY' in key or key in ('ROBOT_TEACHER_ACCESS', 'ROBOT_OUTER_STAGE_CONFIG',
                                       'ROBOT_CUSTOM_LAYOUT_PATH', 'ROBOT_CURRICULUM_MANIFEST',
                                       'ROBOT_EXPLORATION_STEP_LIMIT', 'ROBOT_DIAG_OBJECTS',
                                       'ROBOT_ATTEMPT_RESULT'):
            env.pop(key)
    repo = str(spec['repo'])
    env.update(ROBODOJO_ROOT=repo, ROBOT_RUN_DIR=str(run_dir), EVAL_NUM='1',
               PYTHONPATH=os.pathsep.join([str(CODE_V1), repo, str(Path(repo) / 'XPolicyLab')]),
               ROBOT_IDLE_TIMEOUT=str(SIMULATOR_IDLE_TIMEOUT_S), ROBOT_LAYOUT_ID=str(spec['layout_id']))
    env.pop('ROBOT_MOTION_PROFILE', None)
    env.update(robot_settings(spec['split']))
    if layout_sha256:
        env['ROBOT_LAYOUT_SHA256'] = layout_sha256
    if spec['split'] == 'development':
        env['ROBOT_ATTEMPT_RESULT'] = '1'  # score the attempt when it ends, for the consolidation note
    if spec.get('simulator_home'):
        home = Path(spec['simulator_home'])
        for sub in ('.cache', '.config', '.local/share'):
            (home / sub).mkdir(parents=True, exist_ok=True)
        env.update(HOME=str(home), XDG_CACHE_HOME=str(home / '.cache'), XDG_CONFIG_HOME=str(home / '.config'),
                   XDG_DATA_HOME=str(home / '.local' / 'share'))
    return env


def robot_settings(split):
    """The robot service's settings for an episode of this split (the adapter reads them). The simulator this
    runner starts and the submission entry, which serves the robot inside RoboDojo's own evaluator
    (v2/submission.py), both take them from here, so an evaluation episode moves the robot the same way."""
    settings = {'ROBOT_CONTROLLER_OWNER': 'coding_agent',
                'ROBOT_BUDGET_MODE': 'exploration' if split == 'development' else 'benchmark',
                'ROBOT_SPEED_LIMITS': json.dumps(SPEED_LIMITS),
                'ROBOT_GRIPPER_SQUEEZE': '1'}  # width targets below 0 grip harder
    if MOTION_PROFILE:
        settings['ROBOT_MOTION_PROFILE'] = (MOTION_PROFILE if isinstance(MOTION_PROFILE, str)
                                            else json.dumps(MOTION_PROFILE))
    return settings


def development_layouts(spec):
    """How many seed-0 layouts the task has; development may use any of them."""
    if spec.get('simulator'):
        return spec.get('development_layouts', 26)
    sys.path.insert(0, str(CODE_V1))
    from task_catalog import layouts
    return len(layouts(spec['repo'], spec['task'], 0))


def resolve_layout(spec):
    if spec.get('simulator'):
        test_layouts = spec.get('test_layouts') or {}
        digest = (test_layouts.get(f"{spec['task']}:{spec['seed']}:{spec['layout_id']}") or
                  test_layouts.get(f"{spec['seed']}:{spec['layout_id']}"))
        return {'layout_sha256': digest, 'note': 'test simulator'}
    sys.path.insert(0, str(CODE_V1))
    from task_catalog import resolve
    return resolve(spec['repo'], spec['task'], spec['seed'], spec['layout_id'])


# Known causes of a simulator that dies while starting, found in its log.
STARTUP_HINTS = (
    ('Failed to create change watch', 'errno=28',
     'the inotify watch limit is exhausted (often an editor watching the runs directory); raise '
     'fs.inotify.max_user_watches or close that editor'),
)


def startup_hint(log_path):
    """A short explanation for a known start-up failure, from the simulator log (or '')."""
    try:
        with open(log_path, 'rb') as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 2_000_000))
            text = stream.read().decode('utf-8', 'replace')
    except OSError:
        return ''
    for first, second, hint in STARTUP_HINTS:
        if first in text and second in text:
            return ': ' + hint
    return ''


def kill_simulator(sim):
    """Stop the simulator's process group at once (the runner is being interrupted)."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(sim.pid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            sim.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            continue


def stop_simulator(sim, run_dir, wait_s=45):
    (run_dir / 'stop_requested').touch()
    try:
        sim.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        for sig, grace in ((signal.SIGTERM, 10), (signal.SIGKILL, 10)):
            try:
                os.killpg(sim.pid, sig)
            except ProcessLookupError:
                break
            try:
                sim.wait(timeout=grace)
                break
            except subprocess.TimeoutExpired:
                continue
    return sim.returncode


# ------------------------------------------------------------------ actor

def opening_factory(replay_report=None, source_dir=None):
    """First message: the current images and state line, and after a replay the source's images."""
    def opening(worker):
        reply = worker.call('observe', {'cameras': None}, timeout=180)
        texts, images = ['The episode has started. Current observation:', reply['text']], list(reply['images'])
        if replay_report is not None:
            from v2.replay import note
            texts.insert(0, note(replay_report))
            frame = replay_report.get('source_frame')
            for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
                path = Path(source_dir) / 'recording' / 'observations' / str(frame) / f'{camera}.png'
                if frame and path.is_file():
                    images.append({'label': f"source {replay_report['source']}:{frame} {camera}",
                                   'path': str(path)})
        return [user_message(texts, images)]
    return opening


opening = opening_factory()


def read_summary(run_dir):
    """The session's own final summary (written when the session completes), if any."""
    try:
        return json.loads((Path(run_dir) / 'session' / 'session_summary.json').read_text())
    except (OSError, ValueError):
        return None


def replayed_command_ids(run_dir, depth=100):
    """Earlier command ID -> the ID of its replayed copy in this episode.

    A replayed command gets a new ID, while the model may still name the one it saw in the source
    episode (or further back, when that episode was itself replayed); the replay records chain them.
    """
    mapping, current, run_dir = {}, None, Path(run_dir)
    for _ in range(depth):
        try:
            replay = json.loads((run_dir / 'run.json').read_text()).get('replay') or {}
        except (OSError, ValueError):
            break
        step = {c['source']: c['replayed'] for c in replay.get('commands') or []
                if c.get('source') and c.get('replayed')}
        # step maps the source episode's IDs to run_dir's; carry them on to this episode's IDs.
        step = step if current is None else {old: current[new] for old, new in step.items() if new in current}
        mapping.update({old: new for old, new in step.items() if old not in mapping})
        if not step or not replay.get('source'):
            break
        current, run_dir = step, run_dir.parent / replay['source']
    return mapping


def replayed_from_with(run_dir, command_id, depth=100):
    """The earlier episode of this episode's replay chain whose own log holds command_id, or None."""
    run_dir = Path(run_dir)
    for _ in range(depth):
        try:
            source = (json.loads((run_dir / 'run.json').read_text()).get('replay') or {}).get('source')
        except (OSError, ValueError):
            return None
        if not source:
            return None
        run_dir = run_dir.parent / source
        commands = run_dir / 'recording' / 'commands.jsonl'
        if commands.is_file() and any(command_id in (row['request'].get('id'), row['result'].get('command_id'))
                                      for row in map(json.loads, jsonl_lines(commands.read_text()))):
            return source
    return None


def official_result_hook(run_dir, budget=None, replayed=0):
    """The consolidation note's official result line (development only), or None:
    with the attempt's steps against the evaluation budget (`budget`), counting `replayed` steps.

    The simulator writes attempt_result.json right after the command that ended the attempt; the
    reply to that command can reach the session first, hence the wait.
    """
    from v2 import history

    def hook(wait_s=ATTEMPT_RESULT_WAIT_S):
        path = run_dir / 'attempt_result.json'
        deadline = time.monotonic() + wait_s
        while not path.is_file() and time.monotonic() < deadline:
            time.sleep(0.1)
        try:
            return history.attempt_line(json.loads(path.read_text()), budget, replayed)
        except (OSError, ValueError):
            return None
    return hook


def official_score(run_dir, repo):
    """RoboDojo's own score of the episode in [0, 1] from its _result.json, or None (developer record)."""
    try:
        result = json.loads((Path(run_dir) / 'result.json').read_text())
        official = Path(result['official_result_dir'])
        data = json.loads(((official if official.is_absolute() else Path(repo) / official) /
                           '_result.json').read_text())
    except (OSError, ValueError, KeyError, TypeError):
        return None
    first = next(iter((data.get('details') or {}).values()), None)
    return float(first['score']) if isinstance(first, dict) and first.get('score') is not None else None


# A simulator whose physics has broken: RoboDojo's evaluator abandons such an episode (src/eval_client/main.py), and
# every later step fails the same way, so the physical episode ends.
# The exception line of a traceback (module prefix optional, message possibly empty), not text anywhere in a line.
PHYSX_FAILURE = re.compile(r'^(?:[\w.]+\.)?(PhysX(?:Broken|Fatal)Error(?::.*)?)$', re.M)
SERVICE_LOG_WAIT_S = 1.0  # the simulator answers a failed command, then appends its traceback to service_errors.log


def simulator_failure_hook(episode):
    """Whether the simulator has broken or vanished, from its own state only (never from text the model can write):
    its process has exited, or its developer-only service_errors.log (unreadable to the worker) records a PhysX
    failure. Returns the reason shown to the model, or None. After a failed call it waits briefly for the log."""
    seen = {'offset': 0}

    def check():
        sim = getattr(episode, '_sim', None)
        if sim is not None and sim.poll() is not None:
            return f'the simulator process exited with code {sim.returncode}'
        try:
            with open(episode.run_dir / 'service_errors.log', 'rb') as stream:
                stream.seek(seen['offset'])
                data = stream.read()
        except OSError:
            return None
        cut = data.rfind(b'\n') + 1  # whole lines only; the rest is read next time
        seen['offset'] += cut
        match = PHYSX_FAILURE.search(data[:cut].decode('utf-8', 'replace'))
        return match.group(1).strip() if match else None

    def hook(failed=False):
        deadline = time.monotonic() + (SERVICE_LOG_WAIT_S if failed else 0)
        while True:
            reason = check()
            if reason or time.monotonic() >= deadline:
                return reason
            time.sleep(0.1)
    return hook


def end_review_hook(episode):
    """The readiness review shown when a development actor first chooses end:
    the episode list, this episode last, and the review questions (v2/prompts/end_review.md)."""
    from v2 import history

    def hook(state):
        spec, run_dir = episode.spec, episode.run_dir
        try:
            attempt = json.loads((run_dir / 'attempt_result.json').read_text())
        except (OSError, ValueError):
            attempt = None
        record = episode.record or {}
        current = {'run_id': spec['run_id'], 'task': spec['task'], 'layout_id': spec['layout_id'],
                   'layout': record.get('layout'), 'replay': record.get('replay'), 'attempt_result': attempt,
                   'session': {'outcome': state.get('outcome'), 'stats': state.get('stats')}}
        readiness = ''
        try:
            current['kind'] = history.episode_kind(
                current, history.layouts_played(episode.campaign_dir, spec['task'], exclude=spec['run_id']))
            episodes = '\n'.join(history.episode_list(episode.campaign_dir, episode.history_records, current=current,
                                                      budget=episode.benchmark_step_limit))
            readiness = history.readiness_text(history.episode_rows(episode.campaign_dir, episode.history_records,
                                                                    current), budget=episode.benchmark_step_limit)
        except Exception as exc:  # a damaged record must not block the review
            episodes = f'(the episode list is unavailable: {type(exc).__name__})'
        return context.read_prompt('end_review.md').format(episodes=episodes, readiness=readiness)
    return hook


def decision_hook(run_dir, spec, layouts):
    """Check plan_next_episode against this episode and store the handoff."""
    def hook(args):
        named = args['replay_to']
        if args['replay_to'] is not None:
            commands = run_dir / 'recording' / 'commands.jsonl'
            rows = [json.loads(line) for line in jsonl_lines(commands.read_text())] if commands.is_file() else []
            own = {row['request'].get('id') for row in rows} | {row['result'].get('command_id') for row in rows}
            if named not in own:
                # A command replayed into this episode may be named by its ID in the earlier episode; the stored
                # decision (args) then carries this episode's ID, which the next replay reads.
                args['replay_to'] = replayed_command_ids(run_dir).get(named, named)
            match = [row for row in rows if row['request'].get('id') == args['replay_to'] or
                     row['result'].get('command_id') == args['replay_to']]
            if not match:
                earlier = replayed_from_with(run_dir, named)
                if earlier:
                    raise ValueError(f'replay_to {named} is a command of {earlier} that was not replayed into this '
                                     'episode; name a command of this episode (see robot.history())')
                raise ValueError(f"replay_to {args['replay_to']} is not a command of this episode "
                                 '(see robot.history())')
            if match[0]['request'].get('method') not in PHYSICAL_METHODS:
                raise ValueError('replay_to must name a motion command (move, gripper, move_to_mark, trajectory, '
                                 'mark_pose or hold)')
        if args['decision'] == 'switch_layout' and args['layout_id'] == spec['layout_id']:
            raise ValueError('switch_layout needs a different layout; use reset_same for this layout')
        if args['decision'] == 'switch_layout' and not args['layout_id'] < layouts:
            raise ValueError(f'switch_layout needs a development layout of this task, from 0 to {layouts - 1}')
        record = {'decision': args['decision'], 'next_layout_id': args['layout_id'], 'replay_to': args['replay_to'],
                  'handoff': args['handoff'], 'run_id': spec['run_id'], 'task': spec['task'], 'seed': spec['seed'],
                  'layout_id': spec['layout_id'], 'time': now()}
        if named != args['replay_to']:
            record['replay_to_named'] = named
        write_json(run_dir / 'handoff.json', record)
        if spec.get('knowledge_dir'):
            handoffs = Path(spec['knowledge_dir']) / 'handoffs'
            handoffs.mkdir(parents=True, exist_ok=True)
            write_json(handoffs / f"{spec['run_id']}.json", record)
        copy = f" (this episode's replayed copy of {named})" if named != args['replay_to'] else ''
        target = {'reset_same': f"reset layout {spec['layout_id']}" +
                                (f" and replay up to {args['replay_to']}{copy}" if args['replay_to'] else ''),
                  'switch_layout': f"switch to layout {args['layout_id']}", 'end': 'end development'}
        return f"Decision recorded: {target[args['decision']]}. The session ends here."
    return hook


class Episode:
    """Run one episode. `spec` holds the launch settings; `config` the model settings (with the key)."""

    def __init__(self, spec, config, client=None, knowledge=None):
        self.spec = {**DEFAULTS, **spec}
        validate(self.spec)
        if self.spec.get('official') and not (knowledge or {}).get('snapshot_id'):
            raise ValueError('an official episode needs a frozen release snapshot')
        if self.spec['split'] == 'evaluation' and (knowledge or {}).get('snapshot_id') != self.spec.get('snapshot'):
            raise ValueError("an evaluation episode reads the knowledge snapshot named in its spec")
        self.config, self.client, self.knowledge = config, client, knowledge
        self.run_dir = Path(self.spec['runs_root']) / self.spec['campaign'] / self.spec['run_id']
        self.workspace = self.run_dir / 'workspace'
        self.record = None
        self.history_records, self.history_roots, self.history_recordings = [], [], {}
        self.history_index = None
        self.replay_report = None
        self.benchmark_step_limit = None

    def _worker_factory(self, connection, step_limit):
        spec, run_dir, knowledge = self.spec, self.run_dir, self.knowledge or {}
        # Development experiences join the campaign's experience archive, which the
        # actor may search; evaluation experiences stay in this episode.
        experience_log = run_dir / 'experience.jsonl'
        # Of the harness source only the code the worker runs (tracebacks show its lines): v2/ also holds files
        # the model must not read (holdout notes, tests, every task's prior). The SDK source stays readable.
        read = [self.workspace, run_dir / 'recording', run_dir / 'tool_images', *WORKER_SOURCE,
                SDK_SOURCE, experience_log] + ([connection] if connection else []) + list(spec.get('extra_read') or ())
        if spec['split'] == 'development' and spec.get('knowledge_dir'):
            experience_log = Path(spec['knowledge_dir']) / 'experience' / f"{spec['run_id']}.jsonl"
        # An evaluation episode's seed and layout stay with the developer.
        hidden = spec['split'] == 'evaluation'
        episode = {'run_id': spec['run_id'], 'task': spec['task'], 'seed': None if hidden else spec['seed'],
                   'layout_id': None if hidden else spec['layout_id'], 'split': spec['split'], 'step_limit': step_limit}
        if spec['split'] == 'development' and self.benchmark_step_limit:
            episode['evaluation_budget'] = self.benchmark_step_limit  # shown in the state line for reference
        settings = {'V2_WORKSPACE': str(self.workspace), 'V2_OUTPUT': str(run_dir),
                    'V2_RECORDING': str(run_dir / 'recording'), 'V2_EPISODE': episode,
                    'V2_EXPERIENCE_LOG': str(experience_log)}
        if connection:
            settings['V2_CONNECTION'] = str(connection)
        if knowledge.get('root'):
            read.append(knowledge['root'])
            # A `_random` task is evaluated with its standard variant's snapshot: skills and experiences are
            # looked up under the task the knowledge was developed on.
            settings['V2_KNOWLEDGE'] = {'root': str(knowledge['root']), 'writable': bool(knowledge.get('writable')),
                                        'task': knowledge.get('task') or spec['task'],
                                        'recordings': self.history_recordings, 'session': 'actor',
                                        'layouts': {r['run_id']: r['layout_id'] for r in self.history_records}}
        read += self.history_roots
        if self.history_index is not None:
            settings['V2_HISTORY'] = str(self.workspace / 'history' / 'INDEX.md')
            settings['V2_HISTORY_WATCH'] = [str(path) for path in self._history_watch_roots()]
        settings['V2_GUARD'] = {'read': [str(path) for path in read],
                                'write': [str(self.workspace), str(run_dir / 'tool_images')],
                                'log': str(run_dir / 'guard.jsonl')}
        settings.update(spec.get('worker_settings') or {})
        env = child_environment(self.workspace, settings)

        def factory():
            return WorkerProcess(env, cwd=self.workspace, stderr_path=run_dir / 'worker.stderr')
        return factory

    def _history_watch_roots(self):
        """Where a read counts as opening history/: history/ itself, the run files
        it links, and the knowledge's handoffs. The linked experience logs are left out: they are the experience
        archive that search_experience reads, as an evaluation episode may."""
        knowledge = Path(self.spec['knowledge_dir']).resolve() if self.spec.get('knowledge_dir') else None
        roots = [self.workspace / 'history']
        roots += [Path(root) for root in self.history_roots
                  if knowledge is None or not Path(root).resolve().is_relative_to(knowledge)]
        if knowledge is not None:
            roots.append(knowledge / 'handoffs')
        return roots

    def _session(self, connection, instruction, step_limit):
        spec = self.spec
        episode = {'task': spec['task'], 'seed': spec['seed'], 'layout_id': spec['layout_id'],
                   'split': spec['split'], 'step_limit': step_limit,
                   'benchmark_step_limit': self.benchmark_step_limit}
        development = spec['split'] == 'development'
        if development:
            episode['development_layouts'] = development_layouts(spec)
        handoff = None
        if development:
            from v2 import history
            handoff = history.opening_section(self.campaign_dir, self.history_records)
        instructions = context.actor_instructions(
            episode, instruction, prior=context.load_priors().get(spec['task']),
            knowledge=(self.knowledge or {}).get('text'), handoff=handoff)
        source_dir = self.campaign_dir / self.replay_report['source'] if self.replay_report else None
        hooks = {}
        if development:
            hooks = {'plan_next_episode': decision_hook(self.run_dir, spec, episode['development_layouts']),
                     'official_result': official_result_hook(self.run_dir, self.benchmark_step_limit,
                                                             history.replayed_steps((self.record or {}).get('replay'))),
                     'end_review': end_review_hook(self)}
        if connection is not None:
            hooks['simulator_failure'] = simulator_failure_hook(self)
        config = self.config
        if (development and config.get('compact_threshold_development')
                and config.get('compact_threshold', 0) >= DEFAULT_COMPACT_THRESHOLD):
            # Development compacts later; a compact_threshold set to 0 (server compaction off) or lowered below
            # the default by the operator applies to development too.
            config = {**config, 'compact_threshold': config['compact_threshold_development']}
        return Session(config, self.run_dir / 'session', profile='actor_dev' if development else 'actor_eval',
                       instructions=instructions, worker_factory=self._worker_factory(connection, step_limit),
                       mode=spec['split'], workspace=self.workspace, robot=connection is not None,
                       opening=opening_factory(self.replay_report, source_dir), step_limit=step_limit,
                       hooks=hooks,
                       consolidation=context.read_prompt('consolidation.md') if development else None,
                       client=self.client, cache_key=prompt_cache_key(spec['campaign'], spec['run_id']),
                       api_waits=spec.get('model_api_waits_s', API_WAITS_S),
                       max_model_calls=spec.get('max_model_calls') or 0,
                       redact_endpoint=bool(spec.get('redact_endpoint')),
                       wait_refusals=bool(spec.get('model_api_wait_refusals')))

    def _run_session(self, connection, instruction, step_limit):
        """Run the session, resuming once after incomplete replies or a pause; return (result, error).

        Transient model API failures are waited out inside the session, which keeps the worker.
        """
        resume, retried_incomplete, retried_pause = False, False, False
        while True:
            session = self._session(connection, instruction, step_limit)
            try:
                result = session.run(resume=resume)
            except NeedsAttention as exc:
                if exc.reason == 'incomplete' and not retried_incomplete:
                    retried_incomplete, resume = True, True
                    self._note({'event': 'auto_resume', 'reason': exc.reason})
                    continue
                return None, exc
            except Exception as exc:  # an unexpected harness error; the checkpoint is saved
                self._note({'event': 'session_error', 'error': f'{type(exc).__name__}: {exc}'[:500]})
                return None, exc
            if result['status'] == 'paused' and not retried_pause:
                retried_pause, resume = True, True
                self._note({'event': 'auto_resume', 'reason': 'paused'})
                continue
            return result, None

    def _note(self, record):
        with (self.run_dir / 'episode_events.jsonl').open('a') as stream:
            stream.write(json.dumps({'time': now(), **record}) + '\n')

    @property
    def campaign_dir(self):
        return Path(self.spec['runs_root']) / self.spec['campaign']

    def _prepare_history(self, layout):
        spec = self.spec
        if spec['split'] != 'development':
            if spec.get('parent_run') or spec.get('replay_to'):
                raise ValueError('evaluation episodes have no parent chain or replay')
            return
        from v2 import history
        self.history_records = history.chain(self.campaign_dir, spec.get('parent_run'), spec['task'])
        if spec.get('replay_to'):
            parent = self.history_records[-1] if self.history_records else None
            if parent is None or parent['layout_id'] != spec['layout_id']:
                raise ValueError('replay needs a parent episode on the same layout')
            if (parent.get('layout') or {}).get('layout_sha256') != layout.get('layout_sha256'):
                raise ValueError('replay needs the same layout file as the parent episode')
        self.history_roots, self.history_recordings, self.history_index = history.build(
            self.workspace, self.campaign_dir, spec.get('knowledge_dir'), self.history_records)

    def _replay(self, connection):
        from v2.replay import replay
        from v2.runtime.sdk import Robot
        source = self.spec['parent_run']
        report = replay(Robot(str(connection)), self.campaign_dir / source, self.run_dir, self.spec['replay_to'],
                        source)
        self._note({'event': 'replay', **{k: report[k] for k in ('planned', 'replayed', 'stopped', 'reason')}})
        return report

    def run(self):
        spec, run_dir = self.spec, self.run_dir
        if run_dir.exists():
            raise ValueError(f'{run_dir} already exists; use a new run ID')
        layout = resolve_layout(spec)
        expected = spec.get('expected_layout_sha256')
        if expected and layout.get('layout_sha256') != expected:
            raise ValueError(f"layout {spec['layout_id']} of seed {spec['seed']} does not match the preregistered file")
        lock = Path(spec['lock_file']).open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('another simulator run holds ' + spec['lock_file']) from None
        source = source_identity()
        origin = provenance(spec['repo'])
        if spec.get('official') and not (origin.get('frozen_source_matches') and origin.get('robodojo_same_as_frozen')):
            lock.close()
            raise ValueError('official episodes run from an unmodified frozen copy with the RoboDojo checkout '
                             'it was frozen with')
        self.record = {
            'schema': 'v2_run/1', 'run_id': spec['run_id'], 'campaign': spec['campaign'], 'task': spec['task'],
            'seed': spec['seed'], 'layout_id': spec['layout_id'], 'split': spec['split'],
            'parent_run': spec.get('parent_run'), 'replay_to': spec.get('replay_to'), 'replay': None,
            'knowledge_snapshot': (self.knowledge or {}).get('snapshot_id'),
            'model': self.config['model'], 'reasoning_effort': self.config.get('reasoning_effort'),
            'key_alias': spec.get('key_alias'), 'config_path': spec.get('config_path'),
            'config_hash': config_hash(self.config), 'model_config': public_config(self.config),
            'env_cfg': spec['env_cfg'], 'device_id': spec['device_id'], 'layout': layout,
            'source': source, 'git': origin, 'command': launch_command(),
            'official': bool(spec.get('official')), 'expected_layout_sha256': spec.get('expected_layout_sha256'),
            'status': 'starting', 'wall_started': now(), 'isolation': landlock.isolation_summary()}
        try:
            self._run(self.record, layout, source)
        finally:
            lock.close()
        return self.record

    def _run(self, record, layout, source):
        """Every path, including a second interrupt while the simulator stops, ends in _finalize."""
        run_dir = self.run_dir
        self._sim = self._sim_log = None
        failure, instruction, step_limit = None, None, None
        try:
            try:
                self._start(record, layout)
                instruction, step_limit = self._connect(record)
                result, error = self._run_session(run_dir / 'connection.json', instruction, step_limit)
                record['session'] = result
                if error is not None:
                    record.update(status='needs_attention', error={'type': type(error).__name__,
                                                                   'reason': getattr(error, 'reason', None),
                                                                   'message': str(error)[:2000]})
                else:
                    record['status'] = 'completed' if result['status'] == 'completed' else result['status']
            except BaseException as exc:
                record.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed',
                              error={'type': type(exc).__name__, 'message': str(exc)[:2000]})
                failure = exc
            finally:
                try:
                    if self._sim is not None:
                        record['simulator_exit_code'] = stop_simulator(self._sim, run_dir)
                except BaseException as exc:  # an interrupt while waiting: do not leave the simulator running
                    kill_simulator(self._sim)
                    record['simulator_exit_code'] = self._sim.returncode
                    failure = failure or exc
                finally:
                    if self._sim_log is not None:
                        self._sim_log.close()
            if failure is None and self._needs_recovery(record):
                try:
                    self._recover(record, instruction, step_limit)
                except BaseException as exc:  # interrupted during the consolidation without the robot
                    failure = exc
        finally:
            self._finalize(record, source, clean=failure is None)
        if failure is not None:
            raise failure

    def _start(self, record, layout):
        """Create the run directory and history links, then start the simulator."""
        spec, run_dir = self.spec, self.run_dir
        run_dir.mkdir(parents=True)
        self.workspace.mkdir()
        (run_dir / 'tool_images').mkdir()
        write_json(run_dir / 'run.json', record)
        self._prepare_history(layout)
        env = simulator_environment(spec, run_dir, layout.get('layout_sha256'))
        self._sim_log = (run_dir / 'simulator.log').open('w')
        self._sim = subprocess.Popen(simulator_command(spec, run_dir), env=env, stdout=self._sim_log,
                                     stderr=subprocess.STDOUT, start_new_session=True, cwd=str(HARNESS_ROOT))
        record.update(simulator_pid=self._sim.pid, status='simulator_starting')
        write_json(run_dir / 'run.json', record)

    def _connect(self, record):
        """Wait for the robot service, read the task and budget, and replay if requested."""
        spec, run_dir = self.spec, self.run_dir
        connection = run_dir / 'connection.json'
        started = time.monotonic()
        while not connection.exists():
            if self._sim.poll() is not None:
                raise RuntimeError('simulator exited during startup; see simulator.log' +
                                   startup_hint(run_dir / 'simulator.log'))
            if time.monotonic() - started > spec['startup_timeout_s']:
                raise TimeoutError('simulator startup timeout')
            time.sleep(0.5)
        description = json.loads((run_dir / 'robot_description.json').read_text())
        instruction = description.get('task')
        self.benchmark_step_limit = description.get('benchmark_step_limit')
        step_limit = description.get('step_limit') if spec['split'] == 'evaluation' else None
        if spec['split'] == 'evaluation' and not step_limit:
            raise RuntimeError('evaluation needs the official step limit from the robot service')
        if spec.get('replay_to'):
            self.replay_report = record['replay'] = self._replay(connection)
        record.update(status='running', instruction=instruction, step_limit=step_limit)
        write_json(run_dir / 'run.json', record)
        return instruction, step_limit

    def consolidate_after_stop(self):
        """After a restart: let a stopped development episode consolidate without the robot and decide.

        Used by the manager for an episode that ended without a decision (interrupted, paused or
        stopped). Returns the updated run record.
        """
        from v2 import history
        record = json.loads((self.run_dir / 'run.json').read_text())
        if (record.get('split') != 'development' or (record.get('session') or {}).get('decision') or
                not (self.run_dir / 'session' / 'session.json').is_file()):
            return record
        summary = read_summary(self.run_dir)
        if summary and summary.get('status') == 'completed':
            # Already decided; only the run record missed it.
            record.update(session=summary, status='completed')
            write_json(self.run_dir / 'run.json', record)
            return record
        self.record = record
        self.history_records = history.chain(self.campaign_dir, record.get('parent_run'), record['task'])
        self.history_roots, self.history_recordings, self.history_index = history.build(
            self.workspace, self.campaign_dir, self.spec.get('knowledge_dir'), self.history_records)
        description = json.loads((self.run_dir / 'robot_description.json').read_text())
        self.benchmark_step_limit = description.get('benchmark_step_limit')
        if record.get('status') == 'interrupted':
            record.setdefault('error', {})['message'] = 'the run was interrupted and the simulator was stopped'
        self._recover(record, record.get('instruction'), record.get('step_limit'))
        # The consolidation may run from a newer frozen copy than the episode did.
        record['recovery'].update(source=source_identity(), harness=str(HARNESS_ROOT), time=now())
        write_json(self.run_dir / 'run.json', record)
        return record

    def _needs_recovery(self, record):
        """A development session that stopped before its decision consolidates without the robot."""
        if self.spec['split'] != 'development' or record['status'] == 'completed':
            return False
        reason = (record.get('error') or {}).get('reason')
        return reason not in ('quota', 'protocol') and (self.run_dir / 'session' / 'session.json').is_file()

    def _recover(self, record, instruction, step_limit):
        why = ((record.get('error') or {}).get('message') or record['status'])[:300]
        self._note({'event': 'recovery_consolidation', 'reason': why})
        try:
            result = self._session(None, instruction, step_limit).run(resume=True, physical_ended=why)
        except Exception as exc:
            record['recovery'] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'[:2000]}
            return
        record['recovery'] = {'status': result['status'], 'from': record['status'], 'error': record.get('error')}
        record['session'] = result
        if result['status'] == 'completed':
            record['status'] = 'completed'

    def _finalize(self, record, source, clean=True):
        """Write the final record first; the slow review video comes last and only for a clean finish.

        A session that completed keeps its outcome and decision even when an interrupt came later
        (during cleanup or right after a consolidation without the robot). An evaluation also needs
        the benchmark's result.json: without it the attempt does not count as completed.
        """
        run_dir = self.run_dir
        record['wall_finished'] = now()
        summary = read_summary(run_dir)
        session = record.get('session') or {}
        if summary and summary.get('status') == 'completed' and session.get('status') != 'completed':
            if record.get('status') in ('needs_attention', 'paused', 'failed', 'interrupted'):
                record['recovery'] = {**(record.get('recovery') or {}), 'status': 'completed',
                                      'restored_from_session_summary': True}
            record['session'] = session = summary
        if not clean:
            record['interrupted_during_cleanup'] = session.get('status') == 'completed'
        try:
            record['source_integrity'] = 'verified' if source_identity() == source else 'modified'
        except OSError:
            record['source_integrity'] = 'unknown'
        result_path = run_dir / 'result.json'
        if result_path.is_file():
            try:
                external = json.loads(result_path.read_text())
                record['external_result'] = {key: external.get(key) for key in
                                             ('benchmark_success', 'physical_steps', 'step_limit',
                                              'benchmark_step_limit', 'final_observation_id',
                                              'scoring_stages')}
                record['external_result']['official_score'] = official_score(run_dir, self.spec['repo'])
            except ValueError:
                record['external_result'] = {'error': 'unreadable result.json'}
        attempt_path = run_dir / 'attempt_result.json'
        if attempt_path.is_file():
            try:
                record['attempt_result'] = attempt = json.loads(attempt_path.read_text())
                official = (record.get('external_result') or {}).get('official_score')
                if official is not None and attempt.get('score') is not None:
                    # The note told the model the attempt's score; RoboDojo's own result must agree.
                    record['attempt_result_matches'] = abs(official - attempt['score']) < 1e-6
            except ValueError:
                record['attempt_result'] = {'error': 'unreadable attempt_result.json'}
        failure = (session.get('outcome') or {}).get('reason') if (session.get('outcome') or {}).get(
            'ended_by') == 'simulator_failure' else None
        if failure:
            record['simulator_failure'] = failure
        if session.get('status') == 'completed':
            has_result = 'benchmark_success' in (record.get('external_result') or {})
            if record.get('split') == 'evaluation' and (not has_result or failure):
                # No benchmark result, or the simulator broke (RoboDojo abandons such an episode): run it again.
                record['status'] = 'interrupted'
            else:
                record['status'] = 'completed'
        guard_log = run_dir / 'guard.jsonl'
        record['guard_blocks'] = len(jsonl_lines(guard_log.read_text())) if guard_log.is_file() else 0
        write_json(run_dir / 'run.json', record)
        if clean and not self.spec.get('simulator'):
            try:
                sys.path.insert(0, str(CODE_V1))
                from render_video import render
                video = render(run_dir, self.spec['repo'], self.spec.get('ffmpeg', 'ffmpeg'))
                record['review_video'] = str(video.relative_to(run_dir)) if video else None
            except Exception as error:
                record['review_video_error'] = type(error).__name__
            write_json(run_dir / 'run.json', record)

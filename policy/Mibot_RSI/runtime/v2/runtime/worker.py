"""Persistent Python worker for model sessions.

The worker runs in a child process with its own namespace (robot, display,
display_image, log_experience, ...). The parent sends one JSON request per
line on stdin and reads one JSON reply per line on stdout:

  {"id": 1, "method": "python_exec", "params": {"code": "...", "timeout_s": 60}}
  {"id": 1, "result": {"text": "...", "images": [...], "is_error": false, ...}}

Replies contain compact text (one line per robot action, then stdout and
displayed items) and image references (file paths), never image bytes.
Robot commands are never replayed: after an error the worker reports the
unresolved command ID instead of retrying it.
"""
from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from v2.runtime import steps, view  # noqa: E402

CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')
# When the time limit arrives during a command, the worker waits for that command to finish. Every
# DEADLINE_GRACE_S it checks that the simulator is still stepping (the recording's transitions.jsonl
# grows by one line per control step) and keeps waiting while it is, up to DEADLINE_GRACE_MAX_S in
# all; the command is cancelled only after STALL_CHECKS checks in a row without a step, or at the cap.
DEADLINE_GRACE_S = int(os.environ.get('V2_DEADLINE_GRACE_S', 45))
DEADLINE_GRACE_MAX_S = int(os.environ.get('V2_DEADLINE_GRACE_MAX_S', 1800))
STALL_CHECKS = 2
REPEAT_ALARM_S = 5           # the timeout is raised again if model code swallows it
# Opened skills are shown after the rest of a call's output and outside its length cap, whole, up to this many
# characters per call (about 10k tokens, four or five skills); a skill that does not fit is named, not cut.
SKILL_OUTPUT_CHARS = 40000
STDOUT_CHARS = 20000         # printed text kept in one reply (head and tail)
RPC_MARGIN_S = 60            # parent-side slack beyond timeout + grace (and beyond each extension)


class ExecTimeout(BaseException):
    """python_exec time limit; derived from BaseException so `except Exception` does not hide it."""


class HistoryWatch:
    """Counts Python-level file opens in the development history.

    An audit hook, installed before any model code runs: an `open` of a file under history/, of a file it links (by its
    own path) or under knowledge/handoffs/ counts, whether model code or a helper (history(), add_evidence with a
    history reference) made it. Listing directories does not count (a walk of the workspace lists history/ without
    reading it). The parent adds up the counts of the physical episode: an attempt on a new layout without replay and
    without such reads is "fresh". Native readers (cv2.imread, torch.from_file) make no audit event and are not seen.
    """

    def __init__(self, roots):
        self.plain = [os.path.abspath(str(root)) for root in roots]
        self.real = [os.path.realpath(str(root)) for root in roots]
        self.count = 0

    @staticmethod
    def _under(path, roots):
        return any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots)

    def hook(self, event, args):
        if event != 'open':
            return
        try:
            path = args[0] if args else None
            if path is None or isinstance(path, int):
                return
            path = os.fsdecode(path)
            if self._under(os.path.abspath(path), self.plain) or self._under(os.path.realpath(path), self.real):
                self.count += 1
        except Exception:  # an observer: never change what the code does
            pass


# ---------------------------------------------------------------- child side

class Session:
    """State of one worker process: robot connection, namespace and outputs."""

    def __init__(self, env):
        from v2.runtime.sdk import DeadlineReached, Robot
        from v2.knowledge.store import ExperienceLog
        self.DeadlineReached = DeadlineReached
        self.workspace = Path(env['V2_WORKSPACE'])
        self.output = Path(env['V2_OUTPUT'])
        self.recording = Path(env.get('V2_RECORDING') or self.output / 'recording')
        self.has_recording = bool(env.get('V2_RECORDING'))
        self.images = self.output / 'tool_images'
        self.images.mkdir(parents=True, exist_ok=True)
        self.episode = json.loads(env.get('V2_EPISODE') or '{}')
        self.robot = None
        if env.get('V2_CONNECTION'):
            self.robot = Robot(env['V2_CONNECTION'])
            self.robot.result_observer = self._record_action
            self.robot.wait_limit_s = DEADLINE_GRACE_MAX_S + 60  # a command may outlast the time limit
        self.trace = (self.output / 'tool_trace.jsonl').open('a', encoding='utf-8')
        self.experiences = None
        if env.get('V2_EXPERIENCE_LOG'):
            self.experiences = ExperienceLog(env['V2_EXPERIENCE_LOG'], self.episode)
        self.step_limit = self.episode.get('step_limit')
        # A development episode has no step limit; its state line shows the evaluation budget for reference.
        self.eval_budget = None if self.step_limit else self.episode.get('evaluation_budget')
        self.action_index = 0
        self.last_frame = None
        self.last_step = None
        self.last_state = None
        self.episode_ended = False
        self.commands_since_log = []
        self._reset_call()
        self.namespace = {'__name__': '__v2_session__', 'display': self.display,
                          'display_image': self.display_image, 'rgb': self.rgb}
        if self.robot is not None:
            self.robot.step_report = self.step_report  # models also call it as robot.step_report()
            self.namespace.update(robot=self.robot, step_report=self.step_report)
        if self.experiences is not None:
            self.namespace['log_experience'] = self.log_experience
        self._open_writes = None  # set by serve(): opens the guard for the knowledge helpers' writes
        self.knowledge = None
        self.writing_knowledge = False
        self.notify = None       # set by serve(): sends a progress message for the current request to the parent
        self.history_watch = None  # set by serve() in a development episode: counts reads of history/
        self._history_base = 0
        self.request_id = None
        self._setup_knowledge(json.loads(env.get('V2_KNOWLEDGE') or 'null'), env.get('V2_EXPERIENCE_LOG'))
        if env.get('V2_HISTORY'):
            self.history_index = Path(env['V2_HISTORY'])
            self.namespace['history'] = self.history
        if env.get('V2_VIDEO'):
            self.video = Path(env['V2_VIDEO'])
            self.namespace.update(video_info=self.video_info, frame_at=self.frame_at, frames=self.frames)
            if self.knowledge is not None and self.knowledge.writable:
                self.namespace.update(save_analysis=self.save_analysis, save_frame=self.save_frame)
        self.preloaded = set(self.namespace) | {'__builtins__'}
        # The objects preloaded for model code (kept alive here, so their IDs stay theirs): never a tool's argument.
        self._helper_objects = [value for value in self.namespace.values() if callable(value)]
        self._helper_ids = {id(value) for value in self._helper_objects}

    def _setup_knowledge(self, settings, experience_log):
        if not settings:
            return
        from v2.knowledge.store import Knowledge
        root = settings['root']
        run = {key: self.episode.get(key) for key in ('run_id', 'task', 'seed', 'layout_id')}
        # The task the knowledge belongs to: the standard variant when a `_random` task is evaluated with its
        # snapshot; otherwise this episode's task.
        self.knowledge_task = run['task'] = settings.get('task') or run['task']
        # The episode's own recording stays citable after the robot is gone (consolidation without the robot).
        run.update(recording=str(self.recording) if self.has_recording else None,
                   session=settings.get('session', 'actor'))
        self.knowledge = Knowledge(root, run, settings.get('recordings'), bool(settings.get('writable')),
                                   allow_write=lambda: self._write_window(root), layouts=settings.get('layouts'))
        self.video_session = bool(settings.get('video'))
        inside = experience_log and Path(experience_log).resolve().is_relative_to(Path(root).resolve())
        self.extra_experience = [experience_log] if experience_log and not inside else []
        self.namespace.update(skill=self.skill, skills=self.skills, search_experience=self.search_experience,
                              tool=self.tool)
        self._tools = {}  # name -> (code sha256, module) of the tools imported in this session
        if self.knowledge.writable:
            self.namespace.update(write_skill=self.write_skill, retire_skill=self.retire_skill,
                                  add_evidence=self.add_evidence, update_brief=self.update_brief,
                                  write_tool=self.write_tool, retire_tool=self.retire_tool)

    @contextlib.contextmanager
    def _write_window(self, root):
        """Allow knowledge writes. The time-limit handler defers while the flag is set, so a file and its
        changelog entry are written together (a signal mask would not cover other threads)."""
        self.writing_knowledge = True
        try:
            with (self._open_writes(root) if self._open_writes else contextlib.nullcontext()):
                yield
        finally:
            self.writing_knowledge = False

    # -- per-call accumulation

    def _reset_call(self):
        self.lines, self.displayed, self.images_out = [], [], []
        self.actions, self.logged = [], []
        self.opened, self.knowledge_changes = [], []
        watch = getattr(self, 'history_watch', None)
        self._history_base = watch.count if watch is not None else 0

    def _note_frame(self, observation_id, state=None):
        number = view.frame_number(observation_id)
        if number is not None:
            self.last_frame, self.last_step = observation_id, number
        if state is not None:
            self.last_state = state

    def _record_action(self, method, params, result):
        if method == 'plan':  # a dry run: listed, but not an executed command
            self.lines.append(view.plan_line(params, result))
            if view.ended_by(result):
                self.episode_ended = True
            return
        if method not in view.MOTION_METHODS:
            return
        self.action_index += 1
        self.lines.append(view.action_line(self.action_index, method, params, result))
        self.actions.append({'method': method, 'status': result.get('status'),
                             'termination_reason': result.get('termination_reason'),
                             'command_id': result.get('command_id')})
        if result.get('command_id'):
            self.commands_since_log.append(result['command_id'])
        if result.get('end_observation_id'):
            self._note_frame(result['end_observation_id'], result.get('measured_final_state'))
        if view.ended_by(result) or method == 'finish':
            self.episode_ended = True

    # -- images

    def _image_ref(self, camera, observation_id, png_base64=None, region=None, source_episode=None):
        path = self.recording / 'observations' / str(observation_id) / f'{camera}.png'
        if png_base64 is not None and (region is not None or source_episode is not None or not path.is_file()):
            data = base64.b64decode(png_base64)
            path = self.images / (hashlib.sha256(data).hexdigest() + '.png')
            if not path.exists():
                path.write_bytes(data)
        label = f'{observation_id} {camera}' + (f' region={list(region)}' if region else '')
        self.images_out.append({'label': label, 'camera': camera, 'observation_id': observation_id,
                                'path': str(path)})

    def _observation(self, observation, cameras=None, prefix=''):
        meta = {k: v for k, v in observation.items() if k != 'images'}
        self._note_frame(meta.get('observation_id'), meta.get('robot_state'))
        if meta.get('episode_ended'):
            self.episode_ended = True
        self.displayed.append(prefix + view.observation_text(meta, self.step_limit, self.eval_budget))
        for image in observation.get('images', []):
            if cameras is None or image['camera'] in cameras:
                self._image_ref(image['camera'], meta.get('observation_id'), image.get('png_base64'))

    def display(self, item, cameras=None):
        """Show an observation, action report, inspect result, image list or value."""
        if cameras is not None:
            cameras = [cameras] if isinstance(cameras, str) else list(cameras)
            unknown = set(cameras) - set(CAMERAS)
            if unknown:
                raise ValueError('unknown cameras: ' + ', '.join(sorted(unknown)))
        if isinstance(item, (list, tuple)):
            for entry in item:
                self.display(entry, cameras)
        elif isinstance(item, dict) and 'views' in item:
            for entry in item['views']:
                region = entry.get('region')
                self.displayed.append(f"view {entry.get('observation_id')}"
                                      + (f" ({entry.get('source_episode')})" if entry.get('source_episode') else '')
                                      + (f' region={region}' if region else ''))
                for image in entry.get('images', []):
                    if cameras is None or image['camera'] in cameras:
                        self._image_ref(image['camera'], entry.get('observation_id'), image.get('png_base64'),
                                        region, entry.get('source_episode'))
        elif isinstance(item, dict) and 'images' in item and 'observation_id' in item:
            self._observation(item, cameras)
        elif isinstance(item, dict) and item.get('end_observation_id') and self.robot is not None:
            label = f"after {item.get('command_id', 'command')}: "
            self._observation(self.robot.replay(item['end_observation_id']), cameras, prefix=label)
        else:
            text = item if isinstance(item, str) else json.dumps(item, default=str, ensure_ascii=False)
            self.displayed.append(view.truncate(text, 4000))

    def display_image(self, item, caption='image'):
        """Show an image you made: a PIL image, a NumPy array or a file path."""
        from PIL import Image
        if isinstance(item, (str, Path)):
            image = Image.open(item)
        elif isinstance(item, Image.Image):
            image = item
        else:
            image = Image.fromarray(item)
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='PNG')
        data = buffer.getvalue()
        path = self.images / (hashlib.sha256(data).hexdigest() + '.png')
        if not path.exists():
            path.write_bytes(data)
        self.displayed.append(f'image: {caption}')
        self.images_out.append({'label': str(caption)[:200], 'camera': None, 'observation_id': None,
                                'path': str(path)})

    def rgb(self, source, camera='cam_head'):
        """An image as a NumPy RGB array (height x width x 3, uint8) for your own analysis.

        source: an observation (robot.observe()), an action report (the image after it), a frame ID
        ('frame_000123'), or an image path. Reading images costs no physical steps.
        """
        import numpy as np
        from PIL import Image
        if camera not in CAMERAS:
            raise ValueError('camera must be one of ' + ', '.join(CAMERAS))
        if isinstance(source, dict) and 'images' in source:
            for image in source['images']:
                if image['camera'] == camera:
                    data = base64.b64decode(image['png_base64'])
                    return np.asarray(Image.open(io.BytesIO(data)).convert('RGB'))
            raise ValueError(f'{camera} is not in this observation')
        if isinstance(source, dict) and source.get('end_observation_id'):
            source = source['end_observation_id']
        if isinstance(source, str) and view.frame_number(source) is not None:
            path = self.recording / 'observations' / source / f'{camera}.png'
            if not path.is_file():
                raise ValueError(f'frame {source} has no recorded {camera} image')
            return np.asarray(Image.open(path).convert('RGB'))
        if isinstance(source, (str, Path)):
            return np.asarray(Image.open(source).convert('RGB'))
        raise ValueError('source must be an observation, an action report, a frame ID or an image path')

    def _check_refs(self, refs):
        """Frame and command IDs of this episode must exist in its recording (run-prefixed refs are checked
        when the knowledge is frozen)."""
        commands = None
        for ref in refs or ():
            if not isinstance(ref, str) or ':' in ref:
                continue
            if view.frame_number(ref) is not None:
                if not (self.recording / 'observations' / ref).is_dir():
                    raise ValueError(f'{ref} is not recorded in this episode')
                continue
            if commands is None:
                path = self.recording / 'commands.jsonl'
                commands = set()
                if path.is_file():
                    for line in path.read_text(encoding='utf-8').split('\n'):
                        if line.strip():
                            try:
                                commands.add(json.loads(line)['request'].get('id'))
                            except (ValueError, KeyError):
                                continue
            if ref not in commands:
                raise ValueError(f'{ref} is not a command of this episode')

    def log_experience(self, description, result, insight=None, skills_used=(), frames=()):
        """Record what just happened: situation, action, observation and your judgment of the result."""
        self._check_refs(frames)
        entry = self.experiences.append(description, result, insight, skills_used, frames,
                                        step=self.last_step, frame_id=self.last_frame,
                                        command_ids=self.commands_since_log)
        self.commands_since_log = []
        self.logged.append(entry)
        return entry['id']

    # -- knowledge helpers

    def skill(self, name):
        """Open a skill: shows its text (kept in view after context compaction) and returns it. The text is shown
        once even when the call also prints it."""
        text = self.knowledge.show_skill(name)
        self.opened = [entry for entry in self.opened if entry['name'] != name] + [{'name': name, 'text': text}]
        return text

    def skills(self, query=None):
        """Show the active skills for this task and the general skills, optionally filtered by words."""
        words = [word.lower() for word in str(query or '').split()]
        lines = [line for line in self.knowledge.index_lines(self.knowledge_task)
                 if all(word in line.lower() for word in words)]
        self.displayed.append('\n'.join(lines) or '(no matching skills)')
        return [line.split(' ', 2)[1] for line in lines]

    def tool(self, name):
        """Import one of this task's tools and return it as a module of its functions, classes and constants. The first
        import (and the first after a revision) shows its doc; the module's __doc__ holds it too."""
        from v2.knowledge import task_tools
        from v2.knowledge.store import KnowledgeError
        meta, code = self.knowledge.read_tool(name, self.knowledge_task)
        if meta['status'] != 'active':
            raise KnowledgeError(f'tool {name} is retired')
        cached = self._tools.get(name)
        if cached is not None and cached[0] == meta['sha256']:
            return cached[1]
        path = self.knowledge.tool_paths(name, self.knowledge_task)[0]
        module = task_tools.load(name, code, path, meta['doc'], refuse=self._refuse_tool_argument)
        self._tools[name] = (meta['sha256'], module)
        self.displayed.append(self.knowledge.show_tool(name, self.knowledge_task))
        return module

    def _refuse_tool_argument(self, value, depth=0):
        """Why `value` may not be passed to a tool's function, or None: the robot, its arms
        and grippers, the session helpers, their methods, and functions or classes defined in the model's session.
        Lists, tuples, sets, dicts and object arrays are looked into three levels deep; instances of the model's
        own classes are not."""
        import functools
        import types
        if value is None or isinstance(value, (str, bytes, bool, int, float, complex)):
            return None
        if value is self or value is self.robot or value is self.knowledge or id(value) in self._helper_ids:
            return 'the robot or a session helper'
        module = getattr(type(value), '__module__', None) or ''
        if module in ('robot_sdk', 'v2.runtime.sdk'):
            return 'the robot or a part of it'
        if isinstance(value, types.MethodType):
            return self._refuse_tool_argument(value.__self__, depth)
        if isinstance(value, functools.partial):
            for part in (value.func, *value.args, *value.keywords.values()):
                reason = self._refuse_tool_argument(part, depth + 1)
                if reason:
                    return reason
            return None
        if isinstance(value, types.FunctionType) and value.__globals__ is self.namespace:
            return f'a function defined in your session ({value.__name__})'
        if isinstance(value, type) and value.__module__ == self.namespace.get('__name__'):
            return f'a class defined in your session ({value.__name__})'
        if depth >= 3:
            return None
        items = None
        if isinstance(value, dict):
            items = list(value.keys()) + list(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
        elif type(value).__name__ == 'ndarray' and getattr(value, 'dtype', None) is not None and value.dtype == object:
            items = list(value.ravel())
        for item in (items or ())[:10000]:
            reason = self._refuse_tool_argument(item, depth + 1)
            if reason:
                return reason
        return None

    def search_experience(self, query=None, task=None, layout=None, run=None, result=None, limit=20):
        """Experiences matching all query words; run="current" for this episode. Print the result to read it."""
        return self.knowledge.search_experience(query, task, layout, run, result, limit,
                                                extra_files=self.extra_experience)

    def step_report(self, top=8):
        """Show where this episode's physical steps went, replayed commands included: steps per method, what set
        each motion's duration (a requested speed or duration, a minimum duration, the joint-space speed limit)
        and the costliest commands. Costs no steps; returns the same as a dict."""
        actions, before = [], None
        while True:
            page = self.robot.history(limit=100, before_sequence=before)
            actions = list(page.get('actions') or []) + actions
            before = page.get('next_before_sequence')
            if not before:
                break
        text, data = steps.step_report(actions, top)
        self.displayed.append(text)
        return data

    def history(self):
        """Show the index of your earlier development episodes; their files are under history/<run_id>/."""
        self.displayed.append(self.history_index.read_text() if self.history_index.is_file() else '(no history)')

    # -- video helpers (video analysis sessions)

    def video_info(self):
        """Duration, size, frame rate and frame count of video.mp4."""
        result = subprocess.run(['ffprobe', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-print_format', 'json',
                                 '-show_format', '-show_streams', str(self.video)],
                                capture_output=True, text=True, timeout=60, check=True)
        info = json.loads(result.stdout)
        stream = next(s for s in info['streams'] if s.get('codec_type') == 'video')
        num, _, den = stream.get('avg_frame_rate', '0/1').partition('/')
        fps = float(num) / float(den or 1) if float(den or 1) else None
        return {'duration_s': float(info['format'].get('duration', 0)), 'width': stream.get('width'),
                'height': stream.get('height'), 'fps': fps, 'frames': int(stream.get('nb_frames') or 0) or None}

    def frame_at(self, t):
        """The video frame at t seconds as a PIL image."""
        from PIL import Image
        result = subprocess.run(['ffmpeg', '-v', 'error', '-protocol_whitelist', 'file,pipe', '-ss', f'{float(t):.3f}',
                                 '-i', str(self.video),
                                 '-frames:v', '1', '-f', 'image2pipe', '-vcodec', 'png', '-'],
                                capture_output=True, timeout=60, check=True)
        if not result.stdout:
            raise ValueError(f'no frame at {t} s')
        return Image.open(io.BytesIO(result.stdout)).convert('RGB')

    def frames(self, t0, t1, n):
        """n evenly spaced frames from t0 to t1 seconds, as (t, image) pairs."""
        n = int(n)
        if not 1 <= n <= 64:
            raise ValueError('n must be 1..64')
        times = [float(t0)] if n == 1 else [float(t0) + i * (float(t1) - float(t0)) / (n - 1) for i in range(n)]
        return [(round(t, 3), self.frame_at(t)) for t in times]

    def save_analysis(self, text, reason='video analysis'):
        """Write the video analysis (markdown text); a later call replaces the earlier version."""
        return self._knowledge_write('save_video_analysis', text, reason)

    def save_frame(self, image, name, t):
        """Save a cited frame or crop (PIL image or array); t is its video time in seconds (for a crop, of
        its source frame), or a range such as '9.40-10.48' for a contact sheet."""
        from PIL import Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        buffer = io.BytesIO()
        image.convert('RGB').save(buffer, format='PNG')
        return self._knowledge_write('save_video_frame', name, buffer.getvalue(), t)

    def _knowledge_write(self, op, *args, **kwargs):
        message = getattr(self.knowledge, op)(*args, **kwargs)
        self.displayed.append(message)
        self.knowledge_changes.append({'op': op, 'message': message})
        return message

    def write_skill(self, name, text, reason):
        """Create or revise a skill: frontmatter (when, type, tasks) and a body with How, How to check,
        Failure signals and recovery, Limits."""
        return self._knowledge_write('write_skill', name, text, reason)

    def retire_skill(self, name, reason):
        """Retire a skill that the evidence contradicts; it leaves the index but its history is kept."""
        return self._knowledge_write('retire_skill', name, reason)

    def add_evidence(self, name, verdict, ref, note=''):
        """name: a skill or a tool; verdict: effective or failed; ref: a frame or command ID of this episode (or
        run_id:ref)."""
        return self._knowledge_write('add_evidence', name, verdict, ref, note)

    def write_tool(self, name, code, doc, reason):
        """Save a tool for this task: code is a Python module of functions that compute from what they are passed;
        doc says what each function takes and returns and how to check its output. The code is imported once
        before it is saved."""
        from v2.knowledge import task_tools
        from v2.knowledge.store import KnowledgeError, normalize_newlines
        try:
            task_tools.load(name, normalize_newlines(code), '<write_tool>', register=False)
        except task_tools.ToolError as exc:
            raise KnowledgeError(str(exc)) from None
        except Exception as exc:  # the import itself failed; a time limit (BaseException) still stops the call
            frames = [frame for frame in traceback.extract_tb(exc.__traceback__) if frame.filename == '<write_tool>']
            where = f' at line {frames[-1].lineno}' if frames else ''
            raise KnowledgeError(f'write_tool imports the code before saving it, and the import raised '
                                 f'{type(exc).__name__}: {exc}{where}; nothing was saved') from None
        message = self._knowledge_write('write_tool', name, code, doc, reason)
        self._tools.pop(name, None)
        return message

    def retire_tool(self, name, reason):
        """Retire a tool the evidence contradicts; it is no longer listed or imported, and its history is kept."""
        message = self._knowledge_write('retire_tool', name, reason)
        self._tools.pop(name, None)
        return message

    def update_brief(self, section, text, reason):
        """Replace one section of the task brief with text (no "## " headings; "### " is fine)."""
        return self._knowledge_write('update_brief', section, text, reason, video=self.video_session)

    # -- requests

    def _skills_block(self):
        """The skills opened in this call, whole, within SKILL_OUTPUT_CHARS; the rest are named. Returns (text, shown)."""
        shown, cut, used = [], [], 0
        for entry in self.opened:
            if shown and used + len(entry['text']) > SKILL_OUTPUT_CHARS:
                cut.append(entry['name'])
                continue
            shown.append(entry)
            used += len(entry['text'])
        text = '\n\n'.join(entry['text'] for entry in shown)
        if cut:
            text += (f"\n[Not shown: {', '.join(cut)}; one call shows at most {SKILL_OUTPUT_CHARS} characters of skill "
                     'text. Open them with skill(name) in another call.]')
        return text, shown

    def _save_output(self, text):
        """Keep the full text of an output that is too long to show; returns its path in the workspace, or None."""
        try:
            folder = self.workspace / '.outputs'
            folder.mkdir(exist_ok=True)
            number = len(list(folder.glob('output_*.txt'))) + 1
            while (folder / f'output_{number:04d}.txt').exists():
                number += 1
            path = folder / f'output_{number:04d}.txt'
            path.write_text(text, encoding='utf-8')
            return path
        except OSError:
            return None

    def _reply(self, is_error=False, extra_lines=(), full_extra=None):
        """The reply to the parent. full_extra: extra_lines before their own cuts (printed text), for the saved copy
        of an output that is too long."""
        text = '\n'.join(list(self.lines) + list(self.displayed) + list(extra_lines)).strip()
        full = '\n'.join(list(self.lines) + list(self.displayed) + list(full_extra or extra_lines)).strip()
        saved = self._save_output(full) if len(full) > len(text) or len(text) > view.MAX_OUTPUT_CHARS else None
        if len(text) > view.MAX_OUTPUT_CHARS:
            text = view.truncate(text, saved_path=saved)
        elif saved is not None:  # only the printed text was cut: say where all of it is
            text += f'\n[The full output is saved at {saved}.]'
        skills, shown = self._skills_block()
        text = '\n\n'.join(part for part in (text, skills) if part)
        return {'text': text or 'Done. (No output; use display(...) to see images.)',
                'images': self.images_out, 'is_error': is_error, 'actions': self.actions,
                'experiences': self.logged, 'episode_ended': self.episode_ended,
                'last_frame': self.last_frame, 'last_step': self.last_step,
                'skills_opened': shown, 'knowledge_changes': self.knowledge_changes,
                'history_reads': (self.history_watch.count - self._history_base) if self.history_watch else 0}

    def _motion_progress(self):
        """Size of the recording's per-control-step log, or None if there is none: it grows while the
        simulator steps a command."""
        try:
            return (self.recording / 'transitions.jsonl').stat().st_size
        except OSError:
            return None

    def _extend(self, seconds):
        """Wait `seconds` more for the running command, and let the parent wait as long for the reply."""
        signal.alarm(seconds)
        if self.notify is not None and self.request_id is not None:
            self.notify({'id': self.request_id, 'extend_s': seconds})

    def python_exec(self, code, timeout_s):
        self._reset_call()
        stdout = io.StringIO()
        grace = {'state': None, 'finished': False, 'progress': None, 'since': None, 'command': None, 'idle': 0}

        def on_alarm(signum, frame):
            if grace['finished']:
                return  # the code has ended; never raise inside the cleanup below
            if self.writing_knowledge:
                signal.alarm(1)  # finish the knowledge write first
                return
            robot = self.robot
            if robot is not None and robot.active_command and grace['state'] != 'over':
                # A command is running: let it finish (never cut a motion short) while the simulator
                # keeps stepping; the SDK raises DeadlineReached once it completes.
                progress, clock = self._motion_progress(), time.monotonic()
                if grace['state'] is None:
                    grace.update(state='waiting', progress=progress, since=clock, command=robot.active_command)
                    robot.deadline_pending = True
                    self._extend(DEADLINE_GRACE_S)
                    return
                grace['idle'] = 0 if progress is not None and progress != grace['progress'] else grace['idle'] + 1
                grace['progress'] = progress
                if (robot.active_command == grace['command'] and progress is not None
                        and grace['idle'] < STALL_CHECKS and clock - grace['since'] < DEADLINE_GRACE_MAX_S):
                    self._extend(DEADLINE_GRACE_S)
                    return
                grace['state'] = 'over'  # the simulator stopped stepping, the cap is reached, or not that command
            if robot is not None:
                robot.deadline_passed = True
            signal.alarm(REPEAT_ALARM_S)
            raise ExecTimeout('python_exec time limit reached; the code was interrupted')

        previous = signal.signal(signal.SIGALRM, on_alarm)
        signal.alarm(int(timeout_s))
        failure = None
        try:
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                exec(compile(code, '<python_exec>', 'exec'), self.namespace)
        except BaseException as exc:  # includes KeyboardInterrupt/SystemExit raised by model code
            grace['finished'] = True
            signal.alarm(0)
            failure = exc
        finally:
            grace['finished'] = True
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
            if self.robot is not None:
                self.robot.deadline_pending = self.robot.deadline_passed = False
        extra = []
        printed = stdout.getvalue()
        for entry in self.opened:  # print(skill(name)): the text is shown once, with the opened skills below
            printed = printed.replace(entry['text'], f"[skill {entry['name']}: its text is shown below]", 1)
        failure_lines = self._failure_lines(failure) if failure is not None else []
        if printed:
            extra.append(view.truncate(printed, STDOUT_CHARS))
        reply = self._reply(is_error=failure is not None, extra_lines=extra + failure_lines,
                            full_extra=([printed] if printed else []) + failure_lines)
        self._trace('python_exec', {'code': code, 'timeout_s': timeout_s}, reply)
        return reply

    def _failure_lines(self, failure):
        lines = []
        robot = self.robot
        active = getattr(robot, 'active_command', None) if robot is not None else None
        if active:
            # A command may still be executing. Ask it to stop at the next control
            # boundary and resolve what actually happened; never resubmit it.
            request = getattr(robot, 'active_request', None) or {}  # read before wait() clears it
            note = {'command_id': active, 'resolution': 'unresolved'}
            try:
                robot.cancel(active)
                report = robot.wait(active, timeout=15)
                self._record_action(request.get('method', 'command'), request.get('params', {}), report)
                note['resolution'] = 'resolved'
            except Exception as error:
                note['error'] = f'{type(error).__name__}: {error}'
            lines.append('Interrupted command: ' + json.dumps(note) +
                         ' — do not rerun this code; check robot.status(command_id) if unresolved.')
        if isinstance(failure, self.DeadlineReached):
            lines.append(str(failure))
        elif isinstance(failure, ExecTimeout):
            stack = traceback.extract_tb(failure.__traceback__)
            where = next((f'line {frame.lineno}' for frame in reversed(stack) if frame.filename == '<python_exec>'),
                         'an unknown line')
            lines.append(f'{failure} at {where} of your code; no further robot command was submitted.')
        else:
            lines.append(view.truncate(''.join(traceback.format_exception(
                type(failure), failure, failure.__traceback__)), 8000))
        return lines

    def observe(self, cameras=None):
        self._reset_call()
        self._observation(self.robot.observe(), cameras)
        reply = self._reply()
        self._trace('observe', {'cameras': cameras}, reply)
        return reply

    def inspect(self, selections):
        self._reset_call()
        self.display(self.robot.call('inspect', selections=selections))
        reply = self._reply()
        self._trace('inspect', {'selections': selections}, reply)
        return reply

    def finish(self):
        self._reset_call()
        report = self.robot.finish()
        self.episode_ended = True
        reply = self._reply(extra_lines=[f"Physical episode finished ({report.get('status', 'done')})."])
        self._trace('finish', {}, reply)
        return reply

    def status(self):
        """Current state line and images, used after context compaction."""
        return self.observe()

    def describe(self):
        return self.robot.describe()

    def names(self, limit=60, max_chars=3000):
        """A short list of what model code has defined in this session (variables, functions, classes and imported
        modules), shown after a context compaction: the Python session outlives it. Only
        built-in values are rendered, so no model-defined __repr__ or __len__ runs here."""
        import inspect
        import types
        scalars = (bool, int, float, complex, str, type(None))

        def short(value):
            text = repr(value)
            return text if len(text) <= 60 else None
        entries, modules = [], []
        for name, value in list(self.namespace.items()):
            if name in self.preloaded or name.startswith('_'):
                continue
            kind = type(value)
            try:
                if kind is types.ModuleType:
                    modules.append(name)
                    continue
                if kind is types.FunctionType:
                    entry = f'{name}{inspect.signature(value)}'
                elif isinstance(value, type):
                    entry = f'class {name}'
                elif kind in scalars:
                    text = short(value)
                    entry = f'{name} = {text}' if text else f'{name}: {kind.__name__} ({len(value)} characters)'
                elif kind.__name__ == 'ndarray' and kind.__module__ == 'numpy':
                    entry = f'{name}: ndarray {tuple(value.shape)} {value.dtype}'
                elif kind in (list, tuple, dict, set):
                    items = list(value.items()) if kind is dict else list(value)
                    flat = [part for item in items for part in (item if kind is dict else (item,))]
                    text = short(value) if len(items) <= 12 and all(type(part) in scalars for part in flat) else None
                    entry = f'{name} = {text}' if text else f'{name}: {kind.__name__} of {len(items)}'
                else:
                    entry = f'{name}: {kind.__name__}'
            except Exception:
                entry = f'{name}: {kind.__name__}'
            entries.append(entry[:200])
        shown, used = [], 0
        for entry in entries[:limit]:
            if used + len(entry) > max_chars:
                break
            shown.append(entry)
            used += len(entry) + 2
        text = '; '.join(shown)
        if len(shown) < len(entries):
            text += f'; and {len(entries) - len(shown)} more'
        if modules:
            text += ('; ' if text else '') + 'modules: ' + ', '.join(sorted(modules))
        return {'text': text}

    def _trace(self, name, arguments, reply):
        record = {'time': time.time(), 'name': name, 'arguments': arguments, 'is_error': reply['is_error'],
                  'text': reply['text'], 'images': [image['label'] for image in reply['images']],
                  'actions': reply['actions'], 'experiences': reply['experiences']}
        self.trace.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.trace.flush()

    def handle(self, method, params):
        if method == 'python_exec':
            return self.python_exec(params['code'], params.get('timeout_s', 600))
        if method in ('observe', 'inspect', 'finish', 'status', 'describe') and self.robot is None:
            raise RuntimeError('This session has no robot')
        if method == 'observe':
            return self.observe(params.get('cameras'))
        if method == 'inspect':
            return self.inspect(params['selections'])
        if method == 'finish':
            return self.finish()
        if method == 'status':
            return self.status()
        if method == 'describe':
            return self.describe()
        if method == 'ping':
            return {'ok': True}
        if method == 'names':
            return self.names()
        raise ValueError('unknown method ' + str(method))


PR_SET_PDEATHSIG = 1


def end_with_parent(sig, parent=None):
    """Linux: have `sig` sent to this process when the thread that started it ends, i.e. when its parent process
    ends however it ends (os._exit, SIGKILL, the OOM killer). Returns False when `parent`, the pid that started this
    process (directly, or through a wrapper that did not exec), has already gone: then nothing would send the signal.
    Elsewhere, or if prctl is refused, it only checks `parent`."""
    if sys.platform.startswith('linux'):
        try:
            import ctypes
            ulong = ctypes.c_ulong
            ctypes.CDLL(None, use_errno=True).prctl(ctypes.c_int(PR_SET_PDEATHSIG), ulong(int(sig)), ulong(0),
                                                     ulong(0), ulong(0))
        except (OSError, AttributeError, ValueError):
            pass
    if parent is None or os.getppid() == int(parent):
        return True
    try:  # a wrapper between them (runner_python that did not exec): the evaluator may still run
        os.kill(int(parent), 0)
    except ProcessLookupError:
        return False
    except (PermissionError, ValueError, OverflowError):
        pass
    return True


def serve(env=None):
    """Child entry point: read requests from stdin until EOF."""
    env = env or os.environ
    if env.get('V2_END_WITH_PARENT'):
        # The submission's session subprocess may itself be ended at any moment with the evaluation process; this
        # worker then stops at once rather than finishing whatever the model's code is doing (a closed stdin is
        # read only after that). The session starts it from its main thread, which lives as long as the session.
        end_with_parent(signal.SIGKILL)
    session = Session(env)
    guard_config = json.loads(env.get('V2_GUARD') or 'null')
    if guard_config:
        from urllib.parse import urlsplit
        import ctypes  # noqa: F401  (landlock.apply drives the Landlock syscalls through ctypes)
        from v2.runtime import guard, landlock
        # Network: only this episode's robot service. Writes outside the workspace: only the knowledge helpers.
        url = urlsplit(session.robot.connection['url']) if session.robot is not None else None
        endpoints = [(url.hostname, url.port)] if url is not None else []
        knowledge = json.loads(env.get('V2_KNOWLEDGE') or 'null') or {}
        helpers = [knowledge['root']] if knowledge.get('writable') else []
        log = guard_config.get('log')
        if log:
            open(log, 'a').close()  # the guard appends to its log; create it before the sandbox denies new files
        # Layer 2: kernel-enforced isolation, applied before any model code. Native
        # library calls (cv2, torch, cffi, fork_exec) reach the kernel without a Python audit event, so only
        # Landlock can stop them; it is inherited by ffmpeg/ffprobe and any other child of the worker. It is
        # applied BEFORE importing numpy: landlock_restrict_self binds only threads that exist from now on, and
        # numpy's BLAS pool would otherwise start unsandboxed threads. The knowledge root is granted for the
        # whole session when its knowledge is writable; the Python guard still confines Python-level writes to
        # the helpers' write window. A kernel without Landlock (before 5.13, or with it disabled) runs the worker
        # on the Python guard alone, as the README says (older kernels, e.g. 5.4, have no Landlock); run.json and
        # the evaluator console say so. Where Landlock is
        # there, the worker still refuses to start if it cannot be applied, unless V2_SANDBOX_OPTIONAL (test
        # machines) is set. apply() is a no-op on non-Linux hosts.
        optional = (str(env.get('V2_SANDBOX_OPTIONAL', '')).lower() in ('1', 'true', 'yes')
                    or landlock.abi_version() < 1)
        sandbox = landlock.apply(list(guard_config['read']) + list(guard.system_read_roots()),
                                 list(guard_config['write']) + list(helpers), guard_config.get('executables', ()),
                                 endpoints=endpoints, write_files=[log] if log else (), require=not optional)
        sys.stderr.write('v2 sandbox: ' + json.dumps(sandbox) + '\n')  # worker.stderr (developer-only)
        sys.stderr.flush()
        # The guard refuses every ctypes event; numpy loads ctypes when imported, so import it (now sandboxed)
        # before the guard is installed.
        try:
            import numpy  # noqa: F401
        except ImportError:  # environments without numpy (some test machines)
            pass
        installed = guard.install(guard_config['read'], guard_config['write'], guard_config.get('executables', ()),
                                  log, endpoints=endpoints, helper_roots=helpers)
        installed.sandbox = sandbox
        session._open_writes = installed.allow
    watch = json.loads(env.get('V2_HISTORY_WATCH') or 'null')
    if watch:
        session.history_watch = HistoryWatch(watch)
        sys.addaudithook(session.history_watch.hook)
    # Replies go to a private copy of fd 1; fd 1 itself is pointed at stderr, so
    # prints and native libraries writing to stdout cannot corrupt the protocol.
    protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8')
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    def notify(message):
        # Called from the time-limit handler while model code runs; the reply below is written only
        # after the request has finished, so the two writes never interleave.
        protocol.write(json.dumps(message) + '\n')
        protocol.flush()
    session.notify = notify
    for line in sys.stdin:
        if not line.strip():
            continue
        request = json.loads(line)
        session.request_id = request['id']
        try:
            reply = {'id': request['id'], 'result': session.handle(request['method'], request.get('params') or {})}
        except BaseException as exc:  # keep serving; a late timeout or model-raised exit is only this request's error
            reply = {'id': request['id'], 'error': f'{type(exc).__name__}: {exc}'}
        protocol.write(json.dumps(reply, ensure_ascii=False, default=str) + '\n')
        protocol.flush()


# --------------------------------------------------------------- parent side

class WorkerError(RuntimeError):
    """The worker could not complete a request; physical outcome may be uncertain.

    `delivered` is False when the request never reached the worker, so nothing ran.
    """

    def __init__(self, message, delivered=True):
        super().__init__(message)
        self.delivered = delivered


class RequestError(WorkerError):
    """The worker is alive but rejected or failed one request (for example an invalid inspect)."""


# The two V2_DEADLINE_* values let an operator change the time-limit grace for the worker as for the parent.
# The harness code a worker runs, readable to it so that tracebacks show its lines; the rest of v2/ is not.
WORKER_SOURCE = (HARNESS_ROOT / 'v2' / 'runtime', HARNESS_ROOT / 'v2' / 'knowledge')

INHERITED_ENV = ('PATH', 'USER', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'SYSTEMROOT', 'V2_DEADLINE_GRACE_S',
                 'V2_DEADLINE_GRACE_MAX_S')


def child_environment(workspace, settings, base=None):
    """Environment for a worker: a short allow-list plus V2_* settings.

    API keys and other secrets in the parent environment are not inherited.
    HOME, caches and temporary files point into the workspace.
    """
    base = os.environ if base is None else base
    workspace = Path(workspace)
    env = {key: base[key] for key in INHERITED_ENV if key in base}
    for name in ('.home', '.cache', '.tmp'):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    env.update(HOME=str(workspace / '.home'), XDG_CACHE_HOME=str(workspace / '.cache'),
               MPLCONFIGDIR=str(workspace / '.cache' / 'matplotlib'), TMPDIR=str(workspace / '.tmp'),
               PYTHONPATH=os.pathsep.join([str(HARNESS_ROOT), str(HARNESS_ROOT / 'embodied_agent')]),
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1', PYTHONSAFEPATH='1',
               NO_PROXY='127.0.0.1,localhost,::1', no_proxy='127.0.0.1,localhost,::1')
    for key, value in settings.items():
        if not key.startswith('V2_'):
            raise ValueError('worker settings must be V2_* variables')
        env[key] = value if isinstance(value, str) else json.dumps(value)
    return env


class WorkerProcess:
    """Start a worker child and exchange sequential JSON requests with it."""

    def __init__(self, env, cwd, stderr_path, python=None):
        self.stderr = open(stderr_path, 'a')
        child_env = dict(env)
        # -s: no per-user site-packages (its .pth files run code at startup, before the sandbox, and HOME is
        # model-writable). PYTHONSAFEPATH (set in child_environment) keeps the model-writable cwd off sys.path
        # on Python 3.11+; older Pythons ignore it.
        self.proc = subprocess.Popen([python or sys.executable, '-u', '-s', '-m', 'v2.runtime.worker'],
                                     cwd=str(cwd), env=child_env, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=self.stderr, text=True,
                                     bufsize=1, start_new_session=True)
        self.inbox = queue.Queue()
        self.sequence = 0

        def reader():
            try:
                for line in self.proc.stdout:
                    self.inbox.put(line)
            finally:
                self.inbox.put(None)
        threading.Thread(target=reader, daemon=True).start()

    def call(self, method, params=None, timeout=120):
        if self.proc.poll() is not None:
            raise WorkerError(f'the worker is not running (exit code {self.proc.returncode}); {method} was not run',
                              delivered=False)
        self.sequence += 1
        try:
            self.proc.stdin.write(json.dumps({'id': self.sequence, 'method': method, 'params': params or {}}) + '\n')
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise WorkerError(f'could not send {method} to the worker ({type(exc).__name__}); it was not run',
                              delivered=False) from None
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.inbox.get(timeout=max(0.001, deadline - time.monotonic()))
            except queue.Empty:
                self.kill()
                raise WorkerError(f'worker did not answer {method} in {timeout}s and was stopped; the physical '
                                  'outcome is uncertain — check command status before any further motion') from None
            if line is None:
                raise WorkerError(f'worker exited during {method}; check command status before any further motion')
            try:
                reply = json.loads(line)
            except ValueError:
                continue  # not a reply; the protocol stream carries only JSON lines
            if not isinstance(reply, dict) or reply.get('id') is None or reply['id'] < self.sequence:
                continue  # a stale reply to an abandoned request
            if 'extend_s' in reply and reply['id'] == self.sequence:
                # A command is still running past the time limit and the simulator is stepping: wait longer.
                deadline = max(deadline, time.monotonic() + float(reply['extend_s']) + RPC_MARGIN_S)
                continue
            if reply['id'] != self.sequence:
                self.kill()
                raise WorkerError('unexpected worker reply ID; the worker was stopped without replay')
            if 'error' in reply:
                raise RequestError(reply['error'])
            return reply['result']

    def python_exec(self, code, timeout_s):
        return self.call('python_exec', {'code': code, 'timeout_s': timeout_s},
                         timeout=timeout_s + DEADLINE_GRACE_S + RPC_MARGIN_S)

    def kill(self):
        """Stop the worker now (used when it stops answering)."""
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
        for stream in (self.proc.stdin, self.proc.stdout):
            try:
                stream.close()
            except (OSError, ValueError):
                pass
        self.stderr.close()


if __name__ == '__main__':
    serve()

"""Model loop shared by all v2 sessions.

The loop owns the conversation, its checkpoint and the Python worker. Python and
robot tools run in the worker; plan, finish, next-episode and end-session tools
are handled here. The checkpoint is written before and after every tool call,
so a restart never runs a tool call again when it may have moved the robot:
an interrupted call is reported to the model as uncertain instead.

Stages: an actor session starts in `episode`. A development actor moves to
`consolidation` when the physical episode ends (finish, or the environment ends
it) and stays in the same conversation until plan_next_episode. Curation and
video sessions use the single stage `session` and end with end_session.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.parse

from v2.runtime import context, tools
from v2.runtime.client import ModelAPIError, ResponsesClient, image_refused, public_config
from v2.runtime.worker import RequestError, WorkerError

SCHEMA = 'v2_session/1'
OUTPUT_TYPES = ('function_call', 'message', 'reasoning', 'compaction')
INCOMPLETE_RETRIES = 2
MAX_OUTPUT_TOKENS_CAP = 128000
TRANSCRIPT_CHARS = 80000       # about 20k tokens of recent calls kept by a local compaction
FALLBACK_OVERSHOOT = 1.2       # input this far above the threshold without compaction → local fallback
LONG_CONTEXT_TOKENS = 272000   # the provider's long-context step: the fallback also starts here
NAMES_NOTE = ('Your Python session was not reset: these names you defined still hold their values (print them to '
              'check): {names}')
# After a server compaction the next input must fall below this share of the threshold; otherwise
# the compaction did not free room and the session
# switches to local compaction.
COMPACTION_RELIEF = 0.9
WORKER_RESTARTS = 3            # consecutive worker failures before the session stops
QUOTA_CODES = ('pre_consume_token_quota_failed', 'insufficient_quota', 'insufficient_user_quota')
TRANSIENT_STATUSES = (None, 408, 409, 425, 429, 500, 502, 503, 504)
# Key, payment, permission or quota, and model-not-found refusals. A session with wait_refusals (the RoboDojo
# submission) waits these out like transient failures, since an API gateway can return them transiently;
# otherwise a quota refusal stops the batch.
REFUSAL_STATUSES = (401, 402, 403, 404)
API_WAITS_S = (60, 300, 900)   # waits after the client's own retries, keeping the worker and its variables
TOOL_TIMEOUT_S = 180
# Stuck tool calls: a failed call does not reset the text-only stop.
# - The same call failing the same way REPEATED_FAILURES times in a row (text-only replies in between do not break the
#   run; the call's arguments are part of "the same", python_exec's code included). Three: the model sees the error
#   each time; a second identical try can be a fair retry of a transient fault, a third shows the fault is not
#   transient and the model is not adapting; the text-only stop also allows two.
# - INFRASTRUCTURE_FAILURES failed finish/observe calls (any text) without a working robot reply in between.
# finish and observe have no argument the model can get wrong, so their failures are the harness's or the simulator's:
# the session stops as needs_attention. Other calls failing again and again are the model's own: its physical episode
# ends (an evaluation is scored as it stands), a curation or video session ends, and a stuck consolidation stops.
REPEATED_FAILURES = 3
INFRASTRUCTURE_TOOLS = ('finish', 'observe')
INFRASTRUCTURE_FAILURES = 6
# A python_exec that failed on the robot service's transport (the SDK's own messages) failed for the same reason as a
# failing finish or observe: such failures count as infrastructure too.
TRANSPORT_ERRORS = ('Request outcome uncertain', 'Wait timed out')
SIMULATOR_FAILED = ('The simulator failed ({reason}), so the physical episode has ended: the robot can no longer move '
                    'or be observed, but python_exec still works and the recorded frames and commands remain readable.')
REVIEW_FALLBACK = ('End is not recorded yet: this review comes once. To end development, call plan_next_episode again '
                   'with decision end (and your handoff); reset_same or switch_layout continues development instead.')
REPEATED_ENDED = ('{tool} failed the same way {count} times in a row, so the harness ended the physical episode: the '
                  'robot can no longer move, but python_exec still works and the recorded frames and commands remain '
                  'readable.')

UNCERTAIN = ('Its physical outcome is uncertain: check robot.history(limit=5) and the current images before any '
             'motion, and never resubmit a motion that may already have run.')
RESUME_NOTE = ('The harness restarted. Your Python session is new: variables, functions and imports from earlier '
               'calls are gone; files in your workspace remain.')
ROBOT_KEPT = 'The robot and the scene were not reset.'
TEXT_ONLY = {
    ('episode', 'development'): 'Continue with a tool call. To end this physical episode, call finish(summary, '
                                'self_assessment).',
    ('episode', 'evaluation'): 'Continue with a tool call, or call finish(summary, self_assessment) if you are '
                               'done. Another reply without a tool call ends the episode.',
    ('consolidation', 'development'): 'Continue with a tool call. When the knowledge is updated, call '
                                      'plan_next_episode(...).',
    ('session', None): 'Continue with a tool call, or call end_session(summary) when you are done. Another reply '
                       'without a tool call ends the session.',
}


def failure_signature(name, text, arguments=None):
    """What makes two failed calls "the same": the tool, its arguments (python_exec's code with whitespace
    normalised; free text such as a summary or handoff left out) and the error text with command IDs and whole numbers
    masked."""
    arguments = dict(arguments or {}) if isinstance(arguments, dict) else {}
    for key in ('summary', 'handoff', 'explanation', 'timeout_s'):
        arguments.pop(key, None)
    if 'code' in arguments:
        arguments['code'] = hashlib.sha256(' '.join(str(arguments['code']).split()).encode()).hexdigest()[:16]
    text = re.sub(r'\b[0-9a-f]{8,}\b', '<id>', str(text or ''))
    text = re.sub(r'\b\d+(?:\.\d+)?\b', '<n>', text)  # not digits inside names (x1, x2)
    return name + ' ' + json.dumps(arguments, sort_keys=True, default=str) + ': ' + text[:2000]


def image_count_refused(detail):
    """Whether the provider refused the request for carrying too many images ("... maximum number of images ...";
    a gateway may wrap that error as JSON text)."""
    text = json.dumps(detail or {}).lower()
    return 'maximum number of images' in text and ('exceed' in text or 'allowed' in text)


def summary_refused(detail):
    """Whether the provider refused the request's reasoning summary (an endpoint that does not offer it, e.g. "Your
    organization must be verified to generate reasoning summaries", param reasoning.summary; a gateway may wrap that
    error as JSON text)."""
    text = json.dumps(detail or {}).lower()
    return 'reasoning.summary' in text or 'reasoning summar' in text


class NeedsAttention(RuntimeError):
    """The session stopped at a safe point; check the cause, then resume."""

    def __init__(self, message, reason='error'):
        super().__init__(message)
        self.reason = reason


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def write_private(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, default=str)
    os.replace(temp, path)


def write_private_gzip(path, value):
    """write_private, gzip-compressed: the request log keeps every call's items (text only), which add up."""
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'wb') as stream, gzip.GzipFile(fileobj=stream, mode='wb', mtime=0) as packed:
        packed.write(json.dumps(value, ensure_ascii=False, default=str).encode('utf-8'))
    os.replace(temp, path)


def append_private(path, record):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(fd, 'a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, default=str) + '\n')


def prompt_cache_key(*parts):
    """A prompt_cache_key of at most 64 characters: "v2-" and the parts when they fit, otherwise their start and
    a hash of the whole, so that runs with long campaign and task names keep distinct keys."""
    key = '-'.join(('v2',) + tuple(str(part) for part in parts))
    return key if len(key) <= 64 else key[:47] + '-' + hashlib.sha256(key.encode()).hexdigest()[:16]


def endpoint_record(base_url, redact=False):
    """The model endpoint as a session record keeps it: the URL, or with redact (the submission path, whose
    records stay on the evaluator's machine) only its sha256."""
    return 'sha256:' + hashlib.sha256(base_url.encode()).hexdigest() if redact else base_url


def scrub_endpoint(value, base_url):
    """`value` (text, or a record holding text) with the endpoint's URL, network location and host name replaced,
    also percent-encoded or cut off at the end of a truncated text, for the records of a redacted session: provider
    and connection errors can quote them. Never raises."""
    url = str(base_url or '').rstrip('/')
    found = {url}
    try:
        parts = urllib.parse.urlsplit(url)
        found |= {parts.netloc, parts.hostname or ''}
        host = parts.netloc.rpartition('@')[2]  # the host as written (hostname is lower-cased)
        found.add(host[1:host.find(']')] if host.startswith('[') else host.partition(':')[0])
    except ValueError:
        pass
    found |= {urllib.parse.quote(name, safe='') for name in found}
    names = sorted((name for name in found if name), key=len, reverse=True)
    # A short bare name ("gateway") is replaced only as the same whole word; a dotted or ported one anywhere.
    patterns = [re.compile(re.escape(name) + r'(?![\w-])', re.IGNORECASE) if dotted(name)
                else re.compile(r'(?<![\w.-])' + re.escape(name) + r'(?![\w-])') for name in names]

    def scrub(item):
        if isinstance(item, str):
            for pattern in patterns:
                item = pattern.sub('[model endpoint]', item)
            return cut_name(item, names)
        if isinstance(item, dict):
            return {key: scrub(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [scrub(entry) for entry in item]
        return item
    return scrub(value) if names else value


def dotted(name):
    return '.' in name or ':' in name or '%' in name


def cut_name(text, names):
    """`text` with a dotted name cut off at its end replaced: a tail that starts the name, runs past its first
    separator after any URL scheme and is at least 6 characters past that scheme ("gw.exa"), so ordinary words
    ("mode", "Gateway", "10.0") stay."""
    lowered = text.lower()
    for name in names:
        if not dotted(name):
            continue
        scheme = 0
        for marker in ('://', '%3a%2f%2f'):
            at = name.lower().find(marker)
            if at >= 0:
                scheme = max(scheme, at + len(marker))
        rest = name[scheme:]
        separators = [at for at in (rest.find('.'), rest.find(':'), rest.find('%')) if at >= 0]
        if not separators:
            continue
        shortest = scheme + max(6, min(separators) + 2)
        for size in range(len(name) - 1, shortest - 1, -1):
            if lowered.endswith(name[:size].lower()):
                return text[:-size] + '[model endpoint]'
    return text


def user_message(texts, images=()):
    content = [{'type': 'input_text', 'text': text} for text in texts if text]
    for image in images:
        content.append({'type': 'input_text', 'text': image['label']})
        content.append(context.image_part(image['path']))
    return {'role': 'user', 'content': content}


def quota_refusal(exc):
    """Whether a model API error is a quota refusal (HTTP 403 with a quota code)."""
    detail = exc.detail or {}
    return exc.status == 403 and (detail.get('code') in QUOTA_CODES or
                                  any(code in json.dumps(detail) for code in QUOTA_CODES))


class Session:
    """One model session: an actor episode (with its consolidation), a curation or a video analysis.

    `opening(worker)` returns the first input items. `hooks` may provide
    `plan_next_episode(args) -> text`, which may raise ValueError to reject a
    decision, `official_result(wait_s) -> text or None`, the episode's
    official result for the consolidation note, `end_review(state) -> text`, the
    readiness review shown at a development actor's first end, and
    `simulator_failure(is_error) -> reason or None`, which tells from the simulator's
    own state (not from any text the model could write) when it has broken or
    vanished. `consolidation` is the message
    sent to a development actor when its physical episode ends.
    """

    def __init__(self, config, output, *, profile, instructions, worker_factory, mode=None,
                 workspace=None, opening=None, step_limit=None, hooks=None, consolidation=None,
                 client=None, cache_key=None, max_model_calls=0, robot=True, api_waits=API_WAITS_S,
                 redact_endpoint=False, wait_refusals=False):
        if profile not in tools.PROFILES:
            raise ValueError('unknown tool profile ' + str(profile))
        self.config, self.output = config, Path(output)
        self.workspace = Path(workspace) if workspace else None
        self.profile, self.mode, self.instructions = profile, mode, instructions
        self.worker_factory, self.opening = worker_factory, opening
        self.step_limit, self.hooks, self.consolidation = step_limit, hooks or {}, consolidation
        self.client, self.cache_key, self.max_model_calls = client, cache_key, max_model_calls
        self.robot, self.redact_endpoint = robot, redact_endpoint
        self.api_waits, self.api_wait_index = tuple(api_waits), 0
        self.wait_statuses = TRANSIENT_STATUSES + (REFUSAL_STATUSES if wait_refusals else ())
        self.state_path = self.output / 'session.json'
        self.events_path = self.output / 'events.jsonl'
        self.archive_dir = (self.workspace or self.output) / '.context_archives'
        self.images = context.ImageCache()
        self.worker = None
        self.state = None
        self.worker_failures = 0
        self.request_attempts = {}  # model call -> attempts written to the request log by this process

    # ------------------------------------------------------------ lifecycle

    def run(self, resume=False, physical_ended=None):
        """Run or resume the session. physical_ended (with resume) says why the physical episode was
        stopped by the harness; a development actor then continues with consolidation, without the robot."""
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / 'responses').mkdir(exist_ok=True)
        (self.output / 'requests').mkdir(exist_ok=True)
        lock = (self.output / 'session.lock').open('a')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise RuntimeError('another process is running this session') from None
        try:
            self._load(resume)
            self.client = self.client or ResponsesClient(self.config, retry_observer=self._event)
            self._start_worker()
            if resume:
                self._resume(physical_ended)
            elif not self.state['items']:
                self.state['items'] = list(self.opening(self.worker) if self.opening else [])
                if not self.state['items']:
                    raise ValueError('a session needs an opening message')
                self._save()
            return self._loop()
        except BaseException as exc:
            self._stop(exc)
            raise
        finally:
            if self.worker is not None:
                self.worker.close()
            lock.close()

    def _load(self, resume):
        if self.state_path.exists():
            if not resume:
                raise ValueError('output already holds a session; resume it or choose a new output')
            state = json.loads(self.state_path.read_text())
            if state.get('schema') != SCHEMA:
                raise ValueError('unsupported session checkpoint')
            if state['phase'] == 'completed':
                raise ValueError('session already completed')
            fingerprint = state['fingerprint']
            current = endpoint_record(self.config['base_url'], str(fingerprint['base_url']).startswith('sha256:'))
            if (fingerprint['model'], fingerprint['base_url']) != (self.config['model'], current):
                raise ValueError('resume needs the same model and endpoint')
            self.state = state
            return
        if resume:
            raise ValueError('no session to resume')
        self.state = {
            'schema': SCHEMA, 'profile': self.profile, 'mode': self.mode, 'phase': 'starting',
            'stage': 'episode' if self.mode else 'session', 'started': now(),
            'fingerprint': {'model': self.config['model'],
                            'base_url': endpoint_record(self.config['base_url'], self.redact_endpoint)},
            'instructions': self.instructions, 'items': [], 'calls': 0,
            'pending_tools': [], 'active_tool': None, 'handled_call_ids': [],
            'max_output_tokens': self.config['max_output_tokens'], 'last_input_tokens': 0,
            'local_fallback': False, 'image_count_rebuilt': False, 'text_only': 0, 'incomplete': 0,
            'plan': [], 'opened_skills': {}, 'reminders': None, 'last_step': None,
            'outcome': None, 'decision': None, 'end': None, 'end_requested': False,
            'usage': {'input_tokens': 0, 'cached_tokens': 0, 'cache_write_tokens': 0, 'output_tokens': 0,
                      'reasoning_tokens': 0},
            'stats': {'tools': {}, 'tool_errors': 0, 'action_calls': 0, 'actions': 0, 'experiences': 0,
                      'skills_opened': 0, 'knowledge_changes': 0,
                      'functions_defined': 0, 'image_code_calls': 0, 'images_evicted': 0,
                      'compactions_server': 0, 'compactions_local': 0, 'text_only': 0,
                      'incomplete': 0, 'worker_restarts': 0, 'model_seconds': 0.0,
                      'history_reads_in_episode': 0}}
        public = public_config(self.config)
        public['base_url'] = endpoint_record(self.config['base_url'], self.redact_endpoint)
        write_private(self.output / 'session_config.json', public)
        self._save()

    def _start_worker(self):
        self.worker = self.worker_factory()
        self.worker.call('ping', timeout=60)

    def _resume(self, physical_ended=None):
        state = self.state
        answered = {item.get('call_id') for item in state['items'] if item.get('type') == 'function_call_output'}
        for item in list(state['items']):
            call_id = item.get('call_id')
            if item.get('type') != 'function_call' or call_id in answered:
                continue
            if call_id == state.get('active_tool'):
                text = 'This call was interrupted when the harness stopped. ' + UNCERTAIN
            else:
                text = 'This call was not run: the harness stopped before it started.'
            state['items'].extend(context.tool_result_items(call_id, {'text': text, 'is_error': True}))
            state['handled_call_ids'].append(call_id)
            answered.add(call_id)
        state['pending_tools'], state['active_tool'] = [], None
        state['text_only'] = state['incomplete'] = 0
        state['failure_streak'], state['infrastructure_failures'] = None, 0
        self._event({'event': 'session_resumed', 'from_phase': state['phase']})
        notes, images = [RESUME_NOTE], []
        if physical_ended:
            notes.append(f'The harness stopped the physical episode ({physical_ended}). The robot is no longer '
                         'available; recorded frames and commands remain readable.')
            if state['stage'] == 'episode':
                state['outcome'] = {'ended_by': 'harness_stop', 'reason': physical_ended, 'summary': None,
                                    'self_assessment': None, 'model_call': state['calls'],
                                    'step': state['last_step']}
                if self.mode == 'development' and self.consolidation:
                    state['stage'] = 'consolidation'
                    notes.append(self._official_result(0) + self.consolidation)
                    self._event({'event': 'stage', 'stage': 'consolidation', 'ended_by': 'harness_stop'})
        elif self.robot:
            notes.append(ROBOT_KEPT)
        if self.robot:
            notes.append('Current state:')
            images = self._current_view(notes)
        state['items'].append(user_message(notes, images))
        state['phase'] = 'between_calls'
        self._save()

    def _stop(self, exc):
        if self.state is None or self.state.get('phase') in ('completed', 'paused'):
            return
        self.state['phase'] = 'needs_attention'
        self.state['error'] = {'type': type(exc).__name__, 'message': self._scrub(str(exc))[:2000],
                               'reason': getattr(exc, 'reason', 'interrupted' if isinstance(exc, KeyboardInterrupt)
                                                 else 'error'), 'time': now()}
        try:
            context.archive_window(self.archive_dir, self.state['items'], 'needs_attention', self.state['calls'])
        except Exception:
            pass
        self._save()
        self._event({'event': 'needs_attention', **self.state['error']})

    # ----------------------------------------------------------------- loop

    def _loop(self):
        state = self.state
        reminders = context.Reminders(self.mode, self.step_limit, state['reminders'])
        state['reminders'] = reminders.state
        while True:
            if self.max_model_calls and state['calls'] >= self.max_model_calls:
                raise NeedsAttention('model call limit reached', 'call_limit')
            before = context.image_files(state['items'])
            removed = context.evict_images(state['items'])
            if removed:
                state['stats']['images_evicted'] += removed
                self._event({'event': 'images_evicted', 'images': removed,
                             'files': context.evicted_files(before, context.image_files(state['items']))})
            if state['local_fallback'] and state['last_input_tokens'] >= self._threshold() > 0:
                self._compact_locally()
            response = self._request(state['items'])
            if response is None:
                continue
            outcome = self._handle(response, reminders)
            if outcome is not None:
                return outcome

    def _threshold(self):
        return int(self.config.get('compact_threshold') or 0)

    def _payload(self, items, compaction=True):
        config, state = self.config, self.state
        payload = {'model': config['model'], 'instructions': state['instructions'],
                   'input': context.materialize(items, self.images), 'tools': tools.schemas(state['profile']),
                   'parallel_tool_calls': False, 'store': False, 'include': ['reasoning.encrypted_content'],
                   'max_output_tokens': state['max_output_tokens']}
        summary = None if state.get('reasoning_summary_off') else config.get('reasoning_summary')
        reasoning = {key: value for key, value in (('effort', config.get('reasoning_effort')), ('summary', summary))
                     if value}
        if reasoning:
            payload['reasoning'] = reasoning
        if compaction and self._threshold() and not state['local_fallback']:
            payload['context_management'] = [{'type': 'compaction', 'compact_threshold': self._threshold()}]
        if config.get('prompt_cache_ttl'):
            payload['prompt_cache_options'] = {'ttl': config['prompt_cache_ttl']}
        if self.cache_key:
            payload['prompt_cache_key'] = self.cache_key
        return payload

    def _record_request(self, items, payload, kind='model'):
        """The request log (every call's input is traceable): requests/request_NNNN.json.gz
        written before the request is sent, NNNN = the model call it makes (the response file's number); an
        attempt repeated for the same call (an image withheld or recovered, a resumed session) adds _a2, _a3.
        The record keeps the items as the session holds them, images as file references (never their bytes),
        and every payload field except input and instructions, which the response file carries; the
        instructions' digest ties the two."""
        call = self.state['calls'] + 1
        attempt = self.request_attempts.get(call, 0) + 1
        folder = self.output / 'requests'
        while (folder / (f'request_{call:04d}' + (f'_a{attempt}' if attempt > 1 else '') + '.json.gz')).exists():
            attempt += 1
        self.request_attempts[call] = attempt
        instructions = payload.get('instructions') or ''
        record = {'time': now(), 'model_call': call, 'attempt': attempt, 'kind': kind,
                  'instructions_sha256': hashlib.sha256(instructions.encode('utf-8')).hexdigest(),
                  'instructions_chars': len(instructions),
                  'tools': [tool.get('name') for tool in payload.get('tools') or []],
                  **{key: value for key, value in payload.items() if key not in ('input', 'instructions', 'tools')},
                  'items': items}
        write_private_gzip(folder / (f'request_{call:04d}' + (f'_a{attempt}' if attempt > 1 else '') + '.json.gz'),
                           record)

    def _post(self, payload):
        started = time.monotonic()
        try:
            response = self.client.post(payload)
        except ModelAPIError as exc:
            if 403 not in self.wait_statuses and quota_refusal(exc):
                raise NeedsAttention('model quota exhausted (HTTP 403); stop this batch', 'quota') from exc
            raise
        finally:
            self.state['stats']['model_seconds'] += time.monotonic() - started
        self.state['calls'] += 1
        write_private(self.output / 'responses' / f"response_{self.state['calls']:04d}.json", self._scrub(response))
        self._save()
        return response

    def _drop_summary(self, exc):
        """A request refused for its reasoning summary: stop asking for summaries for the rest of the session and
        return True (the request is then repeated). Summaries are only logged, so an endpoint without them must not
        stop the episode. Reasoning items already received keep their summaries; an endpoint that refuses those as
        input too stops the session as before."""
        if (exc.status not in (400, 403, 422) or not self.config.get('reasoning_summary')
                or self.state.get('reasoning_summary_off') or not summary_refused(exc.detail)):
            return False
        self.state['reasoning_summary_off'] = True
        self._event({'event': 'reasoning_summary_refused', 'provider_error': exc.detail})
        self._save()
        return True

    def _api_wait(self, exc):
        """After the client's own retries: wait the next of api_waits for a transient failure (or a refusal, with
        wait_refusals) and return True (the request is then repeated, keeping the worker); False when it is not one
        of those or no wait is left."""
        if exc.status in self.wait_statuses and self.api_wait_index < len(self.api_waits):
            delay = self.api_waits[self.api_wait_index]
            self.api_wait_index += 1
            self._event({'event': 'model_api_wait', 'status': exc.status, 'code': (exc.detail or {}).get('code'),
                         'wait_s': delay, 'wait': self.api_wait_index, 'of': len(self.api_waits)})
            time.sleep(delay)
            return True
        if quota_refusal(exc):
            raise NeedsAttention('model quota exhausted (HTTP 403) after the waits', 'quota') from exc
        return False

    def _request(self, items):
        """One model request; None when it must be repeated (nothing was executed)."""
        state = self.state
        state['phase'] = 'requesting_model'
        self._save()
        try:
            payload = self._payload(items)
            self._record_request(items, payload)
            response = self._post(payload)
        except ModelAPIError as exc:
            if exc.status == 400 and image_count_refused(exc.detail):
                if self._recover_image_count(items):
                    return None
                raise NeedsAttention(f'model request refused for its number of images after recovery: {exc}',
                                     'model_request') from exc
            if exc.status == 400 and image_refused(exc.detail) and self._withhold_images(items):
                return None
            if self._drop_summary(exc):
                return None
            text = json.dumps(exc.detail or {}).lower()
            if (not state['local_fallback'] and exc.status == 400 and
                    ('context_management' in text or 'compaction' in text)):
                state['local_fallback'] = True
                self._event({'event': 'server_compaction_rejected', 'provider_error': exc.detail})
                self._save()
                return None
            if self._api_wait(exc):
                return None
            raise NeedsAttention(f'model request failed: {exc}',
                                 'model_api' if exc.status in TRANSIENT_STATUSES else 'model_request') from exc
        self.api_wait_index = 0
        status = response.get('status')
        if status != 'completed':
            self._account(response, [], [])
            reason = (response.get('incomplete_details') or {}).get('reason') or status
            state['incomplete'] += 1
            state['stats']['incomplete'] += 1
            self._event({'event': 'model_incomplete', 'model_call': state['calls'], 'status': status,
                         'reason': reason, 'attempt': state['incomplete']})
            if state['incomplete'] > INCOMPLETE_RETRIES:
                raise NeedsAttention(f'model response {status} ({reason}) {state["incomplete"]} times in a row; '
                                     'no tool was run', 'incomplete')
            if reason == 'max_output_tokens':
                state['max_output_tokens'] = min(MAX_OUTPUT_TOKENS_CAP, state['max_output_tokens'] * 2)
            self._save()
            return None
        state['incomplete'] = 0
        return response

    def _recover_image_count(self, items):
        """The provider refused the request for its number of images although the window stays under
        IMAGE_TOTAL_TRIGGER explicit images (an opaque server compaction item may carry more).
        Only the model request is repeated, no tool:
        first with images kept only in the newest messages; if none can be dropped, once more after a local
        compaction whose request carries no images, compacting locally from then on. False when both are used up."""
        state = self.state
        before = context.image_files(items)
        removed = context.evict_images(items, keep=1, trigger=0, total_trigger=0,
                                       total_keep=context.IMAGES_PER_RESULT)
        if removed:
            state['stats']['images_evicted'] += removed
            self._event({'event': 'image_count_refused', 'images_evicted': removed,
                         'files': context.evicted_files(before, context.image_files(items))})
            self._save()
            return True
        if state.get('image_count_rebuilt'):
            return False
        state['image_count_rebuilt'] = True
        self._event({'event': 'image_count_refused', 'rebuild': 'local compaction without images'})
        self._compact_locally(without_images=True)
        state['local_fallback'] = True
        self._save()
        return True

    def _withhold_images(self, items):
        """The provider refused an image: drop the images of the newest message still carrying any,
        and say so there. False when no image is left to drop."""
        for item in reversed(items):
            if context.has_images(item):
                count = context.image_count(item)
                item['content'] = [part for part in item['content'] if part.get('type') != 'input_image'] + [
                    {'type': 'input_text', 'text': f'[{count} image(s) above withheld: the image service refused them '
                                                   '(its content filter misreads some frames). The frames are '
                                                   'recorded: analyze them in code with rgb(frame_id, camera), or '
                                                   'look through another camera or a crop.]'}]
                self.state['stats']['images_withheld'] = self.state['stats'].get('images_withheld', 0) + count
                self._event({'event': 'images_withheld', 'images': count})
                self._save()
                return True
        return False

    def _handle(self, response, reminders):
        state = self.state
        outputs = response.get('output')
        if not isinstance(outputs, list):
            raise NeedsAttention('model response has no output list; no tool was run', 'protocol')
        unsupported = sorted({str(item.get('type')) for item in outputs} - set(OUTPUT_TYPES))
        if unsupported:
            raise NeedsAttention('unsupported model output items: ' + ', '.join(unsupported) +
                                 '; no tool was run', 'protocol')
        calls = [item for item in outputs if item.get('type') == 'function_call']
        ids = [call.get('call_id') for call in calls]
        if (any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids) or
                set(ids) & set(state['handled_call_ids'])):
            raise NeedsAttention('missing or repeated tool call IDs; no tool was run', 'protocol')
        texts = [text for text in (context.message_text(item) for item in outputs
                                   if item.get('type') == 'message') if text]
        # The new items and their pending calls are saved together, before any other I/O.
        state['items'].extend(outputs)
        state['pending_tools'] = ids
        state['phase'] = 'executing_tool' if calls else 'between_calls'
        self._save()
        self._account(response, calls, texts)
        compacted = any(item.get('type') == 'compaction' for item in outputs)
        if (state.get('compacted_before') and not state['local_fallback'] and
                state['last_input_tokens'] > self._threshold() * COMPACTION_RELIEF):
            state['local_fallback'] = True
            self._event({'event': 'server_compaction_ineffective', 'input_tokens': state['last_input_tokens']})
        state['compacted_before'] = compacted
        if compacted:
            archive_id = context.archive_window(self.archive_dir, state['items'], 'server_compaction',
                                                state['calls'])
            state['items'], _ = context.cut_at_latest_compaction(state['items'], outputs)
            state['stats']['compactions_server'] += 1
            self._event({'event': 'context_compacted', 'mode': 'server', 'archive': archive_id,
                         'input_tokens': state['last_input_tokens']})
        elif (self._threshold() and not state['local_fallback'] and
              state['last_input_tokens'] > min(self._threshold() * FALLBACK_OVERSHOOT, LONG_CONTEXT_TOKENS)):
            state['local_fallback'] = True
            self._event({'event': 'server_compaction_missing', 'input_tokens': state['last_input_tokens']})
        self._save()
        notes = []
        if calls:
            state['text_only'] = 0
            for call in calls:
                reply = self._execute(call)
                notes.extend(text for text in reminders.after_tool(call.get('name'), reply, state['plan']) if text)
            self._stuck()
        else:
            state['text_only'] += 1
            state['stats']['text_only'] += 1
            self._event({'event': 'text_only_reply', 'count': state['text_only']})
            if state['text_only'] >= 2:
                return self._second_text_only(texts)
            notes.append(TEXT_ONLY[(state['stage'], self.mode)])
        return self._after_round(notes, compacted)

    def _after_round(self, notes, compacted):
        state = self.state
        if state['end_requested']:
            return self._complete()
        if state['stage'] == 'episode' and not state['outcome'] and state.get('simulator_failure'):
            # A broken or vanished simulator ends the physical episode cleanly.
            state['outcome'] = {'ended_by': 'simulator_failure', 'reason': state['simulator_failure'], 'summary': None,
                                'self_assessment': None, 'model_call': state['calls'], 'step': state['last_step']}
            self._event({'event': 'simulator_failure', 'reason': state['simulator_failure']})
        if state['stage'] == 'episode' and state['outcome']:
            if self.mode != 'development' or not self.consolidation:
                return self._complete()
            state['stage'] = 'consolidation'
            ended_by = state['outcome']['ended_by']
            if ended_by == 'simulator_failure':
                ended = SIMULATOR_FAILED.format(reason=state['outcome']['reason']) + ' ' + self._official_result(0)
            elif ended_by == 'repeated_failure':
                ended = (REPEATED_ENDED.format(tool=state['outcome']['tool'], count=state['outcome']['count']) + ' '
                         + self._official_result())
            else:
                ended = (('You called finish.' if ended_by == 'finish' else 'The environment ended the episode.')
                         + ' The robot can no longer move, but python_exec, observe and inspect still work. '
                         + self._official_result())
            notes.append(ended + self.consolidation)
            self._event({'event': 'stage', 'stage': 'consolidation', 'ended_by': ended_by})
        images = []
        if compacted:
            notes[:0] = self._reattach_texts()
            if self.robot:
                notes.append('Current state:')
                images = self._current_view(notes)
        if notes:
            state['items'].append(user_message(notes, images))
            self._event({'event': 'notes', 'texts': [text[:200] for text in notes]})
        state['phase'] = 'between_calls'
        self._save()
        return None

    def _official_result(self, *wait_s):
        """The development episode's official result, as a sentence to start the consolidation."""
        hook = self.hooks.get('official_result')
        line = hook(*wait_s) if hook else None
        self._event({'event': 'official_result', 'shown': line})
        return line + ' ' if line else ''

    def _second_text_only(self, texts):
        state = self.state
        text = '\n\n'.join(texts)[:4000] or '(no text)'
        if state['stage'] == 'episode' and self.mode == 'evaluation':
            try:
                reply = self.worker.call('finish', timeout=TOOL_TIMEOUT_S)
            except WorkerError as exc:
                raise NeedsAttention(f'finishing after a text-only reply failed: {exc}', 'worker') from exc
            state['outcome'] = {'ended_by': 'text', 'summary': text, 'self_assessment': 'unclear',
                                'model_call': state['calls'], 'step': reply.get('last_step')}
            return self._complete()
        if state['stage'] == 'session':
            state['end'] = {'summary': text, 'ended_by': 'text'}
            return self._complete()
        state['phase'] = 'paused'
        self._save()
        self._event({'event': 'paused', 'reason': 'text_only', 'stage': state['stage']})
        return self._result('paused')

    # ---------------------------------------------------------------- tools

    def _execute(self, call):
        state = self.state
        name, call_id = call.get('name'), call['call_id']
        state['active_tool'] = call_id
        self._save()
        self._event({'event': 'tool_started', 'call_id': call_id, 'name': name})
        started = time.monotonic()
        arguments = None
        try:
            try:
                arguments = json.loads(call.get('arguments') or '{}')
            except ValueError:
                raise ValueError('arguments are not valid JSON') from None
            if name not in tools.PROFILES[state['profile']]:
                raise ValueError(f'{name} is not available in this session')
            arguments = tools.validate(name, arguments)
            reply = self._dispatch(name, arguments)
            self.worker_failures = 0
        except ValueError as exc:
            reply = {'text': 'Invalid tool request: ' + str(exc), 'is_error': True}
        except RequestError as exc:
            reply = {'text': f'{name} failed: {exc}', 'is_error': True}
        except WorkerError as exc:
            reply = self._recover_worker(name, exc)
        self._note_failure(name, arguments, reply)
        state['items'].extend(context.tool_result_items(call_id, reply))
        state['handled_call_ids'].append(call_id)
        state['pending_tools'].remove(call_id)
        state['active_tool'] = None
        self._record(name, arguments, reply)
        self._save()
        self._event({'event': 'tool_completed', 'call_id': call_id, 'name': name,
                     'seconds': round(time.monotonic() - started, 3), 'is_error': bool(reply.get('is_error')),
                     'images': len(reply.get('images') or []), 'actions': len(reply.get('actions') or []),
                     'experiences': len(reply.get('experiences') or []), 'last_step': reply.get('last_step')})
        return reply

    def _note_failure(self, name, arguments, reply):
        """Track stuck calls (REPEATED_FAILURES, INFRASTRUCTURE_FAILURES) and, while the robot is in use, ask the
        episode whether the simulator has broken or vanished."""
        state = self.state
        text, failed = reply.get('text', ''), bool(reply.get('is_error'))
        # Failures of the harness or the simulator, as opposed to the model's own (an argument the tool refused, its
        # code): finish/observe failing on the robot, or python_exec failing on the service's transport.
        infrastructure = failed and (
            (name in INFRASTRUCTURE_TOOLS and not text.startswith('Invalid tool request'))
            or (name == 'python_exec' and any(error in text for error in TRANSPORT_ERRORS)))
        if failed and name != 'update_plan':  # a bookkeeping tool never ends an attempt
            signature = failure_signature(name, text, arguments)
            streak = state.get('failure_streak') or {}
            count = streak['count'] + 1 if streak.get('signature') == signature else 1
            state['failure_streak'] = {'tool': name, 'signature': signature, 'count': count, 'text': text[:500],
                                       'infrastructure': infrastructure}
        elif not failed:
            state['failure_streak'] = None
        if infrastructure:
            state['infrastructure_failures'] = state.get('infrastructure_failures', 0) + 1
        elif not failed and (name in INFRASTRUCTURE_TOOLS or reply.get('actions')):  # the robot answered
            state['infrastructure_failures'] = 0
        check = self.hooks.get('simulator_failure')
        if check is not None and self.robot and state['stage'] == 'episode' and not state.get('simulator_failure'):
            try:
                reason = check(failed)
            except Exception:  # the check must never break the session
                reason = None
            if reason:
                state['simulator_failure'] = reason

    def _stuck(self):
        """Act on a stuck call (see REPEATED_FAILURES): stop the session (needs_attention), end the physical episode,
        or end a curation or video session. Returns None; the round then continues as usual."""
        state = self.state
        streak = state.get('failure_streak') or {}
        repeated = streak.get('count', 0) >= REPEATED_FAILURES
        infrastructure_count = state.get('infrastructure_failures', 0)
        if state['end_requested'] or not (repeated or infrastructure_count >= INFRASTRUCTURE_FAILURES):
            return None
        tool, count = (streak['tool'], streak['count']) if repeated else ('the robot service', infrastructure_count)
        message = (f"{tool} failed the same way {count} times in a row: {streak.get('text', '')[:300]}" if repeated
                   else f"the robot service failed {count} times without a working reply in between: "
                        f"{streak.get('text', '')[:300]}")
        self._event({'event': 'repeated_failure', 'tool': tool, 'count': count, 'message': message})
        state['failure_streak'], state['infrastructure_failures'] = None, 0
        if state['stage'] == 'episode' and (state['outcome'] or state.get('simulator_failure')):
            return None  # the episode ended in this round anyway (an ended evaluation keeps its result)
        if state['stage'] == 'session':
            state['end'] = {'summary': message, 'ended_by': 'repeated_failure'}
            state['end_requested'] = True
            return None
        if not repeated or streak.get('infrastructure') or state['stage'] == 'consolidation':
            raise NeedsAttention(message, 'repeated_failure')
        # The model's own calls keep failing the same way: its attempt ends as it stands (an evaluation is scored).
        # If the robot cannot even finish, the harness is at fault: stop instead (an infrastructure failure, not the
        # model's outcome; README, Failures and restarts).
        if self.robot:
            try:
                self.worker.call('finish', timeout=TOOL_TIMEOUT_S)
            except WorkerError as exc:
                raise NeedsAttention(f'{message}; finishing the attempt failed too: {exc}'[:2000],
                                     'repeated_failure') from exc
        state['outcome'] = {'ended_by': 'repeated_failure', 'reason': message, 'tool': tool, 'count': count,
                            'summary': None, 'self_assessment': None, 'model_call': state['calls'],
                            'step': state['last_step']}
        return None

    def _dispatch(self, name, args):
        state = self.state
        if name == 'python_exec':
            return self.worker.python_exec(args['code'], args['timeout_s'])
        if name in ('observe', 'inspect'):
            return self.worker.call(name, args, timeout=TOOL_TIMEOUT_S)
        if name == 'update_plan':
            state['plan'] = args['plan']
            if self.workspace is not None:
                (self.workspace / 'PLAN.json').write_text(json.dumps(args, ensure_ascii=False, indent=1))
            return {'text': 'Plan updated.'}
        if name == 'finish':
            if state['stage'] != 'episode':
                raise ValueError('the physical episode has already ended')
            reply = self.worker.call('finish', timeout=TOOL_TIMEOUT_S)
            state['outcome'] = {'ended_by': 'finish', 'summary': args['summary'],
                                'self_assessment': args['self_assessment'], 'model_call': state['calls'],
                                'step': reply.get('last_step')}
            return reply
        if name == 'plan_next_episode':
            if state['stage'] != 'consolidation':
                raise ValueError('plan_next_episode is available after the physical episode has ended; '
                                 'call finish first')
            review = self.hooks.get('end_review')
            if args['decision'] == 'end' and review is not None and not state.get('end_review'):
                # Readiness review: the first end of a session is answered with
                # the episode list and the review questions, and nothing is decided; the next plan_next_episode call
                # decides (end again confirms). Once per session, so it can never loop. The first end's arguments
                # are kept with the review.
                state['end_review'] = {'shown_at_call': state['calls'], 'time': now(), 'outcome': None,
                                       'first_end': args, 'text': None}
                try:
                    text = review(state)
                except Exception as exc:
                    text = REVIEW_FALLBACK
                    self._event({'event': 'end_review_failed', 'error': f'{type(exc).__name__}: {exc}'[:300]})
                state['end_review']['text'] = text
                self._event({'event': 'end_review', 'shown_at_call': state['calls']})
                return {'text': text}
            hook = self.hooks.get('plan_next_episode')
            text = hook(args) if hook else 'Decision recorded.'
            if state.get('end_review') and state['end_review'].get('outcome') is None:
                state['end_review']['outcome'] = 'end confirmed' if args['decision'] == 'end' else args['decision']
                self._event({'event': 'end_review_outcome', 'outcome': state['end_review']['outcome']})
            state['decision'], state['end_requested'] = args, True
            return {'text': text}
        if name == 'end_session':
            state['end'] = {'summary': args['summary'], 'ended_by': 'end_session'}
            state['end_requested'] = True
            return {'text': 'Session ended.'}
        raise ValueError(f'unknown tool {name}')

    def _record(self, name, arguments, reply):
        state, stats = self.state, self.state['stats']
        stats['tools'][name] = stats['tools'].get(name, 0) + 1
        stats['tool_errors'] += bool(reply.get('is_error'))
        if reply.get('actions'):
            stats['action_calls'] += 1
            stats['actions'] += len(reply['actions'])
        stats['experiences'] += len(reply.get('experiences') or [])
        stats['skills_opened'] += len(reply.get('skills_opened') or [])
        stats['knowledge_changes'] += len(reply.get('knowledge_changes') or [])
        if name == 'python_exec' and isinstance(arguments, dict):
            code = arguments.get('code', '')
            stats['functions_defined'] += len(re.findall(r'^\s*def ', code, re.M))
            stats['image_code_calls'] += bool(re.search(r'\brgb\(|display_image\(|Image\.|cv2\.|np\.asarray|'
                                                        r'b64decode', code))
        for skill in reply.get('skills_opened') or []:
            state['opened_skills'][skill['name']] = skill['text']
        if reply.get('history_reads') and state['stage'] == 'episode':
            # Reads of history/ during the physical episode: an attempt with none, on a new layout and without
            # replay, is "fresh". A session resumed from a checkpoint made
            # before this was counted has no entry, and its episode stays "unrecorded".
            if 'history_reads_in_episode' in stats:
                stats['history_reads_in_episode'] += reply['history_reads']
        if reply.get('last_step') is not None:
            state['last_step'] = reply['last_step']
        if reply.get('episode_ended') and state['stage'] == 'episode' and not state['outcome']:
            state['outcome'] = {'ended_by': 'environment', 'summary': None, 'self_assessment': None,
                                'model_call': state['calls'], 'step': reply.get('last_step')}

    def _recover_worker(self, name, exc):
        self.worker_failures += 1
        self._event({'event': 'worker_failed', 'name': name, 'error': str(exc)[:500],
                     'consecutive': self.worker_failures})
        try:
            self.worker.close()
        except Exception:
            pass
        self.worker = None
        if self.worker_failures > WORKER_RESTARTS:
            raise NeedsAttention('the Python worker failed repeatedly', 'worker') from exc
        try:
            self._start_worker()
        except Exception as error:
            raise NeedsAttention('the Python worker could not be restarted', 'worker') from error
        self.state['stats']['worker_restarts'] += 1
        restarted = ('It has been restarted: variables, functions and imports from earlier calls are gone; files '
                     'in your workspace remain.')
        if not getattr(exc, 'delivered', True):
            return {'text': f'The Python worker had stopped, so this {name} call was not run ({exc}). ' + restarted,
                    'is_error': True}
        return {'text': f'The Python worker failed during this {name} call ({exc}). ' + restarted + ' ' + UNCERTAIN,
                'is_error': True}

    # ----------------------------------------------------------- compaction

    def _current_view(self, notes):
        """Append the current state line to notes and return current images (no physical steps)."""
        try:
            reply = self.worker.call('status', timeout=TOOL_TIMEOUT_S)
        except RequestError as exc:
            notes.append(f'(current state unavailable: {exc})')
            return []
        except WorkerError as exc:
            notes.append(f'(current state unavailable: {exc}; the Python worker was restarted and its variables '
                         'are gone)')
            try:
                self.worker.close()
            except Exception:
                pass
            self.worker = None
            try:
                self._start_worker()
            except Exception as error:
                raise NeedsAttention('the Python worker could not be restarted', 'worker') from error
            self.state['stats']['worker_restarts'] += 1
            return []
        notes.append(reply.get('text', ''))
        return reply.get('images') or []

    def _reattach_texts(self):
        state = self.state
        texts = ['Earlier context was compacted; the full earlier window is archived in .context_archives/.']
        plan = context.plan_text(state['plan'])
        if plan:
            texts.append(plan)
        for name, text in state['opened_skills'].items():
            texts.append(f'Skill {name} (opened earlier):\n{text}')
        names = self._defined_names()
        if names:
            texts.append(NAMES_NOTE.format(names=names))
        return texts

    def _defined_names(self):
        """What model code defined in the worker (it survives a compaction), or '' when the worker cannot say."""
        if self.worker is None:
            return ''
        try:
            return self.worker.call('names', timeout=60).get('text') or ''
        except (WorkerError, RequestError):
            return ''

    def _compact_locally(self, without_images=False):
        """Fallback when server compaction is unavailable: a summary written by the model, then a rebuilt window.
        `without_images` sends the summary request with every image replaced by its omission note."""
        state = self.state
        archive_id = context.archive_window(self.archive_dir, state['items'], 'local_compaction', state['calls'])
        window = state['items']
        if without_images:
            window = copy.deepcopy(window)
            context.strip_images(window)
        request = window + [user_message([context.read_prompt('compaction.md')])]
        payload = self._payload(request, compaction=False)
        payload['tool_choice'] = 'none'
        state['phase'] = 'requesting_model'
        self._save()
        while True:
            try:
                self._record_request(request, payload, kind='local_compaction')
                response = self._post(payload)
                break
            except ModelAPIError as exc:
                if self._drop_summary(exc):
                    payload = self._payload(request, compaction=False)
                    payload['tool_choice'] = 'none'
                    continue
                if self._api_wait(exc):
                    continue
                raise NeedsAttention(f'local compaction request failed: {exc}',
                                     'model_api' if exc.status in TRANSIENT_STATUSES else 'model_request') from exc
        self.api_wait_index = 0
        summary = '\n\n'.join(text for text in (context.message_text(item) for item in response.get('output') or []
                                                 if item.get('type') == 'message') if text)
        self._account(response, [], [])
        if response.get('status') != 'completed' or not summary.strip():
            raise NeedsAttention('local compaction produced no summary', 'compaction')
        transcript = context.recent_transcript(state['items'], TRANSCRIPT_CHARS)
        items = [user_message([f'Earlier context was compacted locally (archived as {archive_id} in '
                               '.context_archives/). Your handoff summary:\n\n' + summary])]
        if transcript:
            items.append(user_message(['Your most recent tool calls and outputs, already executed (do not repeat '
                                       'them):\n\n' + transcript]))
        notes = self._reattach_texts()[1:]
        images = []
        if self.robot:
            notes.append('Current state:')
            images = self._current_view(notes)
        if notes or images:
            items.append(user_message(notes, images))
        state['items'] = items
        state['last_input_tokens'] = 0
        state['stats']['compactions_local'] += 1
        state['phase'] = 'between_calls'
        self._save()
        self._event({'event': 'context_compacted', 'mode': 'local', 'archive': archive_id,
                     'summary_chars': len(summary), 'transcript_chars': len(transcript)})

    # ---------------------------------------------------------- bookkeeping

    def _account(self, response, calls, texts):
        state = self.state
        usage = response.get('usage') or {}
        values = {'input_tokens': usage.get('input_tokens') or 0,
                  'cached_tokens': (usage.get('input_tokens_details') or {}).get('cached_tokens') or 0,
                  'cache_write_tokens': (usage.get('input_tokens_details') or {}).get('cache_write_tokens') or 0,
                  'output_tokens': usage.get('output_tokens') or 0,
                  'reasoning_tokens': (usage.get('output_tokens_details') or {}).get('reasoning_tokens') or 0}
        for key, value in values.items():
            state['usage'][key] = state['usage'].get(key, 0) + value
        state['last_input_tokens'] = values['input_tokens']
        for text in texts:
            append_private(self.output / 'messages.jsonl', {'time': now(), 'model_call': state['calls'],
                                                            'stage': state['stage'], 'text': text})
        self._event({'event': 'model_response', 'model_call': state['calls'], 'response_id': response.get('id'),
                     'usage': values, 'tools': [call.get('name') for call in calls],
                     'output_types': [item.get('type') for item in response.get('output') or []],
                     'message_chars': sum(len(text) for text in texts)})

    def _save(self):
        write_private(self.state_path, self.state)

    def _event(self, record):
        append_private(self.events_path, {'time': now(), **self._scrub(record)})

    def _scrub(self, value):
        return scrub_endpoint(value, self.config['base_url']) if self.redact_endpoint else value

    def _result(self, status):
        state = self.state
        return {'status': status, 'stage': state['stage'], 'outcome': state['outcome'],
                'decision': state['decision'], 'end': state['end'], 'end_review': state.get('end_review'),
                'model_calls': state['calls'],
                'usage': state['usage'], 'stats': state['stats']}

    def _complete(self):
        state = self.state
        state['phase'], state['finished'] = 'completed', now()
        context.archive_window(self.archive_dir, state['items'], 'session_end', state['calls'])
        self._save()
        result = self._result('completed')
        write_private(self.output / 'session_summary.json', result)
        self._event({'event': 'session_completed', 'outcome': state['outcome'], 'decision': state['decision']})
        return result

from __future__ import annotations
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
from email.utils import parsedate_to_datetime
from ..runtime.io_utils import atomic, read_json
from .tool_images import has_pending_server_tool, lift_tool_result_images
from .archive import finalize_attempt
from .history import DEFAULT_CONTEXT_MAX_BYTES, fingerprint


class APIError(RuntimeError):
    """Transport or unusable-generation failure; reason selects the recovery path."""
    def __init__(self, reason, detail='', *, output_limit_reached=False):
        self.reason = reason
        self.output_limit_reached = output_limit_reached
        super().__init__(reason + (': '+detail if detail else ''))


def output_limit_reached(response):
    """Only explicit provider output-limit markers qualify, never token estimates."""
    details = response.get('incomplete_details')
    return (response.get('stop_reason') == 'max_tokens' or
            (response.get('status') == 'incomplete' and isinstance(details, dict) and
             details.get('reason') == 'max_output_tokens'))


def retry_delay(headers, attempt, now):
    value = next((v for k, v in headers.items() if k.lower() == 'retry-after'), None)
    try:
        delay = float(value)
    except (TypeError, ValueError):
        try:
            delay = parsedate_to_datetime(value).timestamp()-now
        except (TypeError, ValueError, OverflowError, IndexError):
            delay = min(30., 5.*2**min(attempt, 3))
    return max(1., delay) if math.isfinite(delay) else 30.


def context_rejection(result):
    """Known deterministic payload failures; retain legacy retries for generic gateway 400s."""
    error = result.get('error', {})
    message = str(error.get('message', '') if isinstance(error, dict) else error).lower()
    return (('signature' in message and ('thinking' in message or 'invalid' in message))
            or 'prefix_binding_mismatch' in message
            or 'bound to a different conversation' in message
            or any(term in message for term in ('prompt is too long', 'context length exceeded', 'too many images'))
            or (any(term in message for term in ('tool_result', 'tool_use'))
                and any(term in message for term in ('must', 'without', 'not found', 'unexpected', 'matching', 'missing'))))


def image_count(value):
    if isinstance(value, dict):
        return int(value.get('type') in ('image', 'input_image')) + sum(image_count(v) for v in value.values())
    if isinstance(value, list):
        return sum(image_count(v) for v in value)
    return 0


def image_content(path, provider):
    kind = 'image/png' if Path(path).suffix == '.png' else 'image/jpeg'
    data = base64.b64encode(Path(path).read_bytes()).decode()
    if provider == 'anthropic':
        return {'type': 'image', 'source': {'type': 'base64', 'media_type': kind, 'data': data}}
    return {'type': 'input_image', 'image_url': f'data:{kind};base64,{data}'}


def text_content(text, provider):
    return {'type': 'text' if provider == 'anthropic' else 'input_text', 'text': text}


def thinking_summary(response, requested_display, http_status):
    """Readable API-returned summaries only; never reconstruct or alter signed blocks."""
    blocks, redacted = [], []
    content = response.get('content')
    for index, block in enumerate(content if isinstance(content, list) else []):
        if not isinstance(block, dict):
            continue
        if block.get('type') == 'thinking':
            blocks.append({'content_index': index, 'text': block.get('thinking', ''),
                           'signature_present': bool(block.get('signature'))})
        elif block.get('type') == 'redacted_thinking':
            redacted.append(index)
    readable = sum(bool(b['text']) for b in blocks)
    if readable:
        status = 'available' if readable == len(blocks) and not redacted else 'partial'
    elif http_status != 200:
        status = 'response_unavailable'
    elif blocks:
        status = 'empty_thinking'
    elif redacted:
        status = 'redacted_only'
    else:
        status = 'no_thinking_block'
    usage = response.get('usage') if isinstance(response.get('usage'), dict) else {}
    details = usage.get('output_tokens_details')
    return {'source': 'response.json#/content', 'requested_display': requested_display,
            'status': status, 'blocks': blocks, 'redacted_content_indices': redacted,
            'reported_thinking_tokens': details.get('thinking_tokens') if isinstance(details, dict) else None,
            'note': 'API-returned summaries, not full internal reasoning. Missing text is not reconstructed; '
                    'no thinking block does not establish that no reasoning occurred. '
                    'Original signed blocks remain unchanged in response.json and the complete conversation archive. '
                    'Active planning context may omit historical blocks after a window change.'}


class API:
    def __init__(self, root, config):
        self.root, self.config = Path(root), config
        self.provider = config['provider']
        if self.provider not in ('anthropic', 'openai'):
            raise ValueError('Unknown API provider')
        if self.provider == 'openai' and config.get('stream', False) is not False:
            raise ValueError('OpenAI Responses transport requires stream=false')
        self.tool_image_mode = config.get('anthropic_tool_image_mode', 'top_level')
        if self.tool_image_mode not in ('top_level', 'nested'):
            raise ValueError('anthropic_tool_image_mode must be top_level or nested')
        self.model = config['model']
        if any(config.get(k) for k in ('api_accounting', 'api_routes', 'api_pool_state',
                                       'api_global_concurrency', 'api_requests_per_minute')):
            raise ValueError('This package supports a single API credential without shared accounting or pooling.')
        self.http_400_max_attempts = config.get('http_400_max_attempts', 15)
        if type(self.http_400_max_attempts) is not int or not 1 <= self.http_400_max_attempts <= 15:
            raise ValueError('http_400_max_attempts must be an integer from 1 to 15 (including first request)')
        self.index = 0
        self.cache_anchor = None
        self._cache_anchor_digest = None
        self.context_max_bytes = config.get('context_max_bytes', DEFAULT_CONTEXT_MAX_BYTES)
        if type(self.context_max_bytes) is not int or self.context_max_bytes <= 0:
            raise ValueError('context_max_bytes must be a positive integer')

    @staticmethod
    def _cache_prefix(messages, point):
        i, j = point
        return messages[:i] + [{**messages[i], 'content': messages[i]['content'][:j+1]}]

    def payload(self, system, messages, tools, checkpoint=False, history=None):
        if checkpoint and self.provider == 'anthropic' and 'opus' not in self.model.lower():
            raise ValueError('Checkpoint review requires the configured Anthropic Opus model')
        effort = 'low' if checkpoint else 'xhigh'
        tokens = 1536 if checkpoint else 49152
        if self.provider == 'anthropic':
            value = {'model': self.model, 'max_tokens': tokens, 'output_config': {'effort': effort},
                    'thinking': {'type': 'adaptive', 'display': 'summarized'}, 'system': system, 'messages': messages,
                    'metadata': {},
                    'tools': [{'name': t['name'], 'description': t['description'], 'input_schema': t['schema']} for t in tools],
                    'stream': self.config.get('stream', True)}
            if self.config.get('prompt_cache', True):
                control = {'type': 'ephemeral', 'ttl': self.config.get('cache_ttl', '1h')}
                # The configured gateway demonstrably ignored automatic top-level
                # caching. Explicit block breakpoints produced actual cache reads.
                value['system'] = [{'type':'text', 'text':system, 'cache_control':control.copy()}]
                value['messages'] = copy.deepcopy(messages)
                eligible = [(i,j) for i,m in enumerate(messages)
                            if isinstance(m.get('content'),list)
                            for j,b in enumerate(m['content'])
                            if b.get('type') in ('text','image','document','tool_use','tool_result')
                            and (b.get('type') != 'text' or b.get('text'))]
                if eligible:
                    current = eligible[-1]
                    previous = None if checkpoint else self.cache_anchor
                    # A numeric index alone is unsafe after trimming or rewriting.
                    if previous not in eligible or (previous is not None and
                            fingerprint(self._cache_prefix(messages, previous)) != self._cache_anchor_digest):
                        previous = None
                    first_end = history.first_round_end() if history is not None and not checkpoint else None
                    first = next((point for point in reversed(eligible)
                                  if first_end is not None and point[0] <= first_end), None)
                    # Four slots total: system, immutable first round, previous write,
                    # and current tail. The tail writes newly appended input for reuse.
                    for i,j in {first, previous, current} - {None}:
                        value['messages'][i]['content'][j]['cache_control'] = control.copy()
                    if not checkpoint:
                        self.cache_anchor = current
                        self._cache_anchor_digest = fingerprint(self._cache_prefix(messages, current))
            return value
        return {'model': self.model, 'store': False, 'stream': False, 'reasoning': {'effort': effort},
                'include': ['reasoning.encrypted_content'], 'max_output_tokens': tokens,
                'instructions': system, 'input': messages, 'parallel_tool_calls': False,
                'tools': [{'type': 'function', 'name': t['name'], 'description': t['description'],
                           'parameters': t['schema'], 'strict': False} for t in tools]}

    def call(self, payload, req):
        default_env = self.config.get('api_key_env') or ('ANTHROPIC_AUTH_TOKEN' if self.provider == 'anthropic' else 'OPENAI_API_KEY')
        previous_index = self.index
        try:
            return self._call(payload, req, default_env)
        finally:
            if self.index > previous_index:
                for folder in sorted((self.root/'api').glob(f'{self.index:05d}_*/attempt_*')):
                    finalize_attempt(folder)

    def _call(self, payload, req, key_env):
        self.index += 1
        parent = self.root/'api'/f'{self.index:05d}_{req["kind"]}_{req["observation_id"]}'
        deadline = req['deadline']-1
        retry_deadline = None
        attempt = 0
        saw_http_400 = False
        request_images = image_count(payload)
        while True:
            self.check_alive(deadline)
            if retry_deadline is not None and time.time() >= retry_deadline:
                raise APIError('api_retry_window_exhausted', 'raw attempts archived')
            folder = parent/f'attempt_{attempt}'
            folder.mkdir(parents=True)
            request_limit = float(self.config.get('api_request_timeout_s', 1200.))
            timeout = min(request_limit, deadline-time.time(),
                          (retry_deadline-time.time()) if retry_deadline is not None else request_limit)
            if not os.environ.get(key_env):
                raise RuntimeError(f'Missing {key_env}; no API call made')
            url = self.config['base_url'].rstrip('/') + ('/v1/messages' if self.provider == 'anthropic' else '/responses')
            atomic(folder/'request.json', payload)
            if req.get('context_selection') is not None:
                atomic(folder/'context_selection.json', req['context_selection'])
            atomic(folder/'transport.json', {'provider': self.provider, 'url': url, 'key_env': key_env,
                'timeout': timeout, 'idle_timeout': min(timeout, float(self.config.get('api_idle_timeout_s', 180.))),
                'request_headers': {'Content-Type': 'application/json',
                    **({'anthropic-version': '2023-06-01'} if self.provider == 'anthropic' else {})},
                'authentication': 'Authorization: Bearer <credential from key_env; never archived>'})
            # Bound the exact archived request before attempting transport.
            encoded_size = (folder/'request.json').stat().st_size
            call_id = f'{req["run_id"]}/{self.index}/{attempt}/{uuid.uuid4().hex}'
            meta = {**req, 'call_id': call_id, 'requested_model': self.model, 'provider': self.provider,
                    'attempt': attempt, 'retry_deadline': retry_deadline,
                    'api_route_id': 'default', 'started_unix_s': time.time(),
                    'request_sha256': hashlib.sha256((folder/'request.json').read_bytes()).hexdigest(),
                    'request_bytes': encoded_size, 'request_image_count': request_images,
                    'context_max_bytes': self.context_max_bytes,
                    'message_cache_breakpoints': [[i,j] for i,m in enumerate(payload.get('messages', []))
                        if isinstance(m.get('content'),list) for j,b in enumerate(m['content'])
                        if 'cache_control' in b],
                    'http_400_max_attempts': self.http_400_max_attempts}
            if self.provider == 'anthropic':
                # Start with an explicit unavailable record, including for hard timeouts.
                self.record_thinking(folder, payload, req, call_id, {}, None, meta)
            self.save_meta(folder, meta)
            if encoded_size > self.context_max_bytes:
                meta.update(error='api_context_size_limit', retryable=False, request_sent=False,
                            context_max_bytes=self.context_max_bytes)
                self.save_meta(folder, meta)
                raise APIError('api_context_size_limit',
                    'request exceeds configured byte limit; archived without sending or editing the tool chain')
            started = time.monotonic()
            with (folder/'transport.stderr').open('w') as stderr:
                child = subprocess.Popen([sys.executable, str(Path(__file__).with_name('api_worker.py')), str(folder)],
                                         stdout=subprocess.DEVNULL, stderr=stderr)
                hard_timeout = False
                try:
                    while child.poll() is None:
                        self.check_alive(deadline)
                        if time.monotonic()-started >= timeout:
                            hard_timeout = True
                            break
                        time.sleep(min(.25, max(.01, timeout-(time.monotonic()-started))))
                except APIError as error:
                    meta.update(seconds=time.monotonic()-started, error=error.reason)
                    self.save_meta(folder, meta)
                    raise
                finally:
                    if child.poll() is None:
                        child.kill(); child.wait()
            http = read_json(folder/'http.json', {})
            result = read_json(folder/'response.json', {})
            if not isinstance(result, dict):
                result = {}
            partial = read_json(folder/'partial_response.json', {})
            observed = result or (partial if isinstance(partial, dict) else {})
            if hard_timeout:
                http = {**http, 'http_status': None, 'error': 'hard_request_deadline',
                        'seconds': time.monotonic()-started}
            usage = observed.get('usage') if isinstance(observed.get('usage'), dict) else {}
            meta.update(**http, usage=usage, returned_model=observed.get('model'),
                        output_limit_reached=output_limit_reached(observed))
            if self.provider == 'anthropic':
                self.record_thinking(folder, payload, req, call_id, observed, http.get('http_status'), meta,
                                     source='partial_response.json' if not result and partial else 'response.json')
            self.save_meta(folder, meta)
            generation_error = None
            if http.get('http_status') in (None, 200) and (
                    http.get('stream_error_type') == 'generation_incomplete' or
                    (http.get('error') and (observed.get('stop_reason') == 'max_tokens' or
                                           observed.get('status') == 'incomplete'))):
                generation_error = http.get('error') or 'incomplete generated response'
            if http.get('http_status') == 200 and not http.get('error'):
                if not result or not isinstance(result.get('content' if self.provider=='anthropic' else 'output'), list):
                    http['error'] = 'incomplete_api_response'
                    meta['error'] = http['error']
                    if result.get('model') or result.get('type') in ('message', 'response'):
                        generation_error = http['error']
                elif result.get('model') != self.model:
                    meta.update(error='api_model_identity_mismatch', retryable=False)
                    self.save_meta(folder, meta)
                    raise APIError('api_model_identity_mismatch', 'returned model does not match requested model')
                else:
                    try:
                        self.validate_envelope(result)
                    except (TypeError, ValueError, KeyError) as error:
                        http['error'] = 'invalid_response_envelope: '+str(error)
                        meta['error'] = http['error']
                        generation_error = http['error']
                    else:
                        return result
            if generation_error is not None:
                meta.update(error='api_generation_incomplete', generation_error_detail=generation_error,
                            retryable=False, attempts_used=attempt+1)
                self.save_meta(folder, meta)
                raise APIError('api_generation_incomplete', 'raw response archived; no partial tools executed',
                               output_limit_reached=output_limit_reached(observed))
            if http.get('http_status') in (400, 422) and context_rejection(result):
                meta.update(error='api_context_invalid', retryable=False, attempts_used=attempt+1)
                self.save_meta(folder, meta)
                raise APIError('api_context_invalid', 'deterministic context/signature rejection; raw request and response archived')
            saw_http_400 |= http.get('http_status') == 400
            # Once a 400 occurs, the entire logical request has a single wire-attempt
            # budget, including attempts before it. Mixed network errors cannot reset it.
            if saw_http_400 and attempt + 1 >= self.http_400_max_attempts:
                meta.update(error='api_http_400_attempts_exhausted', attempts_used=attempt+1)
                self.save_meta(folder, meta)
                raise APIError('api_http_400_attempts_exhausted', f'{attempt+1} total attempts; raw responses archived')
            retry_statuses = (None, 200, 400, 408, 429, 500, 502, 503, 504, 529)
            if http.get('http_status') not in retry_statuses or http.get('stream_error_type') in ('authentication_error','permission_error','invalid_request_error','not_found_error'):
                raise APIError('api_permanent_error', f'HTTP {http.get("http_status")}; raw response archived')
            if retry_deadline is None and http.get('http_status') != 400:
                retry_deadline = min(deadline, time.time()+float(self.config.get('network_recovery_window_s', 600.)))
            delay = retry_delay(http.get('headers') or {}, attempt, time.time())
            wake = min(time.time()+delay, retry_deadline or deadline)
            meta.update(retry_deadline=retry_deadline, retry_wait_seconds=max(0., wake-time.time()),
                        retry_reason=http.get('error') or f'HTTP {http.get("http_status")}')
            self.save_meta(folder, meta)
            while time.time() < wake:
                self.check_alive(deadline)
                time.sleep(min(.5, wake-time.time()))
            attempt += 1

    def save_meta(self, folder, meta):
        meta['accounting_enabled'] = False
        atomic(folder/'meta.json', meta)

    def check_alive(self, deadline):
        if (self.root/'END.json').exists() or (self.root/'STOP').exists():
            raise APIError('api_episode_cancelled')
        if time.time() >= deadline:
            raise APIError('api_request_deadline')

    def record_thinking(self, folder, payload, req, call_id, result, http_status, meta, source='response.json'):
        summary = thinking_summary(result, payload.get('thinking', {}).get('display'), http_status)
        if source != 'response.json':
            summary['source'] = source+'#/content'
            summary['note'] = ('Partial API response retained in '+source+' and the raw wire archive; '
                               'not appended to planning history or used to execute tools.')
        summary.update({k: req[k] for k in ('run_id', 'task', 'request_id', 'observation_id', 'kind') if k in req})
        summary['call_id'] = call_id
        atomic(folder/'thinking_summary.json', summary)
        meta['thinking_summary_status'] = summary['status']

    def validate_envelope(self, response):
        if self.provider != 'anthropic':
            for block in response['output']:
                if not isinstance(block, dict):
                    raise ValueError('output item must be an object')
                if block.get('type') == 'function_call':
                    if not isinstance(json.loads(block['arguments']), dict):
                        raise ValueError('tool arguments must encode an object')
            return
        ids = set()
        for block in response['content']:
            if not isinstance(block, dict) or not isinstance(block.get('type'), str):
                raise ValueError('content block requires type')
            if block['type'] == 'tool_use':
                call_id, name = block.get('id'), block.get('name')
                if not isinstance(call_id,str) or not call_id or call_id in ids or not isinstance(name,str) or not name:
                    raise ValueError('tool requires unique ID and name')
                ids.add(call_id)
                if not isinstance(block.get('input'),dict):
                    raise ValueError('tool input must be an object')

    def calls(self, response):
        if self.provider == 'anthropic':
            return [{'id': b['id'], 'name': b['name'], 'input': b['input']} for b in response.get('content', []) if b.get('type') == 'tool_use']
        return [{'id': b['call_id'], 'name': b['name'], 'input': json.loads(b['arguments'])}
                for b in response.get('output', []) if b.get('type') == 'function_call']

    def append_response(self, messages, response):
        if self.provider == 'anthropic':
            messages.append({'role': 'assistant', 'content': response['content']})
        else:
            messages.extend(response['output'])  # preserve all reasoning items and tool-call IDs

    def append_results(self, messages, results, *, image_context=None):
        if self.provider == 'anthropic':
            blocks = []
            for result in results:
                i, content = result[:2]
                block = {'type': 'tool_result', 'tool_use_id': i, 'content': content}
                if len(result) > 2 and result[2]:
                    block['is_error'] = True
                blocks.append(block)
            if self.tool_image_mode == 'top_level' and not has_pending_server_tool(messages):
                blocks = lift_tool_result_images(blocks, image_context)
            messages.append({'role': 'user', 'content': blocks})
        else:
            messages.extend({'type': 'function_call_output', 'call_id': r[0], 'output': r[1]} for r in results)

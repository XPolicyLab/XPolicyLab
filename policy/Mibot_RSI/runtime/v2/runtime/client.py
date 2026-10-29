"""Responses API configuration, credentials and HTTP client for the v2 harness.

Standard library only. This module talks to the model endpoint and nothing
else; it never touches the robot, so retrying a model request can never
replay a physical action.
"""
from __future__ import annotations

import email.utils
import hashlib
import http.client
import json
import os
from pathlib import Path
import random
import re
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

HARNESS_ROOT = Path(__file__).resolve().parents[2]

DEFAULTS = {
    'base_url': 'https://api.openai.com/v1',
    'model': 'gpt-6-astra',
    'api_key_env': 'OPENAI_API_KEY',
    'reasoning_effort': 'high',
    'request_timeout_s': 300,
    'api_retries': 6,
    'api_retry_base_delay_s': 2,
    'api_retry_max_delay_s': 60,
    'api_retry_total_timeout_s': 0,
    'max_output_tokens': 32768,
    'compact_threshold': 200000,
    # Development actors compact later: close to, but below, the provider's 272k long-context step.
    # Evaluation, curation and video sessions keep compact_threshold.
    'compact_threshold_development': 260000,
    'prompt_cache_ttl': '30m',
}
# Accepted when given, and absent from the config otherwise, so configs without them keep their hash.
# reasoning_summary asks the provider for readable summaries of the model's reasoning (the reasoning itself is
# only returned encrypted); it does not change the request's reasoning.
OPTIONAL_KEYS = frozenset({'reasoning_summary'})
# v1 credential/config files may carry these; v2 accepts and ignores them.
V1_ONLY_KEYS = frozenset({'max_model_calls', 'compact_after_input_tokens'})
INTEGER_KEYS = ('request_timeout_s', 'api_retries', 'api_retry_base_delay_s',
                'api_retry_max_delay_s', 'api_retry_total_timeout_s', 'max_output_tokens',
                'compact_threshold', 'compact_threshold_development')
REASONING_EFFORTS = ('low', 'medium', 'high', 'xhigh', 'max')
REASONING_SUMMARIES = ('auto', 'concise', 'detailed')


PERMANENT_CODES = ('invalid_request_error', 'context_length_exceeded', 'model_not_found', 'unsupported_parameter',
                   'unknown_parameter', 'content_policy_violation')


def inner_error(detail):
    """The provider's own error object when a gateway wraps it as JSON text in `param` (or {})."""
    param = (detail or {}).get('param')
    if not isinstance(param, str) or not param.lstrip().startswith('{'):
        return {}
    try:
        value = json.loads(param)
    except ValueError:
        return {}
    error = value.get('error') if isinstance(value, dict) else None
    return error if isinstance(error, dict) else {}


def image_refused(detail):
    """Whether the provider refused an image in the request (a content filter, sometimes a false positive)."""
    detail = detail or {}
    inner = inner_error(detail)
    text = ' '.join(str(value) for value in (detail.get('message'), inner.get('message'))).lower()
    return (inner.get('code') == 'content_policy_violation' or detail.get('code') == 'content_policy_violation') \
        and 'image' in text


class ModelAPIError(RuntimeError):
    """A model request that did not produce a response object.

    `status` is the final HTTP status (None for transport failures) and
    `detail` is the provider's error type/code/param/message with any
    credential removed.
    """

    def __init__(self, message, status=None, detail=None, attempts=0):
        super().__init__(message)
        self.status = status
        self.detail = detail or {}
        self.attempts = attempts


def credential_file(alias, directory=None):
    """Resolve a credential alias; the file must live outside the harness and be 0600."""
    if not isinstance(alias, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', alias):
        raise ValueError('Invalid key alias')
    root = Path(directory or Path.home() / '.config' / 'robodojo' / 'credentials').expanduser().resolve()
    path = root / (alias + '.json')
    if path.resolve().is_relative_to(HARNESS_ROOT):
        raise ValueError('Credential alias must live outside the harness repository')
    if path.is_symlink() or not path.is_file():
        raise ValueError('Credential alias does not name a regular file')
    info = path.stat()
    if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError('Credential file must be owned by the current user and mode 0600')
    return path


def load_config(path, credential_path=None, require_key=True, overrides=None):
    """Read a JSON config (and optional credential file) into a validated dict."""
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError('Configuration must be a JSON object')
    unknown = set(value) - set(DEFAULTS) - OPTIONAL_KEYS - V1_ONLY_KEYS - {'api_key'}
    if unknown:
        raise ValueError('Unknown configuration keys: ' + ', '.join(sorted(unknown)))
    config = dict(DEFAULTS)
    config.update({k: v for k, v in value.items() if k in DEFAULTS or k in OPTIONAL_KEYS})
    if overrides:
        unknown = set(overrides) - set(DEFAULTS) - OPTIONAL_KEYS
        if unknown:
            raise ValueError('Unknown override keys: ' + ', '.join(sorted(unknown)))
        config.update(overrides)
    for key in ('base_url', 'model', 'api_key_env'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError('Invalid ' + key)
    url = urllib.parse.urlsplit(config['base_url'])
    if (url.scheme not in ('http', 'https') or not url.netloc or url.username or
            url.password or url.query or url.fragment):
        raise ValueError('base_url must be an HTTP(S) API base without credentials or query')
    for key in INTEGER_KEYS:
        if type(config[key]) is not int or config[key] < 0:
            raise ValueError('Invalid ' + key)
    if not config['request_timeout_s'] or not config['api_retry_max_delay_s'] or not config['max_output_tokens']:
        raise ValueError('Timeout, retry delay cap and output tokens must be positive')
    if config['reasoning_effort'] is not None and config['reasoning_effort'] not in REASONING_EFFORTS:
        raise ValueError('reasoning_effort must be one of ' + ', '.join(REASONING_EFFORTS) + ' or null')
    if config.get('reasoning_summary') is None:
        config.pop('reasoning_summary', None)
    elif config['reasoning_summary'] not in REASONING_SUMMARIES:
        raise ValueError('reasoning_summary must be one of ' + ', '.join(REASONING_SUMMARIES) + ' or null')
    ttl = config['prompt_cache_ttl']
    if ttl is not None and (not isinstance(ttl, str) or not re.fullmatch(r'[1-9][0-9]*[mh]', ttl)):
        raise ValueError('prompt_cache_ttl must be like "30m" or null')
    if credential_path is not None:
        credential = json.loads(Path(credential_path).read_text())
        if not isinstance(credential, dict) or not isinstance(credential.get('api_key'), str):
            raise ValueError('Credential file must contain an api_key string')
        secret = credential['api_key']
    else:
        secret = value.get('api_key') or os.environ.get(config['api_key_env'], '')
    if not isinstance(secret, str):
        raise ValueError('api_key must be a string')
    if require_key and (not secret.strip() or secret.startswith('YOUR_')):
        raise ValueError('No API key: fill api_key in the private config or set ' + config['api_key_env'])
    config['api_key'] = secret
    return config


def public_config(config):
    return {k: v for k, v in config.items() if k != 'api_key'}


def config_hash(config):
    text = json.dumps(public_config(config), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(text.encode()).hexdigest()


def endpoint(config):
    url = config['base_url'].rstrip('/')
    return url if url.endswith('/responses') else url + '/responses'


def bypass_proxy(config):
    """Add the endpoint host to NO_PROXY; the model endpoint must not go through a local proxy."""
    host = urllib.parse.urlsplit(config['base_url']).hostname
    for name in ('NO_PROXY', 'no_proxy'):
        current = [item for item in os.environ.get(name, '').split(',') if item]
        if host not in current:
            os.environ[name] = ','.join(current + [host])
    return host


def redact(text, secret):
    text = str(text)
    if secret:
        text = text.replace(secret, '[REDACTED_KEY]')
    return re.sub(r'sk-[A-Za-z0-9_-]{8,}', '[REDACTED_KEY]', text)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        # Never forward the Authorization header to another location.
        return None


class ResponsesClient:
    RETRYABLE_HTTP = (408, 409, 425, 429, 500, 502, 503, 504)
    # Some API gateways wrap transient upstream failures as HTTP 400.
    # Retry those a bounded number of times; explicit request errors fail fast.
    MAX_BAD_REQUEST_RETRIES = 2

    def __init__(self, config, retry_observer=None, sleeper=time.sleep, random_value=random.random,
                 wall_time=time.time, monotonic=time.monotonic):
        self.config = config
        self.opener = urllib.request.build_opener(NoRedirect)
        self.retry_observer = retry_observer
        self.sleeper = sleeper
        self.random_value = random_value
        self.wall_time = wall_time
        self.monotonic = monotonic

    def _retry_after(self, headers):
        value = headers.get('Retry-After') if headers is not None else None
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except (TypeError, ValueError):
            try:
                target = email.utils.parsedate_to_datetime(value)
                return max(0.0, target.timestamp() - self.wall_time())
            except (TypeError, ValueError, OverflowError):
                return None

    def _delay(self, attempt, headers=None):
        maximum = float(self.config['api_retry_max_delay_s'])
        requested = self._retry_after(headers)
        if requested is not None:
            return min(requested, maximum), 'retry-after'
        base = float(self.config['api_retry_base_delay_s']) * (2 ** attempt)
        return min(base * (0.75 + 0.5 * self.random_value()), maximum), 'exponential-jitter'

    def _error_detail(self, exc):
        try:
            value = json.loads(exc.read(65536).decode('utf-8', errors='replace'))
            value = value.get('error', value) if isinstance(value, dict) else None
        except Exception:
            return {}
        if isinstance(value, str):
            return {'message': redact(value, self.config.get('api_key', ''))[:1000]}
        if not isinstance(value, dict):
            return {}
        return {key: redact(value[key], self.config.get('api_key', ''))[:1000]
                for key in ('type', 'code', 'param', 'message') if value.get(key) is not None}

    @staticmethod
    def _permanent(detail):
        inner = inner_error(detail)
        return (detail.get('type') == 'invalid_request_error' or
                detail.get('code') in PERMANENT_CODES or
                detail.get('param') == '404 NOT_FOUND' or
                # A gateway may wrap the provider's error as JSON text in `param`.
                inner.get('type') == 'invalid_request_error' or inner.get('code') in PERMANENT_CODES)

    def _observe(self, detail):
        if self.retry_observer:
            self.retry_observer(detail)

    def post(self, payload):
        """POST a Responses request and return the decoded response object."""
        data = json.dumps(payload, allow_nan=False).encode()
        total = self.config['api_retry_total_timeout_s']
        deadline = self.monotonic() + total if total else None
        retries = self.config['api_retries']
        bad_request_retries = 0
        for attempt in range(retries + 1):
            remaining = deadline - self.monotonic() if deadline is not None else None
            if remaining is not None and remaining <= 0:
                raise ModelAPIError('Model API retry time budget exhausted', attempts=attempt)
            request = urllib.request.Request(endpoint(self.config), data=data, headers={
                'Content-Type': 'application/json',
                'Authorization': 'Bearer ' + self.config['api_key']})
            timeout = self.config['request_timeout_s']
            if remaining is not None:
                timeout = min(timeout, remaining)
            detail = None
            try:
                with self.opener.open(request, timeout=max(0.001, timeout)) as response:
                    value = json.load(response)
                if not isinstance(value, dict):
                    raise ValueError('Response must be a JSON object')
                return value
            except urllib.error.HTTPError as exc:
                status, headers = exc.code, exc.headers
                detail = self._error_detail(exc)
                exc.close()
                retryable = status in self.RETRYABLE_HTTP or (
                    status == 400 and not self._permanent(detail) and
                    bad_request_retries < self.MAX_BAD_REQUEST_RETRIES)
                if not retryable or attempt == retries:
                    self._observe({'event': 'model_request_failed', 'attempts': attempt + 1,
                                   'status': status, 'provider_error': detail})
                    raise ModelAPIError(f'Model API HTTP {status} after {attempt + 1} attempt(s)',
                                        status=status, detail=detail, attempts=attempt + 1) from None
                if status == 400:
                    bad_request_retries += 1
                delay, source = self._delay(attempt, headers)
                reason = f'Model API HTTP {status}'
            except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                    http.client.HTTPException) as exc:
                if attempt == retries:
                    self._observe({'event': 'model_request_failed', 'attempts': attempt + 1,
                                   'status': None, 'reason': type(exc).__name__})
                    raise ModelAPIError(f'Model API connection failed after {attempt + 1} attempt(s): '
                                        f'{type(exc).__name__}', attempts=attempt + 1) from None
                delay, source = self._delay(attempt)
                reason = 'Model API transport or response error'
            if deadline is not None:
                remaining = deadline - self.monotonic()
                if remaining <= 0:
                    raise ModelAPIError('Model API retry time budget exhausted', attempts=attempt + 1)
                delay = min(delay, remaining)
            event = {'event': 'model_request_retry', 'retry': attempt + 1, 'max_retries': retries,
                     'reason': reason, 'delay_s': round(delay, 3), 'delay_source': source}
            if detail:
                event['provider_error'] = detail
            self._observe(event)
            self.sleeper(delay)
        raise AssertionError('unreachable')

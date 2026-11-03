"""Portable launch configuration; credentials stay in the process environment."""
import json
import os
from pathlib import Path
import sys
import uuid
from urllib.parse import urlsplit

POLICY = Path(__file__).resolve().parents[1]
XPL_ROOT = POLICY.parents[1]
DEFAULT_PREFIX = sys.prefix


def resolve_output_path(value):
    """Resolve output paths against the checkout, independent of the launch cwd."""
    path = Path(value).expanduser()
    return (path if path.is_absolute() else XPL_ROOT/path).resolve()


def create_run_directory(explicit=None):
    """Reserve a fresh run directory; never reuse an existing archive."""
    selected = explicit or os.environ.get('Q0_PREVIEW_RUN_DIR')
    if selected:
        root = resolve_output_path(selected)
    else:
        base = os.environ.get('Q0_PREVIEW_RUN_ROOT') or XPL_ROOT/'runs'/POLICY.name
        root = resolve_output_path(base)/uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    return root


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def config(prefix=None, template=None):
    cfg = json.loads(Path(template or POLICY/'config/agent.opus55.json').read_text())
    if any(k in cfg for k in ('api_key', 'token', 'password', 'headers')):
        raise ValueError('Use an environment variable for API credentials, never configuration literals')
    cfg['base_url'] = os.environ.get('Q0_PREVIEW_API_BASE_URL') or cfg.get('base_url', '')
    url = urlsplit(cfg['base_url'])
    if url.scheme != 'https' or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError('Set Q0_PREVIEW_API_BASE_URL to a credential-free HTTPS endpoint base URL')
    cfg['python_prefix'] = str(Path(prefix or cfg.get('python_prefix') or sys.prefix).resolve())
    if not (Path(cfg['python_prefix'])/'bin/python').is_file():
        raise ValueError('Agent Python prefix must contain bin/python')
    if any(cfg.get(k) for k in ('calibration_mode', 'api_routes', 'api_accounting',
                               'api_pool_state', 'api_global_concurrency', 'api_requests_per_minute')):
        raise ValueError('Development calibration, API pooling and accounting are not supported.')
    cfg['api_key_env'] = cfg.get('api_key_env', 'Q0_PREVIEW_API_KEY')
    if not cfg['api_key_env'].isidentifier():
        raise ValueError('Invalid API credential environment variable name')
    if cfg.get('sandbox', 'docker') not in ('docker', 'bwrap', 'bwrap-sudo'):
        raise ValueError('An isolated Python sandbox is required')
    return cfg

"""Bind VAP launch roles to one canonical evaluation request."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys


def canonical_request(arguments):
    if len(arguments) < 6:
        raise ValueError('Expected BENCH TASK CKPT ENV_CFG ACTION SEED')
    policy_dir = Path(os.environ.get('VAP_POLICY_DIR', Path.cwd())).expanduser().resolve()
    checkpoint = Path(arguments[2]).expanduser()
    # The official server starts in policy/vap while its client starts at the
    # benchmark root. Match the standard resolver for path-valued and bare names.
    is_path = checkpoint.is_absolute() or '/' in arguments[2] or bool(os.altsep and os.altsep in arguments[2])
    if is_path:
        if not checkpoint.is_absolute():
            checkpoint = policy_dir / checkpoint
        checkpoint = checkpoint.resolve()
    else:
        run_name = '-'.join((arguments[0], arguments[2], arguments[3], arguments[4], arguments[5]))
        combined = (policy_dir / 'checkpoints' / run_name).resolve()
        verbatim = (policy_dir / 'checkpoints' / checkpoint).resolve()
        checkpoint = combined if combined.exists() else verbatim
    checkpoint = str(checkpoint)
    return {
        'schema': 1,
        'bench': arguments[0],
        'task': arguments[1],
        'checkpoint': checkpoint,
        'env_cfg_type': arguments[3],
        'action_type': arguments[4],
        'seed': arguments[5],
        'mode': os.environ.get('EVAL_ENV_TYPE', 'sim'),
        'eval_num': os.environ.get('EVAL_NUM', 'native'),
        'debug_obs_encoded': os.environ.get('DEBUG_OBS_ENCODED', '0'),
    }


def serialize(request):
    return json.dumps(request, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def server_request(request):
    # A batch server serves several tasks and is not passed the client's eval count.
    return {key: value for key, value in request.items() if key not in {'task', 'eval_num'}}


def run_id(request):
    task = re.sub(r'[^A-Za-z0-9_-]+', '_', request['task'])
    seed = re.sub(r'[^A-Za-z0-9_-]+', '_', request['seed'])
    digest = hashlib.sha256(serialize(request).encode()).hexdigest()
    return f"vap_{task}_seed{seed}_{digest}"


def main():
    if not sys.argv[1:]:
        raise ValueError('Expected BENCH TASK CKPT ENV_CFG ACTION SEED')
    write = sys.argv[1] == '--write'
    validate = sys.argv[1] == '--validate'
    if write or validate:
        target = Path(sys.argv[2])
        arguments = sys.argv[3:]
    else:
        target = None
        arguments = sys.argv[2:] if sys.argv[1] in {'--shell', '--request'} else sys.argv[1:]
    request = canonical_request(arguments)
    if write or validate or sys.argv[1] == '--request':
        payload = serialize(server_request(request))
    else:
        payload = serialize(request)
    if write:
        # The directory is exclusively reserved by common.sh before this write.
        with (target / '.vap_request.json').open('x', encoding='utf-8') as stream:
            stream.write(payload + '\n')
        return
    if validate:
        try:
            stored = (target / '.vap_request.json').read_bytes()
        except OSError as error:
            raise ValueError(f'Missing VAP request signature: {target}') from error
        if stored.decode('utf-8') != payload + '\n':
            raise ValueError(f'VAP request signature mismatch: {target}')
        return
    if sys.argv[1] == '--shell':
        print('export ROBODOJO_RUN_ID=' + shlex.quote(run_id(request)))
        return
    print(payload)


if __name__ == '__main__':
    try:
        main()
    except (OSError, UnicodeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)

"""Mibot-RSI deployment loop: one RoboDojo evaluation episode per call, run by the Mibot-RSI agent.

The agent runs inside the RoboDojo evaluation client, like the organisers' GPT_6_Astra_Direct_EEF entry: this
module starts the harness's robot service in this process and the model session in a subprocess, and the
policy server (model.py) serves nothing. The work is done by the harness entry v2.submission; this file only
finds the harness runtime and reads deploy.yml. It diverges from demo_policy/deploy.py because the actions come
from a closed-loop agent, not from a policy-server action chunk (README, Evaluation).

The harness runtime is $MIBOT_RSI_HARNESS (a directory) when set, else runtime/ next to this file: an eval-only
export of one Mibot-RSI commit, whose runtime/EXPORT.json lists every file's sha256 (the runtime refuses to
run when a file differs).
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

POLICY_DIR = Path(__file__).resolve().parent


def _harness_root():
    harness = os.environ.get('MIBOT_RSI_HARNESS')
    root = Path(harness).expanduser() if harness else POLICY_DIR / 'runtime'
    # runtime/ must carry its EXPORT.json: the runtime checks every file against it (MIBOT_RSI_HARNESS may name
    # another copy of the runtime instead).
    if (root / 'v2' / 'submission.py').is_file() and (harness or (root / 'EXPORT.json').is_file()):
        return root.resolve()
    _stop(f'[Mibot-RSI] no harness runtime (with its EXPORT.json) at {root} (README, Installation)')


def _stop(message):
    """Leave the evaluation process (rc 1: a new process would fail the same way). An Exception would make RoboDojo
    skip every layout in turn, and a SystemExit leaves Isaac Sim's shutdown hanging."""
    print(message, file=sys.stderr, flush=True)
    sys.stdout.flush()
    os._exit(1)


def _submission():
    try:
        return _load_submission()
    except Exception as exc:  # an Exception would make RoboDojo skip every layout in turn
        _stop(f'[Mibot-RSI] cannot load the harness runtime: {type(exc).__name__}: {exc}')


def _load_submission():
    root = _harness_root()
    loaded = sys.modules.get('v2')
    if loaded is not None and Path(getattr(loaded, '__file__', '') or '').resolve().parent.parent != root:
        _stop(f'[Mibot-RSI] another module named v2 is already loaded ({loaded.__file__})')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from v2 import submission
    return submission


def _deploy_cfg():
    try:
        import yaml
        with open(POLICY_DIR / 'deploy.yml', encoding='utf-8') as stream:
            return yaml.safe_load(stream) or {}
    except Exception as exc:
        _stop(f'[Mibot-RSI] cannot read deploy.yml: {type(exc).__name__}: {exc}')


def eval_one_episode(TASK_ENV, model_client):
    _submission().eval_one_episode(TASK_ENV, model_client, policy_dir=POLICY_DIR, deploy_cfg=_deploy_cfg())


def eval_one_episode_batch(TASK_ENV, model_client):
    _submission().eval_one_episode_batch(TASK_ENV, model_client, policy_dir=POLICY_DIR, deploy_cfg=_deploy_cfg())

"""Validate official results, including explicitly reported unstable layouts."""
import math
import re
import hashlib
import json
from pathlib import Path


def verify_source(root, expected_digest):
    root = Path(root)
    files = {str(path.relative_to(root)): path.read_text()
             for path in root.rglob('*') if path.is_file() and path.suffix in ('.py','.md','.json','.yml')}
    actual = hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest()
    if actual != expected_digest:
        raise ValueError('Immutable source digest mismatch')
    return actual


def audit_result(manifest, official, log_text):
    requested = set(manifest['layouts'])
    episodes = list(official['details'].values())
    actual = {row['layout_id'] for row in episodes}
    if len(actual) != len(episodes) or not actual <= requested:
        raise ValueError('Unexpected or duplicate official layouts')
    missing = sorted(requested - actual)
    summaries = re.findall(r'Success nums:\s*(\d+),\s*Fail nums:\s*(\d+),\s*Unstable nums:\s*(\d+)', log_text)
    if missing:
        if not summaries or int(summaries[-1][2]) != len(missing):
            raise ValueError('Missing layouts lack official unstable evidence')
        if sum(map(int, summaries[-1][:2])) != len(episodes):
            raise ValueError('Official terminal episode counts mismatch')
    if official['eval_time'] != len(episodes):
        raise ValueError('Official evaluation count mismatch')
    for row in episodes:
        if type(row['success']) is not bool or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1:
            raise ValueError('Invalid official episode')
    if episodes:
        expected_success = sum(row['success'] for row in episodes) / len(episodes)
        expected_score = 100 * sum(row['score'] for row in episodes) / len(episodes)
        if abs(official['success_rate'] - expected_success) > .001 or abs(official['score'] - expected_score) > .001:
            raise ValueError('Official aggregate mismatch')
    return {'excluded_layouts': missing, 'exclusion_reason': 'Official unstable environment exclusion' if missing else None,
            'official_terminal_counts': list(map(int, summaries[-1])) if summaries else None}

#!/usr/bin/env python3
"""Verify the 50-task RoboTwin 100-episode result counts."""
import json
from pathlib import Path


def main():
    path = Path(__file__).resolve().parent / 'results' / 'robotwin_c2r_100ep.json'
    result = json.loads(path.read_text())
    rows = result['per_task']
    assert len(rows) == 100
    assert len({r['task'] for r in rows}) == 50
    assert len({(r['task'], r['condition']) for r in rows}) == 100
    counts = {}
    for condition in ('clean', 'random'):
        group = [r for r in rows if r['condition'] == condition]
        assert len(group) == 50 and all(r['episodes'] == 100 for r in group)
        counts[condition] = sum(r['successes'] for r in group)
        assert counts[condition] == result['counts'][condition]['successes']
    assert counts == {'clean': 4408, 'random': 4239}
    assert result['counts']['combined'] == {'successes': 8647, 'episodes': 10000}
    print('clean=88.16 random=84.78 overall=86.47 (8647/10000)')


if __name__ == '__main__':
    main()

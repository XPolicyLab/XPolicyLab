"""Read current RGB/proprioception requests and write one controller-authored call."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import time
from client import atomic_json, validate_reply


def pending(root):
    sessions = []
    for folder in (root / 'sessions').glob('*'):
        if (folder / 'episode-ended.json').exists():
            continue
        descriptor = folder / 'session.json'
        if descriptor.is_file():
            sessions.append((json.loads(descriptor.read_text(encoding='utf-8'))['created_at'], folder))
    for _, folder in sorted(sessions):
        for request in sorted(folder.glob('request-*.json')):
            if not request.with_name(request.name.replace('request-', 'reply-')).exists():
                value = json.loads(request.read_text(encoding='utf-8'))
                value['images'] = [str(folder / name) for name in value['images']]
                return value
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream-root', default=os.environ.get('CYPHER_TEMPRE_STREAM_ROOT'))
    sub = parser.add_subparsers(dest='command', required=True)
    observation = sub.add_parser('observe')
    observation.add_argument('--wait', type=float, default=20)
    reply = sub.add_parser('act')
    reply.add_argument('session')
    reply.add_argument('turn', type=int)
    reply.add_argument('reply_file', type=Path)
    sub.add_parser('episodes')
    args = parser.parse_args()
    if not args.stream_root:
        parser.error('Set CYPHER_TEMPRE_STREAM_ROOT or --stream-root.')
    root = Path(args.stream_root).resolve()
    if args.command == 'observe':
        if not 0 <= args.wait <= 30:
            parser.error('Wait must be within0..30seconds.')
        deadline = time.monotonic() + args.wait
        while True:
            value = pending(root)
            if value:
                print(json.dumps(value, ensure_ascii=False))
                return
            if (root / 'stop-agent.json').exists():
                print(json.dumps({'status': 'stream_stopped'}))
                return
            if time.monotonic() >= deadline:
                print(json.dumps({'status': 'waiting', 'same_agent_continues': True}))
                return
            time.sleep(.2)
    if args.command == 'episodes':
        print(json.dumps([json.loads(p.read_text(encoding='utf-8'))
                          for p in sorted((root / 'sessions').glob('*/episode-ended.json'))]))
        return
    if not re.fullmatch('[0-9a-f]{32}', args.session) or args.turn < 1:
        parser.error('Invalid session or turn.')
    folder = root / 'sessions' / args.session
    if (folder / 'episode-ended.json').exists():
        raise RuntimeError('This robot episode has ended; no stale action is relayed.')
    request = json.loads((folder / f'request-{args.turn:03}.json').read_text(encoding='utf-8'))
    reply = validate_reply(json.loads(args.reply_file.read_text(encoding='utf-8-sig')), request['tools'])
    target = folder / f'reply-{args.turn:03}.json'
    if target.exists():
        raise FileExistsError('A reply already exists; no replacement or duplicate action.')
    atomic_json(target, reply)
    print(json.dumps({'accepted_by_transport': True, 'session': args.session, 'turn': args.turn,
                      'native_arrival_not_yet_observed': True}))


if __name__ == '__main__':
    main()

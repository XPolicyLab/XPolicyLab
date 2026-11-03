"""Read-only archive verification and context/SSE reconstruction; no API or robot calls."""
import argparse
import copy
import importlib
import json
from pathlib import Path
import sys

POLICY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POLICY.parents[2]))
PACKAGE = 'XPolicyLab.policy.'+POLICY.name+'.harness.'
archive = importlib.import_module(PACKAGE+'archive')
history = importlib.import_module(PACKAGE+'history')
MessageStream = importlib.import_module(PACKAGE+'api_worker').MessageStream


def without_cache(messages):
    value = copy.deepcopy(messages)
    for message in value:
        if isinstance(message.get('content'), list):
            for block in message['content']:
                block.pop('cache_control', None)
    return value


def reconstruct_sse(path):
    parser, data = MessageStream(), []
    for line in path.read_text().splitlines()+['']:
        if not line:
            if data:
                parser.feed(json.loads('\n'.join(data)))
                data.clear()
                if parser.complete:
                    return parser.snapshot()
        elif line.startswith('data:'):
            data.append(line[5:].lstrip(' '))
    raise ValueError('SSE has no complete message_stop')


def audit_session(root):
    root = Path(root)
    state_path = root/'conversation_state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    messages = state.get('messages')
    prefix_path = root/'planning_prefix.json'
    prefix = json.loads(prefix_path.read_text()) if prefix_path.exists() else None
    if prefix is not None:
        prefix_digest = history.fingerprint({k: prefix[k] for k in ('system', 'tools')})
        if prefix.get('sha256', prefix_digest) != prefix_digest:
            raise ValueError('frozen planning prefix digest mismatch')
    if (root/'history_journal').exists():
        messages, rounds = archive.read_history_journal(root/'history_journal')
        # A crash may leave the final snapshot behind the committed journal.
        if state.get('messages', []) != messages[:len(state.get('messages', []))]:
            raise ValueError('conversation snapshot disagrees with committed journal')
    summary = {'attempts': 0, 'integrity_verified': 0, 'legacy_or_unfinalized': 0,
               'contexts_reconstructed': 0, 'sse_responses_reconstructed': 0,
               'uncommitted_history_records': len(list((root/'history_journal').glob('*.pending'))),
               'response_statuses': {}, 'warnings': []}
    for folder in sorted((root/'api').glob('*/attempt_*')):
        summary['attempts'] += 1
        if (folder/'archive_manifest.json').exists():
            manifest = archive.verify_attempt(folder)
            if manifest['request_matches_worker'] is False:
                raise ValueError('worker sent a different request: '+str(folder))
            status = manifest['response_status']
            summary['integrity_verified'] += 1
        else:
            status = 'legacy_or_unfinalized'
            summary['legacy_or_unfinalized'] += 1
        summary['response_statuses'][status] = summary['response_statuses'].get(status, 0)+1
        request_path = folder/'request.json'
        if not request_path.exists():
            summary['warnings'].append(str(folder.relative_to(root))+': no submitted request body')
            continue
        request = json.loads(request_path.read_text())
        for name in ('transport_worker.json', 'meta.json'):
            if (folder/name).exists():
                saved = json.loads((folder/name).read_text()).get('request_sha256')
                if saved and archive.file_record(request_path)['sha256'] != saved:
                    raise ValueError('request digest mismatch: '+str(folder))
        selection_path = folder/'context_selection.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection.get('mode') in ('first_recent', 'full'):
                if prefix is not None:
                    if selection.get('planning_prefix_sha256', prefix_digest) != prefix_digest:
                        raise ValueError('planning prefix attribution mismatch: '+str(folder))
                    if 'messages' in request:
                        system = request['system']
                        if isinstance(system, list):
                            system = '\n'.join(block['text'] for block in system)
                        tools = [{'name':t['name'], 'description':t['description'], 'input_schema':t['schema']}
                                 for t in prefix['tools']]
                    else:
                        system = request['instructions']
                        tools = [{'type':'function', 'name':t['name'], 'description':t['description'],
                                  'parameters':t['schema'], 'strict':False} for t in prefix['tools']]
                    if system != prefix['system'] or request['tools'] != tools:
                        raise ValueError('request differs from frozen planning prefix: '+str(folder))
                if messages is None:
                    summary['warnings'].append(str(folder.relative_to(root))+': history unavailable; exact request still retained')
                else:
                    rebuilt = history.reconstruct_active(messages, selection)
                    if rebuilt != without_cache(request.get('messages', request.get('input', []))):
                        raise ValueError('selected history differs from submitted context: '+str(folder))
                    summary['contexts_reconstructed'] += 1
        response_path = folder/'response.json'
        capture_path = folder/'wire_capture.json'
        redacted = (json.loads(capture_path.read_text()).get('credential_redactions', 0)
                    if capture_path.exists() else 0)
        if response_path.exists() and response_path.stat().st_size and not redacted:
            response = json.loads(response_path.read_text())
            if (folder/'response.sse').exists():
                if reconstruct_sse(folder/'response.sse') != response:
                    raise ValueError('SSE reconstruction differs from response: '+str(folder))
                summary['sse_responses_reconstructed'] += 1
            elif (folder/'response.wire').exists():
                if json.loads((folder/'response.wire').read_bytes()) != response:
                    raise ValueError('wire JSON differs from response: '+str(folder))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_session(args.session), ensure_ascii=False, indent=2))

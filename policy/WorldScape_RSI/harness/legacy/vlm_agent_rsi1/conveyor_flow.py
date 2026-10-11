"""RSI-1 stage gate. Evidence is model visual judgment, not oracle reward."""
import json
from pathlib import Path

ORDER = ['basket', 'intercept', 'transfer', 'place', 'lift', 'finish']
REQUIRED = {'intercept': 'basket_retained', 'transfer': 'target_retained',
            'place': 'over_basket', 'lift': 'target_inside_left_clear'}

def validate(packet, state):
    stage = packet.get('stage', state.get('stage', 'basket'))
    previous = state.get('stage', 'basket')
    if stage not in ORDER + ['recover', 'abort']:
        raise ValueError('Unknown conveyor stage')
    evidence = packet.get('evidence', {})
    if stage in ORDER and previous in ORDER:
        if ORDER.index(stage) > ORDER.index(previous) + 1:
            raise ValueError('Cannot skip stage checks')
    if stage != previous and stage in REQUIRED:
        if not isinstance(evidence.get(REQUIRED[stage]), str) or len(evidence[REQUIRED[stage]]) < 15:
            raise ValueError('Missing explicit visual evidence: ' + REQUIRED[stage])
    cmds = list(packet.get('commands', []))
    macro = packet.get('macro')
    if macro:
        if macro['name'] not in ['grasp_probe', 'moving_grasp_probe']:
            raise ValueError('Unknown macro')
        arm = macro['arm']
        if arm not in ['left', 'right']:
            raise ValueError('Invalid arm')
        delta = macro['retreat_cm']
        if len(delta) != 3 or not (2 <= delta[2] <= 12) or sum(v*v for v in delta) > 225:
            raise ValueError('Probe must be bounded and lift 2..12 cm')
        if cmds:
            raise ValueError('Macro cannot mix unverified dependent actions')
        close = f'{arm} gripper close'
        if macro['name'] == 'moving_grasp_probe':
            lead = macro['lead_cm']
            if not 0 < lead <= 5:
                raise ValueError('Moving closure lead must be 0..5 cm along belt +X')
            close = f'{arm} catch3d {lead} 0 0'
        cmds = [close, f'{arm} move3d {delta[0]} {delta[1]} {delta[2]}']
    if not 1 <= len(cmds) <= 4:
        raise ValueError('Expected 1..4 commands')
    if stage not in ['basket', 'recover', 'abort'] and any(c == 'right gripper open' for c in cmds):
        raise ValueError('Keep basket grip during target handling')
    if stage in ['intercept', 'transfer'] and any(c == 'left gripper open' for c in cmds) and previous == 'transfer':
        raise ValueError('Target release requires place stage')
    if stage == 'abort' and any(not c.startswith('wait') for c in cmds):
        raise ValueError('Abort drains native budget without reset motions')
    if cmds == state.get('commands') and state.get('repeat', 0) >= 1 and not all(c.startswith('wait') for c in cmds):
        raise ValueError('Repeated identical action: revise approach')
    new = {'stage': stage, 'commands': cmds, 'evidence': evidence,
           'repeat': state.get('repeat', 0)+1 if cmds == state.get('commands') else 0}
    result = dict(packet, commands=cmds)
    return result, new

def require_live(root, request_id):
    root = Path(root).resolve()
    run = root.parent
    if (run/'original_source/terminal_result.json').exists():
        raise ValueError('Episode already terminated')
    latest = Path('/ML-vePFS/model_partner/ken/RoboDawn/task_control/pick_from_conveyor_by_image/latest_run.txt')
    if Path(latest.read_text().strip()).resolve() != run:
        raise ValueError('Not the current run')
    pending = json.loads((root/'pending.json').read_text())
    directory = Path(pending['directory']).resolve()
    if pending['request_id'] != request_id or directory != root/request_id:
        raise ValueError('Stale request identity')
    telemetry = json.loads((directory/'telemetry.json').read_text())
    if telemetry.get('request_id') != request_id or telemetry.get('status') != 'pending':
        raise ValueError('Request is not pending')
    return pending

def publish(root, request_id, packet):
    root = Path(root)
    pending = require_live(root, request_id)
    if pending['request_id'] != request_id:
        raise ValueError('Stale request')
    directory = Path(pending['directory'])
    if (directory/'response.json').exists():
        raise ValueError('Already responded')
    statefile = root/'conveyor_flow_state.json'
    state = json.loads(statefile.read_text()) if statefile.exists() else {}
    response, new = validate(packet, state)
    from commands import parse_command_list
    _, errors = parse_command_list(response['commands'])
    if errors:
        raise ValueError('Rejected before simulation: ' + '; '.join(errors))
    new['request_id'] = request_id
    (directory/'flow_decision.json').write_text(json.dumps(new, indent=2))
    temp = directory/'response.flow.tmp'
    temp.write_text(json.dumps({'request_id':request_id,'response':response}))
    require_live(root, request_id)
    temp.replace(directory/'response.json')
    statefile.write_text(json.dumps(new, indent=2))
    print(json.dumps(new))

if __name__ == '__main__':
    import sys
    raw = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    publish(sys.argv[2], raw['request_id'], raw['response'])

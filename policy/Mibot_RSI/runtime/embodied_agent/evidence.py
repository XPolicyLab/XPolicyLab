"""Read-only visual evidence queries plus append-only agent annotations.

Only recorded RGB, robot commands and proprioception are used. No simulator
imports, object inference, task predicates or automatic grasp labels belong here.
"""
import base64
import io
import json
import re
import time
from pathlib import Path


class EvidenceQueryError(ValueError):
    """An explicitly checked evidence request, not a simulator failure."""


def jsonl(path):
    if not Path(path).exists():
        return []
    lines = Path(path).read_text().splitlines(keepends=True)
    rows = []
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except ValueError:
            # A live writer may not have finished the final line yet.
            if index != len(lines) - 1 or line.endswith('\n'):
                raise
    return rows


class Evidence:
    def __init__(self, recording):
        self.root = Path(recording)

    def history(self, method=None, arm=None, status=None, label=None,
                before_sequence=None, limit=10):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise EvidenceQueryError('limit must be 1..100')
        if before_sequence is not None and (type(before_sequence) is not int or before_sequence < 1):
            raise EvidenceQueryError('before_sequence must be a positive integer')
        annotations = {}
        for note in jsonl(self.root / 'annotations.jsonl'):
            annotations.setdefault(note['command_id'], []).append(note)
        found = []
        for sequence, event in enumerate(jsonl(self.root / 'commands.jsonl'), 1):
            req, result = event['request'], event['result']
            notes = annotations.get(req['id'], [])
            if method is not None and req['method'] != method: continue
            if arm is not None and req.get('params', {}).get('arm') not in (arm, 'both'): continue
            if status is not None and result.get('status') != status: continue
            if label is not None and not any(label in n['label'] for n in notes): continue
            if before_sequence is not None and sequence >= before_sequence: continue
            summary = {k: result[k] for k in (
                'status', 'termination_reason', 'start_observation_id', 'end_observation_id',
                'path_fraction', 'tcp_error', 'tracking_lag', 'gripper_width_error_m', 'gripper', 'timing')
                if k in result}
            found.append({'sequence': sequence, 'command_id': req['id'], 'method': req['method'],
                          'requested': req.get('params', {}), **summary, 'annotations': notes})
        page = found[-limit:]
        return {'actions': page, 'matching_count': len(found),
                'next_before_sequence': page[0]['sequence'] if len(found) > len(page) else None,
                'order': 'execution order, oldest first within page',
                'note': 'Labels are agent-authored interpretations; reached does not prove a grasp.'}

    def command(self, command_id):
        for event in jsonl(self.root / 'commands.jsonl'):
            if event['request']['id'] == command_id:
                return event
        raise EvidenceQueryError('Unknown or unfinished command ID')

    def annotate(self, target_command_id, label, expected='', observed=''):
        command_id = target_command_id
        self.command(command_id)
        for value, cap in ((label, 160), (expected, 2000), (observed, 2000)):
            if not isinstance(value, str) or len(value) > cap:
                raise EvidenceQueryError('Annotation must be bounded text')
        if not label.strip(): raise EvidenceQueryError('label is required')
        note = {'command_id': command_id, 'label': label, 'expected': expected,
                'observed': observed, 'author': 'agent', 'wall_time_s': time.time()}
        with (self.root / 'annotations.jsonl').open('a') as stream:
            stream.write(json.dumps(note, ensure_ascii=False) + '\n')
        return note

    def metadata(self, observation_id):
        if not isinstance(observation_id, str) or not re.fullmatch(r'frame_[0-9]+', observation_id):
            raise EvidenceQueryError('Invalid observation ID')
        path = self.root / 'observations' / observation_id / 'observation.json'
        if not path.is_file(): raise EvidenceQueryError('Observation is not available in this archive: ' + observation_id)
        return json.loads(path.read_text())

    def inspect(self, selections):
        """Each selection references an observation OR a command boundary."""
        from PIL import Image
        if not isinstance(selections, list) or not 1 <= len(selections) <= 6:
            raise EvidenceQueryError('Select 1..6 views at a time')
        views = []
        for selection in selections:
            if not isinstance(selection, dict): raise EvidenceQueryError('Each selection must be an object')
            if set(selection) - {'observation_id', 'command_id', 'phase', 'camera', 'region'}:
                raise EvidenceQueryError('Unknown selection field')
            oid, cid = selection.get('observation_id'), selection.get('command_id')
            if (oid is None) == (cid is None):
                raise EvidenceQueryError('Select either observation_id or command_id')
            if cid is not None:
                phase = selection.get('phase')
                if phase not in ('before', 'after'): raise EvidenceQueryError('phase must be before or after')
                result = self.command(cid)['result']
                oid = result.get('start_observation_id' if phase == 'before' else 'end_observation_id')
            elif 'phase' in selection:
                raise EvidenceQueryError('phase applies only to command selections')
            meta = self.metadata(oid)
            camera = selection.get('camera', 'cam_head')
            if camera not in [c['name'] for c in meta['cameras']] or not re.fullmatch(r'[A-Za-z0-9_-]+', camera):
                raise EvidenceQueryError('Camera is not present in this observation')
            path = self.root / 'observations' / oid / (camera + '.png')
            if not path.is_file(): raise EvidenceQueryError('Original image is missing: ' + str(path))
            data = path.read_bytes()
            with Image.open(io.BytesIO(data)) as original:
                width, height = original.size
                region = selection.get('region')
                if region is not None:
                    if not isinstance(region, list) or len(region) != 4 or any(type(x) is not int for x in region):
                        raise EvidenceQueryError('region is [x, y, width, height] in original pixels')
                    x, y, w, h = region
                    if x < 0 or y < 0 or w <= 0 or h <= 0 or x+w > width or y+h > height:
                        raise EvidenceQueryError('region must be inside the original image')
                    buffer = io.BytesIO()
                    original.crop((x, y, x+w, y+h)).save(buffer, format='PNG')
                    data = buffer.getvalue()
            views.append({'observation_id': oid, 'camera': camera,
                          'command_id': cid, 'phase': selection.get('phase'),
                          'control_step': meta.get('control_step'), 'control_time_s': meta.get('control_time_s'),
                          'wall_time_s': meta.get('wall_time_s'), 'synchronization': meta.get('synchronization'),
                          'original_size': [width, height], 'region': region,
                          'robot_state': meta.get('robot_state'),
                          'camera_calibration': (meta.get('camera_calibration') or {}).get(camera),
                          'images': [{'camera': camera, 'png_base64': base64.b64encode(data).decode()}]})
        return {'views': views, 'note': 'Historical RGB views; inspection does not move or restore the world.'}

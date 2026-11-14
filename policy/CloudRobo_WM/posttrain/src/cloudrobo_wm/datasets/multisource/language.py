from collections import Counter
from collections.abc import Mapping
import re
VERSION = 'multisource.language.v1'
DEFAULT = dict(mode='episode', missing='episode', invalid='exclude', gaps='exclude')

def language_policy(source, override=None):
    value = dict(DEFAULT)
    supplied = source.get('language', {}) if override is None else override
    if not isinstance(supplied, Mapping) or set(supplied) - set(DEFAULT):
        raise ValueError('Unknown language configuration; expected mode/missing/invalid/gaps')
    value.update(supplied)
    choices = dict(mode={'episode', 'segments'}, missing={'episode'}, invalid={'exclude', 'episode', 'error'}, gaps={'exclude', 'episode'})
    for key, allowed in choices.items():
        if value[key] not in allowed:
            raise ValueError(f'language.{key} must be one of {sorted(allowed)}')
    return value

def parse_segments(value, length):
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError('action_config must be a list')
    result, previous = ([], 0)
    for i, row in enumerate(value):
        if not isinstance(row, dict):
            raise ValueError(f'action_config[{i}] must be an object')
        a, b, text = (row.get('start_frame'), row.get('end_frame'), row.get('action_text'))
        if type(a) is not int or type(b) is not int or (not 0 <= a < b <= length) or (a < previous):
            raise ValueError(f'action_config[{i}] has unordered/overlapping/out-of-range boundaries')
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f'action_config[{i}] has no instruction')
        result.append(dict(start=a, end=b, instruction=text, segment_id=i))
        previous = b
    return result

def episode_instruction(ep):
    tasks = ep.get('tasks') or []
    return tasks[0] if tasks else ''

def training_regions(ep, policy):
    n = ep['length']
    whole = dict(start=0, end=n, instruction=episode_instruction(ep), segment_id=-1)
    if policy['mode'] == 'episode':
        return [dict(whole, language_status='episode')]
    try:
        segments = parse_segments(ep.get('metadata', {}).get('action_config'), n)
    except ValueError as exc:
        if policy['invalid'] == 'error':
            raise ValueError(f"episode {ep['episode_id']}: {exc}") from exc
        if policy['invalid'] == 'exclude':
            return []
        return [dict(whole, language_status='invalid_fallback')]
    if not segments:
        return [dict(whole, language_status='missing_fallback')]
    regions, previous = ([], 0)
    for segment in segments:
        if policy['gaps'] == 'episode' and previous < segment['start']:
            regions.append(dict(whole, start=previous, end=segment['start'], language_status='gap_fallback'))
        regions.append(dict(segment, language_status='segment'))
        previous = segment['end']
    if policy['gaps'] == 'episode' and previous < n:
        regions.append(dict(whole, start=previous, language_status='gap_fallback'))
    return regions

def training_intervals(ep, policy):
    for region in training_regions(ep, policy):
        for a, b in ep['intervals']:
            start, end = (max(a, region['start']), min(b, region['end']))
            if start < end:
                view = dict(ep, instruction=region['instruction'], language_segment_id=region['segment_id'], language_status=region['language_status'], supervision_end=end)
                yield (view, start, end)

def instructions(manifest, override=None):
    policy = language_policy(manifest['source'], override)
    return {view['instruction'] for ep in manifest['episodes'] for view, _, _ in training_intervals(ep, policy)}

def audit_language(manifest, override=None):
    policy = language_policy(manifest['source'], override)
    counts, invalid, placeholders = (Counter(), [], [])
    for ep in manifest['episodes']:
        counts['episodes'] += 1
        try:
            segments = parse_segments(ep.get('metadata', {}).get('action_config'), ep['length'])
            counts['segments'] += len(segments)
            counts['missing'] += not bool(segments)
            for segment in segments:
                if re.search('\\b(?:var|variable)\\s*:\\s*\\S+', segment['instruction'], re.IGNORECASE):
                    counts['placeholder_segments'] += 1
                    if len(placeholders) < 20:
                        placeholders.append(dict(episode_id=ep['episode_id'], **segment, decision='review'))
        except ValueError as exc:
            counts['invalid'] += 1
            invalid.append(dict(episode_id=ep['episode_id'], reason=str(exc)))
        regions = list(training_intervals(ep, policy))
        counts['supervision_frames'] += sum((b - a for _, a, b in regions))
        counts['regions'] += len(regions)
        for view, a, b in regions:
            counts[view['language_status'] + '_frames'] += b - a
    return dict(version=VERSION, policy=policy, counts=dict(counts), invalid=invalid, placeholder_examples=placeholders)

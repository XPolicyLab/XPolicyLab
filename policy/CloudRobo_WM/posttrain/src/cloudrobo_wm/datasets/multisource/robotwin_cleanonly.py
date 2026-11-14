import copy
import json
import math
from cloudrobo_wm.storage import DataPath as Path
import numpy as np
from .schema import STATE_DIM, ACTION_DIM, file_sha256, fingerprint, within
from .robotwin_joint14 import INDICES, native_fields
from .stats import StreamingStats
PROFILE = 'robotwin_joint14_cleanonly_v1'
STATS_KEY = 'robotwin_aloha_joint14_cleanonly_v1'
PROTOCOL = 'robotwin.clean50.c2c_c2r.v1'

def population(episodes):
    return [dict(episode_id=e['episode_id'], task=e['clean_task'], length=e['length'], data_sha256=e['data_sha256']) for e in episodes]

def select_manifest(original, selection, task_names, *, episodes_per_task=50):
    if selection.get('version') != 1 or selection.get('protocol') != PROTOCOL:
        raise ValueError('Explicit clean-only selection protocol is required')
    proof = selection.get('provenance', {})
    if proof.get('kind') not in {'publisher_mapping', 'raw_metadata', 'trajectory_identity', 'user_selected_scene_index'} or not proof.get('description'):
        raise ValueError('Clean selection needs provenance; episode-order inference is forbidden')
    if not proof.get('evidence'):
        raise ValueError('Clean selection requires evidence files with SHA256')
    for evidence in proof['evidence']:
        if file_sha256(evidence['path']) != evidence['sha256']:
            raise ValueError('Clean selection evidence changed')
    source_rows = {int(e['episode_id']): e for e in original['episodes']}
    if len(source_rows) != len(original['episodes']):
        raise ValueError('Duplicate source episode ID')
    tasks = set(task_names)
    counts = dict.fromkeys(tasks, 0)
    selected, seen = ([], set())
    for row in selection['episodes']:
        eid = int(row['episode_id'])
        if eid in seen or eid not in source_rows:
            raise ValueError('Duplicate or unknown selected episode')
        seen.add(eid)
        ep = copy.deepcopy(source_rows[eid])
        if row.get('regime') != 'demo_clean' or row.get('task') not in tasks:
            raise ValueError('Randomized/unknown regime or task in clean selection')
        if row.get('data_sha256') != ep['data_sha256'] or row.get('length') != ep['length']:
            raise ValueError('Clean selection data identity mismatch')
        if ep['intervals'] != [[0, ep['length']]]:
            raise ValueError('Clean benchmark requires complete demonstrations')
        ep.update(split='train', clean_task=row['task'], collection_regime='demo_clean')
        counts[row['task']] += 1
        selected.append(ep)
    if not tasks or any((n != episodes_per_task for n in counts.values())):
        raise ValueError(f'Expected exactly {episodes_per_task} clean demonstrations per task: {counts}')
    selected.sort(key=lambda e: int(e['episode_id']))
    contract = dict(protocol=PROTOCOL, selection_sha256=fingerprint(selection), population_sha256=fingerprint(population(selected)), raw_frames=sum((e['length'] for e in selected)), task_counts=dict(sorted(counts.items())), episodes_per_task=episodes_per_task, statistics_population='all_selected_raw_frames_once', validation_fraction=0.0)
    result = copy.deepcopy(original)
    result.update(episodes=selected, validation_fraction=0.0, exclusions=[dict(episode_id=e['episode_id'], reason='not_in_verified_clean_selection') for e in original['episodes'] if int(e['episode_id']) not in seen])
    source = result['source']
    source.update(supervision_profile=PROFILE, stats_key=STATS_KEY, fields=native_fields(), action_label_mode='native_action', action_target_offset_frames=1, clean_contract=contract)
    source.pop('sidecar_root', None)
    source.pop('native_stats_sha256', None)
    return result

def validate_manifest(manifest):
    contract = manifest['source']['clean_contract']
    if contract.get('protocol') != PROTOCOL or contract.get('statistics_population') != 'all_selected_raw_frames_once':
        raise ValueError('Unknown clean-only population contract')
    counts = dict.fromkeys(contract['task_counts'], 0)
    seen = set()
    for ep in manifest['episodes']:
        if ep['episode_id'] in seen or ep.get('collection_regime') != 'demo_clean' or ep['split'] != 'train' or (ep.get('clean_task') not in counts) or (ep['intervals'] != [[0, ep['length']]]):
            raise ValueError('Invalid or contaminated clean-only manifest')
        seen.add(ep['episode_id'])
        counts[ep['clean_task']] += 1
    if counts != contract['task_counts'] or not counts or any((n != contract['episodes_per_task'] for n in counts.values())) or (manifest['validation_fraction'] != 0) or (sum((e['length'] for e in manifest['episodes'])) != contract['raw_frames']) or (fingerprint(population(manifest['episodes'])) != contract['population_sha256']):
        raise ValueError('Clean-only population fingerprint/count mismatch')

def compute_group(manifest):
    import pyarrow.parquet as pq
    validate_manifest(manifest)
    accumulators = {k: StreamingStats(14, compute_quantiles=False) for k in ('state', 'action')}
    for ep in manifest['episodes']:
        path = within(Path(manifest['root']), ep['data_path'])
        if file_sha256(path) != ep['data_sha256']:
            raise ValueError(f'Changed clean parquet: {path}')
        table = pq.read_table(path, columns=['episode_index', 'frame_index', 'observation.state', 'action'])
        table = table.slice(ep.get('row_offset', 0), ep['length'])
        if len(table) != ep['length'] or not np.all(np.asarray(table['episode_index']) == ep['episode_id']) or (not np.array_equal(np.asarray(table['frame_index']), np.arange(ep['length']))):
            raise ValueError('Clean episode/frame alignment mismatch')
        for kind, col in [('state', 'observation.state'), ('action', 'action')]:
            values = np.asarray(table[col].to_pylist(), np.float32)
            if values.shape != (ep['length'], 14) or not np.isfinite(values).all():
                raise ValueError('Clean native labels must be finite [T,14]')
            accumulators[kind].update(values, np.ones_like(values, bool))
    group = dict(profile=PROFILE, mode='z-score', native_indices=INDICES, clean_contract=copy.deepcopy(manifest['source']['clean_contract']), count_semantics='valid_raw_frames_once_population_std_ddof0')
    for kind, dim in [('state', STATE_DIM), ('action', ACTION_DIM)]:
        native = accumulators[kind].finish()
        group[kind] = {}
        for name in ('mean', 'std', 'count', 'q01', 'q99'):
            x = np.zeros(dim, np.float64)
            x[INDICES[kind]] = native[name]
            group[kind][name] = x.tolist()
        group[kind]['quantiles_computed'] = False
    group['native_stats_sha256'] = fingerprint(group)
    manifest['source']['native_stats_sha256'] = group['native_stats_sha256']
    return group

def validate_group(group):
    from .robotwin_joint14 import NATIVE_STATS_SHA256
    payload = {k: v for k, v in group.items() if k != 'native_stats_sha256'}
    if group.get('profile') != PROFILE or group.get('mode') != 'z-score' or group.get('native_indices') != INDICES or (group.get('native_stats_sha256') in (None, NATIVE_STATS_SHA256)) or (group['native_stats_sha256'] != fingerprint(payload)) or (group.get('clean_contract', {}).get('protocol') != PROTOCOL):
        raise ValueError('Clean-only statistics profile/fingerprint mismatch')
    for kind, dim in [('state', STATE_DIM), ('action', ACTION_DIM)]:
        active = np.zeros(dim, bool)
        active[INDICES[kind]] = True
        count = np.asarray(group[kind]['count'])
        std = np.asarray(group[kind]['std'])
        if count.shape != (dim,) or not np.array_equal(count > 0, active) or (std < 0).any() or (not np.all(count[active] == group['clean_contract']['raw_frames'])):
            raise ValueError('Clean-only statistics require exactly the native 14 slots')

def validate_initialization_mode(cfg):
    mode = cfg.get('clean_initialization', 'wan_interpolated')
    if mode not in {'wan_interpolated', 'pretrained_policy'}:
        raise ValueError('Unknown clean_initialization mode')
    if cfg.get('pretrained_ckpt') and mode != 'pretrained_policy':
        raise ValueError('Clean-only baseline starts from Wan/interpolated ActionDiT, not a policy checkpoint')
    if mode == 'pretrained_policy':
        if not cfg.get('resume') and (not cfg.get('pretrained_ckpt')):
            raise ValueError('Pretrained clean experiment requires pretrained_ckpt')
        if cfg.get('pretrained_ckpt') and cfg.get('pretrained_ckpt_strict_shape', True):
            raise ValueError('Pretrained clean experiment requires tensor-wise non-strict stage initialization')
        if cfg.get('resume') and cfg.get('pretrained_ckpt'):
            raise ValueError('resume and pretrained_ckpt are mutually exclusive')

def validate_initialization(cfg, dataset):
    sources = [a.source for a in dataset.adapters] if hasattr(dataset, 'adapters') else []
    if not any((s.get('supervision_profile') in {PROFILE, 'robotwin_joint14_eef_cleanonly_zscore_v1'} for s in sources)):
        return
    validate_initialization_mode(cfg)
    if cfg.get('resume'):
        path = Path(str(cfg['resume'])).resolve()
        if not path.is_dir():
            raise ValueError('Clean-only resume requires a full-state directory')
        meta = json.loads((path / 'trainer_state.json').read_text())
        if meta.get('multisource_contract') != dataset.checkpoint_contract:
            raise ValueError('Clean-only resume rejected: data/statistics contract differs')


def validate_evaluation_contract(contract):
    from .robotwin_joint14 import validate_source_group
    groups = contract['statistics']['groups']
    if set(groups) != {STATS_KEY} or groups[STATS_KEY].get('profile') != PROFILE:
        raise ValueError('Clean-only evaluator refuses a mixed-data/other experiment checkpoint')
    sources = contract.get('sources', [])
    if len(sources) != 1 or sources[0]['source'].get('stats_key') != STATS_KEY:
        raise ValueError('Clean-only checkpoint needs exactly one verified clean source')
    validate_source_group(sources[0]['source'], groups[STATS_KEY])

import hashlib
import json
from cloudrobo_wm.storage import DataPath as Path
import numpy as np
from .schema import file_sha256, fingerprint, within

def stable_split(identity, fraction=0.005, seed=42):
    if not 0 <= fraction < 1:
        raise ValueError('validation fraction must be in [0,1)')
    h = hashlib.sha256(f'{seed}:{identity}'.encode()).digest()
    return 'val' if int.from_bytes(h[:8], 'big') / 2 ** 64 < fraction else 'train'

def build_lerobot_manifest(root, source, *, val_fraction=0.005, seed=42, episode_ids=None):
    root = Path(root)
    info = json.loads((root / 'meta/info.json').read_text())
    v3 = str(info['codebase_version']).startswith('v3')
    meta_paths = [root / 'meta/info.json']
    if v3:
        import pyarrow.parquet as pq
        paths = sorted((root / 'meta/episodes').rglob('*.parquet'))
        meta_paths += paths
        episodes = []
        for path in paths:
            cols = [c for c in pq.ParquetFile(path).schema_arrow.names if not c.startswith('stats/')]
            episodes.extend(pq.read_table(path, columns=cols).to_pylist())
        starts = {}
        for ep in episodes:
            key = (ep['data/chunk_index'], ep['data/file_index'])
            starts[key] = min(starts.get(key, ep['dataset_from_index']), ep['dataset_from_index'])
    else:
        path = root / 'meta/episodes.jsonl'
        meta_paths.append(path)
        episodes = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        starts = {}
    intervals_by_episode = source.get('intervals', {})
    excluded = source.get('excluded_episodes', {})
    if source.get('cleaning'):
        if source.get('adapter') != 'agibot':
            raise ValueError('AgiBot idle cleanup requires the AgiBot adapter')
        from .agibot_cleaning import cleaned_population
        intervals_by_episode, excluded = cleaned_population(root, source, episodes)
    if source.get('curation'):
        from .curation_report import curated_population
        intervals_by_episode, excluded = curated_population(root, source, episodes, intervals_by_episode, excluded)
    records, identities, hashes, exclusions = ([], set(), {}, [])
    official = info.get('splits', {})
    has_validation = any((k in official for k in ('validation', 'val', 'test')))
    for ep in episodes:
        eid, length = (int(ep['episode_index']), int(ep['length']))
        if eid in identities or length <= 0:
            raise ValueError(f'Duplicate/empty episode {eid}')
        identities.add(eid)
        if str(eid) in excluded:
            exclusions.append(dict(episode_id=eid, reason=excluded[str(eid)]))
            continue
        if episode_ids is not None and eid not in episode_ids:
            exclusions.append(dict(episode_id=eid, reason='explicit episode subset'))
            continue
        if ep.get('calibration_path'):
            meta_paths.append(within(root, ep['calibration_path']))
        fmt = dict(episode_index=eid, episode_chunk=eid // int(info.get('chunks_size', 1000)))
        if v3:
            fmt.update(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index'])
        data_path = info['data_path'].format(**fmt)
        if data_path not in hashes:
            hashes[data_path] = file_sha256(within(root, data_path))
        row = int(ep['dataset_from_index'] - starts[fmt['chunk_index'], fmt['file_index']]) if v3 else 0
        split = stable_split(f"{source['source_id']}:{eid}", val_fraction, seed)
        if has_validation:
            split = 'excluded'
            for key, bounds in official.items():
                start, end = [int(x) for x in bounds.split(':')]
                if start <= eid < end:
                    split = 'train' if key == 'train' else 'val'
                    break
        cameras = {}
        for slot, key in source.get('cameras', {}).items():
            if key not in info['features']:
                continue
            vf = dict(fmt, video_key=key)
            offset = 0.0
            if v3:
                prefix = 'videos/' + key
                vf.update(chunk_index=ep[prefix + '/chunk_index'], file_index=ep[prefix + '/file_index'])
                offset = float(ep[prefix + '/from_timestamp'])
            relative = info['video_path'].format(**vf)
            stat = within(root, relative).stat()
            cameras[slot] = dict(path=relative, start_seconds=offset, fps=float(info['features'][key].get('info', {}).get('video.fps', info['fps'])), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        records.append(dict(episode_id=eid, length=length, data_path=data_path, data_sha256=hashes[data_path], row_offset=row, split=split, intervals=intervals_by_episode.get(str(eid), [[0, length]]), cameras=cameras, tasks=ep.get('tasks', []), metadata=ep))
    return dict(version=1, source=source, root=str(root.resolve()), info=info, metadata_hashes={str(p.relative_to(root)): file_sha256(p) for p in meta_paths}, episodes=records, exclusions=exclusions, split_seed=seed, validation_fraction=val_fraction)

def build_video_manifest(root, source, episodes, *, val_fraction=0.005, seed=42):
    if source.get('adapter') != 'human' or source.get('protocol') != 'video_only':
        raise ValueError('Video manifests require a human/video_only source')
    root = Path(root)
    rows = []
    seen = set()
    for item in episodes:
        eid = str(item['episode_id'])
        length = int(item['length'])
        if eid in seen or length <= 0:
            raise ValueError('Duplicate or empty video episode')
        seen.add(eid)
        cameras = {}
        for slot, spec in item['cameras'].items():
            from .schema import CAMERAS
            if slot not in CAMERAS or float(spec['fps']) <= 0:
                raise ValueError('Invalid camera slot/media FPS')
            stat = within(root, spec['path']).stat()
            cameras[slot] = dict(spec, start_seconds=float(spec.get('start_seconds', 0)), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if not cameras:
            raise ValueError('Video episode has no RGB camera')
        split = item.get('split', stable_split(f"{source['source_id']}:{eid}", val_fraction, seed))
        if split not in ('train', 'val', 'excluded'):
            raise ValueError('Unknown split')
        rows.append(dict(episode_id=eid, length=length, cameras=cameras, split=split, intervals=item.get('intervals', [[0, length]]), tasks=item.get('tasks', [])))
    return dict(version=1, source=source, root=str(root.resolve()), info={}, metadata_hashes={}, episodes=rows, exclusions=[], split_seed=seed, validation_fraction=val_fraction)

class WindowIndex:

    def __init__(self, manifest, split='train', window_stride=1, language=None):
        if manifest['version'] != 1 or window_stride < 1:
            raise ValueError('Unsupported manifest/stride')
        self.manifest, self.stride, self.split = (manifest, int(window_stride), split)
        from .language import language_policy, training_intervals
        self.language = language_policy(manifest.get('source', {}), language)
        self.intervals, counts = ([], [])
        for ep in manifest['episodes']:
            if ep['split'] != split:
                continue
            previous = -1
            for start, end in ep['intervals']:
                if not 0 <= start < end <= ep['length'] or start < previous:
                    raise ValueError('Invalid or overlapping episode intervals')
                previous = end
            for view, start, end in training_intervals(ep, self.language):
                self.intervals.append((view, start, end))
                counts.append((end - start + self.stride - 1) // self.stride)
        self.ends = np.cumsum(counts, dtype=np.int64)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def locate(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        i = int(np.searchsorted(self.ends, index, side='right'))
        ep, start, end = self.intervals[i]
        offset = index - (int(self.ends[i - 1]) if i else 0)
        return (ep, start + offset * self.stride, end)

    @property
    def digest(self):
        value = dict(manifest=self.manifest, split=self.split, window_stride=self.stride)
        if self.language['mode'] != 'episode':
            from .language import VERSION
            value['language_view'] = dict(version=VERSION, **self.language)
        return fingerprint(value)

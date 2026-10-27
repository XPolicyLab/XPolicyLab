"""Prepared train-only normalization, immutable splits and visual-cache reader."""
from bisect import bisect_right
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import hashlib
import json
import multiprocessing
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, default_collate
from wam.posttrain.robocasa_paths import resolve_source
from .robocasa_delta import CONTRACT, RoboCasaNormalizer, masks, project, prompt_for, validate_modality
from .robocasa_visual_cache import RoboCasaVisualCache, read_json, load_plan, digest, file_hash, file_identity, check_episode_source, atomic_json, locked, require
SCHEMA = 'wam.robocasa.posttrain.v1'
REMARKS_SCHEMA = 'wam.robocasa.posttrain.v2'

def action_space_module(name):
    from . import robocasa_delta, robocasa_abs47, robocasa_delta80_slot18
    spaces = {'delta80': robocasa_delta, 'absolute47': robocasa_abs47, 'delta80_slot18': robocasa_delta80_slot18}
    require(name in spaces, 'Unknown RoboCasa action space')
    return spaces[name]

def package_space(package):
    space = action_space_module(package.get('action_space', 'delta80'))
    require(package['contract'] == space.CONTRACT, 'Package/action-space contract mismatch')
    return space

def episode_remarks(path):
    result = {}
    for row in map(json.loads, Path(path).read_text().splitlines()):
        idx, remarks = (row['episode_index'], row.get('remarks'))
        require(type(idx) is int and idx not in result, 'Duplicate/invalid episode metadata ID')
        require(isinstance(remarks, str) and bool(remarks.strip()), f'Missing remarks for episode {idx}')
        result[idx] = prompt_for(remarks)
    return result

def text_name(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest() + '.t5_len128.wan22ti2v5b.pt'

def select_episodes(plan, *, val_ratio=0.01, fewshot_ratio=1.0, seed=42, smoke=False):
    require(0 < val_ratio < 1 and 0 < fewshot_ratio <= 1, 'Invalid split/fewshot fraction')
    groups = defaultdict(list)
    for ep in plan['episodes']:
        groups[ep['task']].append(ep)
    selected = []
    if smoke:
        require(plan['scope'] == 'smoke' and len(plan['episodes']) >= 2, 'Use a smoke cache plan')
        return [{**e, 'split': 'val' if i == len(plan['episodes']) - 1 else 'train'} for i, e in enumerate(plan['episodes'])]
    require(plan['scope'] == 'full24_all_episodes', 'Formal training requires the full cache plan')
    for task, episodes in sorted(groups.items()):
        ordered = sorted(episodes, key=lambda e: digest([seed, 'holdout', task, e['episode']]))
        nval = max(1, int(len(ordered) * val_ratio))
        require(nval < len(ordered), 'No training episodes after holdout')
        val, train = (ordered[:nval], ordered[nval:])
        train = sorted(train, key=lambda e: digest([seed, 'fewshot', task, e['episode']]))
        train = train[:max(1, int(len(train) * fewshot_ratio))]
        selected.extend(({**e, 'split': split} for split, rows in (('train', train), ('val', val)) for e in rows))
    return sorted(selected, key=lambda e: (e['task'], e['episode']))

def numeric_episode(episode, action_space='delta80'):
    space = action_space_module(action_space)
    check_episode_source(episode)
    table = pq.read_table(resolve_source(episode['parquet']), columns=['observation.state', 'action', 'episode_index', 'timestamp', 'annotation.human.coarse_action'], use_threads=False)
    n = episode['length']
    require(len(table) == n and all((e == episode['episode'] for e in table['episode_index'].to_pylist())), 'Numeric episode alignment changed')
    times = np.asarray(table['timestamp'].to_pylist())
    require(np.isfinite(times).all() and np.allclose(np.diff(times), 0.05, atol=1e-06, rtol=0), 'Not native 20Hz')
    state = torch.tensor(np.asarray(table['observation.state'].to_pylist()), dtype=torch.float32)
    action = torch.tensor(np.asarray(table['action'].to_pylist()), dtype=torch.float32)
    state, action = space.project(state, action)
    state_valid, action_valid = space.masks()
    valid = action_valid[None].expand(n, -1).clone()
    if action_space in ('delta80', 'delta80_slot18'):
        valid[-1, 7:13] = valid[-1, 47:53] = False
    return (state, action, valid, table['annotation.human.coarse_action'].to_pylist())

def _statistics_worker_init():
    torch.set_num_threads(1)

def _episode_statistics(ep, action_space='delta80'):
    space = action_space_module(action_space)
    state, action, action_mask, ids = numeric_episode(ep, action_space)
    bounds = {}
    if ep['split'] == 'train':
        for kind, value, valid in (('state', state, space.masks()[0][None].expand_as(state)), ('action', action, action_mask)):
            bounds[kind] = (value.masked_fill(~valid, float('inf')).amin(0), value.masked_fill(~valid, -float('inf')).amax(0), valid.sum(0))
            if action_space == 'absolute47':
                values = value.double()
                bounds[kind] += (values.mean(0), ((values - values.mean(0)) ** 2).sum(0))
    return (bounds, sorted(set(ids)))

def _iter_episode_statistics(episodes, workers, action_space='delta80'):
    compute = partial(_episode_statistics, action_space=action_space)
    require(type(workers) is int and workers > 0, 'workers must be a positive integer')
    if workers == 1:
        yield from map(compute, episodes)
        return
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=_statistics_worker_init) as pool:
        source, pending = (iter(episodes), deque())
        for _ in range(workers * 2):
            ep = next(source, None)
            if ep is not None:
                pending.append(pool.submit(compute, ep))
        while pending:
            yield pending.popleft().result()
            ep = next(source, None)
            if ep is not None:
                pending.append(pool.submit(compute, ep))

def prepare_package(plan_path, output, *, val_ratio=0.01, fewshot_ratio=1.0, seed=42, smoke=False, workers=1, action_space='delta80'):
    space = action_space_module(action_space)
    dim = len(space.masks()[0])
    root, plan = (Path(output), load_plan(plan_path))
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / 'prepare.lock'):
        require(not (root / 'package.json').exists(), 'Package already exists; use a new directory')
        episodes = select_episodes(plan, val_ratio=val_ratio, fewshot_ratio=fewshot_ratio, seed=seed, smoke=smoke)
        stats = {'contract': space.CONTRACT, 'mode': 'z-score' if action_space == 'absolute47' else 'minmax', 'purpose': 'smoke' if smoke else 'training', 'population': 'selected_training_native_rows_once_valid_dimensions_only'}
        lows = {k: torch.full((dim,), float('inf')) for k in ('state', 'action')}
        highs = {k: torch.full((dim,), -float('inf')) for k in lows}
        counts = {k: torch.zeros(dim, dtype=torch.long) for k in lows}
        means = {k: torch.zeros(dim, dtype=torch.float64) for k in lows}
        m2 = {k: torch.zeros(dim, dtype=torch.float64) for k in lows}
        metadata, tasks, prompts = ({}, {}, set())
        for i, (ep, (bounds, ids)) in enumerate(zip(episodes, _iter_episode_statistics(episodes, workers, action_space))):
            task = ep['task']
            if task not in tasks:
                meta = Path(plan['data_root']) / task / 'meta'
                validate_modality(read_json(meta / 'modality.json'))
                tasks[task] = episode_remarks(meta / 'episodes.jsonl')
                for name in ('tasks.jsonl', 'episodes.jsonl', 'modality.json'):
                    path = str(meta / name)
                    metadata[path] = {**file_identity(path), 'sha256': file_hash(path)}
            require(ep['episode'] in tasks[task], 'Missing episode remarks; refusing task-ID fallback')
            ep['prompt'] = tasks[task][ep['episode']]
            ep['prompts'] = {str(idx): ep['prompt'] for idx in set(ids)}
            prompts.update(ep['prompts'].values())
            for kind, values in bounds.items():
                low, high, count = values[:3]
                if action_space == 'absolute47':
                    mean, episode_m2 = values[3:]
                    total = counts[kind] + count
                    delta = mean - means[kind]
                    m2[kind] += episode_m2 + delta.square() * counts[kind] * count / total.clamp_min(1)
                    means[kind] += delta * count / total.clamp_min(1)
                lows[kind] = torch.minimum(lows[kind], low)
                highs[kind] = torch.maximum(highs[kind], high)
                counts[kind] += count
            if i % 100 == 0:
                print(f'[prepare] {i + 1}/{len(episodes)} numeric episodes', flush=True)
        for kind in lows:
            stats[kind] = {'min': lows[kind].masked_fill(counts[kind] == 0, 0).tolist(), 'max': highs[kind].masked_fill(counts[kind] == 0, 0).tolist(), 'count': counts[kind].tolist()}
            if action_space == 'absolute47':
                stats[kind].update(mean=means[kind].tolist(), std=(m2[kind] / counts[kind].clamp_min(1)).clamp_min(0).sqrt().tolist())
        membership = {split: [[e['task'], e['episode']] for e in episodes if e['split'] == split] for split in ('train', 'val')}
        stats['train_membership_sha256'] = digest(membership['train'])
        atomic_json(root / 'norm.json', stats)
        package = {'schema': REMARKS_SCHEMA, 'contract': space.CONTRACT, 'purpose': stats['purpose'], 'action_space': action_space, 'prompt_source': 'episodes.jsonl:remarks', 'plan_path': str(Path(plan_path).resolve()), 'plan_sha256': plan['sha256'], 'norm_sha256': file_hash(root / 'norm.json'), 'episodes': episodes, 'seed': seed, 'val_ratio': val_ratio, 'fewshot_ratio': fewshot_ratio, 'membership': membership, 'metadata': metadata, 'prompts': sorted(prompts)}
        package['sha256'] = digest(package)
        atomic_json(root / 'package.json', package)
        print(json.dumps({'package': str(root), 'prompts': len(prompts), 'episodes': {k: len(v) for k, v in membership.items()}, 'sha256': package['sha256']}), flush=True)
    return package

def load_package(root, *, allow_smoke=False):
    root = Path(root)
    package = read_json(root / 'package.json')
    sha = package.pop('sha256', None)
    require(package['schema'] in (SCHEMA, REMARKS_SCHEMA), 'Posttraining package contract mismatch')
    package_space(package)
    if package['schema'] == REMARKS_SCHEMA:
        require(package.get('prompt_source') == 'episodes.jsonl:remarks', 'Missing episode prompt provenance')
    package['sha256'] = sha
    require(package['purpose'] == 'training' or (allow_smoke and package['purpose'] == 'smoke'), 'Refusing smoke statistics in formal training')
    membership = package['membership']
    require(set(map(tuple, membership['train'])).isdisjoint(map(tuple, membership['val'])), 'Episode split leakage')
    for path, record in package['metadata'].items():
        require(resolve_source(path).is_file(), 'Task metadata missing')
    return package

class RoboCasaTextCache:

    def __init__(self, root, package):
        self.root = Path(root) / 'text'
        self.manifest = read_json(self.root / 'COMPLETE.json')
        self.values = {}

    def get(self, prompt):
        if prompt not in self.values:
            name = text_name(prompt)
            path = self.root / name
            obj = torch.load(path, map_location='cpu', weights_only=True)
            context, mask = (obj['context'], obj['mask'])
            require(obj['prompt'] == prompt and context.shape == (128, 4096) and (mask.shape == (128,)) and (mask.dtype == torch.bool) and bool(torch.isfinite(context).all()), 'Invalid T5 cache')
            self.values[prompt] = (context.masked_fill(~mask[:, None], 0), torch.ones_like(mask))
        return tuple((x.clone() for x in self.values[prompt]))

class RoboCasaPosttrainDataset(Dataset):

    def __init__(self, package_root, cache_root, *, split='train', allow_smoke=False, domain='robocasa', pretrained_norm_stats=None, val_set_proportion=0.01, processor=None, max_open_episodes=8, validation_panel_size=128, smoke_selection=None):
        require(split in ('train', 'val') and domain == 'robocasa', 'Invalid RoboCasa reader split/domain')
        self.root, self.split = (Path(package_root), split)
        self.package = load_package(self.root, allow_smoke=allow_smoke)
        self.action_space = self.package.get('action_space', 'delta80')
        self.space = package_space(self.package)
        if pretrained_norm_stats is not None:
            require(Path(pretrained_norm_stats).resolve() == (self.root / 'norm.json').resolve(), "Use this package's normalization")
        self.normalizer = self.space.RoboCasaNormalizer(read_json(self.root / 'norm.json'))
        self.visual = RoboCasaVisualCache(cache_root, self.package['plan_path'], max_open_episodes=max_open_episodes)
        complete = read_json(Path(cache_root) / 'COMPLETE.json')
        require(complete['status'] == 'complete' and complete['windows'] == self.visual.plan['windows'], 'Cache must be globally complete')
        self.text = RoboCasaTextCache(self.root, self.package)
        self.entries = [e for e in self.package['episodes'] if e['split'] == split]
        self.smoke_selection_sha256 = None
        if smoke_selection:
            require(allow_smoke, 'Smoke selection cannot be used for formal training')
            selection = read_json(smoke_selection)
            keys = set(map(tuple, selection[split]))
            require(1 <= len(keys) <= 8, 'Smoke selection must contain 1..8 episodes per split')
            self.entries = [ep for ep in self.entries if (ep['task'], ep['episode']) in keys]
            require(len(self.entries) == len(keys), 'Smoke selection membership mismatch')
            self.smoke_selection_sha256 = file_hash(smoke_selection)
        elif self.package['purpose'] == 'training':
            from wam.posttrain.robocasa_paths import require_migrated_cache
            require_migrated_cache(cache_root, sorted({ep['task'] for ep in self.package['episodes']}))
        self.ends = np.cumsum([e['windows'] for e in self.entries]).tolist()
        require(bool(self.ends), 'Empty split')
        self.numeric, self.max_open = (OrderedDict(), int(max_open_episodes))
        if split == 'val':
            n = min(int(validation_panel_size), len(self))
            self.fixed_validation_panel = [{'index': int((i + 0.5) * len(self) / n), 'source': domain, 'seed': 42 + i, 'embodiment_id': self.space.CONTRACT['embodiment_id']} for i in range(n)]

    def __len__(self):
        return self.ends[-1]

    @property
    def sample_episode_ranges(self):
        return tuple(zip([0, *self.ends[:-1]], self.ends))

    def __getstate__(self):
        return {k: v for k, v in {**self.__dict__, 'numeric': OrderedDict()}.items() if k != 'space'}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.space = package_space(self.package)

    def __getitem__(self, index):
        require(0 <= index < len(self), 'Window index out of range')
        pos = bisect_right(self.ends, int(index))
        ep = self.entries[pos]
        row = int(index) - (self.ends[pos - 1] if pos else 0)
        start = ep['starts'][row] if 'starts' in ep else row
        key = (ep['task'], ep['episode'])
        if key not in self.numeric:
            self.numeric[key] = numeric_episode(ep, self.action_space)
            while len(self.numeric) > self.max_open:
                self.numeric.popitem(last=False)
        self.numeric.move_to_end(key)
        state, actions, valid, ids = self.numeric[key]
        steps = torch.arange(start, start + 32)
        action_pad = steps >= ep['length']
        indices = steps.clamp_max(ep['length'] - 1)
        mask = valid[indices] & ~action_pad[:, None]
        action = self.normalizer.forward(actions[indices], 'action').masked_fill(~mask, 0)
        proprio = self.normalizer.forward(state[start:start + 1], 'state')
        visual = self.visual.get(*key, start)
        visual.pop('mapping')
        prompt = ep['prompt'] if self.package['schema'] == REMARKS_SCHEMA else ep['prompts'][str(ids[start])]
        context, context_mask = self.text.get(prompt)
        state_valid, action_valid = self.space.masks()
        return {**visual, 'action': action, 'action_mask': mask, 'action_is_pad': action_pad, 'action_dim_is_pad': ~action_valid, 'proprio': proprio, 'proprio_is_pad': torch.zeros(1, dtype=torch.bool), 'proprio_dim_is_pad': ~state_valid, 'embodiment_id': torch.tensor(self.space.CONTRACT['embodiment_id'], dtype=torch.long), 'prompt': prompt, 'context': context, 'context_mask': context_mask, 'sample_id': int(index), 'video_layout': 'latent_horizontal', 'video_view_names': 'main|left|right'}
    collate_fn = staticmethod(default_collate)

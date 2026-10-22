"""Read immutable FastWAM LIBERO dual48 caches with native WAM action labels."""
from collections import OrderedDict
import hashlib
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from .joint96_posttrain import Joint96PosttrainDataset
from wam.model.modules.codecs.joint_visual_encoder import encoder_source_sha256
from wam.training.checkpoint import _joint96_identity

def file_hash(path):
    """File existence only; runtime content hashing is disabled."""
    Path(path).resolve(strict=True)
    return None

def read_json(path):
    return json.loads(Path(path).read_text())

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(f'LIBERO Joint96 cache: {message}')

class CacheRGBTransform:
    """Deterministic RGB reference matching the cache's pre-encoder branch."""

    def __call__(self, image):
        return torch.nn.functional.interpolate(image.float(), size=(256, 256), mode='bicubic', align_corners=False).clamp(0, 1)

class PairedLiberoCache:

    def __init__(self, *, vae_root, stvae_root, source_root, source_sha256=None, initialization_checkpoint, max_open_shards=32, pair_binding_file=None):
        if str(source_root) == 'bundled://dino-stvae-v1':
            from .joint96_cache_contracts import pair_contracts
        else:
            source = Path(source_root).resolve(strict=True)
            package = source / 'src/fastwam'
            for name, module in tuple(sys.modules.items()):
                if name == 'fastwam' or name.startswith('fastwam.'):
                    path = getattr(module, '__file__', None)
                    require(not path or Path(path).resolve().is_relative_to(package), 'conflicting FastWAM import')
            if str(source / 'src') not in sys.path:
                sys.path.insert(0, str(source / 'src'))
            pair_contracts = importlib.import_module('fastwam.datasets.libero_wan_vae_stvae_dataset').pair_contracts
        if pair_binding_file is None:
            self.roots, self.contracts = pair_contracts(vae_root, [Path(stvae_root) / view for view in ('head', 'wrist')])
        else:
            from .libero_stvae_rebind import load_bound_pair
            self.roots, self.contracts = load_bound_pair(vae_root, stvae_root, pair_binding_file, pair_contracts)
        payload = torch.load(initialization_checkpoint, map_location='cpu', weights_only=True, mmap=True)
        identity = _joint96_identity(payload.get('joint96_metadata'))
        encoder = self.contracts[2]['encoder']
        for key in ('teacher_checkpoint_state', 'teacher_output_source', 'teacher_preprocessing', 'dinov3_source_commit'):
            require(encoder[key] == identity[key], f'pretraining/cache encoder mismatch: {key}')
        from wam.model.modules.codecs.dino_stvae.provenance import canonical_source_identity
        expected_source = canonical_source_identity(dict(source_root=str(source_root), source_sha256=source_sha256))
        require(identity['channel_order'] == ['wan_vae48', 'stvae48'], 'channel order mismatch')
        self.snapshot = self.contracts[2]['dataset_snapshot']
        self.length = int(self.contracts[0]['dataset_length'])
        self.max_open_shards = int(max_open_shards)
        require(self.max_open_shards > 0, 'max_open_shards must be positive')
        self._open = OrderedDict()
        self._validated = {}
        self._checked_payloads = set()
        self._source_roots = {}
        self._payloads = {}
        global_start = 0
        for dataset_index, entry in enumerate(self.snapshot['roots']):
            root = Path(entry['root']).resolve()
            self._payloads.update({str(root / path): (size, mtime) for path, size, mtime in entry['payload_inventory']})
            episodes = [json.loads(line) for line in (root / 'meta/episodes.jsonl').read_text().splitlines() if line.strip()]
            info = read_json(root / 'meta/info.json')
            require(len(episodes) == info['total_episodes'], 'episode count mismatch')
            starts, local_start = ({}, 0)
            for episode in episodes:
                ep = int(episode['episode_index'])
                require(ep not in starts, 'duplicate episode')
                starts[ep] = (local_start, int(episode['length']))
                local_start += int(episode['length'])
            self._source_roots[str(root)] = (dataset_index, global_start, starts)
            global_start += local_start
        require(global_start == self.length, 'source frame count mismatch')

    def __getstate__(self):
        return {**self.__dict__, '_open': OrderedDict(), '_checked_payloads': set()}

    def _check_source_files(self, source, episode):
        paths = [source.meta.get_data_file_path(episode)] + [source.meta.get_video_file_path(episode, key) for key in ('observation.images.image', 'observation.images.wrist_image')]
        for relative in paths:
            path = Path(source.root).resolve() / relative
            name = str(path)
            if name not in self._checked_payloads:
                require(path.is_file(), f'missing source payload: {path}')
                self._checked_payloads.add(name)

    def _shard(self, view_index, rank):
        key = (view_index, rank)
        if key in self._open:
            self._open.move_to_end(key)
            return self._open[key]
        if key in self._validated:
            path, count, hashes, mappings = self._validated[key]
            features = np.memmap(path, mode='r', dtype=np.uint16, shape=(count, 48, 3, 16, 16))
            return self._remember(key, (features, hashes, mappings))
        contract = self.contracts[view_index]
        root = self.roots[view_index] / 'shards' / f'rank_{rank:05d}'
        manifest = read_json(root / 'manifest.json')
        complete = read_json(root / 'COMPLETE.json')
        require(manifest['status'] == complete['status'] == 'complete', f'incomplete shard: {root}')
        from wam.utils.runtime_contract import semantic_contract
        require(semantic_contract(manifest['contract']) == semantic_contract(contract) and manifest['rank'] == rank, 'foreign shard contract')
        require(manifest['dtype'] == 'bfloat16' and manifest['storage_dtype'] == 'uint16' and (manifest['feature_shape'] == [48, 3, 16, 16]), 'feature format mismatch')
        mapping = read_json(root / manifest['window_mapping'])
        indices = np.load(root / manifest['sample_indices'], allow_pickle=False)
        valid = np.load(root / manifest['valid'], allow_pickle=False)
        expected = np.arange(rank, self.length, int(contract['world_size']))
        require(np.array_equal(indices, expected) and valid.shape == indices.shape and valid.all(), 'invalid shard indices')
        require([row['idx'] for row in mapping] == expected.tolist(), 'mapping indices mismatch')
        require(int(manifest['shard_length']) == len(indices), 'shard length mismatch')
        hashes = None
        features_path = root / manifest['features']
        require(features_path.stat().st_size == len(indices) * 48 * 3 * 16 * 16 * 2, 'feature file size mismatch')
        features = np.memmap(features_path, mode='r', dtype=np.uint16, shape=(len(indices), 48, 3, 16, 16))
        mappings = np.asarray([bytes.fromhex(digest(row)) for row in mapping], dtype='V32')
        result = (features, hashes, mappings)
        self._validated[key] = (features_path, len(indices), hashes, mappings)
        return self._remember(key, result)

    def _remember(self, key, result):
        self._open[key] = result
        while len(self._open) > self.max_open_shards:
            self._open.popitem(last=False)
        return result

    def get(self, expected_mapping):
        index = int(expected_mapping['idx'])
        require(0 <= index < self.length, 'cache index outside dataset')
        expected_hash = bytes.fromhex(digest(expected_mapping))
        views = []
        for view_index, contract in enumerate(self.contracts):
            world = int(contract['world_size'])
            features, hashes, mappings = self._shard(view_index, index % world)
            row = index // world
            require(bytes(mappings[row]) == expected_hash, 'episode/frame/timestamp/padding mismatch')
            array = np.array(features[row], copy=True)
            value = torch.from_numpy(array).view(torch.bfloat16)
            require(bool(torch.isfinite(value).all()), 'nonfinite feature')
            views.append(value)
        head = torch.cat([views[0], views[2]], dim=0)
        wrist = torch.cat([views[1], views[3]], dim=0)
        return torch.cat([head, torch.zeros_like(head), wrist], dim=-1).contiguous()

class LiberoJoint96CachedDataset(Joint96PosttrainDataset):

    def __init__(self, *, visual_cache, **kwargs):
        require(kwargs.get('domain') == 'libero', 'only LIBERO is supported')
        for name in ('image_color_jitter', 'image_camera_jitter', 'image_sensor_noise', 'image_domain_jitter'):
            require(not (kwargs.get(name) or {}).get('enabled', False), f'disable {name} for fixed visual caches')
        self.visual_cache = PairedLiberoCache(**visual_cache)
        super().__init__(**kwargs)
        self._mapping_sources, self._source_ends = ([], [])
        end = 0
        for source in self.lerobot_dataset.multi_dataset._datasets:
            require(str(Path(source.root).resolve()) in self.visual_cache._source_roots, 'dataset root absent from cache')
            columns = source.hf_dataset.with_format('numpy', columns=['episode_index', 'frame_index', 'timestamp'])
            metadata = {key: np.asarray(columns[key]).reshape(-1) for key in ('episode_index', 'frame_index', 'timestamp')}
            self._mapping_sources.append((source, metadata, end))
            end += source.num_frames
            self._source_ends.append(end)

    def _read_window(self, raw_index):
        return self.lerobot_dataset.get_item_with_offset_factory(raw_index, offsets_factory=lambda _: (list(range(0, 33, 4)), list(range(33)), list(range(32))), processor_method='preprocess_without_images', decode_images=False)

    def _expected_mapping(self, raw_index):
        source_id = int(np.searchsorted(self._source_ends, raw_index, side='right'))
        source, columns, selected_start = self._mapping_sources[source_id]
        local_row = raw_index - selected_start
        episode = int(columns['episode_index'][local_row])
        frame = int(columns['frame_index'][local_row])
        root = str(Path(source.root).resolve())
        dataset_index, global_start, episodes = self.visual_cache._source_roots[root]
        episode_start, length = episodes[episode]
        offsets = np.arange(0, 33, 4)
        frame_indices = np.minimum(frame + offsets, length - 1)
        selected_rows = local_row + frame_indices - frame
        require(np.all(columns['episode_index'][selected_rows] == episode) and np.array_equal(columns['frame_index'][selected_rows], frame_indices), 'noncontiguous source episode')
        self.visual_cache._check_source_files(source, episode)
        source_rows = episode_start + frame_indices
        return {'idx': global_start + episode_start + frame, 'dataset_index': dataset_index, 'dataset_root': root, 'local_row': episode_start + frame, 'episode_index': episode, 'frame_index': frame, 'camera_keys': ['observation.images.image', 'observation.images.wrist_image'], 'source_local_rows': source_rows.tolist(), 'source_global_rows': (source_rows + global_start).tolist(), 'source_frame_indices': frame_indices.tolist(), 'source_timestamps': columns['timestamp'][selected_rows].astype(float).tolist(), 'image_is_pad': (frame + offsets >= length).tolist(), 'rgb_offsets': offsets.tolist(), 'latent_causal_endpoint_offsets': [0, 16, 32]}

    def _visual_inputs(self, sample, raw_index):
        mapping = self._expected_mapping(raw_index)
        require(sample['image_is_pad'].tolist() == mapping['image_is_pad'], 'reader/cache padding differs')
        return {'video_latents': self.visual_cache.get(mapping), 'video_shape': torch.tensor([3, 3, 9, 256, 256]), 'camera_is_pad': torch.tensor([False, True, False]), 'visual_cache_index': mapping['idx']}

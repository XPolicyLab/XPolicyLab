from __future__ import annotations
from collections import OrderedDict
import json
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .adapters import ADAPTERS
from .schema import CAMERAS, fingerprint, within
from .stats import StatsRegistry, default_contract
from .sampling import CoverageSampler
from .timing import time_grids
from crwm.text_conditioning import PROMPT, MODEL_ID, TOKENIZER_ID, text_contract, cache_filename, cached_context

class CleanDataset(torch.utils.data.Dataset):

    def __init__(self, manifests, stats_path=None, *, split='train', horizon=32, stride=1, window_stride=1, weights=None, samples_per_epoch=None, context_len=128, require_full_coverage=True, normalize=True, mixture=None, normalization_mode=None, text_model_id=MODEL_ID, tokenizer_model_id=TOKENIZER_ID, allow_legacy_text_cache=True, pointmap_depth_validity='legacy_clip'):
        from crwm.pointmap_validity import validate_depth_validity
        self.pointmap_depth_validity = validate_depth_validity(pointmap_depth_validity)
        self.text_contract = text_contract(context_len, text_model_id, tokenizer_model_id)
        self.allow_legacy_text_cache = bool(allow_legacy_text_cache)
        self._legacy_text_warned = False
        self.manifests = [json.loads(Path(p).read_text()) if not isinstance(p, dict) else p for p in manifests]
        if len(self.manifests) != 1 or self.manifests[0]['source']['adapter'] != 'robotwin':
            raise ValueError('Exactly one RoboTwin Clean manifest is required')
        self.adapters = [ADAPTERS[m['source']['adapter']](m, split=split, horizon=horizon, stride=stride, window_stride=window_stride, language=None) for m in self.manifests]
        all_adapters = list(self.adapters)
        if split != 'train':
            self.adapters = [a for a in self.adapters if len(a) > 0]
        self.leaf_sizes = [len(a) for a in self.adapters]
        groups = {}
        for a in self.adapters:
            signature = {k: a.source.get(k) for k in ('embodiment', 'coordinate_frame', 'endpoint', 'protocol', 'fields')}
            key = a.source['stats_key']
            if key in groups and groups[key] != signature:
                raise ValueError(f'Statistics group {key} combines incompatible source semantics')
            groups[key] = signature
        if not self.leaf_sizes or any((n <= 0 for n in self.leaf_sizes)):
            raise ValueError(f'Empty {split} sampling leaf: {self.leaf_sizes}')
        self.ends = np.cumsum(self.leaf_sizes)
        self.horizon, self.stride, self.context_len = (horizon, stride, context_len)
        if horizon != 32:
            raise ValueError('The initial five-stream recipe requires 32 actions / nine visual frames')
        self.weights, self.samples_per_epoch = (weights, samples_per_epoch)
        if mixture is not None:
            raise ValueError('This release supports one fixed Clean sampling population')
        self.require_full_coverage = bool(require_full_coverage)
        self.contract = default_contract(manifests=[a.index.digest for a in self.adapters], horizon=horizon, stride=stride, window_stride=window_stride, training_split='train', time_layout='latent_groups_4_pairs.v1')
        from .manifest import WindowIndex
        self.contract['manifests'] = [WindowIndex(m, 'train', window_stride, None).digest for m in self.manifests]
        self.stats = StatsRegistry.load(stats_path, self.contract) if stats_path else None
        if self.stats:
            for adapter in all_adapters:
                if adapter.source.get('supervision_profile') != self.stats.groups[adapter.source['stats_key']].get('profile'):
                    raise ValueError('Source and statistics supervision profiles differ')
                if adapter.source.get('supervision_profile'):
                    from .robotwin_joint14 import validate_source_group
                    validate_source_group(adapter.source, self.stats.groups[adapter.source['stats_key']])
        if normalize and self.stats is None:
            raise ValueError('Generate training statistics before constructing a normalized dataset')
        self.normalize = normalize
        if self.stats and normalization_mode is not None:
            if any((g['mode'] != normalization_mode for g in self.stats.groups.values())):
                raise ValueError('Statistics normalization mode differs from the selected training stage')
        conditions = sorted({(a.source['embodiment'], a.source['protocol'], a.source['stats_key']) for a in all_adapters})
        self.conditions = {str(i): dict(embodiment=x[0], protocol=x[1], stats_key=x[2], normalization=self.stats.groups[x[2]]['mode'] if self.stats else 'uncomputed') for i, x in enumerate(conditions)}
        self.condition_ids = {x: i for i, x in enumerate(conditions)}
        self._videos, self._text = (OrderedDict(), OrderedDict())

    @property
    def checkpoint_contract(self):
        result = dict(data=self.contract, text=self.text_contract, stats_hash=self.stats.digest if self.stats else None, conditions=self.conditions, statistics=self.stats.to_dict() if self.stats else None, sources=[dict(source=a.source, metadata_hashes=a.manifest.get('metadata_hashes', {}), sidecar_provenance=a.sidecar.manifest['provenance'] if a.sidecar else None) for a in self.adapters])
        if self.pointmap_depth_validity != 'legacy_clip':
            result['pointmap_depth_validity'] = self.pointmap_depth_validity
        return result

    def make_sampler(self, **kwargs):
        return CoverageSampler(self.leaf_sizes, weights=self.weights, samples_per_epoch=self.samples_per_epoch, manifest_hash=fingerprint(self.contract), **kwargs)

    def __len__(self):
        return int(self.ends[-1])

    def locate(self, index):
        leaf = int(np.searchsorted(self.ends, index, side='right'))
        if index < 0 or leaf >= len(self.adapters):
            raise IndexError(index)
        return (leaf, index - (int(self.ends[leaf - 1]) if leaf else 0))

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_videos'], state['_text'] = (OrderedDict(), OrderedDict())
        return state

    def _video(self, adapter, spec, indices, slot, ep=None):
        path = within(adapter.root, spec['path'])
        key = str(path)
        if key not in self._videos:
            import decord
            from .media_identity import verify_video_identity
            verify_video_identity(path, spec)
            self._videos[key] = decord.VideoReader(str(path), num_threads=1)
            if len(self._videos) > 8:
                self._videos.popitem(last=False)
        self._videos.move_to_end(key)
        vr = self._videos[key]
        raw = np.asarray(indices) + round(spec.get('start_seconds', 0) * spec['fps'])
        if np.any(raw >= len(vr)) or np.any(raw < 0):
            raise ValueError('Video frame mapping outside media')
        frames = vr.get_batch(raw.tolist()).asnumpy()
        rgb = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 127.5 - 1
        hw = (256, 320) if slot == 'cam_high' else (224, 224)
        return F.interpolate(rgb, size=hw, mode='bilinear', align_corners=False, antialias=True).permute(1, 0, 2, 3)

    def _context(self, source, ep):
        from .language import episode_instruction
        task = ep.get('instruction', episode_instruction(ep))
        prompt = source.get('prompt_template', PROMPT).format(task=task)
        cache = Path(source['text_cache'])
        contract = self.text_contract
        key = str(cache / cache_filename(prompt, contract))
        if key not in self._text:
            payload = torch.load(key, map_location='cpu', weights_only=True)
            worker = torch.utils.data.get_worker_info()
            warn_legacy = not self._legacy_text_warned and os.environ.get('RANK', '0') == '0' and (worker is None or worker.id == 0)
            self._text[key] = cached_context(payload, prompt, contract, allow_legacy=self.allow_legacy_text_cache, warn_legacy=warn_legacy)
            self._legacy_text_warned |= 'text_contract' not in payload
            if len(self._text) > 128:
                self._text.popitem(last=False)
        self._text.move_to_end(key)
        context, mask = self._text[key]
        if context.ndim != 2 or context.shape[0] != self.context_len or mask.shape != (self.context_len,):
            raise ValueError(f'Invalid text context: {key}')
        return (context, mask)

    def __getitem__(self, item):
        index, occurrence = item if isinstance(item, tuple) else (int(item), int(item))
        leaf, local = self.locate(index)
        adapter = self.adapters[leaf]
        try:
            labels, ep, anchor, end = adapter.labels(local)
            source = adapter.source
            offsets = np.arange(9) * 4 * self.stride
            frame_valid = anchor + offsets < end
            indices = np.minimum(anchor + offsets, end - 1)
            per_cam, camera_valid = ({}, np.zeros(3, bool))
            for i, slot in enumerate(CAMERAS):
                spec = ep.get('cameras', {}).get(slot)
                camera_valid[i] = spec is not None
                per_cam[slot] = self._video(adapter, spec, indices, slot, ep) if spec else torch.zeros((3, 9, *((256, 320) if i == 0 else (224, 224))))
            context, context_mask = self._context(source, ep)
            key = source['stats_key']
            clipping = []
            if self.normalize:
                for kind in ('state', 'action'):
                    labels[kind], report = self.stats.normalize(key, kind, labels[kind], labels[kind + '_valid'], report=True)
                    clipping.append([report['clipped'], report['valid']])
            grid, tv = time_grids(source.get('time_protocol'), anchor, self.stride, self.horizon, offsets, end)
            condition = (source['embodiment'], source['protocol'], key)
            sample = {k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in labels.items()}
            sample['episode_id'] = str(labels['episode_id'])
            sample['normalization_counts'] = torch.tensor(clipping or [[0, 0], [0, 0]])
            sample.update(proprio=sample['state'][None], per_cam=per_cam, context=context, context_mask=context_mask, instruction=ep.get('instruction', ''), language_segment_id=torch.tensor(ep.get('language_segment_id', -1)), language_status=ep.get('language_status', 'episode'), action_is_pad=~sample['action_valid'].any(-1), image_is_pad=torch.from_numpy(~frame_valid), camera_valid=torch.from_numpy(camera_valid), modality_valid=torch.tensor([camera_valid.any(), True, camera_valid.any(), False]), time_grid={k: torch.from_numpy(v) for k, v in grid.items()}, time_valid={k: torch.from_numpy(v) for k, v in tv.items()}, condition_id=torch.tensor(self.condition_ids[condition]), source_id=source['source_id'], stats_key=key, sample_position=torch.tensor(occurrence), window_index=torch.tensor(index), pointmap_half_pixel=torch.tensor(False), camera_intrinsics=torch.eye(3).repeat(3, 1, 1), pointmap_pixel_valid=torch.zeros(9, 384, 320, dtype=torch.bool), per_cam_depth={slot: torch.zeros(9, 240, 320, dtype=torch.int32) for slot in CAMERAS})
            sample['modality_valid'][1] = sample['action_valid'].any()
            if source.get('auxiliary'):
                from .auxiliary import load_auxiliary
                load_auxiliary(sample, adapter, ep, anchor, indices, frame_valid, pointmap_depth_validity=self.pointmap_depth_validity)
            return sample
        except Exception as exc:
            raise RuntimeError(f"source={adapter.source['source_id']} window={local} occurrence={occurrence}: {exc}") from exc

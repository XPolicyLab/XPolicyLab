"""One reproducible crop/resize choice for every physical RGB chunk."""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from wam.datasets.local_robot_package import read_json, readonly_db, unpack
from wam.datasets.multiview_plan import file_sha256, stable_hash


def choose_resize(seed, identity):
    # Stateless: rank, worker scheduling and camera/view identity cannot change it.
    return int(stable_hash([int(seed), 'rgb_resize_crop_v1', identity]), 16) % 2 == 0


class RGBVariants:
    def __init__(self, root, *, expected_sha256, package_sha256):
        self.root = Path(root)
        self.fingerprint = file_sha256(self.root / 'manifest.json')
        manifest = read_json(self.root / 'manifest.json')
        if (
            self.fingerprint != expected_sha256
            or read_json(self.root / 'READY.json')['manifest_sha256'] != self.fingerprint
            or manifest['schema'] != 'phoenix.robot37.rgb_variants.v1'
            or manifest['package_sha256'] != package_sha256
            or manifest['policy'] != 'physical_chunk_hash_50_50_v1'
            or manifest['resize_probability'] != 0.5
        ):
            raise ValueError('RGB variants manifest/policy/package identity mismatch')
        for name, key in (
            ('videos.sqlite', 'video_index_sha256'),
            ('root_audit.jsonl', 'root_audit_sha256'),
        ):
            if file_sha256(self.root / name) != manifest[key]:
                raise ValueError('RGB variants artifact changed: ' + name)
        self.available_roots = frozenset(manifest['roots'])
        self.excluded_roots = frozenset(manifest['excluded_roots'])
        if self.available_roots & self.excluded_roots:
            raise ValueError('Excluded roots cannot have an RGB variant')
        self.contract = dict(
            manifest_sha256=self.fingerprint,
            policy=manifest['policy'],
            resize_probability=0.5,
        )
        self._db, self._pid = None, None

    def __getstate__(self):
        return dict(self.__dict__, _db=None, _pid=None)

    @property
    def db(self):
        if self._db is None or self._pid != os.getpid():
            if self._db is not None:
                self._db.close()
            self._db = readonly_db(self.root / 'videos.sqlite')
            self._pid = os.getpid()
        return self._db

    @lru_cache(maxsize=32)
    def _episode(self, source_root, episode_index):
        row = self.db.execute(
            'SELECT payload FROM videos WHERE source_root=? AND episode_index=?',
            (source_root, episode_index),
        ).fetchone()
        if row is None:
            raise ValueError('Published RGB root is missing an episode binding')
        return unpack(row[0])

    def select(self, record, *, seed, identity):
        source = record['source_root']
        if source in self.excluded_roots:
            raise ValueError('Excluded RGB root reached the reader')
        if source not in self.available_roots:
            return record, 'existing_only'
        alternate = self._episode(source, int(record['episode_index']))
        for key in (
            'episode_index',
            'length',
            'dataset_from_index',
            'dataset_to_index',
            'fps',
        ):
            if alternate[key] != record[key]:
                raise ValueError(
                    'RGB variant disagrees with the frozen numeric episode: ' + key
                )
        selected = record['selected_views']
        if any(view['key'] not in alternate['views'] for view in selected):
            raise ValueError('RGB alternative is missing a selected base/wrist camera')
        if not choose_resize(seed, identity):
            return record, 'crop_resize'
        # Use the new per-episode offsets and video shard, not a prefix replacement.
        views = [dict(view, **alternate['views'][view['key']]) for view in selected]
        return dict(record, selected_views=views), 'direct_resize'


def load_selection(root, *, expected_sha256, package_sha256, parent_plan_sha256):
    root = Path(root)
    fingerprint = file_sha256(root / 'manifest.json')
    selection = read_json(root / 'manifest.json')
    if (
        fingerprint != expected_sha256
        or read_json(root / 'READY.json')['manifest_sha256'] != fingerprint
        or selection['schema'] != 'phoenix.robot37.rgb_selection.v1'
        or selection['package_sha256'] != package_sha256
        or selection['parent_plan_sha256'] != parent_plan_sha256
    ):
        raise ValueError('RGB exclusion selection identity mismatch')
    return selection, fingerprint

from cloudrobo_wm.storage import DataPath as Path
import hashlib
import json
from .image_augmentation import AugmentedRequiredModalitiesDataset
from cloudrobo_wm.text_conditioning import cache_filename

class InstructionOverlayAugmentedDataset(AugmentedRequiredModalitiesDataset):

    def __init__(self, *, instruction_overlay, **kwargs):
        self.overlay_root = Path(instruction_overlay['root'])
        self.overlay_group = instruction_overlay.get('source_group', 'clean')
        if kwargs.get('split', 'train') != 'train' or self.overlay_group != 'clean':
            raise ValueError('This experiment admits only the original Clean training population')
        raw = (self.overlay_root / 'metadata/selected_training_reader_rows.json').read_bytes()
        selected = json.loads(raw)
        if selected['revision_id'] != instruction_overlay['revision_id']:
            raise ValueError('Instruction revision mismatch')
        self.overlay_sha = hashlib.sha256(raw).hexdigest()
        if self.overlay_sha != instruction_overlay['reader_rows_sha256']:
            raise ValueError('Selected instruction list SHA mismatch')
        jobs_raw = (self.overlay_root / 'cache/cache_jobs.json').read_bytes()
        if hashlib.sha256(jobs_raw).hexdigest() != instruction_overlay['jobs_file_sha256']:
            raise ValueError('Instruction cache jobs SHA mismatch')
        jobs = json.loads(jobs_raw)
        if jobs['revision_id'] != selected['revision_id'] or not jobs['no_evaluation_labels']:
            raise ValueError('Non-training or wrong-revision cache')
        byfile = {row['filename']: row for row in jobs['jobs']}
        self.overlay_rows = {}
        for row in selected['reader_episodes']:
            if row['source_group'] != self.overlay_group:
                continue
            key = (row['task'], int(row['episode_id']))
            if key in self.overlay_rows:
                raise ValueError(('Duplicate instruction', key))
            if row['kind'] != 'train' or row['selected_instruction_index'] != 0:
                raise ValueError(('Unexpected selected instruction', key))
            text = row['selected_instruction']
            prompt = selected['prompt_template'].format(task=text)
            if text != row['tasks0'] or hashlib.sha256(text.encode()).hexdigest() != row['instruction_sha256']:
                raise ValueError(('Instruction SHA mismatch', key))
            if prompt != row['prompt'] or hashlib.sha256(prompt.encode()).hexdigest() != row['prompt_sha256']:
                raise ValueError(('Prompt mismatch', key))
            if byfile[row['filename']]['prompt'] != prompt:
                raise ValueError(('Job mismatch', key))
            self.overlay_rows[key] = row
        self.overlay_revision = selected['revision_id']
        super().__init__(**kwargs)
        self._overlay_episode_rows = {}
        population = set()
        for adapter in self.adapters:
            if adapter.source['adapter'] != 'robotwin':
                raise ValueError('Overlay only supports RoboTwin')
            for ep in adapter.manifest['episodes']:
                key = (ep['clean_task'], int(ep['episode_id']))
                if key not in self.overlay_rows:
                    raise ValueError(('Missing instruction', key))
                row = self.overlay_rows[key]
                if cache_filename(row['prompt'], self.text_contract) != row['filename']:
                    raise ValueError(('Cache filename mismatch', key))
                if not (self.overlay_root / 'cache' / row['filename']).is_file():
                    raise FileNotFoundError(row['filename'])
                eid = str(ep['episode_id'])
                if eid in self._overlay_episode_rows:
                    raise ValueError(('Ambiguous episode', eid))
                self._overlay_episode_rows[eid] = row
                population.add(key)
        if len(population) != 2500 or population != set(self.overlay_rows):
            raise ValueError(('Original Clean population changed', len(population)))

    def _context(self, source, ep, *, anchor=None, occurrence=0):
        row = self.overlay_rows[ep['clean_task'], int(ep['episode_id'])]
        derived_source = {**source, 'text_cache': str(self.overlay_root / 'cache')}
        derived_ep = {**ep, 'instruction': row['selected_instruction']}
        return super()._context(derived_source, derived_ep)

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        sample['instruction'] = self._overlay_episode_rows[str(sample['episode_id'])]['selected_instruction']
        return sample

    @property
    def checkpoint_contract(self):
        value = dict(super().checkpoint_contract)
        value['training_instruction_overlay'] = dict(revision_id=self.overlay_revision, source_group=self.overlay_group, selected_reader_rows_sha256=self.overlay_sha, root=str(self.overlay_root), text_cache=str(self.overlay_root / 'cache'), selected_index=0, original_manifests_unchanged=True)
        return value

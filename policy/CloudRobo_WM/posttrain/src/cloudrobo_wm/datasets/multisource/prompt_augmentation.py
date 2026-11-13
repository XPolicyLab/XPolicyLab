from collections import OrderedDict
from cloudrobo_wm.storage import DataPath as Path
import hashlib, json, mmap, struct
from .instruction_overlay import InstructionOverlayAugmentedDataset
from cloudrobo_wm.text_conditioning import cached_context, cache_filename

class CleanPromptIndex:

    def __init__(self, root, manifest_sha, text_contract):
        self.root = Path(root)
        raw = (self.root / 'manifest.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_sha:
            raise ValueError('Prompt index manifest SHA changed')
        self.manifest = json.loads(raw)
        if self.manifest.get('version') != 'robotwin.clean-prompt-augmentation.v1' or not self.manifest.get('ready_for_training') or self.manifest.get('evaluation_data_included') or (self.manifest['text_contract'] != text_contract):
            raise ValueError('Unadmitted prompt augmentation contract')
        self.handles = {}
        self.maps = {}
        for name, row in self.manifest['files'].items():
            f = (self.root / name).open('rb')
            if f.seek(0, 2) != row['bytes']:
                raise ValueError('Prompt index length changed')
            self.handles[name] = f
            self.maps[name] = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self):
        for v in self.maps.values():
            v.close()
        for v in self.handles.values():
            v.close()
        self.maps = {}
        self.handles = {}

    def select(self, episode_id, anchor, occurrence, seed, epoch_samples):
        if not 0 <= episode_id < self.manifest['episode_id_bound'] or anchor < 0 or occurrence < 0:
            raise ValueError('Invalid prompt sample key')
        a, b = struct.unpack_from('<QQ', self.maps['episode_offsets.u64le'], episode_id * 8)
        if a == b:
            raise ValueError('Episode outside verified Clean population')
        epoch = occurrence // epoch_samples
        key = json.dumps([self.manifest['revision_id'], seed, epoch, episode_id, anchor, occurrence], separators=(',', ':')).encode()
        position = a + int.from_bytes(hashlib.sha256(key).digest(), 'big') % (b - a)
        pid = struct.unpack_from('<I', self.maps['candidate_ids.u32le'], position * 4)[0]
        digest = self.maps['prompt_shas.ascii64'][pid * 64:(pid + 1) * 64].decode()
        x, y = struct.unpack_from('<QQ', self.maps['prompt_text_offsets.u64le'], pid * 8)
        prompt = self.maps['prompt_texts.utf8'][x:y].decode()
        if hashlib.sha256(prompt.encode()).hexdigest() != digest:
            raise ValueError('Selected prompt SHA differs')
        prefix, suffix = self.manifest['prompt_template'].split('{task}')
        if not prompt.startswith(prefix) or (suffix and (not prompt.endswith(suffix))):
            raise ValueError('Prompt prefix differs')
        instruction = prompt[len(prefix):len(prompt) - len(suffix) if suffix else None]
        filename = cache_filename(prompt, self.manifest['text_contract'])
        if not filename.startswith(digest):
            raise ValueError('Cache filename mismatch')
        return dict(prompt=prompt, instruction=instruction, sha256=digest, path=str(self.root / 'vectors' / filename), epoch=epoch)

class PromptAugmentedInstructionDataset(InstructionOverlayAugmentedDataset):

    def __init__(self, *, prompt_augmentation=None, **kwargs):
        self.prompt_config = dict(prompt_augmentation or {})
        self.prompt_enabled = bool(self.prompt_config.get('enabled', False))
        self._prompt_index = None
        self._prompt_vectors = OrderedDict()
        self._prompt_selected = None
        super().__init__(**kwargs)
        if self.prompt_enabled:
            if self.prompt_config['seed'] != 42 or not isinstance(self.prompt_config['epoch_samples'], int) or self.prompt_config['epoch_samples'] <= 0:
                raise ValueError('Unexpected prompt sampling configuration')
            if not 0 <= self.prompt_config.get('max_cached_vectors', 64) <= 128:
                raise ValueError('Bounded prompt LRU required')
            index = self._index()
            actual = sorted((int(v['episode_id']) for v in self._overlay_episode_rows.values()))
            if actual != index.manifest['clean_episode_ids']:
                raise ValueError('Prompt and actual Clean populations differ')
            index.close()
            self._prompt_index = None

    def _index(self):
        if self._prompt_index is None:
            self._prompt_index = CleanPromptIndex(self.prompt_config['root'], self.prompt_config['manifest_sha256'], self.text_contract)
        return self._prompt_index

    def __getstate__(self):
        state = super().__getstate__()
        state['_prompt_index'] = None
        state['_prompt_vectors'] = OrderedDict()
        state['_prompt_selected'] = None
        return state

    def make_sampler(self, **kwargs):
        sampler = super().make_sampler(**kwargs)
        if self.prompt_enabled:
            world = sampler.contract['world_size']
            if sampler.samples_per_epoch != self.prompt_config['epoch_samples']:
                raise ValueError('Prompt epoch and actual sampler differ')
        return sampler

    def _context(self, source, ep, *, anchor=None, occurrence=0):
        if not self.prompt_enabled:
            return super()._context(source, ep, anchor=anchor, occurrence=occurrence)
        if anchor is None:
            raise ValueError('Augmented text needs the actual window anchor')
        selected = self._index().select(int(ep['episode_id']), int(anchor), int(occurrence), self.prompt_config['seed'], self.prompt_config['epoch_samples'])
        self._prompt_selected = selected
        key = selected['sha256']
        if key in self._prompt_vectors:
            value = self._prompt_vectors.pop(key)
            self._prompt_vectors[key] = value
            return value
        import torch
        payload = torch.load(selected['path'], map_location='cpu', weights_only=True)
        value = cached_context(payload, selected['prompt'], self.text_contract, allow_legacy=False)
        if self.prompt_config.get('max_cached_vectors', 64):
            self._prompt_vectors[key] = value
            while len(self._prompt_vectors) > self.prompt_config.get('max_cached_vectors', 64):
                self._prompt_vectors.popitem(last=False)
        return value

    def __getitem__(self, item):
        self._prompt_selected = None
        sample = super().__getitem__(item)
        if self.prompt_enabled:
            if self._prompt_selected is None:
                raise ValueError('No selected prompt for this complete clip')
            sample['instruction'] = self._prompt_selected['instruction']
            sample['prompt_sha256'] = self._prompt_selected['sha256']
        return sample

    @property
    def checkpoint_contract(self):
        value = dict(super().checkpoint_contract)
        if self.prompt_enabled:
            value['training_prompt_augmentation'] = {**self.prompt_config, 'version': 'robotwin.clean-prompt-augmentation.v1', 'selection': 'sha256(seed,epoch,episode,anchor,global_occurrence,revision)', 'clip_shared': True, 'original_sampling_weights_unchanged': True}
        return value

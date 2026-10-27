"""Online RGB augmentation with strict clean selection and native joint14 labels."""
import json
from pathlib import Path
import torch
from .joint96_posttrain import Joint96PosttrainDataset
from .lerobot.domain_aug_dataset import DomainAugmentedRobotVideoDataset, sample_domain
from .lerobot.robot_video_dataset import DEFAULT_PROMPT
from .robotwin_joint96_io import BoundedJoint96RGB, official_text_padding
from .text_embedding_cache import TaskIndexShardedTextEmbeddingCache

def clean_metadata(root):
    root=Path(root);info=json.loads((root/'meta/info.json').read_text())
    if info['codebase_version']!='v2.1' or info['total_episodes']!=27500 or info['fps']!=50 or info['robot_type']!='aloha':
        raise ValueError('Expected audited ALOHA / 27500-episode RoboTwin v2.1 export')
    if any(info['features'][k]['shape']!=[14] for k in ('action','observation.state')):
        raise ValueError('Expected native joint14')
    episodes=[]
    for line in (root/'meta/episodes.jsonl').read_text().splitlines():
        d=json.loads(line)
        if int(d['episode_index'])%550<50:episodes.append([int(d['episode_index']),int(d['length'])])
    episodes.sort()
    if len(episodes)!=2500 or len({e[0] for e in episodes})!=2500:
        raise ValueError('Expected all 2500 official clean episodes')
    return info,episodes

class RobotwinJoint96OnlineDataset(BoundedJoint96RGB,DomainAugmentedRobotVideoDataset,Joint96PosttrainDataset):
    def __init__(self,*,text_cache_source_root=None,**kwargs):
        if kwargs.get('domain')!='robotwin' or 'visual_cache' in kwargs:
            raise ValueError('Online RoboTwin only; no visual cache')
        # Accept the legacy argument from saved configs without importing it.
        self.text_cache_source_root=text_cache_source_root;self._text_cache=None
        super().__init__(**kwargs)
        datasets=self.lerobot_dataset.multi_dataset._datasets
        if len(datasets)!=1:raise ValueError('Expected one canonical data root')
        self._raw_dataset=datasets[0];self._task_lookup=self._raw_dataset.meta.task_to_task_index
        _,episodes=clean_metadata(self.dataset_dirs[0])
        if len(self)!=sum(max(0,n-4) for _,n in episodes) or [int(x) for x in self._raw_dataset.episodes]!=[e[0] for e in episodes]:
            raise ValueError('Clean episode/window selection changed')

    def __getitem__(self,index):
        previous=self._domain_params
        self._domain_params=sample_domain(self.image_domain_jitter,self.is_training_set,self.video_view_names)
        try:return super().__getitem__(index)
        finally:self._domain_params=previous

    def _get_cached_text_context(self,prompt):
        if self._text_cache is None:
            bank=TaskIndexShardedTextEmbeddingCache(self.text_embedding_cache_dir,expected_context_len=128)
            if bank.embedding_dim!=4096 or bank.manifest['prompt_template']!=DEFAULT_PROMPT:
                raise ValueError('T5 template/width mismatch')
            # get() validates the actual task prompt against its indexed cache.
            # Do not require a fingerprint of the dataset metadata file.
            self._text_cache=bank
        prefix=DEFAULT_PROMPT.split('{task}')[0]
        if not prompt.startswith(prefix):raise ValueError('Unexpected T5 prompt')
        context,mask=self._text_cache.get(self._task_lookup[prompt[len(prefix):]],prompt)
        while len(self._text_cache._open_shards)>32:self._text_cache._open_shards.pop(next(iter(self._text_cache._open_shards)))
        return official_text_padding(context,mask)

"""Opt-in Robot37 prompt transfer for RoboTwin's ARX5-based ALOHA asset."""
import copy
import hashlib
from pathlib import Path
import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf

def validate_robotwin_prompt(cfg):
    get=lambda k,d=None:OmegaConf.select(cfg,k,default=d)
    if not get('joint96_posttrain.enabled') or get('local_robot_joint96.enabled') or get('resume') is not None or get('local_robot_softprompt.init_checkpoint') is not None:
        raise ValueError('RoboTwin requires fresh weights-only initialization')
    if any(get(f'data.{s}.domain')!='robotwin' for s in ('train','val')):
        raise ValueError('RoboTwin-only prompt recipe')
    if get('joint96_posttrain.robotwin_softprompt.embodiment_id')!=2:
        raise ValueError('Pinned ARX5 mechanical family uses pretrained slot 2')
    if (get('model.action_dit_config.num_embodiments'),get('model.action_dit_config.soft_prompt_tokens'))!=(28,8):
        raise ValueError('Keep complete 28-slot / 8-token prompt table')

class RobotwinPromptDataset(Dataset):
    def __init__(self,dataset,registry):
        if registry.content['embodiments'][2]!='arx_x5' or 2 not in registry.assigned_ids:
            raise ValueError('Wrong pinned ARX5 registry')
        self.dataset=dataset;self.embodiment_registry=registry
        if hasattr(dataset,'fixed_validation_panel'):
            self.fixed_validation_panel=copy.deepcopy(dataset.fixed_validation_panel)
            for row in self.fixed_validation_panel:row['embodiment_id']=2
    def __getattr__(self,name):
        dataset=self.__dict__.get('dataset')
        if dataset is None:raise AttributeError(name)
        return getattr(dataset,name)
    def __len__(self):return len(self.dataset)
    def __getitem__(self,index):return {**self.dataset[index],'embodiment_id':torch.tensor(2,dtype=torch.long)}
    def collate_fn(self,samples):
        for sample in samples:
            value=sample.get('embodiment_id')
            if not isinstance(value,torch.Tensor) or value.dtype!=torch.long or value.ndim!=0 or int(value)!=2:
                raise ValueError('Expected scalar ARX5 prompt id')
        collate=getattr(self.dataset,'collate_fn',None)
        if collate is None:
            from torch.utils.data import default_collate
            batch=default_collate(samples)
        else:batch=collate(samples)
        ids=torch.stack([s['embodiment_id'] for s in samples])
        if 'embodiment_id' in batch and not torch.equal(batch['embodiment_id'],ids):raise ValueError('Prompt ID changed in collate')
        batch['embodiment_id']=ids
        return batch

def wrap_robotwin_prompt(model,train,val,cfg,registry):
    if val is None or train is val:raise ValueError('Need distinct training and validation readers')
    model.soft_prompt_training_contract={'domain':'robotwin','embodiment_id':2,'embodiment_name':'arx_x5',
        'simulator_asset':'aloha-agilex/arx5_description_isaac.urdf',
        'action_space':'native_joint14_to_canonical80','normalization_sha256':None,
        'cached_visuals':False}
    return tuple(RobotwinPromptDataset(ds,registry) for ds in (train,val))

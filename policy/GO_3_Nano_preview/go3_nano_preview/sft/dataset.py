from go3_nano_preview.sft.data.unified_dataset.vid_pretrain_dataset_with_errors import MixMultiFrame
from policy_core.utils.WAM_policy.WAM_policy_config import as_dataset_kwargs

class RobotwinDataset(MixMultiFrame):
    def __getitem__(self, index):
        return self.get_batch_new(int(index) % self.true_length)

def resolve_WAM_policy_dataset_class(dataset_type):
    if dataset_type != 'mix_multi_frame':
        raise ValueError('Only the reference mix_multi_frame dataset is supported')
    return RobotwinDataset

def make_dataset(cfg):
    kwargs = as_dataset_kwargs(cfg.data.train)
    cls = resolve_WAM_policy_dataset_class(kwargs.pop('dataset_type'))
    kwargs.pop('metadata_index_manifest', None)
    kwargs.update(target_dim=cfg.model.action.action_in_channels,
        mode='train', use_sampling_weights=True,
        force_goal_on_empty_caption=cfg.data.force_goal_on_empty_caption)
    return cls(**kwargs)

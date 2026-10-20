from .dataset import CleanDataset

class RequiredModalitiesDataset(CleanDataset):

    def __init__(self, *, required_modalities, **kwargs):
        self.required_modalities = tuple(required_modalities)
        if set(self.required_modalities) - {'pointmap'}:
            raise ValueError('Only pointmap and flow are supported as required modalities')
        super().__init__(**kwargs)
        if 'pointmap' in self.required_modalities:
            for adapter in self.adapters:
                auxiliary = adapter.source.get('auxiliary', {})
                if not (auxiliary.get('depth') or auxiliary.get('depth_root')):
                    raise ValueError(f"Source {adapter.source['source_id']} requires pointmap supervision, but its manifest does not enable depth. Rebuild the manifest and its matching statistics using the pointmap source recipe.")

    @property
    def checkpoint_contract(self):
        return {**super().checkpoint_contract, 'required_modalities': list(self.required_modalities)}

    def __getitem__(self, item):
        sample = super().__getitem__(item)
        if 'pointmap' in self.required_modalities and (not bool(sample['modality_valid'][3])):
            raise RuntimeError(f"Required pointmap supervision is missing: source={sample['source_id']} episode={sample['episode_id']} window={sample['window_index']}. Check all three depth videos and camera intrinsics; refusing a silent zero loss.")
        if 'flow' in self.required_modalities and (not bool(sample['modality_valid'][4])):
            raise RuntimeError(f"Required flow supervision is missing: source={sample['source_id']} episode={sample['episode_id']} window={sample['window_index']}; check all three flow caches")
        return sample

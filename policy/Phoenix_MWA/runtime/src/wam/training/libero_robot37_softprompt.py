"""Opt-in LIBERO transfer of the complete Robot37 embodiment-prompt table."""
import copy
import hashlib
from pathlib import Path

import torch
from torch.utils.data import Dataset
from omegaconf import OmegaConf


def validate_libero_prompt(cfg):
    get = lambda key, default=None: OmegaConf.select(cfg, key, default=default)
    if not get("joint96_posttrain.enabled") or get("local_robot_joint96.enabled"):
        raise ValueError("LIBERO prompts require a fresh weights-only posttraining recipe")
    if get("resume") is not None or get("local_robot_softprompt.init_checkpoint") is not None:
        raise ValueError("Load the already-trained prompt with init_checkpoint, not old-to-prompt migration")
    if any(get(f"data.{split}.domain") != "libero" for split in ("train", "val")):
        raise ValueError("This prompt mapping is LIBERO-only")
    if get("joint96_posttrain.libero_softprompt.embodiment_id") != 0:
        raise ValueError("LIBERO Panda must use pretrained embodiment_id=0")
    if (get("model.action_dit_config.num_embodiments"), get("model.action_dit_config.soft_prompt_tokens")) != (28, 8):
        raise ValueError("Preserve the 28-slot / 8-token Robot37 prompt structure")


class LiberoPromptDataset(Dataset):
    def __init__(self, dataset, registry, embodiment_id=0):
        if registry.content["embodiments"][embodiment_id] != "franka_panda" or embodiment_id not in registry.assigned_ids:
            raise ValueError("The pinned registry does not assign slot 0 to Panda")
        self.dataset, self.embodiment_registry, self.embodiment_id = dataset, registry, embodiment_id
        if hasattr(dataset, "fixed_validation_panel"):
            self.fixed_validation_panel = copy.deepcopy(dataset.fixed_validation_panel)
            for row in self.fixed_validation_panel:
                row["embodiment_id"] = embodiment_id

    def __getattr__(self, name):
        dataset = self.__dict__.get("dataset")
        if dataset is None:
            raise AttributeError(name)
        return getattr(dataset, name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "embodiment_id": torch.tensor(self.embodiment_id, dtype=torch.long)}

    def collate_fn(self, samples):
        for sample in samples:
            value = sample.get("embodiment_id")
            if not isinstance(value, torch.Tensor) or value.dtype != torch.long or value.ndim or int(value) != self.embodiment_id:
                raise ValueError("Expected scalar int64 LIBERO Panda prompt id")
        collate = getattr(self.dataset, "collate_fn", None)
        if collate is None:
            from torch.utils.data import default_collate
            return default_collate(samples)
        batch = collate(samples)
        ids = torch.stack([sample["embodiment_id"] for sample in samples])
        if "embodiment_id" in batch and not torch.equal(batch["embodiment_id"], ids):
            raise ValueError("Base collator modified embodiment IDs")
        batch["embodiment_id"] = ids
        return batch


def wrap_libero_prompt(model, train, val, cfg, registry):
    if val is None or train is val:
        raise ValueError("LIBERO posttraining requires distinct train/validation readers")
    model.soft_prompt_training_contract = {
        "domain": "libero", "embodiment_id": 0, "embodiment_name": "franka_panda",
        "frame_contract": cfg.joint96_posttrain.libero_frame_contract,
        "normalization_sha256": None,
        "cached_visuals": bool(cfg.joint96_posttrain.cached_visuals),
    }
    return tuple(LiberoPromptDataset(ds, registry) for ds in (train, val))

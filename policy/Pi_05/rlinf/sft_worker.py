"""Reuse v0.4 native SFT; customize only XPolicyLab data and checkpoint metadata."""

from .sft import build_dataloader, save_metadata
from rlinf.workers.sft.fsdp_vla_sft_worker import FSDPVlaSftWorker


class Pi05SftWorker(FSDPVlaSftWorker):
    def build_dataloader(self, data_paths, eval_dataset=False):
        if eval_dataset:
            raise ValueError(
                "VLA SFT validation is unsupported; run policy evaluation separately."
            )
        return build_dataloader(self.cfg, self._world_size)

    def save_checkpoint(self, save_path, step=0):
        super().save_checkpoint(save_path, step)
        if self._rank == 0:
            save_metadata(self.cfg, save_path)

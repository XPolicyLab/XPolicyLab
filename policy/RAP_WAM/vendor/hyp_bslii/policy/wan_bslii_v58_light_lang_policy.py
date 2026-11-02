"""WanBSLIIv58LightLangPolicy -- v58_light + per-task LANGUAGE conditioning.

NEW FILE, purely additive: a thin subclass of WanBSLIIv58LightPolicy.  No existing
policy / dataset / config / psi_policy file is modified.

Mechanism
---------
Training never loads T5/umt5, so per-task text contexts are precomputed offline into
a table ``[35, 512, 4096]`` (float32, already right-padded with zeros past each task's
true token count).  This policy:

  * holds that table as a non-persistent buffer (moves with ``.to(device)``);
  * stashes the current batch's ``task_index`` ([B] Long) at the top of
    ``compute_loss`` (which every training forward path -- base compute_loss,
    DAOPSR ``_ccrt_chunk1_loss``, PUGA ``_puga_forward_video`` -- funnels through);
  * overrides ``_build_context_list(batch_size)`` so that, whenever a task_index is
    stashed, it returns per-sample ``[512, 4096]`` contexts pulled from the table
    (in the model's text-embedding dtype).  ``WanJointActionPolicy._build_text_context``
    then right-pads/stacks them and feeds ``model.text_embedding`` -- exactly the
    path the empty ``[0, 4096]`` contexts use today, so cross-attn now actually sees
    the task instruction.

When ``task_index`` is absent (e.g. inference / a non-lang batch) it transparently
falls back to the parent's empty-context behaviour.
"""
from typing import Any, Dict, List, Optional

import torch

from hyp_bslii.policy.wan_bslii_v58_light_policy import WanBSLIIv58LightPolicy


class WanBSLIIv58LightLangPolicy(WanBSLIIv58LightPolicy):
    def __init__(self, *args, task_ctx_path: str, **kwargs):
        super().__init__(*args, **kwargs)
        loaded = torch.load(task_ctx_path, map_location="cpu")
        tab = loaded["embedding_table"].float()  # [35, text_len, text_dim]
        if tab.dim() != 3:
            raise ValueError(f"task_ctx table must be 3D [N, L, D], got {tuple(tab.shape)}")
        # register as a non-persistent buffer so .to(device) moves it with the module
        self.register_buffer("_task_ctx", tab, persistent=False)
        self._cur_task_index: Optional[torch.Tensor] = None
        print(
            f"[v58_light_lang] task_ctx table {tuple(tab.shape)} loaded from {task_ctx_path}"
        )

    # every training forward path funnels through compute_loss -> stash task_index here
    def compute_loss(self, batch: dict, training: bool = True):
        self._cur_task_index = batch.get("task_index")
        try:
            return super().compute_loss(batch, training=training)
        finally:
            self._cur_task_index = None

    # consume the stashed task_index; return per-sample [text_len, text_dim] contexts
    def _build_context_list(self, batch_size: int) -> List[torch.Tensor]:
        ti = self._cur_task_index
        if ti is None:
            return super()._build_context_list(batch_size)
        md = self.model.text_embedding[0].weight.dtype
        ctx = self._task_ctx  # buffer already on self.device
        return [
            ctx[int(i)].to(device=self.device, dtype=md)
            for i in ti
        ]


__all__ = ["WanBSLIIv58LightLangPolicy"]

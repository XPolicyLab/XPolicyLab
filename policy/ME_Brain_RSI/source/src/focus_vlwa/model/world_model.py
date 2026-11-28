"""WorldModelExpert operations for Focus-VLWA."""

import logging

import torch


class WorldModelExpert:
    def embed_world_model(self, noisy_world_model: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """WM tokens = [1 event-action | noisy world_state ×270]. att=0 so they share the
        action-expert block (Action ↔ WM bidirectional; VLM still isolated)."""

        def world_model_projection_func(noisy_world_model):
            return self.world_model_input_projection(noisy_world_model)

        embs = self._apply_checkpoint(world_model_projection_func, noisy_world_model)
        bsize, horizon = embs.shape[:2]
        pad_masks = torch.ones(bsize, horizon, dtype=torch.bool, device=embs.device)
        att = [0] * horizon
        att_masks = torch.tensor(att, dtype=embs.dtype, device=embs.device)[None, :].expand(bsize, horizon)
        return embs, pad_masks, att_masks

    def copy_action_expert_into_world_model(self):
        """Initialize WM Gemma + in/out proj from the action expert (pretrained copy)."""
        if not self.use_world_model:
            return
        wm = self.joint_experts.world_model_expert
        act = self.joint_experts.action_expert
        missing, unexpected = wm.load_state_dict(act.state_dict(), strict=True)
        if missing or unexpected:
            raise RuntimeError(f"WM copy mismatch missing={missing} unexpected={unexpected}")
        self.world_model_input_projection.load_state_dict(self.action_input_projection.state_dict())
        self.world_model_output_projection.load_state_dict(self.action_output_projection.state_dict())
        logging.info(
            "Copied action expert → WM expert (gemma + in/out proj), tokens=%s (event 1 + world_state %s) dim=%s",
            self.world_model_total_horizon,
            self.world_model_horizon,
            self.world_model_dim,
        )

    def freeze_world_model_expert(self, *, skip_supervision: bool = False) -> int:
        """Stop WM expert / in-out proj updates. Forward still runs so action can attend."""
        if not self.use_world_model:
            return 0
        mods = [self.world_model_input_projection, self.world_model_output_projection]
        wm = getattr(self.joint_experts, "world_model_expert", None)
        if wm is not None:
            mods.append(wm)
        n = 0
        for mod in mods:
            for p in mod.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    n += 1
        self.compute_world_model_supervision = not skip_supervision
        logging.info("Froze WM expert parameters n=%s; compute_supervision=%s", n, not skip_supervision)
        return n

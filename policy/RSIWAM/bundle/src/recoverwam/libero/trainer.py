"""Two-stage LIBERO training on the shared resumable trainer."""

from pathlib import Path
from recoverwam.trainer import JointTrainer


def validate_preflight_config(cfg):
    probe = cfg.get("preflight")
    if not probe or not probe.get("enabled", False):
        return None
    if cfg.train_mode not in {"video_only", "action_only"}:
        raise ValueError("LIBERO preflight requires video_only or action_only")
    expected = int(probe.get("expected_steps", 20))
    if int(cfg.max_steps) != expected or expected <= 0:
        raise ValueError("Probe max_steps must equal expected_steps")
    return dict(
        stage="video" if cfg.train_mode == "video_only" else "action",
        expected_steps=expected,
        memory_headroom_fraction=float(probe.get("memory_headroom_fraction", 0.1)),
        output_file=str(probe.get("output_file") or Path(cfg.output_dir) / "probe.json"),
        video_size=list(cfg.data.train.video_size),
        contract_kwargs=dict(
            video_frames=17, action_horizon=32, action_dim=7, proprio_steps=1, proprio_dim=8
        ),
    )


class LiberoTrainer(JointTrainer):
    supported_modes = {("video_only", "full"), ("action_only", "action_only")}
    _validate_preflight_config = staticmethod(validate_preflight_config)

    def _apply_train_mode(self, model):
        model.eval().requires_grad_(False)
        model.set_train_mode(self.train_mode)
        if self.train_mode == "video_only":
            model.dit.train().requires_grad_(True)
        else:
            model.action_expert.train().requires_grad_(True)
            model.proprio_encoder.train().requires_grad_(True)
            if any(p.requires_grad for p in model.video_expert.parameters()):
                raise RuntimeError("Action stage must freeze the complete Video expert")

    def _collect_trainable_params(self, model):
        params = [p for p in model.parameters() if p.requires_grad]
        if not params:
            raise ValueError("No trainable LIBERO parameters")
        return params

    def _uses_video_action_lr_groups(self):
        return False

    def _save_weights_checkpoint(self, step_tag, *, state_dict_overrides=None):
        if self._deepspeed_zero_stage() == 3:
            raise ValueError("The released LIBERO recipe uses ZeRO-2; ZeRO-3 is unsupported")
        if self.train_mode == "video_only":
            return super()._save_weights_checkpoint(
                step_tag, state_dict_overrides=state_dict_overrides
            )
        path = str(Path(self.weights_dir) / f"{step_tag}.pt")
        model = self.accelerator.unwrap_model(self.model)
        model.save_action_checkpoint(
            path, self.frozen_video_checkpoint, step=self.global_step, resolved_cfg=self.cfg
        )
        return path

    def evaluate(self):
        import torch

        if self._deepspeed_zero_stage() >= 3:
            raise ValueError("LIBERO diagnostics require ZeRO stage < 3")
        model = self.accelerator.unwrap_model(self.model)
        modes = [(m, m.training) for m in model.modules()]
        result = None
        try:
            if self.accelerator.is_main_process and self.evaluator is not None:
                model.eval()
                with torch.no_grad():
                    result = self.evaluator.evaluate(model, self.global_step)
        finally:
            for module, training in modes:
                module.training = training
        self.accelerator.wait_for_everyone()
        return result

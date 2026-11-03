"""Single-model RecoverWAM inference, with no runtime recovery agent."""
from pathlib import Path
import hashlib

from recoverwam.policy import RoboDojoRecoverWAMPolicy
from .lora import install_action_lora, load_adapter


class RecoverWAMPolicy(RoboDojoRecoverWAMPolicy):
    def __init__(self, *args, recovery_adapter=None, base_identity=None, recovery_mode="auto", **kwargs):
        if recovery_mode not in ("auto", "on", "off"):
            raise ValueError("Invalid recovery mode; 'on' is an oracle diagnostic only")
        if recovery_adapter is not None:
            if not base_identity:
                raise ValueError("A recovery adapter requires the base checkpoint identity")
            checkpoint = kwargs.get("checkpoint_path", args[0] if args else None)
            manifest = Path(checkpoint).resolve().parent / "manifest.json"
            if hashlib.sha256(manifest.read_bytes()).hexdigest() != base_identity:
                raise ValueError("Actual checkpoint manifest differs from the adapter base")
        super().__init__(*args, **kwargs)
        self.recovery_adapter_metadata = None
        if recovery_adapter is not None:
            import torch
            from .bridge import FORMAT as BRIDGE_FORMAT, load_bridge
            payload = torch.load(recovery_adapter, map_location="cpu", weights_only=True)
            if payload.get("format") == BRIDGE_FORMAT:
                self.recovery_adapter_metadata = load_bridge(
                    recovery_adapter, self, expected_base_identity=base_identity)
                self.model.recovery_mode = recovery_mode
            else:
                if recovery_mode != "auto":
                    raise ValueError("Recovery gate modes do not apply to global LoRA")
                install_action_lora(self, rank=8, alpha=8, gradient_checkpointing=False)
                self.recovery_adapter_metadata = load_adapter(
                    Path(recovery_adapter), self, expected_base_identity=base_identity)
            self.model.eval()
            self.model.action_expert.use_gradient_checkpointing = False
            self.model.mot.mot_checkpoint_mixed_attn = False


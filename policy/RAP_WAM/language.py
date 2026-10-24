"""Original trained-row lookup and Wan UMT5 fallback for unseen instructions."""
import threading
import torch
from .runtime import historical_module


class LanguageContext:
    def __init__(self, tasks, wan_dir, tokenizer_path):
        self.lookup = {text.strip().lower(): i for i, text in enumerate(tasks)}
        self.wan_dir = wan_dir
        self.tokenizer_path = str(tokenizer_path or wan_dir / "google/umt5-xxl")
        self.encoder = None
        self.cache = {}
        self.lock = threading.Lock()

    def resolve_index(self, instruction):
        return self.lookup.get(instruction.strip().lower())

    @torch.no_grad()
    def encode_zero_shot(self, instruction, device, dtype):
        text = instruction.strip()
        with self.lock:
            if text not in self.cache:
                if self.encoder is None:
                    cls = historical_module("psi_policy.model.wan.runtime").T5EncoderModel
                    self.encoder = cls(text_len=512, dtype=torch.bfloat16, device=str(device),
                                       checkpoint_path=str(self.wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
                                       tokenizer_path=self.tokenizer_path)
                self.cache[text] = self.encoder([text], str(device))[0].detach().float().cpu()
            return self.cache[text].to(device=device, dtype=dtype)

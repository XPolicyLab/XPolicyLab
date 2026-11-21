"""Text encoder placeholder for visual-only inference profiles."""
from torch import nn

class DisabledT5(nn.Module):
    def encode_text(self, instructions):
        raise RuntimeError("T5 is disabled for the direct-prefix profile")

"""Model-owned inference RNGs, independent of training and environment seeds."""

import torch


def inference_generator(model: torch.nn.Module, device: str | torch.device) -> torch.Generator:
    """Lazily create one entropy-seeded stream per model and sampling device.

    These generators are runtime state, not checkpoint buffers. Episode/cache
    resets do not touch them; torch.manual_seed only affects global generators.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if not hasattr(model, "_inference_generators"):
        model._inference_generators = {}
    if device not in model._inference_generators:
        generator = torch.Generator(device=device)
        generator.seed()
        model._inference_generators[device] = generator
    return model._inference_generators[device]

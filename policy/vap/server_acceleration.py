"""Initialize optional inference accelerations on the model owner thread.

The standard server calls this once, before listening. Experiment servers that
import ExclusiveServer retain their own initialization and are not patched twice.
"""
import json
import os
from pathlib import Path
import time

from .config_env import ACCELERATION_DEFAULTS, MAPPING


def options(config):
    result = {}
    for key, default in ACCELERATION_DEFAULTS.items():
        value = os.environ.get(MAPPING[key], config.get(key, default))
        if isinstance(value, bool):
            result[key] = value
        elif str(value).lower() in ('1', 'true', '0', 'false'):
            result[key] = str(value).lower() in ('1', 'true')
        else:
            raise ValueError(f'{key} must be a boolean, got {value!r}')
    return result


def warm_text(runtime):
    import torch

    prompter = runtime.model._model.prompter
    from .text_device import DeviceRoutedQwenPrompter
    routed = isinstance(prompter, DeviceRoutedQwenPrompter)
    provider = prompter.provider if routed else prompter
    if type(provider).__name__ != 'Qwen35OnlinePrompter':
        raise ValueError('text warmup requires Qwen35OnlinePrompter')
    started = time.monotonic()
    device = torch.device(runtime.device)
    text_device = torch.device(prompter.target_device) if routed else device
    devices = sorted({d.index if d.index is not None else torch.cuda.current_device()
                      for d in (device, text_device) if d.type == 'cuda'})
    with torch.random.fork_rng(devices=devices):
        if text_device.type == 'cuda':
            with torch.cuda.device(text_device):
                provider._load(text_device)
        else:
            provider._load(text_device)
        context = prompter.encode_prompt('warmup', device=device)
        if not torch.isfinite(context).all():
            raise ValueError('Nonfinite Qwen warmup output')
        for index in devices:
            torch.cuda.synchronize(index)
        del context
    return dict(status='ready_before_listen', seconds=time.monotonic()-started,
                device=str(device), text_device=str(text_device), encoder=type(prompter).__name__,
                rng_preserved=True, policy_history_untouched=True)


def install_exact(runtime, *, pixel, time_cache):
    from .exact_acceleration import install
    return install(runtime, pixel=pixel, time_cache=time_cache)


def install_prefill(runtime, *, rope, layout):
    from .prefill_acceleration import install_for_serving
    return install_for_serving(runtime, rope=rope, layout=layout)


def initialize(model):
    """Run through model.executor, after loading and before opening the socket."""
    requested = options(model.model_cfg)
    runtime = model.backend.policy.runtime
    if getattr(runtime, '_vap_standard_acceleration_initialized', False):
        raise RuntimeError('Standard server accelerations already initialized')
    out = Path(os.environ['VAP_RUN_DIR'])
    if requested['text_encoder_warmup']:
        proof = warm_text(runtime)
        (out/'text_warmup.json').write_text(json.dumps(proof, indent=2))
        print('VAP_TEXT_WARMUP_COMPLETE', json.dumps(proof), flush=True)
    if requested['pixel_normalization_cache'] or requested['action_time_cache']:
        install_exact(runtime, pixel=requested['pixel_normalization_cache'],
                      time_cache=requested['action_time_cache'])
    prefill = dict(enabled=False, rope=False, layout_cache=False, m16_enabled=False)
    if requested['prefill_rope_fusion'] or requested['prefill_layout_cache']:
        adapter = install_prefill(runtime, rope=requested['prefill_rope_fusion'],
                                  layout=requested['prefill_layout_cache'])
        prefill = adapter.metadata()
    (out/'prefill_acceleration_ready.json').write_text(json.dumps(prefill, indent=2))
    proof = dict(requested=requested, prefill=prefill,
                 module_directory=str(Path(__file__).resolve().parent),
                 equality_gate='first complete A0 and A1; fail closed',
                 initialized_before_listen=True)
    (out/'server_acceleration.json').write_text(json.dumps(proof, indent=2))
    runtime._vap_standard_acceleration_initialized = True
    print('VAP_SERVER_ACCELERATION_READY', json.dumps(proof), flush=True)
    return proof

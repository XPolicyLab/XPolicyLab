"""Configuration/lifecycle tests. Numerical action parity uses real-model gates."""
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import threading

import pytest
import torch

from XPolicyLab.policy.vap import server_acceleration as accel
from XPolicyLab.policy.vap.config_env import ACCELERATION_DEFAULTS, MAPPING, values
from XPolicyLab.policy.vap.text_device import DeviceRoutedQwenPrompter


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch, tmp_path):
    for key in ACCELERATION_DEFAULTS:
        monkeypatch.delenv(MAPPING[key], raising=False)
    monkeypatch.setenv('VAP_RUN_DIR', str(tmp_path))


def test_options_override_and_reject_typo(monkeypatch):
    assert not any(accel.options({}).values())
    assert all(accel.options(dict.fromkeys(ACCELERATION_DEFAULTS, True)).values())
    monkeypatch.setenv('VAP_PREFILL_EXACT_ROPE', '0')
    assert accel.options({'prefill_rope_fusion': True})['prefill_rope_fusion'] is False
    monkeypatch.setenv('VAP_PREFILL_EXACT_ROPE', 'fasle')
    with pytest.raises(ValueError, match='prefill_rope_fusion'):
        accel.options({})


@pytest.mark.parametrize('pixel,time_cache,rope,layout', [
    (False, False, False, False), (True, False, False, False),
    (False, True, False, False), (False, False, True, False),
    (False, False, False, True), (True, True, True, True),
])
def test_independent_installation_on_owner_thread(monkeypatch, pixel, time_cache, rope, layout):
    calls = []
    caller = threading.get_ident()
    def exact(runtime, **kwargs):
        calls.append(('exact', kwargs, threading.get_ident()))
    def prefill(runtime, **kwargs):
        calls.append(('prefill', kwargs, threading.get_ident()))
        return SimpleNamespace(metadata=lambda: dict(enabled=True, **kwargs))
    monkeypatch.setattr(accel, 'install_exact', exact)
    monkeypatch.setattr(accel, 'install_prefill', prefill)
    monkeypatch.setattr(accel, 'warm_text', lambda _: pytest.fail('warmup disabled'))
    runtime = SimpleNamespace()
    model = SimpleNamespace(model_cfg=dict(text_encoder_warmup=False,
        pixel_normalization_cache=pixel, action_time_cache=time_cache,
        prefill_rope_fusion=rope, prefill_layout_cache=layout),
        backend=SimpleNamespace(policy=SimpleNamespace(runtime=runtime)))
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(accel.initialize, model).result()
        with pytest.raises(RuntimeError, match='already initialized'):
            executor.submit(accel.initialize, model).result()
    assert result['initialized_before_listen']
    assert [c[0] for c in calls] == (['exact'] if pixel or time_cache else []) + (['prefill'] if rope or layout else [])
    for kind, kwargs, thread in calls:
        assert thread != caller
        assert kwargs == (dict(pixel=pixel, time_cache=time_cache) if kind == 'exact' else dict(rope=rope, layout=layout))


@pytest.mark.parametrize('routed', [False, True])
def test_qwen_warmup_preserves_rng_and_episode_state(routed):
    class Qwen35OnlinePrompter:
        def _load(self, device):
            self.loaded_device = str(device)
            torch.rand(3)
        def encode_prompt(self, prompt, device, positive=True):
            assert prompt == 'warmup'
            return torch.rand(4, device=device)
    provider = Qwen35OnlinePrompter()
    prompter = DeviceRoutedQwenPrompter(provider, 'cpu') if routed else provider
    marker = object()
    runtime = SimpleNamespace(device='cpu', history=marker,
        model=SimpleNamespace(_model=SimpleNamespace(prompter=prompter)))
    before = torch.random.get_rng_state().clone()
    proof = accel.warm_text(runtime)
    assert torch.equal(before, torch.random.get_rng_state())
    assert runtime.history is marker and provider.loaded_device == 'cpu'
    assert proof['rng_preserved'] and proof['policy_history_untouched']


def test_legacy_deployment_config_gets_defaults():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(accel.__file__).parent/'deploy.yml').read_text())
    for key in ACCELERATION_DEFAULTS:
        cfg.pop(key)
    env = values(cfg)
    assert all(env[MAPPING[key]] == '0' for key in ACCELERATION_DEFAULTS)

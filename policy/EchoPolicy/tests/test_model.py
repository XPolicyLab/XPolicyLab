"""Contract tests with real JAX on CPU and a deterministic sampling fixture."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("echo_model", Path(__file__).parents[1] / "model.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_active_environment_order_and_reset():
    model = object.__new__(module.Model)
    model.infer_batch = lambda items: {"actions": [x["env_idx"] for x in items]}
    model.update_obs_batch([{"env_idx": 7}, {"env_idx": 2}, {"env_idx": 9}])
    assert model.get_action_batch([9, 7]) == [9, 7]
    assert model.get_action() == 7
    assert model.get_action_batch([]) == []
    with pytest.raises(KeyError):
        model.get_action_batch([99])
    model.reset()
    with pytest.raises(RuntimeError):
        model.get_action_batch()
    with pytest.raises(ValueError, match="Duplicate"):
        model.update_obs_batch([{"env_idx": 1}, {"env_idx": 1}])


@pytest.mark.parametrize("count,expected_calls", [(1, 1), (160, 1), (161, 2)])
def test_fixed_batch_padding_preserves_output_order(monkeypatch, count, expected_calls):
    monkeypatch.setattr(module, "encode_obs", lambda obs, *_: obs)
    monkeypatch.setattr(module.model_lib.Observation, "from_dict", lambda x: x)
    model = object.__new__(module.Model)
    model.batch_size = 160
    model.action_type = "joint"
    model.robot_action_dim_info = None
    calls = []

    def sample(rng, inputs):
        calls.append(inputs["state"].shape)
        return np.repeat(np.asarray(inputs["state"])[:, None, :], 2, axis=1)

    model.policy = SimpleNamespace(
        _input_transform=lambda x: x, _output_transform=lambda x: x,
        _rng=jax.random.key(0), _sample_actions=sample, _sample_kwargs={},
    )
    result = model.infer_batch([{"state": np.full(14, i, dtype=np.float32)} for i in range(count)])
    assert calls == [(160, 14)] * expected_calls
    assert len(result["actions"]) == count
    for i, action in enumerate(result["actions"]):
        assert action.shape == (2, 14)
        assert np.isfinite(action).all()
        np.testing.assert_array_equal(action, np.full((2, 14), i))

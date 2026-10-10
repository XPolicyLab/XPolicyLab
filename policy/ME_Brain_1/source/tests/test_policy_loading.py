from types import SimpleNamespace

from focus_vlwa.configs.model import FocusVLWAConfig
from focus_vlwa.inference import policy as policy_module


def test_inference_cast_happens_after_checkpoint_load(monkeypatch, tmp_path):
    events = []

    class FakeModel:
        def __init__(self, config):
            self.joint_experts = SimpleNamespace(
                to_bfloat16_for_selected_params=lambda dtype: events.append(("cast", dtype))
            )

        def to(self, device):
            events.append(("device", str(device)))
            return self

        def eval(self):
            return self

    monkeypatch.setattr(policy_module, "FocusVLWA", FakeModel)
    monkeypatch.setattr(policy_module, "FocusVLWATokenizer", lambda *args: None)
    monkeypatch.setattr(policy_module, "load_norm_stats", lambda *args: {})
    monkeypatch.setattr(policy_module, "load_checkpoint_weights", lambda *args: events.append(("load", None)))
    policy_module.FocusVLWAPolicy(tmp_path, device="cpu", config=FocusVLWAConfig())
    assert events == [("load", None), ("cast", "bfloat16"), ("device", "cpu")]

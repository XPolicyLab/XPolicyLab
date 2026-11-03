"""Shared offline native adapter fixtures; no simulator or model calls."""
pytest_plugins = ["XPolicyLab.policy.SenseNova-RoboRSI.tests.test_hybrid_backend"]


import pytest


@pytest.fixture(autouse=True)
def offline_provider_environment(monkeypatch):
    """Tests use explicit synthetic routing and never inherit deployment settings."""
    monkeypatch.setenv("ROBORSI_CODEX_BASE_URL", "https://provider.example.invalid/v1")
    monkeypatch.setenv("ROBORSI_CODEX_MODEL", "fixture-model")

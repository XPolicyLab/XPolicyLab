"""Check the policy source-link contract."""

import importlib.util
from pathlib import Path

import pytest


def load_script(name, relative):
    specification = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / relative)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_source_link_is_required_and_used(tmp_path, monkeypatch):
    module = load_script("focus_source", "_source.py")
    policy = tmp_path / "policy"
    policy.mkdir()
    monkeypatch.setattr(module, "__file__", str(policy / "_source.py"))
    monkeypatch.setattr(module.sys, "path", list(module.sys.path))
    with pytest.raises(FileNotFoundError, match="Missing source checkout link"):
        module.configure_source()
    repository = tmp_path / "source"
    (repository / "src/focus_vlwa").mkdir(parents=True)
    (policy / "focus-vlwa").symlink_to(repository, target_is_directory=True)
    assert module.configure_source() == repository
    assert str(repository / "src") in module.sys.path


def test_client_reads_source_from_deploy_config(tmp_path, monkeypatch):
    import sys

    from XPolicyLab.policy.ME_Brain_1 import hist_live

    policy = tmp_path / "policy"
    source = policy / "checkout" / "src"
    source.mkdir(parents=True)
    (policy / "deploy.yml").write_text("focus_vlwa_source: checkout/src\n")
    monkeypatch.setattr(hist_live, "__file__", str(policy / "hist_live.py"))
    monkeypatch.setattr(sys, "path", sys.path[:])
    monkeypatch.delenv("FOCUS_VLWA_SOURCE", raising=False)
    hist_live._configure_client_source()
    assert sys.path[0] == str(source)


def test_client_environment_source_precedes_deploy_config(tmp_path, monkeypatch):
    import sys

    from XPolicyLab.policy.ME_Brain_1 import hist_live

    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "deploy.yml").write_text("focus_vlwa_source: missing/src\n")
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(hist_live, "__file__", str(policy / "hist_live.py"))
    monkeypatch.setattr(sys, "path", sys.path[:])
    monkeypatch.setenv("FOCUS_VLWA_SOURCE", str(source))
    hist_live._configure_client_source()
    assert sys.path[0] == str(source)

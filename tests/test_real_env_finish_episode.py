from __future__ import annotations

from pathlib import Path

import pytest
from robodojo.env_client.runner import _ensure_pipeline_paths

repo_root = Path(__file__).resolve().parents[2]
_ensure_pipeline_paths(str(repo_root))

from task_env.episode_result import should_prompt_episode_result


@pytest.mark.parametrize(
    ("deploy_cfg", "is_tty", "expected"),
    [
        ({"protocol": "robodojo_ws"}, True, False),
        ({}, True, False),
        ({"protocol": "legacy_tcp"}, True, True),
        ({"protocol": "legacy_tcp"}, False, False),
    ],
)
def test_should_prompt_episode_result(
    deploy_cfg: dict[str, str],
    is_tty: bool,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin.isatty", lambda: is_tty)
    assert should_prompt_episode_result(deploy_cfg) is expected

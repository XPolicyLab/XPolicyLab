from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from typing import Any, Iterator
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from robodojo_fixtures import platform_dispatch

from robodojo.dispatch import build_trial_runs
from robodojo.env_client import (
    EnvClientBaselineConfig,
    baseline_deploy_cfg_view,
    debug_env_client_deploy_cfg_view,
    dispatch_trial_to_deploy_cfg,
    dispatch_trial_to_request,
    trial_request_to_deploy_cfg,
)
from robodojo.schemas import DispatchPayload
from robodojo.servers.env_client_server import (
    EnvClientServerConfig,
    EnvClientServerState,
    create_server,
    session_dispatch_path,
    session_start_path,
)
from robodojo.trial.config import build_trial_run_config


def _baseline(**overrides: object) -> EnvClientBaselineConfig:
    return EnvClientBaselineConfig.model_validate(
        {
            "dataset_name": "demo_dataset",
            "task_name": "lift-cube",
            "env_cfg_type": "arx_x5",
            "policy_name": "demo_policy",
            "host": "localhost",
            "port": 19000,
            **overrides,
        }
    )


def _platform_trial(*, task_name: str = "lift-cube"):
    dispatch = DispatchPayload.model_validate(platform_dispatch())
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    return dispatch, trial_run, _baseline(task_name=task_name)


def _real_baseline(**overrides: object) -> EnvClientBaselineConfig:
    return _baseline(
        eval_env="real",
        root_dir="/data/X-Robot-Pipeline",
        action_type="ee",
        **overrides,
    )


def _reference_deploy_cfg(
    config,
    baseline: EnvClientBaselineConfig,
    *,
    eval_episode_num: int | None = None,
) -> dict[str, Any]:
    parsed = urlparse(config.policy_server_url)
    reference = {
        "dataset_name": baseline.dataset_name,
        "task_name": config.task_name or baseline.task_name,
        "env_cfg_type": config.env_cfg_type or baseline.env_cfg_type,
        "policy_name": config.policy_name or baseline.policy_name,
        "protocol": baseline.protocol,
        "host": parsed.hostname or baseline.host,
        "port": parsed.port or baseline.port,
        "policy_server_url": config.policy_server_url,
        "evaluation_id": config.evaluation_id,
        "action_case_id": config.action_case_id,
        "trial_id": config.trial_id,
        "repeat_index": config.repeat_index,
        "eval_episode_num": (
            eval_episode_num if eval_episode_num is not None else baseline.eval_episode_num
        ),
        "eval_batch": config.eval_batch,
    }
    if baseline.eval_env == "real":
        reference["eval_env"] = "real"
        reference["root_dir"] = baseline.root_dir
        reference["action_type"] = baseline.action_type
    return reference


@contextmanager
def _running_server(
    *,
    run_trial,
    baseline: EnvClientBaselineConfig,
    tmp_path,
) -> Iterator[tuple[Any, EnvClientServerState]]:
    state = EnvClientServerState(
        baseline=baseline,
        config=EnvClientServerConfig(
            artifact_root=tmp_path / "artifacts",
            upload_s3=False,
            notify_webhook=False,
        ),
        run_trial=run_trial,
    )
    server = create_server("127.0.0.1", 0, state)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _post(port: int, path: str, payload: dict[str, Any]):
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=2) as response:
        return response.status


def _assert_deploy_cfg_matches(
    deploy_cfg: dict[str, Any],
    reference: dict[str, Any],
    *,
    baseline: EnvClientBaselineConfig,
) -> None:
    if baseline.eval_env == "real":
        assert baseline_deploy_cfg_view(deploy_cfg) == reference
    else:
        assert debug_env_client_deploy_cfg_view(deploy_cfg) == reference


def test_dispatch_trial_to_request_resolves_platform_fields():
    dispatch = DispatchPayload.model_validate(
        platform_dispatch(
            evaluation_plan={
                "repeat_count": 1,
                "task": {"id": "lift-cube", "env_cfg_type": "arx_x5"},
                "trials": [
                    {
                        "trial_id": "case-1-r01",
                        "trial_index": 1,
                        "action_case_id": "case-1",
                        "instruction": "stack the bowls",
                        "finish_url": "https://example.test/finish/",
                    }
                ],
            },
        )
    )
    trial_run = build_trial_runs(dispatch, evaluation_id="eval-1")[0]
    request = dispatch_trial_to_request(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
        eval_episode_num=1,
    )

    assert request.evaluation_id == "eval-1"
    assert request.trial_id == "case-1-r01"
    assert request.action_case_id == "case-1"
    assert request.policy_server_url == "ws://127.0.0.1:19000"
    assert request.case_meta["instruction"] == "stack the bowls"
    assert request.case_meta["env_cfg_type"] == "arx_x5"
    assert request.case_meta["task_name"] == "lift-cube"
    assert request.case_meta["policy_name"] == "demo_policy"
    assert request.overrides == {"eval_episode_num": 1}


def test_dispatch_trial_to_deploy_cfg_matches_debug_env_client_one_shot():
    dispatch, trial_run, baseline = _platform_trial()
    deploy_cfg = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=1,
    )
    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
        eval_env="debug",
    )
    reference = _reference_deploy_cfg(config, baseline, eval_episode_num=1)

    _assert_deploy_cfg_matches(deploy_cfg, reference, baseline=baseline)


def test_session_start_matches_dispatch_trial_to_deploy_cfg(tmp_path):
    dispatch, trial_run, baseline = _platform_trial()
    expected = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=1,
    )
    captured: list[dict[str, Any]] = []

    with _running_server(
        run_trial=lambda deploy_cfg: captured.append(deploy_cfg)
        or {"status": "completed", "trial_id": deploy_cfg["trial_id"]},
        baseline=baseline,
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        assert (
            _post(port, session_dispatch_path("eval-1"), platform_dispatch()) == 200
        )
        assert _post(port, session_start_path("eval-1", 1), {}) == 200

    _assert_deploy_cfg_matches(
        captured[0],
        debug_env_client_deploy_cfg_view(expected),
        baseline=baseline,
    )


def test_trial_request_path_matches_dispatch_trial_to_deploy_cfg():
    dispatch, trial_run, baseline = _platform_trial()
    via_dispatch = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=1,
    )
    request = dispatch_trial_to_request(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
        eval_episode_num=1,
    )
    via_request = trial_request_to_deploy_cfg(request, baseline)

    assert debug_env_client_deploy_cfg_view(via_dispatch) == debug_env_client_deploy_cfg_view(
        via_request
    )


def test_dispatch_trial_to_deploy_cfg_real_preserves_baseline_fields():
    dispatch, trial_run, baseline = _platform_trial()
    baseline = _real_baseline()
    deploy_cfg = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=None,
    )

    assert deploy_cfg["eval_env"] == "real"
    assert deploy_cfg["root_dir"] == "/data/X-Robot-Pipeline"
    assert deploy_cfg["eval_episode_num"] == baseline.eval_episode_num


def test_dispatch_trial_to_request_real_omits_eval_episode_override():
    dispatch, trial_run, _ = _platform_trial()
    request = dispatch_trial_to_request(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
        eval_episode_num=None,
        eval_env="real",
    )

    assert request.overrides == {}


def test_dispatch_trial_to_deploy_cfg_real_matches_baseline_view():
    dispatch, trial_run, baseline = _platform_trial()
    baseline = _real_baseline()
    deploy_cfg = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=None,
    )
    config = build_trial_run_config(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
        eval_env="real",
    )
    reference = _reference_deploy_cfg(config, baseline)

    _assert_deploy_cfg_matches(deploy_cfg, reference, baseline=baseline)


def test_session_start_real_matches_dispatch_trial_to_deploy_cfg(tmp_path):
    dispatch, trial_run, baseline = _platform_trial()
    baseline = _real_baseline()
    expected = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=None,
    )
    captured: list[dict[str, Any]] = []

    with _running_server(
        run_trial=lambda deploy_cfg: captured.append(deploy_cfg)
        or {"status": "completed", "trial_id": deploy_cfg["trial_id"]},
        baseline=baseline,
        tmp_path=tmp_path,
    ) as (server, _state):
        port = server.server_address[1]
        assert (
            _post(port, session_dispatch_path("eval-1"), platform_dispatch()) == 200
        )
        assert _post(port, session_start_path("eval-1", 1), {}) == 200

    _assert_deploy_cfg_matches(
        captured[0],
        baseline_deploy_cfg_view(expected),
        baseline=baseline,
    )


def test_trial_request_path_real_matches_dispatch_trial_to_deploy_cfg():
    dispatch, trial_run, baseline = _platform_trial()
    baseline = _real_baseline()
    via_dispatch = dispatch_trial_to_deploy_cfg(
        dispatch,
        trial_run,
        baseline,
        evaluation_id="eval-1",
        eval_episode_num=None,
    )
    request = dispatch_trial_to_request(
        dispatch,
        trial_run,
        evaluation_id="eval-1",
        eval_episode_num=None,
        eval_env="real",
    )
    via_request = trial_request_to_deploy_cfg(request, baseline)

    assert baseline_deploy_cfg_view(via_dispatch) == baseline_deploy_cfg_view(via_request)

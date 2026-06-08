"""Platform-shaped dispatch JSON for RoboDojo tests."""

from __future__ import annotations

from typing import Any


def platform_dispatch(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "message_type": "dispatch",
        "evaluation_id": "eval-1",
        "task_id": "lift-cube",
        "model_name": "demo-policy",
        "policy_server_url": "ws://127.0.0.1:19000",
        "evaluation_plan": {
            "repeat_count": 2,
            "trials": [
                {
                    "trial_id": "case-1-r01",
                    "trial_index": 1,
                    "action_case_id": "case-1",
                    "finish_url": (
                        "https://example.test/api/v1/internal/eval/"
                        "eval-1/trials/1/finish/"
                    ),
                },
                {
                    "trial_id": "case-1-r02",
                    "trial_index": 2,
                    "action_case_id": "case-1",
                    "finish_url": (
                        "https://example.test/api/v1/internal/eval/"
                        "eval-1/trials/2/finish/"
                    ),
                },
                {
                    "trial_id": "case-2-r01",
                    "trial_index": 3,
                    "action_case_id": "case-2",
                    "finish_url": (
                        "https://example.test/api/v1/internal/eval/"
                        "eval-1/trials/3/finish/"
                    ),
                },
                {
                    "trial_id": "case-2-r02",
                    "trial_index": 4,
                    "action_case_id": "case-2",
                    "finish_url": (
                        "https://example.test/api/v1/internal/eval/"
                        "eval-1/trials/4/finish/"
                    ),
                },
            ],
        },
        "artifact": {
            "bucket": "robodojo-artifacts",
            "prefix": "evaluations/eval-1/",
        },
    }
    payload.update(overrides)
    return payload

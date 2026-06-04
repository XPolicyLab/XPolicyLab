from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyServerPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str = Field(min_length=1)
    connection_mode: str = Field(min_length=1)


class EvaluationTrialPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_case_id: str = Field(min_length=1)


class EvaluationPlanPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: str | None = None
    repeat_count: int = Field(ge=1)
    trials: list[EvaluationTrialPayload] = Field(min_length=1)


class DispatchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    evaluation_id: str = Field(min_length=1)
    policy_server: PolicyServerPayload
    evaluation_plan: EvaluationPlanPayload
    artifact: dict[str, Any]
    webhook: dict[str, Any]


@dataclass
class TrialRecord:
    trial_id: str
    action_case_id: str
    trial_index: int
    repeat_index: int
    case_meta: dict[str, Any]
    status: str = "not_executed"
    metrics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_manifest_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "trial_id": self.trial_id,
            "action_case_id": self.action_case_id,
            "trial_index": self.trial_index,
            "repeat_index": self.repeat_index,
            "case_meta": self.case_meta,
            "status": self.status,
            "video_key": f"videos/{self.trial_id}.mp4",
        }
        if self.started_at is not None:
            entry["started_at"] = self.started_at
        if self.finished_at is not None:
            entry["finished_at"] = self.finished_at
        if self.error is not None:
            entry["error"] = self.error
        return entry

    def to_metrics_entry(self) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "trial_id": self.trial_id,
            "action_case_id": self.action_case_id,
            "repeat_index": self.repeat_index,
            "status": self.status,
            "metrics": self.metrics,
        }
        if self.error is not None:
            entry["error"] = self.error
        return entry

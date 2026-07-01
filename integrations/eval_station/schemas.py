from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvaluationTrialPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    action_case_id: str = Field(min_length=1)
    trial_index: int = Field(ge=1)
    trial_id: str | None = None
    repeat_index: int | None = None
    finish_url: str = ""
    instruction: str = ""


class EvaluationTaskPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(min_length=1)
    name: str = ""
    env_cfg_type: str = ""
    control_frequency_hz: int | None = None


class EvaluationPlanPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: EvaluationTaskPayload | None = None
    repeat_count: int = Field(default=1, ge=1)
    trials: list[EvaluationTrialPayload] = Field(min_length=1)


class CallbackPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    hmac_secret_ref: str = ""


class ArtifactPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    bucket: str = ""
    prefix: str = ""


class DispatchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = ""
    model_name: str = ""
    policy_server_url: str = Field(min_length=1)
    evaluation_plan: EvaluationPlanPayload
    callback: CallbackPayload = Field(default_factory=CallbackPayload)
    artifact: ArtifactPayload = Field(default_factory=ArtifactPayload)

    @model_validator(mode="before")
    @classmethod
    def reject_evaluation_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "evaluation_id" in data:
            raise ValueError("evaluation_id must be provided separately")
        return data

    @property
    def hmac_secret_ref(self) -> str:
        return self.callback.hmac_secret_ref


@dataclass
class TrialRecord:
    trial_id: str
    action_case_id: str
    trial_index: int
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
            "status": self.status,
            "metrics": self.metrics,
        }
        if self.error is not None:
            entry["error"] = self.error
        return entry

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PolicyServerPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    url: str = Field(min_length=1)
    connection_mode: str = Field(min_length=1)


class EvaluationPlanPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: str | None = None
    repeat_count: int = Field(ge=1)
    trials: list[dict[str, Any]]


class DispatchPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    evaluation_id: str = Field(min_length=1)
    policy_server: PolicyServerPayload
    evaluation_plan: EvaluationPlanPayload
    artifact: dict[str, Any]
    webhook: dict[str, Any]

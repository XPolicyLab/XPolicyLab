import logging
from datetime import datetime
from logging import getLevelName
from typing import Annotated, Literal
from pydantic import BeforeValidator
from pydantic_settings import BaseSettings, SettingsConfigDict

def _parse_bool_lenient(value):
    return value.lower() in ("1", "true", "yes", "on") if isinstance(value, str) else bool(value)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    logging_level: str | None = getLevelName(logging.INFO)
    run_name: str = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    world_size: int = 1
    local_rank: int = 0
    seed: int = 42
    mode: Literal[
        "train",
        "norm_stats",
        "infer",
        "robot_infer",
        "evaluate",
        "serve",
    ] = "train"
    infer_checkpoint_dir: str = ""
    WAM_policy_infer_robot_name: str = ""
    WAM_policy_infer_stat_path: str = ""
    WAM_policy_infer_dense_action_steps: int | None = None
    WAM_policy_infer_return_head_waist: bool = False

settings = Settings()

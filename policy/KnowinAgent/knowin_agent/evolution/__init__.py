"""M4 self-improvement: practice runs, review merges, shared-library curation, gates and the pipeline."""

from knowin_agent.evolution.curate import CurationError, Curator, TaskEvidence
from knowin_agent.evolution.gates import (ScorePoint, StallGate, StallRefused, admit_after_stall, pg80_decision,
                                  pg80_from_history, stall_notice, stall_threshold)
from knowin_agent.evolution.pipeline import Pipeline, PipelineError, plan_steps
from knowin_agent.evolution.review import FinishedProcess, Reviewer, ReviewError, command_launcher
from knowin_agent.evolution.runner import EpisodeRunner, FakeRunner, finish_episode, prepare_episode
from knowin_agent.evolution.snippet import check_snippet
from knowin_agent.evolution.types import (Budget, EpisodeResult, EpisodeSpec, PipelineSpec, equal_budget_specs,
                                  spec_from_switches)

__all__ = [
    "Budget", "CurationError", "Curator", "EpisodeResult", "EpisodeRunner", "EpisodeSpec", "FakeRunner",
    "FinishedProcess", "Pipeline", "PipelineError", "PipelineSpec", "ReviewError", "Reviewer", "ScorePoint",
    "StallGate", "StallRefused", "TaskEvidence", "admit_after_stall", "check_snippet", "command_launcher",
    "equal_budget_specs", "finish_episode", "pg80_decision", "pg80_from_history", "plan_steps", "prepare_episode",
    "spec_from_switches", "stall_notice", "stall_threshold",
]

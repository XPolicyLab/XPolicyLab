"""M3 task knowledge: scoring notes, recipes, official task source and object models, each behind a switch."""

from knowin_agent.knowledge.brief import DEFAULTS, SCORING_LEVELS, Brief, knowledge_switches, task_brief
from knowin_agent.knowledge.tasks import DIMENSIONS, base_task, task_dimension, task_names

__all__ = ["DEFAULTS", "DIMENSIONS", "SCORING_LEVELS", "Brief", "base_task", "knowledge_switches", "task_brief",
           "task_dimension", "task_names"]

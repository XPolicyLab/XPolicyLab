# coding=utf-8
"""Local execution skills; model weights and task programs are external inputs."""


def catalogue() -> list[dict]:
    return [
        {
            "name": "pi05",
            "description": "Frozen OpenPI pi0.5 inference from the current observation and subgoal",
            "requires": [
                "official OpenPI submodule installed in the policy environment",
                "checkpoint",
                "train_config_name",
                "normalization_id",
            ],
        },
        {
            "name": "code-policy",
            "description": "Code-policy backend: GPT selects a task-specific sub-skill, then adapts its frozen experience and successful program",
            "requires": [
                "code_skills and code_selection_prompt (or a trusted diagnostic code_path)",
                "configured LLM transport for sub-skill selection and generation",
                "program-specific dependencies and parameters",
            ],
        },
    ]

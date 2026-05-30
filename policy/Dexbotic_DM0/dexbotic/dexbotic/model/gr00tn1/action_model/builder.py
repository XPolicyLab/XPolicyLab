from dexbotic.model.gr00tn1.action_model.action_models import (
    FlowmatchingActionHead,
    FlowmatchingActionHeadConfig,
)


def build_action_model(config):
    model_type = config.action_model_type

    if "fm" in model_type:
        action_head_cfg = FlowmatchingActionHeadConfig(**config.action_head_cfg)
        action_model = FlowmatchingActionHead(action_head_cfg)

    return action_model

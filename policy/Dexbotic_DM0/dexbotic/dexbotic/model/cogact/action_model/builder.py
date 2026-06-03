from dexbotic.exp.utils import require_config_keys
from dexbotic.model.cogact.action_model.action_models import ActionModel, LinearModel


@require_config_keys(["action_model_type", "hidden_size", "action_dim", "chunk_size"])
def build_action_model(config):
    model_type = config.action_model_type
    in_channels = config.action_dim
    token_size = config.hidden_size
    future_action_window_size = config.chunk_size - \
        1  
    past_action_window_size = 0

    if "Linear" in model_type:
        action_model = LinearModel(model_type=model_type,
                                   token_size=token_size,
                                   in_channels=in_channels,
                                   future_action_window_size=future_action_window_size,
                                   past_action_window_size=past_action_window_size)
    elif "DiT" in model_type:
        action_model = ActionModel(model_type=model_type,
                                   token_size=token_size,
                                   in_channels=in_channels,
                                   future_action_window_size=future_action_window_size,
                                   past_action_window_size=past_action_window_size)

    return action_model

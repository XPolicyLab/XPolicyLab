"""Action Expert model configuration"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


# Modified from transformers.model.llama.configuration_llama.LlamaConfig
class ActionExpertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ActionExpert`](based on InternLM2).
    It is used to instantiate an ActionExpert model according to the specified arguments, defining the model
    architecture.
    """

    model_type = "action_expert"
    _auto_class = "AutoConfig"

    def __init__(  # pylint: disable=W0102
        self,
        input_hidden_size=2048,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        hidden_act="silu",
        state_dim=8,
        action_dim=7,
        action_chunk_size=30,
        state_token_num=3,  # time + ctrl_freq + state
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        bias=True,
        rope_theta=10000,
        rope_scaling=None,
        attn_implementation="eager",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_position_embeddings = max_position_embeddings
        self.input_hidden_size = input_hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.state_token_num = state_token_num
        self.bias = bias

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        self.attn_implementation = attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = "eager"

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor < 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {rope_scaling_factor}")


class LatentPlannerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ActionExpert`](based on InternLM2).
    It is used to instantiate an ActionExpert model according to the specified arguments, defining the model
    architecture.
    """

    model_type = "intermidiate_action_expert"
    _auto_class = "AutoConfig"

    def __init__(  # pylint: disable=W0102
        self,
        vocab_size=32,
        input_hidden_size=2048,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=64,
        hidden_act="silu",
        action_dim=1,
        state_token_num=0,  # intermediate config should concatenate none
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        bias=False,
        rope_theta=10000,
        rope_scaling=None,
        attn_implementation="eager",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.max_position_embeddings = max_position_embeddings
        self.input_hidden_size = input_hidden_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.action_dim = action_dim
        self.state_token_num = state_token_num
        self.bias = bias
        self.vocab_size = vocab_size

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self._rope_scaling_validation()

        self.attn_implementation = attn_implementation
        if self.attn_implementation is None:
            self.attn_implementation = "eager"

    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor < 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float >= 1, got {rope_scaling_factor}")

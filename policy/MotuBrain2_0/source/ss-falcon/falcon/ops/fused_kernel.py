import torch
import torch.nn.functional as F

from falcon.ops.triton_glu import triton_bias_reglu_impl as bias_reglu_impl
from falcon.ops.triton_glu import triton_bias_swiglu_impl as bias_swiglu_impl

# ---------------------------------------------------------------------------
# GeGLU  (gelu gating, torch.compile — uses PyTorch's exact erf GELU)
# ---------------------------------------------------------------------------


@torch.compile
def geglu(y):
    y_1, y_2 = torch.chunk(y, 2, -1)
    return F.gelu(y_1) * y_2


def bias_geglu_impl(input, fp8_input_store=False):
    ori_shape = input.shape
    assert len(ori_shape) in [2, 3]
    input = input.view(-1, ori_shape[-1])
    output = geglu(input)
    return output if len(ori_shape) == 2 else output.view(ori_shape[0], ori_shape[1], -1)


# ---------------------------------------------------------------------------
# Activation dispatch helpers
# ---------------------------------------------------------------------------

_GLU_FN_MAP = {
    "relu": bias_reglu_impl,
    "silu": bias_swiglu_impl,
    "gelu": bias_geglu_impl,
}

_ACT_FN_MAP = {
    "relu": F.relu,
    "silu": F.silu,
    "gelu": F.gelu,
}


def get_glu_fn(activation: str):
    """Return the fused GLU function for the given activation name."""
    if activation not in _GLU_FN_MAP:
        raise ValueError(f"Unsupported activation '{activation}' for gated MLP. Choose from {list(_GLU_FN_MAP.keys())}")
    return _GLU_FN_MAP[activation]


def get_act_fn(activation: str):
    """Return the plain activation function for the given activation name."""
    if activation not in _ACT_FN_MAP:
        raise ValueError(f"Unsupported activation '{activation}'. Choose from {list(_ACT_FN_MAP.keys())}")
    return _ACT_FN_MAP[activation]

from collections.abc import Callable
from typing import Any

from flax import nnx
from flax import struct
import jax
import optax

from openpi.models import model as _model
from openpi.shared import array_typing as at


@at.typecheck
@struct.dataclass
class TrainState:
    step: at.Int[at.ArrayLike, ""]
    params: nnx.State
    model_def: nnx.GraphDef[_model.BaseModel]
    opt_state: optax.OptState
    tx: optax.GradientTransformation = struct.field(pytree_node=False)

    ema_decay: float | None = struct.field(pytree_node=False)
    ema_params: nnx.State | None = None


@at.typecheck
def tree_to_info(tree: at.PyTree, interp_func: Callable[[Any], str] = str) -> str:
    """Converts a PyTree into a human-readable string for logging. Optionally, `interp_func` can be provided to convert
    the leaf values to more meaningful strings.
    """
    tree, _ = jax.tree_util.tree_flatten_with_path(tree)
    return "\n".join(f"{jax.tree_util.keystr(path)}: {interp_func(value)}" for path, value in tree)


@at.typecheck
def array_tree_to_info(tree: at.PyTree) -> str:
    """Converts a PyTree of arrays into a human-readable string for logging."""
    return tree_to_info(tree, lambda x: f"{x.shape}@{x.dtype}")


def init_gradient_accumulator(params: at.PyTree) -> at.PyTree:
    """Creates a zero-filled gradient accumulator matching a trainable parameter tree."""
    return jax.tree.map(jax.numpy.zeros_like, params)


def accumulate_gradients(accumulator: at.PyTree, grads: at.PyTree, accumulation_steps: int) -> at.PyTree:
    """Adds one micro-batch gradient to an accumulator as part of a gradient mean."""
    scale = 1.0 / accumulation_steps
    return jax.tree.map(lambda accumulated, grad: accumulated + grad * scale, accumulator, grads)


def train_rng(
    rng: at.KeyArrayLike,
    optimizer_step: at.Int[at.ArrayLike, ""],
    micro_step: at.Int[at.ArrayLike, ""],
    accumulation_steps: int,
) -> at.KeyArrayLike:
    """Derives a deterministic, distinct RNG for every micro-batch."""
    return jax.random.fold_in(rng, optimizer_step * accumulation_steps + micro_step)

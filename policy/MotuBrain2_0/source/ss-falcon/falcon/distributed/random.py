import contextlib
import logging
import random
from typing import Any, Union

import numpy as np
import torch
import torch.distributed as dist
from torch import _C
from torch.cuda import _lazy_call
from torch.cuda import device as device_ctx_manager
from torch.distributed.checkpoint.stateful import Stateful

from falcon.utils.deterministic import is_deterministic_enabled, set_deterministic_enabled

# Default name for the parallel rng tracker.
FALCON_DATA_PARALLEL_RNG_TRACKER_NAME = "data-parallel-rng"


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda",
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.

    Arguments:
        device (int): The gpu to retrieve the rng state

    This function is adapted from torch.cuda.random.get_rng_state()"""

    return torch.cuda.random.get_rng_state(device=device)


def _set_cuda_rng_state(new_state: torch.Tensor, device: int = -1):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
        device (int): The gpu to retrieve the rng state

    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]

            default_generator.set_state(new_state)

    _lazy_call(cb)


def _remove_trailing_dash_and_digits(text: str) -> str:
    return text.rstrip("-0123456789")


class CudaRNGStatesTracker(Stateful):
    """Tracker for the cuda RNG states.

    Using the `add` method, a cuda rng state is initialized based on
    the input `seed` and is assigned to `name`. Later, by forking the
    rng state, we can perform operations and return to our starting
    cuda state.
    """

    def __init__(self):
        self.dp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.reset()

    def is_initialized(self):
        """Checks if the internal RNG state has been set wirth set_states()."""
        return self._is_initialized

    def reset(self):
        """Set to the initial state (no tracker)."""

        # Track if initialized.
        self._is_initialized = False

        # Map from a string name to the cuda rng state.
        self.states_ = {}

        # Seeds are just for book keeping and ensure no seed is set twice.
        self.seeds_ = set()

    def get_states(self):
        """Get rng states. Copy the dictionary so we have direct
        pointers to the states, not just a pointer to the dictionary."""
        states = {}
        for name in self.states_:
            states[name] = self.states_[name]
        return states

    def set_states(self, states):
        """Set the rng states. For efficiency purposes, we do not check
        the size of seed for compatibility."""
        self._is_initialized = True
        self.states_ = states

    def add(self, name, seed):
        """Track the rng state."""
        self._is_initialized = True
        # Check seed is not already used.
        if seed in self.seeds_:
            raise Exception("seed {} already exists".format(seed))
        self.seeds_.add(seed)
        # Check that state is not already defined.
        if name in self.states_:
            raise Exception("cuda rng state {} already exists".format(name))

        # Get the current rng state.
        orig_rng_state = torch.cuda.get_rng_state()
        # Set the new state and store it.
        torch.cuda.manual_seed(seed)
        self.states_[name] = torch.cuda.get_rng_state()
        # Reset rng state to what it was.
        _set_cuda_rng_state(orig_rng_state)

    @contextlib.contextmanager
    def fork(self, name=FALCON_DATA_PARALLEL_RNG_TRACKER_NAME):
        """Fork the cuda rng state, perform operations, and exit with
        the original state."""
        # Check if we have added the state
        if name not in self.states_:
            raise Exception("cuda rng state {} is not added".format(name))
        # Store current rng state.
        orig_cuda_rng_state = _get_cuda_rng_state()
        # Set rng state to the desired one
        _set_cuda_rng_state(self.states_[name])
        # Record cpu RNG state
        cpu_rng_state = torch.get_rng_state()
        # Do the stuff we wanted to do.
        try:
            yield
        finally:
            # Throw a warning if cpu RNG state changed
            if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                logging.getLogger(__name__).warning("CPU RNG state changed within GPU RNG context")
            # Update the current rng state for later use.
            self.states_[name] = _get_cuda_rng_state()
            # And set the state to the original state we started with.
            _set_cuda_rng_state(orig_cuda_rng_state)

    def state_dict(self) -> dict[str, Any]:
        return {f"{key}-{self.dp_rank}": value for key, value in self.states_.items()}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        for key, value in state_dict.items():
            if _remove_trailing_dash_and_digits(key) in self.states_:
                self.states_[_remove_trailing_dash_and_digits(key)] = value


# RNG tracker object.
FALCON_CUDA_RNG_STATE_TRACKER = None
FALCON_CUDA_RNG_STATE_TRACKER_INITIALIZED = False


def initialize_cuda_rng_tracker():
    """Create the RNG tracker."""
    global FALCON_CUDA_RNG_STATE_TRACKER
    global FALCON_CUDA_RNG_STATE_TRACKER_INITIALIZED
    if FALCON_CUDA_RNG_STATE_TRACKER_INITIALIZED:
        return

    # Get the base tracker class
    base_tracker = CudaRNGStatesTracker
    tracker_class = base_tracker

    FALCON_CUDA_RNG_STATE_TRACKER = tracker_class()
    FALCON_CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def parallel_cuda_manual_seed(seed: int):
    """Initialize model parallel cuda seed.

    This function should be called after the model parallel is
    initialized. Also, no torch.cuda.manual_seed should be called
    after this function. Basically, this is replacement for that
    function.
    One set of RNG states are tracked:
    default state: This is for data parallelism and is the same among a set of model parallel GPUs
    but different across different model parallel groups. This is used for example for dropout
    in the non-tensor-model-parallel regions.
    """
    data_parallel_seed = seed
    initialize_cuda_rng_tracker()
    FALCON_CUDA_RNG_STATE_TRACKER.reset()
    # Set the default state.
    torch.cuda.manual_seed(data_parallel_seed)
    FALCON_CUDA_RNG_STATE_TRACKER.add(FALCON_DATA_PARALLEL_RNG_TRACKER_NAME, data_parallel_seed)


def get_rng_states_tracker() -> CudaRNGStatesTracker:
    """Get cuda rng tracker."""
    initialize_cuda_rng_tracker()
    assert FALCON_CUDA_RNG_STATE_TRACKER is not None, "CudaRNGStatesTracker is not initialized"
    return FALCON_CUDA_RNG_STATE_TRACKER  # type: ignore

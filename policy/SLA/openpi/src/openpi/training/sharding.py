import contextlib
import logging

import jax
import numpy as np

BATCH_AXIS = "batch"
FSDP_AXIS = "fsdp"
# In FSDP, we shard the data across both the batch and FSDP axes.
DATA_AXIS = (BATCH_AXIS, FSDP_AXIS)

# These adaptive RMSNorm projections operate on data-sharded activations. Sharding their output dimension makes XLA
# transpose the activation layout during the backward pass and replicate it across data-parallel groups.
_REPLICATED_FSDP_PATH_PARTS = (
    "['final_norm_1']['Dense_0']['kernel']",
    "['pre_attention_norm_1']['Dense_0']['kernel']",
    "['pre_ffw_norm_1']['Dense_0']['kernel']",
    "['time_mlp_in']['kernel']",
    "['time_mlp_out']['kernel']",
)


class _MeshState:
    active_mesh: jax.sharding.Mesh | None = None


def _device_rows(devices: list[jax.Device], num_fsdp_devices: int, num_nodes: int) -> list[list[jax.Device]]:
    devices_per_node = len(devices) // num_nodes
    # The launcher assigns contiguous process indices to each node. Sorting also keeps devices local to a multi-GPU
    # process adjacent, so this supports both one-process-per-GPU and one-process-per-node launch layouts.
    ordered_devices = sorted(devices, key=lambda device: (device.process_index, device.local_hardware_id))
    rows = []
    for node_start in range(0, len(ordered_devices), devices_per_node):
        node_devices = ordered_devices[node_start : node_start + devices_per_node]
        rows.extend(
            node_devices[start : start + num_fsdp_devices] for start in range(0, devices_per_node, num_fsdp_devices)
        )
    return rows


def make_mesh(
    num_fsdp_devices: int,
    devices: list[jax.Device] | None = None,
    *,
    num_nodes: int = 1,
) -> jax.sharding.Mesh:
    devices = list(jax.devices() if devices is None else devices)
    if len(devices) % num_fsdp_devices != 0:
        raise ValueError(
            f"Number of devices {len(devices)} must be divisible by the number of FSDP devices {num_fsdp_devices}."
        )
    return jax.sharding.Mesh(
        np.asarray(_device_rows(devices, num_fsdp_devices, num_nodes), dtype=object),
        (BATCH_AXIS, FSDP_AXIS),
    )


@contextlib.contextmanager
def set_mesh(mesh: jax.sharding.Mesh):
    """Plumbing the mesh deep into the module tree is extremeley cumbersome; until the JAX team lands a better API, a
    custom context manager like this one is the recommended way to maintain a reference to a global mesh. This is only used
    in `activation_sharding_constraint` below."""
    if _MeshState.active_mesh is not None:
        raise ValueError("Cannot nest set_mesh context managers.")
    _MeshState.active_mesh = mesh
    try:
        yield
    finally:
        _MeshState.active_mesh = None


def activation_sharding_constraint(pytree):
    if _MeshState.active_mesh is None:
        return pytree
    return jax.lax.with_sharding_constraint(
        pytree,
        jax.sharding.NamedSharding(_MeshState.active_mesh, jax.sharding.PartitionSpec(DATA_AXIS)),
    )


def fsdp_sharding(
    pytree,
    mesh: jax.sharding.Mesh,
    *,
    min_size_mbytes: int = 4,  # 4 MiB
    log: bool = False,
):
    """Apply FSDP sharding to a pytree of arrays based on the mesh shape.

    Args:
        pytree: A pytree to be apply sharding specified by the mesh, note that only array types (eg. contains .shape attr)
          will be considered for sharding.
        mesh: The mesh being used for applying sharding on to pytree.
        min_size_mbytes: The minimum size of the array in MiB to be considered for sharding, any array smaller than this
          will be replicated.
        log: If true, will log the sharding decisions for arrays that are being considered for sharding.

    Returns:
        The sharded pytree.
    """
    min_size_bytes = min_size_mbytes * 2**20

    def _shard_arr(kp, array: jax.ShapeDtypeStruct):
        path = jax.tree_util.keystr(kp)
        if any(part in path for part in _REPLICATED_FSDP_PATH_PARTS):
            if log:
                logging.info("Replicating %s to preserve data-sharded adaptive RMSNorm activations", path)
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # if fsdp is not actually going to be used, replicate everything to avoid extraneous logging
        if mesh.shape[FSDP_AXIS] == 1:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate scalar and vector arrays
        if not hasattr(array, "shape"):
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        if len(array.shape) < 2:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        # replicate small arrays
        if (arr_size := np.prod(array.shape) * np.dtype(array.dtype).itemsize) < min_size_bytes:
            return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

        # shard matrices and larger tensors along the largest axis that is divisible by the fsdp dimension
        axes = np.argsort(array.shape)[::-1]
        spec = [None] * len(axes)
        for i in axes:
            if array.shape[i] % mesh.shape[FSDP_AXIS] == 0:
                if log:
                    logging.info(f"Sharding {path} of shape {array.shape} ({arr_size / 2**20:.2f} MiB) along axis {i}")
                spec[i] = FSDP_AXIS
                return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(*spec))

        # replicate if no valid sharding was found
        if log:
            logging.warning(
                f"Could not find a valid sharding for {path} of shape {array.shape} with mesh of shape {mesh.shape}"
            )
        return jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    return jax.tree_util.tree_map_with_path(_shard_arr, pytree)

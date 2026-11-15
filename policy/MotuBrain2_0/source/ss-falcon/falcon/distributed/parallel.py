import logging
import os
from dataclasses import dataclass

import torch
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

logger = logging.getLogger(__name__)


global WORLD_MESH
WORLD_MESH = None
global NODE_GROUP
NODE_GROUP = None
global NODE_GROUP_INITIALIZED
NODE_GROUP_INITIALIZED = False
global NODE_GROUP_RANKS
NODE_GROUP_RANKS = None


def get_world_mesh() -> DeviceMesh | None:
    global WORLD_MESH
    return WORLD_MESH


def get_cp_group() -> torch.distributed.ProcessGroup | None:
    global WORLD_MESH
    if WORLD_MESH is None or WORLD_MESH.mesh_dim_names is None or "cp" not in WORLD_MESH.mesh_dim_names:
        return None

    return WORLD_MESH["cp"].get_group()


def get_cp_size() -> int:
    global WORLD_MESH
    if WORLD_MESH is None or WORLD_MESH.mesh_dim_names is None or "cp" not in WORLD_MESH.mesh_dim_names:
        return 1

    return WORLD_MESH["cp"].size()


def get_cp_rank() -> int:
    global WORLD_MESH
    if WORLD_MESH is None or WORLD_MESH.mesh_dim_names is None or "cp" not in WORLD_MESH.mesh_dim_names:
        return 0

    return WORLD_MESH["cp"].get_local_rank()

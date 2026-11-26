"""Fixed X5 / single table extension for the workspace CuRobo source snapshot.

Only the world cost receives a subset. Native self/bounds keep the full model.
Scene, installation and sphere mapping are immutable for the solver lifetime.
"""
from copy import copy
from dataclasses import dataclass, fields
import math

import torch
from curobo._src.cost.cost_scene_collision import SceneCollisionCost
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.robot.types.joint_types import JointType
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg


def table_signature(scene):
    """Accept exactly the current product's one named cuboid scene."""
    if not isinstance(scene, SceneCfg) or len(scene) != 1:
        raise ValueError("Expected a single-table SceneCfg")
    table = scene[0]
    if type(table) is not Cuboid or table.name != "table":
        raise ValueError("Expected only the cuboid named table")
    dims, pose = tuple(table.dims), tuple(table.pose)
    if len(dims) != 3 or len(pose) != 7:
        raise ValueError("Malformed table geometry")
    if not all(math.isfinite(v) for v in dims + pose) or min(dims) <= 0:
        raise ValueError("Invalid table geometry")
    if not math.isclose(sum(v * v for v in pose[3:]), 1.0, abs_tol=1e-5):
        raise ValueError("Table quaternion must be normalized")
    return dims, pose


@dataclass(frozen=True)
class FixedX5Table:
    geometry: tuple
    root: str = "base_link"

    @classmethod
    def from_scene(cls, scene, *, robot_family, fix_root_link, root):
        # These arguments must come from the actual articulation configuration.
        if robot_family != "x5" or fix_root_link is not True or root != "base_link":
            raise ValueError("Only the fixed X5 base_link installation is supported")
        geometry = table_signature(scene)
        if geometry[1][3:] != (1., 0., 0., 0.):
            raise ValueError("The fixed table preflight requires an axis-aligned table")
        return cls(geometry, root)

    def check_scene(self, scene):
        if table_signature(scene) != self.geometry:
            raise ValueError("Solver scene differs from the installed table contract")

    @property
    def dimensions(self):
        return self.geometry[0]

    @property
    def center(self):
        return self.geometry[1][:3]

    def keep_indices(self, kin):
        if self.root != "base_link" or kin.base_link != self.root:
            raise ValueError("Mount permission must name the kinematic root")
        root_index = kin.link_name_to_idx_map[self.root]
        if int(kin.joint_map_type[root_index]) != JointType.FIXED.value:
            raise ValueError("Root must have a fixed joint type")
        root_spheres = kin.get_sphere_index_from_link_name(self.root)
        if root_spheres.numel() == 0:
            raise ValueError("Expected X5 root collision spheres")
        mask = torch.ones_like(kin.link_sphere_idx_map, dtype=torch.bool)
        mask[root_spheres] = False
        keep = mask.nonzero().view(-1)
        if keep.numel() == 0:
            raise ValueError("Installation must retain moving robot spheres")
        return keep


class FixedRootSceneCost(SceneCollisionCost):
    """Native cost with a world-only sphere subset and independent buffer."""
    def __init__(self, original, keep):
        config = copy(original.config)
        config.update_num_spheres(keep.numel())
        super().__init__(config)
        self.keep = keep
        if not original.enabled:
            self.disable_cost()

    def forward(self, state, idxs_env_query=None, trajectory_dt=None):
        world_state = KinematicsState(
            robot_spheres=state.robot_spheres.index_select(-2, self.keep)
        )
        return super().forward(world_state, idxs_env_query, trajectory_dt)


class FixedRootCostManager(RobotCostManager):
    def initialize_from_config(self, config, transition_model,
                               scene_collision_checker=None, **kwargs):
        if not isinstance(config.installation, FixedX5Table):
            raise ValueError("Missing fixed installation contract")
        if scene_collision_checker is None:
            raise ValueError("Missing installed table checker")
        config.installation.check_scene(scene_collision_checker.scene_model)
        kin = transition_model.robot_model.config.kinematics_config
        self._world_keep = config.installation.keep_indices(kin)
        super().initialize_from_config(
            config, transition_model, scene_collision_checker, **kwargs
        )

    def register_cost(self, name, component):
        if name == "scene_collision":
            component = FixedRootSceneCost(component, self._world_keep)
        super().register_cost(name, component)


@dataclass
class FixedRootCostManagerCfg(RobotCostManagerCfg):
    installation: FixedX5Table | None = None

    def __post_init__(self):
        super().__post_init__()
        self.class_type = FixedRootCostManager

    @staticmethod
    def create(data_dict, **kwargs):
        base = RobotCostManagerCfg.create(data_dict, **kwargs)
        return FixedRootCostManagerCfg(**{
            f.name: getattr(base, f.name)
            for f in fields(RobotCostManagerCfg) if f.init
        })


def bind_solver_config(solver_config, installation):
    """Call on internal single-planner IK/trajopt and standalone IK before init."""
    core = solver_config.core_cfg
    scene_cfg = getattr(core, "scene_collision_cfg", None)
    if scene_cfg is not None:
        installation.check_scene(scene_cfg.scene_model)
    for rollout in [*core.optimizer_rollout_configs, core.metrics_rollout_config]:
        if rollout.cost_manager_config_instance_type is not FixedRootCostManagerCfg:
            raise ValueError("Factory did not propagate the fixed-root manager type")
        for config in rollout.get_cost_manager_configs():
            if type(config) is not FixedRootCostManagerCfg:
                raise ValueError("Unexpected rollout manager configuration")
            config.installation = installation


def bind_single_planner_config(config, installation):
    installation.check_scene(config.scene_collision_cfg.scene_model)
    # Current native product uses plan_joint. Unused graph is removed explicitly.
    config.graph_planner_config = None
    bind_solver_config(config.ik_solver_config, installation)
    bind_solver_config(config.trajopt_solver_config, installation)


def assert_solver_installation(solver, installation):
    """Fail before warmup if a constructed rollout bypassed the contract."""
    rollouts = solver.core.get_all_rollout_instances() + list(solver.core.additional_metrics_rollouts.values())
    for rollout in rollouts:
        for manager in rollout._cost_manager_list:
            if (type(manager) is not FixedRootCostManager
                    or manager.config.installation != installation):
                raise ValueError("Constructed rollout bypassed fixed-root manager")
            world = manager.get_cost("scene_collision")
            if manager.config.scene_collision_cfg is not None:
                if type(world) is not FixedRootSceneCost:
                    raise ValueError("Constructed rollout bypassed fixed-root world cost")
                installation.check_scene(world.config.scene_collision_checker.scene_model)


def preflight_components(checker, q, keep):
    """Query native full self/bounds and subset world independently.

    Return strict self/bounds and strict world masks plus original spheres.
    The caller applies the existing nondeepening escape rule to table geometry;
    a strict world mask alone would reject every initially overlapping escape.
    """
    if checker.self_collision_cost is None or checker.collision_constraint is None:
        raise ValueError("Preflight requires complete self and installed world checks")
    state = checker.get_kinematics(q)
    spheres = state.robot_spheres

    def clear(value):
        if value.shape[:2] != q.shape[:2] or value.ndim < 2:
            raise ValueError("Unexpected native preflight cost shape")
        value = value.reshape(*q.shape[:2], -1)
        return (torch.isfinite(value) & (value == 0)).all(dim=-1)

    valid = clear(checker.get_bound(q)) & clear(checker.get_self_collision_distance(spheres))
    world = checker.get_collision_constraint(spheres.index_select(-2, keep))
    if not torch.isfinite(world).all():
        raise ValueError("Nonfinite installed world constraint")
    return valid, clear(world), spheres

"""Static target encoding for public factories with observation-only robots."""
from pathlib import Path
from types import SimpleNamespace
import json
from rsi.resource_locations import load_manifest as load_resource_manifest


def clock_environment(environment):
    """Keep the actor environment; supply only missing clock descriptions.

    These declarations encode commands, not measurements of fingers/contact.
    The original clock's sampler, initial normalized feedback and subsequent
    ACK history stay unchanged. No planner or native manager is constructed.
    """
    if hasattr(environment.robot_manager, 'robot_list'):
        return environment
    root = Path(__file__).resolve().parents[4]
    declaration = load_resource_manifest(root / 'configs/public-x5-numeric-tools.json')
    parameters = declaration['public_robot_parameters']
    robots = tuple(SimpleNamespace(type='target', arm_name=side,
        gripper_name=side.replace('arm', 'ee'),
        gripper_scale=tuple(parameters['gripper_scale']),
        gripper_move={'sign': parameters['sign'], 'mimic': tuple(parameters['mimic'])})
        for side in declaration['planner_robot_names'])
    manager = SimpleNamespace(robot_list=robots,
        process_name=lambda name: name if name.endswith('_state') else name + '_joint_state')
    return SimpleNamespace(robot_manager=manager, step_lim=environment.step_lim)

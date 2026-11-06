---
name: mature_sampled_motion_observed_start
description: "CheckedSampledMotion plans sampled mature motions from observed-start own-URDF pose or joint inputs, checked native cuRobo routes, and measured-joint-arc replay. It maps operation names to contact, carry, free, home, contact_correction, retreat, or view profiles and returns sampled joint positions or failure/budget results."
---

# Mature sampled motion observed start

## Function

CheckedSampledMotion plans sampled mature motions from observed-start own-URDF pose or joint inputs, checked native cuRobo routes, and measured-joint-arc replay. It maps operation names to contact, carry, free, home, contact_correction, retreat, or view profiles and returns sampled joint positions or failure/budget results.

## Inputs

- CheckedSampledMotion(urdf)
- planner: cuRobo planner exposing robot_cfg, device_cfg, motion_planner.scene_collision_checker, plan_joint, and plan_path
- payload: operation in the bounded vocabulary, side, observation.state with <side>_arm_joint_state and <side>_ee_pose, pose or joint_target, and remaining_actions
- validate(side, planner, positions) positions to collision-check

## Outputs

- plan() returns {'status':'Success','positions': sampled positions,'dense_waypoints': len(dense),'official_waypoints': len(sampled positions)}
- failure returns {'status':'Fail','positions':[],'dense_waypoints':0,'official_waypoints':0}; budget returns {'status':'Budget','positions':[],'dense_waypoints':len(dense),'official_waypoints':0}
- a [RIEMANN SAMPLED OWN ENDPOINT] JSON log reports commanded_joints, requested_own_pose, own_fk_position_error_m, own_fk_rotation_error_radians, public_robot_inputs_only, trajectory_changed, and is_contact_or_object_state_sensor
- validate() returns bool(all collision-checker validate results)

## Callable interface

Implementation: [implementation.py](implementation.py).

- `CheckedSampledMotion: __init__(self, urdf); reset(self); validate(self, side, planner, positions); plan(self, planner, payload)`

## Constraints

- OPERATIONS maps sampled_contact->contact, sampled_carry->carry, sampled_free->free, sampled_home->home, sampled_correction->contact_correction, sampled_retreat->free, sampled_view->view; CONSTRAINED is contact, carry, contact_correction, retreat
- route order: constrained Cartesian continuity first when required, then checked joint route, then checked pose fallback; sampled_home uses plan_joint to joint_target
- sample_path must succeed with remaining_actions; reject Budget rather than truncate and preserve all replay knots and the profile terminal hold
- never invokes a teacher or receives an environment object; a checked route is not proof of grasp or task success, and current public visual verification plus independent native scorer remain necessary

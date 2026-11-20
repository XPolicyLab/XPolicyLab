---
name: public_urdf_joint_planning
description: "Defines JointPlanner over an own URDF and public state to compute six-joint FK, Jacobians, damped least-squares IK, multi-seed IK, TCP/line planning, and pose solving; plan_joint samples joint-space interpolation at a bounded max_dq."
---

# Public urdf joint planning

## Function

Defines JointPlanner over an own URDF and public state to compute six-joint FK, Jacobians, damped least-squares IK, multi-seed IK, TCP/line planning, and pose solving; plan_joint samples joint-space interpolation at a bounded max_dq.

## Inputs

- urdf: own robot URDF model/path accepted by OwnURDFKinematics with preserve_declared_limits=True
- state: dict containing left_arm_joint_state, left_ee_pose, right_arm_joint_state, and right_ee_pose for public observed bases
- target poses as 4x4 numpy arrays and q_start/q_goal/q_seed/q_ref as six-element joint vectors in radians
- optional limits Nx2 array, step_m meters, step_rad radians, offset/TCP_X meters, and grip meters clamped to [0, GRIP_MAX]

## Outputs

- plan_joint returns a list of six-element joint vectors excluding q_start and ending at q_goal
- ik returns (q, position_error_m, rotation_error_rad) for the best encountered iterate; ik_multi returns (q, ep, er)
- plan_line returns a list of joint vectors or raises PlanError on IK failure or per-step jump greater than MAX_LINE_JUMP=0.25 rad
- solve returns a wrapped joint solution or raises PlanError if no seed meets ep<0.002 m and er<0.02 rad; fk/jacobian/tcp/link_frames/link_box_corners return numerical transforms or named geometry

## Callable interface

Implementation: [implementation.py](implementation.py).

- `PlanError: exception or data type`
- `rot_y(a)`
- `axis_angle(axis, angle)`
- `make_T(R=None, p=None)`
- `rot_err(R_target, R_cur)`
- `wrap(a)`
- `plan_joint(q_start, q_goal, max_dq=0.06)`
- `approach_pose(p, azimuth, pitch)`
- `JointPlanner: __init__(self, urdf, state); fk_chain(self, arm, q); fk(self, arm, q); tcp_from_link6(self, T6, offset=TCP_X); link6_from_tcp(self, Ttcp, offset=TCP_X); jacobian(self, arm, q); ik(self, arm, T_target, q_seed, iters=150, tol_p=0.0005, tol_r=0.005, w_rot=0.35, limits=None); ik_multi(self, arm, T_target, q_seed, extra_seeds=(), **kw); tcp(self, arm, q, offset=TCP_X); plan_line(self, arm, q_start, T_goal, step_m=STEP_M, step_rad=STEP_RAD, offset=TCP_X); solve(self, arm, T, q_ref); link_frames(self, arm, q, grip=0.044); link_box_corners(self, arm, q, grip=0.044, pad=0.006)`

## Constraints

- Six joints only; IK defaults to 150 iterations per seed, damping 0.01, step cap 0.3 rad, and convergence gates tol_p=0.0005 m and tol_r=0.005 rad unless overridden
- World bases come from public observed own arm joint state and EE pose, not simulator state; no object handles, scoring, or measured jaw API are used
- link_frames/link_box_corners treat grip as a clamped argument and do not represent measured physical finger width or contact
- line planning accepts ep<=0.002 m and er<=0.03 rad after multi-seed fallback; all results are numerical hypotheses

## Provenance

GO-3 PR214 `e4178ee754413cd792d3a37e0411ea45c58761c3`: selected FK/Jacobian/IK/line and joint sampling mathematics. World bases are recovered from public own-robot EE poses; fixed world bases and measured-width conversions are excluded.

## Robot guide

### Setup
- Two ARX X5 arms, left and right, with bases at world x = -0.30 m and +0.30 m, y = -0.45 m, z = 0.765 m.
- World frame: +x from the left base toward the right base, +y forward across the table, +z up.
- The TCP is the grasp centre between the finger pads. TCP frame: +x along the fingers (the approach direction), +y across the jaws. Relative commands use axes frozen when the command starts; rotations turn about the TCP.
- Gripper opening is 0 to 88.8 mm. The fingers are position-controlled: closed on an object, each presses with a force that grows with how far the commanded width lies below the width the object allows (about 11.5 N per finger per centimetre). A command below 0, down to -19 mm, squeezes harder; a thin object held at a 0 mm command gets only a light grip. `gripper_width_m` is estimated from one measured joint. When a closing gripper stops wider than commanded, something is usually between the fingers, but only images confirm a grasp. The fingers take about 0.4 s to travel their range; a shorter duration_s ends the command with the fingers still moving.
- cam_head is fixed above and behind the robot and sees the whole table. cam_left_wrist and cam_right_wrist move with each gripper and look along the fingers; a free arm can serve as a movable camera.
- The simulation pauses between commands; only motion, or `robot.hold`, advances it, so after a release nothing settles or falls until the next command. One control step is 1/25 s of motion: moving 20 cm at 0.2 m/s costs 25 steps. Looking, computing and reading cost no steps.
- Steps count simulated time for the whole robot, not per arm: while one arm moves on its own, the other arm's time is spent too. In `execute_trajectory` both arms move during the same steps, so two independent 1 s motions cost 25 steps together instead of 50 one after the other. Each waypoint gives a target for one or both arms and one duration that must respect both arms' speed limits; if either arm stops (IK failure, tracking error), the trajectory stops for both. The state line's return≈k is the number of steps robot.move_to_mark('initial') would take from the current pose.
- `step_report()` shows where this episode's physical steps went: the steps per method and per what set each motion's duration (a speed or duration you requested, a default speed, a minimum duration, or the joint-space speed limit), and the costliest commands (step_report(top=20) lists more).
- Simulated motion runs about 9 times slower than real time (each action line shows its wall time): a 3 s motion takes about 30 s. Set python_exec's timeout_s with that in mind.

### Python SDK
```python
left, right = robot.arm('left'), robot.arm('right')
left.move_by(dx_m=0, dy_m=0, dz_m=0, frame='world', speed_m_s=0.2)    # frame: world | tcp
left.move_to(position_m=[x, y, z], quaternion_xyzw=None, speed_m_s=0.2, speed_rad_s=0.5)  # world frame; None keeps orientation
left.rotate_by(axis='z', angle_deg=10, frame='tcp', speed_rad_s=0.5)      # about the TCP
left.gripper.set_width(width_m=0.04, duration_s=0.4); left.gripper.open(); left.gripper.close()
left.move_by(dz_m=0.08, speed_m_s=0.2, gripper_width_m=0.088)    # the gripper moves toward this width during the motion
left.move_to_mark('initial', speed_rad_s=1.0)   # this arm back to its pose at the start of the episode
robot.move_to_mark('initial')              # both arms at once, in the steps of the slower arm
# move_to_mark moves the joints straight to the start pose (no Cartesian path, no IK), so it can leave a pose where Cartesian moves keep failing; robot.move_to_mark does not check the two arms' paths against each other
robot.execute_trajectory([{'left': {'delta_m': [0, 0, 0.05]}, 'right': {'delta_m': [0, 0, 0.05]},
                           'duration_s': 1.0}])                                        # both arms at once
left.move_to(position_m=[x, y, z], quaternion_xyzw=q, dry_run=True)                 # a check only: nothing moves, no step is used
obs = robot.observe()                      # images and measured state; costs no steps
robot.replay('frame_000123')               # a recorded observation
robot.hold(duration_s=1.0)                 # both arms stay still while this much simulated time passes (25 steps per second)
robot.status(command_id)
```
- Limits: speed up to 0.4 m/s and 2 rad/s (about 114°/s); `move_to_mark` moves the joints at 3 rad/s, or slower with `speed_rad_s` (a joint speed). A single motion or waypoint lasts at most 20 s. A trajectory, or its dry run, is checked before it starts: if any waypoint lasts longer than 20 s or its `duration_s` would move an arm faster than these limits, the whole trajectory is rejected without moving.
- Waypoint entries accept `position_m`, `quaternion_xyzw`, `delta_m`, `rotation_axis`, `angle_deg` and `frame`.
- `gripper_width_m` on `move_by`, `move_to`, `rotate_by` or in a waypoint's arm entry starts that gripper toward the width with the motion's first step; the arm does not wait for the fingers, the action line ends with the width measured when the motion ends (the fingers may still be moving), and the width stays the target afterwards, as with `set_width`. To grasp, arrive first and then close with `set_width`; `gripper_width_m` suits opening or pre-opening while the arm moves away or approaches.
- `dry_run=True` on `move_to`, `move_by`, `rotate_by` or `execute_trajectory` checks a motion without moving: nothing moves and no step is used. The report says whether the straight path would stop on `ik_failed` or `joint_jump_or_speed_limit` and where, how many steps the motion would take, and each joint's planned range. It cannot predict contact, `tracking_error`, slipping or final accuracy, so `feasible` is not a guarantee. It is listed as `dry run move L … → feasible 29steps 1.2s (nothing moved)` or `→ would_stop ik_failed:left at 40% after 12steps (nothing moved)`.
- A move carries the TCP along a straight line (move_to_mark: the joints straight to the start pose); the whole arm, the wrist and the open fingers sweep the space in between. Nothing checks that sweep against the other arm or against objects: dry_run checks only reach, joint limits and the arm against itself and a table plane 2.5 cm below the real table top (so it can pass a path that dips up to 2.5 cm into the table), and a parked arm stays where it is. A contact that pushes a moving arm less than about 2 cm off its path is not reported (a move_to_mark is checked only where it ends).
- `display(x)` shows an observation or an action report (the images after it); `display(x, cameras=['cam_head'])` selects cameras. `display_image(img, caption)` shows an image you made, such as a crop, an overlay or a side-by-side comparison. Images arrive only after the whole python_exec call ends.
- `obs['robot_state'][arm]['tcp']` holds `position_m` and `quaternion_xyzw`; `obs['images']` is a list of `{'camera', 'png_base64'}`, and `rgb(obs, 'cam_left_wrist')` gives one as a NumPy array. Do not print whole observations.

### Reading results
Every command in a python_exec call is listed as one line, for example
`#23 c1a2b3c4 move L Δ(+0.0,+15.0,+0.0)mm world 25mm/s → reached err 0.1mm 15steps 0.6s`,
where `c1a2b3c4` is the command ID (use it with robot.status, the inspect tool and in your notes). A displayed
observation or report adds a state line `L (x,y,z)mm rpy(roll,pitch,yaw)° grip width | R ... | step n | frame_id`.
In a development episode the step part reads `step n (eval budget N; return≈k)`: N is the evaluation budget, shown for reference and not enforced in development.

- `reached`: the motion completed within 3 mm and 0.06 rad. It says nothing about a grasp or the task.
- `partial`: the command stopped early; the reason follows.
  - `ik_failed:<arm> at N%`: no joint solution along this straight path at that point. A different orientation or route may still reach the region.
  - `joint_jump_or_speed_limit:<arm> at N%`: at that point, following this straight path at this speed would need a joint of that arm to exceed the 5 rad/s joint-speed limit, or to switch to another arm configuration. Moving slower lowers the joint speed and sometimes passes; otherwise change the orientation or the route.
  - `tracking_error`, `final_tolerance_not_met`: motion stopped for safety or accuracy. A `tracking_error` also names, for each arm that fell off its path, the joint furthest from its commanded position, e.g. `(R joint3 at -0.034 rad, commanded -0.084 rad)`.
  - `gripper_width_not_reached_contact_or_tracking_unknown`: the fingers stopped away from the commanded width, often because an object is between them.
  - A trajectory names the waypoint where it stopped and why, in the same terms, e.g. `(1/2 waypoints reached; waypoint 2 stopped by joint_jump_or_speed_limit:left at 85%)`.
- `rejected`: nothing moved, for example `duration_exceeds_timeout` (a motion or waypoint longer than 20 s), or for a trajectory `exceeds_speed_limit`, which names each waypoint and arm that is too fast and the shortest `duration_s` within the limits, e.g. `(waypoint 2 R 2.60 rad/s > 2 rad/s, needs duration_s ≥1.3)`.
- `benchmark_terminal` or `episode_already_ended`: the environment has ended the episode; no further motion is possible. The environment ends an episode as soon as the task's scoring rule is met, or when an evaluation's step budget is used up, or when the task's scoring rule says an action fails it.
- `! uncommanded motion of a held arm`: an arm that should have stayed still moved; inspect the named frame.
- When a command's outcome is uncertain (an exception or lost reply), check `robot.status(command_id)` before moving again. Never resubmit a motion that may already have run.
- A python_exec time limit never interrupts a running motion: the motion completes, then the rest of the code is skipped.

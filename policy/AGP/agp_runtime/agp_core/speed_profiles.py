"""Empty-gripper speed profile F, two-arm guard F-g, held-object guard D and the further motion switches below, as a
worker mixin.

Three switches (PART_A; agp_core/features.py, opt-in, all off by default). With all three off, agp_core/official_worker.py
does not compose this mixin for them and the worker class is unchanged (nothing of this module is imported).

    OURS_FREE_TOPT   F    empty-gripper transit moves and home re-timed along their own path
    OURS_ARM_GUARD   F-g  clearance between the two arms: before a stage starts and before every control step
    OURS_HOLD_GUARD  D    a lift or transport move of an arm that holds something stops when the observation and the
                          wrist image both show it lost it

Official observation only. The gripper value n of the official observation (<arm>_ee_joint_state) is the commanded
finger target clamped to the measured finger +-10.8 mm (env/robot_manager/control_manager.py process_gripper_val,
gripper_eps 0.2 of the scale [-0.01, 0.044] m), mapped to [0, 1]: n = 1 with the gripper commanded open means the fingers
are no more than 10.8 mm from fully open (no clamp; the fingers are at least 67 mm apart); n = 0 with the gripper
commanded closed means the finger is within 0.8 mm of closed (nothing between the pads); 0 < n with the gripper
commanded closed means the fingers stopped (or still move) apart: something is between them.

F (fixed parameters: 4 rad/s, 20 rad/s^2, TCP 1 m/s, 3 rad/s)
  Which motions: move_ee / move_delta with speed_profile transit, and home, of an arm that is empty when the stage is
  prepared: gripper commanded open and n = 1 (a commanded open has converged), or commanded closed with both signals of
  D: the reading shows the fingers fully closed (n = 0) and the pad gap estimated from the arm's wrist image in the same
  observation (agp_core/visual_gap.py, quality ok) is below HOLD_LOST_GAP_M = 1.5 mm (empty_state). n = 0 alone is not
  empty: a held object pushed against something can drive one finger shut and leave the reading at 0 while the object is
  still between the pads (an object dropped once F sped such an arm up); a closed gripper reading n = 0 whose estimate
  is unavailable, not quality ok or shows the pads apart counts as holding (F does not re-time it; H, when on, does:
  retime_holding). The image is looked at only for a gripper commanded closed whose reading is 0 (once per arm and
  observation; the record ours_speed.empty_check).
  Not re-timed: precision moves, move_joints (always precision in the runtime), a move with max_samples (a checkpoint
  keeps the sample rate it was asked for), an arm that holds something, an open that has not converged (the move
  keeps the old rate).
  How: the worker prepares the move exactly as before (same IK, planner path, joint and Cartesian resampling: the
  "dense" path); the dense polyline is then re-timed (retime_free) to the fastest motion along it from rest to rest
  under these limits, all joints synchronised (every joint follows the same path parameter):
    joint speed      <= 4 rad/s (FREE_JOINT_SPEED)             per joint, per control step
    joint accel.     <= 20 rad/s^2 (FREE_JOINT_ACCEL)          per joint, second difference of the control steps,
                                                               including the change of direction at path corners
    TCP speed        <= 1 m/s (FREE_TCP_SPEED), rotation <= 3 rad/s (FREE_TCP_ROTATION)
    low altitude     TCP below 10 cm above the table top: <= 0.25 m/s                                     (LOW_*)
    landing zone     a move ending below 10 cm: its last 3 cm of TCP path <= 0.12 m/s and <= 1 rad/s    (LANDING_*)
    take-off zone    a move starting below 6 cm: its first 2 cm of TCP path <= 0.12 m/s, translation only (TAKEOFF_*)
  The table top is the policy planner's public table (env_cfg/scene/<scene>.yml), in the arm's base frame. Samples lie
  on the dense polyline (linear interpolation between its points), so the commanded path is the dense path; the
  preflight (table, self collision, joint limits, wrist table) runs on the dense path, as before the re-timing.
  The worker's tracking-error stop for these motions is FREE_TRACKING_STOP_RAD = 0.05 rad instead of 0.08 (in
  simulator probes, 4 rad/s, 20 rad/s^2 trapezoids track within 0.024 rad).
  Approximate homes (agp_core/approx_home.py) take the same time law: the steps a home needs, the target search and the
  budget tiers use it (ApproxHomeMixin asks _home_time_law); other homes are re-timed here.

F-g (margins are engineering values, see below)
  Shapes: agp_core/arm_guard.py (public URDF, x5_link_hulls.json, the colliders' approximation of ARX.usd; fingers at the
  observed opening). A held object has no envelope (arm_guard.HELD_RADIUS_M = 0): the robot's own links, fingers
  included, are checked. Both arms, every motion stage (any arm with move_ee, move_delta, move_joints or home).
  Layer 1, before a stage starts: both arms' commanded samples and, between consecutive samples, the official per-step
  interpolation (eval_env.process_control_info: 8 substeps from the measured joints toward the target at i/9, then the
  target), both arms at the same instants (a static arm at its measured joints). A stage keeps GUARD_MARGIN_M = 25 mm,
  or, when it starts closer than that, never comes closer than its start minus 2 mm (it may separate the arms). When
  it does not: in a stage where both arms move, the start of one arm is delayed by 1..GUARD_DELAY_MAX hold steps (the
  delay that adds the fewest steps to the stage, then the smallest); otherwise the stage is refused before any motion
  with the links, the control step and the distance.
  Layer 2, before every control step of a stage (trajectory and settle samples): the measured joints of both arms
  (official observation) and the targets about to be sent, with the same interpolation; below GUARD_HARD_M = 10 mm (or
  the stage's start clearance minus 2 mm, if smaller) both arms stop before the step is sent.
  Why 25 mm and 10 mm: the colliders of the simulator are the convex hulls of the same meshes (link3: a convex
  decomposition, V-HACD with a voxel of about 1.7 mm and up to 10 % concavity), so the hull distance is the simulator's
  up to the hull simplification (a few mm); the commanded joints lead the measured ones by up to 0.024 rad in F motions
  (about 12 mm at a 0.5 m lever) and both arms lag alike; the interpolated instants are up to 0.018 rad apart (about
  5 mm at the TCP). Layer 1's 25 mm covers these (5 + 12 + 5 mm and a few mm to spare); layer 2 sees the measured
  joints, so only the hull simplification and the step's own interpolation remain: 10 mm. A normal stage planned at
  >= 25 mm never comes below 10 mm in layer 2; reaching it means the arms left their plan (contact, a push).

D
  Which arms: an arm that holds something when the stage starts (gripper commanded closed, n > 0) and lifts or carries
  it in the stage (hold_guard_phase; the worker does not see the runtime intent, so the phase comes from the motion):
    transport  a move_ee / move_delta at transit speed (the runtime allows transit only for the intents transfer,
               retreat and observe; a place that P split runs its fast part this way, intent transfer), a home
    lift       a precision move (move_ee, move_delta, move_joints) whose last scheduled sample puts the TCP at least
               HOLD_LIFT_RISE_M = 1 mm higher than at the start (the grasp lift; a retreat up at precision)
  Not watched: a precision move that does not go up (a contact, a place descent: with P its precision tail, a
  precision transfer sideways), hold, gripper. In our recorded runs, all 300 precision moves of a holding arm with
  intent grasp rose by 4.7 mm or more; 218 of the 219 with intent place did not rise (the other rose 10.4 mm and
  ended "Cartesian endpoint no longer settling"); no precision move of a holding arm had another intent.
  When it stops: after every control step (trajectory and settle) both signals hold for a watched arm: the official
  observation shows the fingers fully closed (n <= CLOSED_EMPTY) and the pad gap estimated from the arm's wrist image in
  the same observation (agp_core/visual_gap.py, quality ok) is below HOLD_LOST_GAP_M = 1.5 mm (visual_gap's closed-empty
  bound). n = 0 alone does not stop: a thin rim between the pads reads n = 0 from a pad gap of about 2.4 mm down (the
  clamp of the official reading). An estimate that is unavailable or not quality ok does not stop either. The stop
  ends the stage before the next step: held_short with stop_reason '<arm> held object lost: ...' (both signals named).

Records: motion_diagnostics/*.json get `ours_speed` (what applied, the re-timing record, the guard's verdict, the
D state: hold_guard {holding_at_start, observed_at_start, watched, phase, reason}, hold_trace (the reading after every
step of a watched arm), hold_checks (the image looked at after a step whose reading showed n = 0: reading, quality,
pad gap, lost), hold_losses (the stop); empty_check (only when the stage's preparation looked at the arm's wrist
image for F's or H's empty/holding decision: reading, quality, pad gap, empty)); the npz gets `ours_dense` (the dense
path) for re-timed motions; arm_guard.jsonl in the session has one line per guarded stage (layer 1 verdict, delays,
refusals, layer 2 stops); speed_profiles.json the switches and constants.

Three more switches (PART_B), opt-in, all off by default. With these three off, everything above is exactly as
described (no PART_B record is written, no PART_B code path is taken).

    OURS_GRIP_OBS      G  a gripper action ends when the official observation shows the fingers stopped or open enough
    OURS_PREC_ZONE     P  precision only near contact. Here: the lift of a held object (its first 1 cm at precision);
                          the contact and place splits are made by the run (agp_core/prec_zone.py, agp_core/runner.py)
    OURS_HOLD_PROFILE  H  transit moves while holding: Cartesian s-curve 0.2 m/s, 1 m/s^2, 1 rad/s, joints 2 rad/s

G (GRIP_*)
  A gripper action keeps its command (the worker sends it at every later step, so fingers that are still moving keep
  moving during the next motion), but its stage ends early: the plan keeps its 10 samples as the maximum, and the stage
  ends after the control step at which the rule holds for every gripper arm of the stage and every other arm of the
  stage has sent all its samples (the settle phase of such an arm then runs as before).
    close  the reading after the step shows the fingers fully closed (n <= CLOSED_EMPTY), or, from the second step
           on, the finger travel between the readings of two consecutive steps is below GRIP_STILL_M = 0.2 mm
           (|n_k - n_(k-1)| * 54 mm: while the official clamp acts, n follows the measured finger joint)
    open   the reading shows the fingers fully open (n >= OPEN_CONVERGED: the clamp no longer acts, each finger is at
           least 33.2 mm open, the pad gap at least 67.2 mm), or the pad gap derived from the reading exceeds the gap
           at the start of the open by GRIP_OPEN_WIDER_M = 30 mm. Pad gap = q7 + q8 + 0.807 mm (agp_core/visual_gap.py);
           q7 from n: the clamped target t = -10 mm + n * 54 mm is the finger -10.8 mm while closing (fingers stopped
           apart: q7 = t + 10.8 mm; n = 0: closed, q7 = 0) and +10.8 mm while opening (q7 = t - 10.8 mm); q8 from the
           public mimic (q8 = 1.0 * q7 + 0).
  Fallback when an arm's reading is not in the observation (or not finite) at the start of the action: the finger
  joints estimated from the arm's wrist image (OfficialBaseWorker._gripper_estimate, agp_core/visual_gap.py) with the same
  rule (close: both readings quality ok and both fingers moved less than 0.2 mm, or the estimated gap below 1.5 mm,
  visual_gap's closed-empty bound; open: gap at least the start estimate + 30 mm, or at least 67.2 mm); when that
  estimate is not quality ok either, the fixed 10 samples. A step whose reading is missing decides nothing.

H (HOLD_*)
  Which motions: move_ee (linear and plan) and move_delta with speed_profile transit, of an arm that holds something
  when the stage is prepared (commanded closed and not empty by F's two signals: n > 0, or n = 0 without a quality-ok
  wrist image showing the pads closed on nothing; retime_holding), without max_samples. Not: home and
  move_joints (outside the decision), precision moves (P below), empty arms (F). D's watch at the stage start and F-g's
  held flag keep the reading-only rule (holding_state: n > 0).
  How: the worker prepares the move as before (the dense path); retime_parts re-times it along that polyline with one
  path parameter u, the time at the caps (per segment the largest of TCP length / 0.2 m/s, rotation / 1 rad/s and
  L-infinity joint length / 2 rad/s, the caps planned at CAP_SAFETY), and a raised-cosine (s-curve) speed profile from
  rest to rest whose peak rate is the caps and whose peak acceleration is 1 m/s^2 where translation binds (5 rad/s^2
  where rotation binds, 10 rad/s^2 where a joint binds).
  Every step is then checked at 100 %: TCP speed <= 0.2 m/s, rotation <= 1 rad/s, joints <= 2 rad/s, TCP acceleration
  (second difference of the sampled TCP positions, from and to rest) <= 1 m/s^2; a violation lowers that part's rate by
  15 % and re-times (at most HOLD_ITERATIONS rounds), else the move keeps the worker's samples. The preflight checks the
  dense path (as for F); the tracking stop stays 0.08 rad.

P, the worker's half (agp_core/prec_zone.py PREC_*; the contact and place splits are made by the run)
  The lift of a held object: a precision move_delta or linear move_ee of an arm that holds something (as for H), going
  up nearly vertically (target TCP within PREC_VERTICAL_XY_M = 1 cm horizontally of the start, rotation change at most
  PREC_MAX_TURN_RAD = 0.05 rad) by at least PREC_LIFT_HEAD_M + PREC_MIN_FAST_M (1.5 cm), without max_samples: its first
  PREC_LIFT_HEAD_M = 1 cm of TCP path keeps the precision rate (3 cm/s, 0.108 rad/s, 0.3 rad/s, constant, as before),
  the rest follows H's s-curve from that speed when OURS_HOLD_PROFILE is on, otherwise the transit rate of a closed
  gripper (0.12 m/s, 0.42 rad/s, 0.9 rad/s, constant). The worker does not see the runtime intent: a precision move
  straight up while holding is a lift.

Five more switches (the tuple R2A), opt-in, all off by default. With the five off, everything above is exactly as
described (no "r2a" record is written, none of their code paths is taken, the class binds the same switch set). Each
acts only together with the switches named (R2A_NEEDS, effective_r2a; otherwise it changes nothing):

    OURS_GUARD_V2        F-g v2 (with OURS_ARM_GUARD): layer 1 at GUARD_V2_MARGIN_M = 15 mm, layer 2 unchanged
                         (GUARD_HARD_M); the delay search may hold the later arm of a two-arm stage for up to the other
                         arm's whole stage (its run samples) instead of GUARD_DELAY_MAX, still the delay that adds the
                         fewest steps, then the smallest; a conflict between the two end poses (both arms at their last
                         samples, the instant every delay ends with) is refused without the search; the refusal keeps
                         the links, control step, distance and margin and adds that only this pair conflicts
                         (guard_stage, GUARD_V2_PAIR / GUARD_V2_SINGLE). The runtime half (a step no arm started keeps
                         its measurements and arm states) is agp_core/runtime_ext.py.
    OURS_GF_OPEN         G x F (with OURS_GRIP_OBS and OURS_FREE_TOPT): an open that G ended by its wide-enough rule (the
                         pad gap 3 cm wider than at the start, before n = 1; observation or wrist-image source) leaves
                         the gripper empty for F (_sp_gf_open, _sp_empty_state) until the arm's next gripper action:
                         F re-times its transit moves and home as after a converged open (the reason names G x F).
    OURS_HOLD_CLEARANCE  held clearance (no other switch needed): a transit move_ee (linear or plan) or move_delta of an
                         arm that holds something (commanded closed and not empty by F's two signals, empty_state) whose
                         target lies more than CLEARANCE_LATERAL_M = 1 cm away horizontally keeps the held object's
                         lowest point at least CLEARANCE_M = 3 cm above the table along its lateral part. Lowest point =
                         TCP height above the table - the TCP height above the table at the close that started the grasp
                         (the worker's own record of the arm's last executed close, _sp_close_tcp; counted from the
                         table, conservative), so the TCP stays at least close height + 3 cm high. A start below that is
                         first lifted vertically, an end below it is reached by moving across above it and then
                         descending vertically, both when both are low; a plan path that dips below it (or starts or
                         ends below it) takes this path instead: lift, move across (planned when the move was a plan and
                         that plan stays high enough, else straight), descend. Each part is prepared by the worker's own
                         linear or plan preparation; H (when on and holding) re-times each part from rest to rest. A
                         part that cannot be prepared refuses the move with the reason (CLEARANCE_REFUSAL). A height
                         within CLEARANCE_TOL_M below the required one counts as reaching it (engineering value).
                         Precision and vertical moves, homes and move_joints are not changed. The result's arm gets
                         `note` (one sentence, CLEARANCE_NOTE) when a lift or a final descent was added, and
                         ours_hold_clearance.
    OURS_DROP_STOP       drop stop (with OURS_HOLD_GUARD): during a lift or transport D watches, when the official
                         reading of a watched arm falls to 0 (n <= CLOSED_EMPTY after a reading above it, from the stage
                         start on) and D's own loss stop does not apply at that step, the motion stops at that control
                         step with DROP_STOP_REASON; D's both-signal stop wins when it applies at the same step.
    OURS_HOLD_WATCH2     watch2 (with OURS_HOLD_GUARD): D watches a commanded-closed arm at the start of a lift or
                         transport unless the reading shows the fingers fully closed AND the wrist-image estimate
                         (quality ok) shows the pads < 1.5 mm apart (F's empty rule); without the switch D watches only
                         when n > 0. D's stop rule is unchanged.
  Records (only with the switch on): motion_diagnostics ours_speed.hold_clearance (every transit move of the arm, applied
  or why not), drop_stops, hold_guard.start_rule / start_check (watch2), free_topt.empty naming G x F, the arm guard's
  delay as before; the result's ours_guard_delay (v2, a delayed arm), ours_hold_clearance and note (held clearance);
  speed_profiles.json "r2a" with the switches and constants; arm_guard.jsonl as before (threshold, delays, refusals).

One more switch (the tuple R3I), opt-in, off by default. With it off (or without H), everything above is exactly as
described (no "r3i" record is written, none of its code paths is taken, the class binds the same switch set as without
it):

    OURS_HOLD_VERIFIED_FAST  (with OURS_HOLD_PROFILE): the verified fast hold. Invisible to the model: no prompt text,
                         no result text.
  The grasp of an arm is "verified" by the first lift after its close: the first stage after the arm's executed close in
  which the arm (commanded closed) makes a lift by D's phase rule (hold_guard_phase: a precision move whose last
  scheduled sample is >= HOLD_LIFT_RISE_M higher than its start). Its readings are taken after every control step at the
  points where D records hold_trace; with t the finite readings, the pad gap from the official reading
  (VERIFY_GAP_PER_READING_M x n + VERIFY_GAP_OFFSET_M, 108 mm x n + 2.4 mm, fitted on simulator replays) of the first
  minus that of the last may be at most VERIFY_SHRINK_M = 0.5 mm, and the last reading must be > 0 (n > CLOSED_EMPTY). A
  lift that executed no step leaves the arm waiting for its first lift. Every later held transit move that H re-times
  (retime_holding: transit move_ee / move_delta of a holding arm without max_samples, the held clearance path's parts
  included) of a verified arm then takes hold_parts_fast: H's s-curve with the TCP cap FAST_HOLD_TCP_SPEED = 0.3 m/s and
  the peak acceleration FAST_HOLD_TCP_ACCEL = 2 m/s^2, H's rotation (1 rad/s) and joint (2 rad/s) caps and every other
  rule of H unchanged (the values tested in simulator replays). Every other grasp keeps H (0.2 m/s, 1 m/s^2); no
  object-type check (bowls slip in their lift and stay at H by the signal). The flag is cleared on the arm's next
  executed open, on its next executed close (which starts a new grasp waiting for its lift), and on any later reading of
  the arm (after every control step of every stage, and when a move of the arm is prepared) whose pad gap is more than
  VERIFY_SHRINK_M below the gap at the end of the verifying lift. A place split's fast part (agp_core/prec_zone.py, a held
  transit move) of a verified arm thereby runs at the fast law; the run reads the arm's state for its step estimate from
  the session file hold_verified.json (written after every stage that changed a state). Records: motion_diagnostics
  ours_speed.hold_verified (the arm's state at the stage start, the lift's readings and verdict, a clearing, and for a
  held transit move whether the fast law applied, with the samples of both laws); speed_profiles.json "r3i".

One more switch (SPIN), opt-in, off by default. With it off, everything above is exactly as described (no twist record
is written, no twist code path is taken, the class binds the same switch set as without it):

    OURS_SPIN_FAST   the in-place twist: a precision move that only turns the gripper about its own axis
  Which motions (spin_check, on the worker's prepared "dense" path, start included, public URDF FK in the arm's base
  frame): move_ee (linear or plan, a new rotation) and move_delta (drot_deg) at speed_profile precision -- any intent,
  holding or not (the worker does not see the intent; precision is what grasp, contact, place, manipulate and a slow
  move run at) -- without max_samples (a checkpoint keeps the sample rate it was asked for), not already re-timed (P's
  lift), whose whole path
    keeps the TCP within SPIN_MAX_SHIFT_M = 5 mm of its start,
    keeps the tool axis (the TCP's z axis, toward the fingertips) within SPIN_MAX_AXIS_DEG = 2 deg of its start direction
      (a twist about the gripper's own axis; a tilt does not qualify), and
    turns the gripper: the rotation from the start to the end is at least SPIN_MIN_TURN_RAD (one precision sample,
      0.00432 rad; engineering value: a smaller turn is one sample at the precision rate already).
  Not: transit moves (F and H time them), home and move_joints (outside the decision).
  How: the worker prepares the move exactly as before (the dense path at the precision rate); retime_parts re-times it
  along that polyline with spin_parts: one raised-cosine (s-curve) part from rest to rest, caps TCP rotation
  SPIN_TCP_ROTATION = 1 rad/s and joints SPIN_JOINT_SPEED = 2 rad/s (H's values), translation SPIN_TCP_SPEED = 0.03 m/s
  (the precision cap, unchanged), the caps planned at CAP_SAFETY; the peak rate acceleration is H's where rotation binds
  (SPIN_ROT_ACCEL = 5 rad/s^2; 0.15 m/s^2 with the 0.03 m/s cap where translation binds, 10 rad/s^2 where a joint
  binds). Every step is then checked at 100 % (retime_parts: rotation <= 1 rad/s, joints <= 2 rad/s, TCP <= 0.03 m/s,
  TCP acceleration <= SPIN_TCP_ACCEL = 1 m/s^2, H's bound: the real planner's IK leaves up to about 0.5 mm between
  neighbouring points of a twist's dense path, which a bound at 0.15 m/s^2 would read as acceleration and slow the twist
  for); far inside the official 5 rad/s per joint (env/robot_manager/robot_config/x5.py velocity_limit_sim). No zone of
  F applies: the landing zone's rotation cap (1 rad/s) is a cap of F's transit law near the table, and a twist has no
  descent. The re-timing is used only when it needs fewer samples than the worker's own precision samples (engineering
  value: a short turn combined with a few mm of translation can take longer from rest to rest than at the constant
  precision rate); otherwise, or when no admissible timing is found, the move keeps the worker's samples and the record
  says why. The preflight checks the dense path (as for F and H); the tracking stop stays the worker's 0.08 rad; F-g, D
  and every other guard act on the re-timed samples as on any others.
  Records (only with the switch on): motion_diagnostics ours_speed.spin_fast (the check and the re-timing record) or
  spin_fast_skipped (why not) for every precision move_ee / move_delta; the npz gets `ours_dense`; speed_profiles.json
  "spin". Nothing in a result changes (the prompt's speeds line says it, agp_core/PROMPT.md under the switch).

One more switch (YAW), opt-in, off by default. With it off, everything above is exactly as described (no yaw-flip
record is written, no yaw-flip code path is taken, the class binds the same switch set as without it). Invisible to the
model: no prompt text, no result text.

    OURS_YAW_FLIP    the symmetric-gripper flip: a two-finger gripper turned 180 deg about its own axis (the TCP z axis,
                     toward the fingertips; for a top-down target yaw psi and psi + 180 deg) gives the same grasp
  Each arm has a frame: "requested" (the TCP is where the agent's orientations put it) or "flipped" (the TCP is the
  agent's orientation turned by YAW_FLIP = Rz(180 deg) in the TCP's own frame, R_actual = R_agent Rz(pi)). Both arms
  start "requested".
  Which motions: every move_ee (linear and plan, any intent; its rotation is explicit: the runtime requires one, and the
  run fills the current orientation in when the agent gives none).
    Empty gripper (F's rule, _sp_empty_state: commanded open and converged, or commanded closed with both signals of
    an empty close, or G x F's wide open when OURS_GF_OPEN acts): the worker prepares the move twice, exactly as without
    the switch (IK, plan or straight path, resampling, F / spin re-timing), once with the requested rotation R and once
    with R Rz(pi), and compares the two goals (the last commanded sample) with the current joints by the L-infinity
    joint change, the measure migration.kinematics.nearest_solution uses for the IK's own branch choice, on the actual
    joint values within the simulator's limits (joint6 is limited to +-179.9 deg and cannot wrap, so a turn past its
    limit shows as the long way round or another branch). The flipped goal is executed when it is closer by more than
    YAW_FLIP_MIN_GAIN_RAD = 0.05 rad (engineering value: near ties keep the request); when only one of the two can be
    prepared (plan: IK or planning rejected; linear: IK rejected at a sample, an IK branch discontinuity, the Cartesian
    resampling rejected), that one; when neither, the failure of the candidate in the arm's current frame (for a
    "requested" arm the same refusal as without the switch). The flipped candidate is not prepared when the requested
    one is prepared and changes no joint by more than YAW_FLIP_SKIP_RAD (about 0.278 rad): two goals whose TCP
    orientations differ by about 180 deg are at least (pi - 0.1) / 6 = 0.507 rad apart in L-infinity (each joint turns
    the TCP by at most its own angle), so the flipped goal cannot be closer by the margin (the choice is the same, one
    preparation is saved). The executed candidate sets the arm's frame.
    Holding (not empty by that rule): no choice; the arm keeps its frame (R executed for a "requested" arm, R Rz(pi) for a
    "flipped" one), so the held object turns exactly as the agent asked: an arm that flipped while empty and then closed
    carries on in that frame, and nothing a holding arm does changes it.
  home and move_joints set the frame back to "requested" (the agent gives joints, or the start pose); move_delta,
  gripper and hold keep it (move_delta turns the measured TCP, so it is consistent in either frame). A frame is taken
  over only when the stage executed at least one control step.
  What the model sees is unchanged: every pose the worker reports is presented in the arm's frame, the achieved pose of
  a result (achieved_tcp_pose_base; _errors) and, through the RGB-only observer's camera model (AgentFrameModel), the
  observation's tcp_pose_base and every camera's T_base_tcp_by_arm (calibration.json; so a measurement's point_tcp is in
  the agent's frame too); the rotation error and target_reached are checked against the executed target, which is the
  requested one turned about the tool axis, so a flipped move reports reached with the same small error. The joints,
  the robot's link poses (robot_state.json, the silhouettes), the camera poses and the images stay physical. No result
  field and no prompt text is added. Without the RGB-only observer (no official conditions) nothing is flipped.
  Records (only with the switch on): motion_diagnostics ours_speed.yaw_flip for every move_ee (the frame before and
  after, the empty decision, each candidate's rotation, preparation outcome, goal joints, joint change, L-infinity and
  L1 distance, joint6 margin, samples and the choice with its reason) and ours_speed.yaw_frame for every other motion
  (the arm's frame for its reported pose); speed_profiles.json "yaw_flip" (the constants).

Grip width (OURS_GRIP_WIDTH, agp_core/grip_width.py):
the width action commands a partial gripper value 0 < c < 1 (open is 1.0, close 0.0; with the switch off no partial
value is ever commanded and none of the following is reached). The rules above read it as follows (partial_command,
at_target, lost_bound_m): F's two signals become the reading equal to c (the clamp does not act: the fingers within
10.8 mm of the target; for a close n = 0) and the wrist-image pad gap (quality ok) below the target gap + 1.0 mm
(grip_width.MARGIN_M, the width outcome's own margin; for a close 1.5 mm): a partly open gripper at its gap with nothing
between is empty (F, and every rule that asks F's question), otherwise holding (H, P's lift, the held clearance); a
reading above c (held more than 10.8 mm wider) is holding, one below c has not converged (width_empty_state). D: the
reading alone cannot show an object near a partial target, so the watch starts by the two signals (as OURS_HOLD_WATCH2
does for a close) and stops on them at the commanded width (hold_loss with the command; the drop stop: the reading
falling to c); holding_state (F-g's held flag) is n > c. G: a width action runs its 10 samples (agp_core/grip_width.py
GripWidthWorkerMixin._sp_grip_rule); G x F only follows an open. A verified fast hold is cleared by a width; the held
clearance records the TCP height of a width as of a close.

Precision scale: one more switch (P2X), opt-in, off by default. With it off, everything above is exactly as described
(the worker's preparation is called unchanged, no P2X record is written, no P2X code path is taken, the class binds the
same switch set as without it). It acts by itself (the mixin is composed for it alone).

    OURS_PRECISION_2X   every precision-segment limit x PRECISION_2X_SCALE = 2.5 (the scale, whatever the name says)
  Which limits (the official speed limits, 5 rad/s per joint and the per-step interpolation, are not ours and unchanged):
    the worker's precision preparation   move_ee (linear and plan), move_delta and move_joints at speed_profile precision
                  (any intent: grasp, contact, place, manipulate, a slow move, a place split's precision tail; with or
                  without max_samples): migration/worker.py _prepare_from_state samples them at rate x dt x 0.6 per
                  sample; the mixin calls it with dt x 2.5 for that preparation only (the way agp_core/worker_ext.py's
                  OURS_FAST_PRECISION does, which official conditions keep off), so every limit of it scales together:
                  TCP 0.03 -> 0.075 m/s and 0.108 -> 0.27 rad/s (the linear move's Cartesian samples and resample_tcp),
                  joints 0.3 -> 0.75 rad/s (resample_joints of plan, linear IK points and move_joints). The plan records
                  joint_velocity_limit_rad_s 1.25 (0.5 x 2.5, the rate x 0.6 is the sampled joint speed) and
                  ours_precision_scale 2.5. Transit, home, hold and the gripper keep their preparation.
    P's lift head (OURS_PREC_ZONE)       lift_parts: the first 1 cm at PRECISION_CAPS x 2.5.
    the in-place twist (OURS_SPIN_FAST)  its translation cap SPIN_TCP_SPEED, the precision cap, x 2.5 (0.075 m/s); the
                  rotation (1 rad/s) and joint (2 rad/s) caps are H's values and stay; the rate acceleration is given
                  where rotation binds (5 rad/s^2), so where translation binds it is 0.375 m/s^2 (0.15 at scale 1); the
                  check bound SPIN_TCP_ACCEL (1 m/s^2) stays. SPIN_MIN_TURN_RAD (one precision sample's turn) x 2.5; a
                  twist is re-timed only when it needs fewer samples than the (scaled) precision samples.
    F's zones (OURS_FREE_TOPT)           the landing zone (the last 3 cm of a move ending below 10 cm) 0.12 -> 0.30 m/s
                  and 1 -> 2.5 rad/s, the take-off zone (the first 2 cm of a move starting below 6 cm, the same constant
                  ZONE_TCP_SPEED) 0.12 -> 0.30 m/s, for F's moves and approximate homes (FreeLaw). The low-altitude cap
                  (0.25 m/s below 10 cm) is not a precision limit and stays.
  Not scaled: transit and H's caps, accelerations of F and H, the tracking-error stops (0.08 / 0.05 rad), settle rules,
  the gripper. The precision moves keep their constant-rate law (no acceleration limit is involved).
  Records (only with the switch on): speed_profiles.json "precision_2x" (the scale and the scaled limits);
  motion_diagnostics ours_speed.precision_scale (the scale for a precision move prepared at the scaled limits, else
  None); F's info.limits show the scaled zones. agp_core/prec_zone.py's place split estimates the precision steps at the
  scaled rate (agp_core/runner.py passes the switch). A session records its own scale in speed_profiles.json.

Plan guard: one more switch (PG), opt-in, off by default. With it off, everything above is exactly as described
(nothing is checked, no plan guard record is written, the class binds the same switch set as without it). It acts by
itself (the mixin is composed for it alone).

    OURS_PLAN_GUARD   a plan that would swing the arm around (an IK goal on another arm branch) is not executed
  Which motions: every move_ee and move_delta in plan mode (args mode "plan"): holding or empty, any intent and speed
  profile, with or without max_samples; under OURS_YAW_FLIP every candidate the flip prepares (an empty arm's requested
  and flipped candidates, a holding arm's one in its frame), so a refused candidate counts as not prepared and the other
  one is chosen when the flip may choose (yaw_flip_choice); the held clearance's (OURS_HOLD_CLEARANCE) plan across
  between the raised ends too (refused: the straight path across, as when that plan dips). Not checked: linear moves
  (the worker refuses their branch jumps itself, "Linear IK branch discontinuity"), home (approximate or not) and
  move_joints.
  Rule (plan_guard_check): right after the worker's own preparation (before the held clearance, F, H, P and the in-place
  twist re-time it), the TCP path along the prepared joint samples (the start and every commanded sample, through the
  public URDF FK of arm_guard.ArmKinematics, the FK of F's re-timing) is compared with the straight distance between its
  first and last TCP position; the plan is refused when the path is longer than PLAN_GUARD_RATIO = 3 times the straight
  distance AND longer than PLAN_GUARD_MIN_PATH_M = 0.6 m (in 3860 move_ee / move_delta of 113 recorded episodes this
  criterion selects the 5 plans whose IK goal lay on another arm branch and nothing else; the 4 other plans longer than
  0.6 m were about 1.1 times their straight distance).
  Refusal: the worker's own refusal of a plan ({"ok": False, "error": "Planning rejected", "motion_started": False,
  "motion_executed": False}) plus `reason` (plan_guard_reason: the guard, the path, the straight distance, the ratio, the
  bounds and what to do instead) and `ours_plan_guard` (the check); nothing moves. The compact view shows the reason
  with the error (agp_core/compact.py); the prompt says it in one sentence (agp_core/PROMPT.md, with OURS_P_MOTION).
  Records (only with the switch on): speed_profiles.json "plan_guard" (the bounds); PLAN_GUARD_LOG_FILE in the session
  (one line per checked preparation, refused or not); motion_diagnostics ours_speed.plan_guard for every move_ee and
  move_delta (the executed plan's check, None for a linear move); a yaw-flip candidate refused by the guard carries
  plan_guard in its yaw_flip record.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from agp_core import features

PART_A = ("free_topt", "arm_guard", "hold_guard")
PART_B = ("grip_obs", "prec_zone", "hold_profile")
FLAGS = PART_A + PART_B
MOTIONS = {"move_ee", "move_delta", "move_joints", "home"}

# ---- F: limits
FREE_JOINT_SPEED = 4.0          # rad/s, every joint
FREE_JOINT_ACCEL = 20.0         # rad/s^2, every joint
FREE_TCP_SPEED = 1.0            # m/s
FREE_TCP_ROTATION = 3.0         # rad/s
# ---- F: zones near the table
LOW_HEIGHT_M = 0.10             # TCP below this height above the table top ...
LOW_TCP_SPEED = 0.25            # ... moves at most this fast (m/s)
LANDING_HEIGHT_M = 0.10         # a move ending below this height ...
LANDING_LENGTH_M = 0.03         # ... covers its last 3 cm of TCP path ...
TAKEOFF_HEIGHT_M = 0.06         # a move starting below this height ...
TAKEOFF_LENGTH_M = 0.02         # ... covers its first 2 cm of TCP path ...
ZONE_TCP_SPEED = 0.12           # ... at most this fast (m/s)
ZONE_ROTATION = 1.0             # landing zone rotation (rad/s)
# ---- F: engineering values
CAP_SAFETY = 0.98               # TCP caps are planned at 98 % (the TCP between two dense points is not exactly straight)
CORNER_SHARE = 0.5              # first guess at a path corner: half of a step's acceleration budget for the turn
RETIME_ITERATIONS = 80          # slow-down rounds for samples that break the per-joint acceleration at corners
RETIME_SLOWDOWN = 0.85
FREE_TRACKING_STOP_RAD = 0.05   # tracking-error stop of re-timed motions (worker: 0.08)
OPEN_CONVERGED = 1.0 - 1e-6     # n at or above: a commanded open has converged
CLOSED_EMPTY = 1e-6             # n at or below: the fingers are fully closed
# ---- F-g
GUARD_MARGIN_M = 0.025          # layer 1
GUARD_HARD_M = 0.010            # layer 2
GUARD_START_TOLERANCE_M = 0.002
GUARD_DELAY_MAX = 10            # control steps (0.4 s)
GUARD_EXACT_ABOVE_M = 0.05      # distances are computed exactly up to this much above the threshold (reports)
INTERP_SUBSTEPS = 9             # eval_env.process_control_info: 8 interpolated substeps at i/9, then the target
# ---- D (both signals; lift and transport only)
HOLD_LOST_GAP_M = 0.0015        # wrist-image pad gap (quality ok) below which fully closed fingers hold nothing
                                # (1.5 mm = agp_core/visual_gap.py CLOSED_EMPTY_GAP_M)
HOLD_LIFT_RISE_M = 0.001        # engineering value: a precision move of a holding arm is a lift when its last
                                # scheduled sample puts the TCP at least this much higher than at the start
# ---- G: the end rule
GRIP_STILL_M = 0.0002           # a close ends when two consecutive readings differ by less than this finger travel
GRIP_OPEN_WIDER_M = 0.030       # an open ends when the pad gap exceeds the gap at its start by this
# ---- G: public constants and engineering values
GRIP_EPS = 0.2                  # env/robot_manager/control_manager.py process_gripper_val gripper_eps (fraction of the
                                # gripper scale): the official clamp of the finger target, 0.2 * 54 mm = 10.8 mm
GRIP_PAD_GAP0_M = 0.000807      # pad separation at finger joints 0 (agp_core/visual_gap.py, the public finger meshes)
GRIP_MIN_STILL_STEPS = 2        # the still rule compares the readings of two steps of the action
GRIP_CLOSED_GAP_M = 0.0015      # fallback (wrist image): agp_core/visual_gap.py CLOSED_EMPTY_GAP_M, nothing between the pads
# ---- H: limits
HOLD_TCP_SPEED = 0.2            # m/s
HOLD_TCP_ACCEL = 1.0            # m/s^2, peak of the raised-cosine ramps
HOLD_ROTATION = 1.0             # rad/s
HOLD_JOINT_SPEED = 2.0          # rad/s
# ---- H: engineering values
HOLD_ITERATIONS = 20            # slow-down rounds (15 % each, RETIME_SLOWDOWN) for a step that breaks a limit
HOLD_ACCEL_TOLERANCE = 1e-6     # relative tolerance of the acceleration check (the s-curve meets it by construction on
                                # a straight path; curvature of a plan path can add to it, then the rate is lowered)
# ---- the worker's own rates (migration/worker.py: limits x 0.6), used for P's lift
PRECISION_CAPS = {"tcp": 0.05 * 0.6, "rot": 0.18 * 0.6, "joint": 0.5 * 0.6}       # 0.03 m/s, 0.108 rad/s, 0.3 rad/s
TRANSIT_CLOSED_CAPS = {"tcp": 0.20 * 0.6, "rot": 0.7 * 0.6, "joint": 1.5 * 0.6}   # 0.12 m/s, 0.42 rad/s, 0.9 rad/s
HOLD_CAPS = {"tcp": HOLD_TCP_SPEED, "rot": HOLD_ROTATION, "joint": HOLD_JOINT_SPEED}
# ---- the five switches of R2A (see the module docstring)
R2A = ("guard_v2", "gf_open", "hold_clearance", "drop_stop", "hold_watch2")
R2A_NEEDS = {"guard_v2": ("arm_guard",), "gf_open": ("grip_obs", "free_topt"), "hold_clearance": (),
             "drop_stop": ("hold_guard",), "hold_watch2": ("hold_guard",)}
GUARD_V2_MARGIN_M = 0.015       # F-g v2 layer 1 (15 mm instead of 25 mm); layer 2 stays GUARD_HARD_M = 10 mm
CLEARANCE_M = 0.03              # held clearance: the held object's lowest point at least this high above the table
CLEARANCE_LATERAL_M = 0.01      # held clearance: moves whose target lies more than this away horizontally
CLEARANCE_TOL_M = 0.001         # engineering value: within this below the required height counts as reaching it
GUARD_V2_PAIR = ("Only this pair of motions conflicts: run these two one after the other or move one target away, "
                 "and keep the rest of the plan with both arms moving together.")
GUARD_V2_SINGLE = ("Only this motion conflicts, with the other arm where it stands: move the other arm away first or "
                   "move this target away, and keep the rest of the plan with both arms moving together.")
DROP_STOP_REASON = ("{arm} gripper reading fell to 0 while holding: the object may have slipped or been pushed; check "
                    "the wrist image before moving on")
CLEARANCE_REFUSAL = ("Held clearance: the held object must stay {clear:.0f} cm above the table while moving across, so "
                     "the TCP must stay at least {need:.1f} cm above it ({close:.1f} cm at the close plus "
                     "{clear:.0f} cm); the path that lifts first, moves across and lowers at the end is not reachable "
                     "here ({part}: {error})")


def effective_r2a(flags):
    """{R2A name: on} with each switch on only when it is set and every switch it acts with is on (R2A_NEEDS)."""
    return {name: bool(flags.get(name)) and all(bool(flags.get(need)) for need in R2A_NEEDS[name]) for name in R2A}


# ---- the verified fast hold, R3I (see the module docstring)
R3I = ("hold_verified_fast",)
R3I_NEEDS = {"hold_verified_fast": ("hold_profile",)}
VERIFY_SHRINK_M = 0.0005        # the pad gap may shrink by at most 0.5 mm in the first lift (in simulator replays,
                                # 1 mm would have sped up a "2" that dropped, 0.5 mm not)
VERIFY_GAP_PER_READING_M = 0.108   # pad gap = 108 mm x n + 2.4 mm while the clamp acts (fitted on simulator replays,
VERIFY_GAP_OFFSET_M = 0.0024       # |error| <= 0.4 mm while an object blocks the fingers)
FAST_HOLD_TCP_SPEED = 0.3       # m/s, tested in simulator replays
FAST_HOLD_TCP_ACCEL = 2.0       # m/s^2, the same replays; H's other limits unchanged
HOLD_VERIFIED_FILE = "hold_verified.json"   # the arms' states for the run's split estimate (session file)
# ---- SPIN: the in-place twist (see the module docstring)
SPIN = ("spin_fast",)
SPIN_MAX_SHIFT_M = 0.005        # the TCP within 5 mm of its start along the whole path
SPIN_MAX_AXIS_DEG = 2.0         # the tool axis within about 2 deg of its start direction along the whole path
SPIN_MIN_TURN_RAD = 0.18 * 0.04 * 0.6   # one precision sample's turn, 0.00432 rad (migration/worker.py sampling)
SPIN_TCP_ROTATION = HOLD_ROTATION       # 1 rad/s (H's value)
SPIN_JOINT_SPEED = HOLD_JOINT_SPEED     # 2 rad/s (H's value)
SPIN_TCP_SPEED = PRECISION_CAPS["tcp"]  # 0.03 m/s (the translation cap unchanged)
SPIN_ROT_ACCEL = HOLD_TCP_ACCEL / HOLD_TCP_SPEED * HOLD_ROTATION   # 5 rad/s^2: H's peak where rotation binds (the
                                # profile's peak rate acceleration; 0.15 m/s^2 where translation binds, 10 rad/s^2 where a
                                # joint binds)
SPIN_TCP_ACCEL = HOLD_TCP_ACCEL         # 1 m/s^2: the TCP acceleration bound every step is checked against (H's); the
                                # IK of the real planner leaves up to about 0.5 mm between neighbouring points of a
                                # twist's dense path, too much for a check at 0.15 m/s^2
# ---- P2X: every precision-segment limit scaled (see the module docstring)
P2X = ("precision_2x",)
PRECISION_2X_SCALE = 2.5        # the factor of OURS_PRECISION_2X (see the module docstring)
PRECISION_MOTIONS = ("move_ee", "move_delta", "move_joints")   # the worker's precision preparations


def precision_scale(flags):
    """The factor of every precision-segment limit: PRECISION_2X_SCALE with OURS_PRECISION_2X bound, else 1.0."""
    return PRECISION_2X_SCALE if (flags or {}).get("precision_2x") else 1.0


def precision_caps(scale=1.0):
    """The worker's precision rates (PRECISION_CAPS) times scale; PRECISION_CAPS itself (a copy) at 1."""
    return dict(PRECISION_CAPS) if scale == 1.0 else {k: v * scale for k, v in PRECISION_CAPS.items()}


def zone_limits(scale=1.0):
    """retime_free's `limits` for F's landing and take-off zones at scale: None at 1 (retime_free's own constants)."""
    if scale == 1.0:
        return None
    return {"zone_v": ZONE_TCP_SPEED * scale, "zone_w": ZONE_ROTATION * scale}


def spin_parts(precision_scale=1.0):
    """The in-place twist: one s-curve part over the whole path, rotation 1 rad/s, joints 2 rad/s, TCP 0.03 m/s (the
    precision cap; x precision_scale with OURS_PRECISION_2X), the peak rate acceleration of H where rotation binds
    (rot_accel 5 rad/s^2; 0.15 m/s^2 where translation binds at 0.03 m/s); every step's TCP acceleration checked against
    H's 1 m/s^2 (accel)."""
    return [{"name": "in-place twist s-curve (plan A)", "tcp_m": None,
             "caps": {"tcp": SPIN_TCP_SPEED * precision_scale if precision_scale != 1.0 else SPIN_TCP_SPEED,
                      "rot": SPIN_TCP_ROTATION, "joint": SPIN_JOINT_SPEED},
             "law": "scurve", "accel": SPIN_TCP_ACCEL, "rot_accel": SPIN_ROT_ACCEL}]


def spin_check(start, dense, tcp_of, min_turn=SPIN_MIN_TURN_RAD):
    """(qualifies, record) of the in-place twist for the joint path start -> dense (module docstring): the largest TCP
    distance from the start, the largest angle between the tool axis (TCP z) and its start direction, the turn from the
    start to the end (and the largest along the path), each against its bound; the reason names the first bound missed.
    min_turn: one precision sample's turn (SPIN_MIN_TURN_RAD x the precision scale)."""
    Q = np.vstack([np.asarray(start, float)[None], np.asarray(dense, float).reshape(-1, len(start))])
    frames = np.asarray(tcp_of(Q), float)
    pos, axis = frames[:, :3, 3], frames[:, :3, 2]
    shift = float(np.max(np.linalg.norm(pos - pos[0], axis=1)))
    unit = axis / np.linalg.norm(axis, axis=1)[:, None]
    tilt = float(np.degrees(np.max(np.arccos(np.clip(unit @ unit[0], -1.0, 1.0)))))
    rel = np.einsum("ji,njk->nik", frames[0, :3, :3], frames[:, :3, :3])          # R_0^T R_k
    turns = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    turn = float(turns[-1])
    record = {"tcp_shift_m": round(shift, 6), "axis_change_deg": round(tilt, 4), "turn_rad": round(turn, 6),
              "path_turn_rad": round(float(turns.max()), 6), "end_shift_m": round(float(np.linalg.norm(pos[-1] - pos[0])), 6),
              "bounds": {"tcp_shift_m": SPIN_MAX_SHIFT_M, "axis_change_deg": SPIN_MAX_AXIS_DEG,
                         "min_turn_rad": min_turn}}
    if shift > SPIN_MAX_SHIFT_M:
        why = (f"not in place: the TCP moves {shift * 1000:.1f} mm from its start (at most "
               f"{SPIN_MAX_SHIFT_M * 1000:.0f} mm)")
    elif tilt > SPIN_MAX_AXIS_DEG:
        why = (f"not a twist about the gripper's own axis: that axis tilts by {tilt:.2f} deg (at most "
               f"{SPIN_MAX_AXIS_DEG:g} deg)")
    elif turn < min_turn:
        why = f"does not turn the gripper: {turn:.5f} rad (at least {min_turn:.5f} rad)"
    else:
        why = None
    record.update(qualifies=why is None, reason=why or "in-place twist")
    return why is None, record


PR = ("prec_roll",)
PREC_ROLL_SPEED = 1.0
PREC_JOINT6_SPEED = 1.0
PREC_ROLL_TERMS = ("tcp", "swing", "roll", "joint", "joint6")
PREC_ROLL_TCP_TERMS = ("tcp", "swing", "roll")


def roll_axis(kin):
    joint = next(entry for entry in kin.order if entry[2] == kin.model.ee_link)
    T = np.asarray(kin.T_link6_tcp, float)
    along = np.asarray(joint[4], float) / np.linalg.norm(joint[4])
    axis = T[:3, :3].T @ along
    return {"joint": joint[0], "index": int(kin.arm_index[joint[0]]), "axis": axis / np.linalg.norm(axis),
            "tcp_offset_m": float(np.linalg.norm(np.cross(T[:3, 3], along)))}


def prec_roll_caps(scale=1.0, joints_only=False):
    caps = precision_caps(scale)
    joints = {"joint": caps["joint"], "joint6": PREC_JOINT6_SPEED}
    if joints_only:
        return joints
    return {"tcp": caps["tcp"], "swing": caps["rot"], "roll": PREC_ROLL_SPEED, **joints}


def roll_metrics(Q, tcp_of, axis, index, tcp=True):
    Q = np.asarray(Q, float)
    dq = np.abs(np.diff(Q, axis=0))
    out = {"joint": np.delete(dq, index, axis=1).max(axis=1), "joint6": dq[:, index]}
    if not tcp:
        return out
    from scipy.spatial.transform import Rotation
    axis = np.asarray(axis, float)
    frames = np.asarray(tcp_of(Q), float)
    rel = np.einsum("nij,nik->njk", frames[:-1, :3, :3], frames[1:, :3, :3])
    quat = Rotation.from_matrix(rel).as_quat()
    vector, w = quat[:, :3], np.abs(quat[:, 3])
    along = vector @ axis
    across = np.linalg.norm(vector - along[:, None] * axis[None, :], axis=1)
    out.update(tcp=np.linalg.norm(np.diff(frames[:, :3, 3], axis=0), axis=1),
               swing=2.0 * np.arctan2(across, np.sqrt(w * w + along * along)),
               roll=2.0 * np.arctan2(np.abs(along), w))
    return out


def retime_roll(start, dense, tcp_of, axis, index, caps, dt=0.04):
    start = np.asarray(start, float)
    raw = np.vstack([start[None], np.asarray(dense, float).reshape(-1, len(start))])
    keep = [0]
    for i in range(1, len(raw)):
        if np.max(np.abs(raw[i] - raw[keep[-1]])) > 1e-12:
            keep.append(i)
    P = raw[keep]
    end = raw[-1]
    names = [k for k in PREC_ROLL_TERMS if k in caps]
    tcp = any(k in caps for k in PREC_ROLL_TCP_TERMS)
    safety = {k: CAP_SAFETY if k in PREC_ROLL_TCP_TERMS else 1.0 for k in names}
    info = {"law": "prec_roll", "dense_samples": int(len(raw) - 1), "dt_s": float(dt),
            "caps": {k: float(caps[k]) for k in names}, "cap_safety": CAP_SAFETY}
    if len(P) < 2:
        info.update(samples=1, note="no motion")
        return [end.copy()], info
    seg = roll_metrics(P, tcp_of, axis, index, tcp)
    info.update(segments=int(len(P) - 1), path={k: round(float(seg[k].sum()), 6) for k in names})
    scale = 1.0
    for iteration in range(HOLD_ITERATIONS + 1):
        times = np.vstack([seg[k] / (caps[k] * safety[k] * scale) for k in names])
        u_seg = times.max(axis=0)
        u, law = const_positions(float(u_seg.sum()), dt)
        u_cum = np.r_[0.0, np.cumsum(u_seg)]
        samples = []
        for value in u:
            i = int(min(len(u_seg) - 1, max(0, np.searchsorted(u_cum, value, side="right") - 1)))
            f = 0.0 if u_seg[i] <= 0 else min(1.0, max(0.0, (value - u_cum[i]) / u_seg[i]))
            samples.append(P[i] + f * (P[i + 1] - P[i]))
        samples[-1] = end.copy()
        rates = roll_metrics(np.vstack([start[None], np.asarray(samples)]), tcp_of, axis, index, tcp)
        rates = {k: rates[k] / dt for k in names}
        bad = [k for k in names if np.any(rates[k] > caps[k] * (1 + (1e-6 if k in PREC_ROLL_TCP_TERMS else 1e-9)))]
        if not bad:
            bound = times.argmax(axis=0)
            info.update(samples=len(samples), seconds=law["seconds"], iterations=iteration, rate_scale=round(scale, 6),
                        binding={k: int(np.sum(bound == j)) for j, k in enumerate(names)},
                        max_rates={k: float(rates[k].max()) for k in names})
            return [np.asarray(q, float).copy() for q in samples], info
        if iteration == HOLD_ITERATIONS:
            break
        scale *= RETIME_SLOWDOWN
    info.update(samples=None, failed="no admissible timing within the slow-down rounds", iterations=HOLD_ITERATIONS,
                rate_scale=round(scale, 6))
    return None, info


# ---- PG: a plan that would swing the arm around is refused (see the module docstring)
PG = ("plan_guard",)
PLAN_GUARD_RATIO = 3.0          # refused when the TCP path is longer than this times the straight distance ...
PLAN_GUARD_MIN_PATH_M = 0.6     # ... and longer than this
PLAN_GUARD_LOG_FILE = "plan_guard.jsonl"   # one line per checked plan preparation of the worker (refused or not)


def plan_guard_check(start, points, tcp_of, ratio=PLAN_GUARD_RATIO, min_path=PLAN_GUARD_MIN_PATH_M):
    """The plan guard's check of the joint path start -> points (the worker's prepared samples): the TCP path length
    along the samples, the straight distance between its first and last TCP position, their ratio (None for a path that
    ends where it starts) and whether the plan is refused (path > ratio x straight and path > min_path)."""
    start = np.asarray(start, float)
    Q = np.vstack([start[None], np.asarray(points, float).reshape(-1, len(start))])
    pos = np.asarray(tcp_of(Q), float)[:, :3, 3]
    path = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(pos[-1] - pos[0]))
    return {"tcp_path_m": round(path, 4), "straight_m": round(straight, 4),
            "ratio": round(path / straight, 2) if straight > 1e-9 else None,
            "max_ratio": ratio, "min_path_m": min_path, "samples": int(len(Q) - 1),
            "refused": bool(path > min_path and path > ratio * straight)}


def plan_guard_reason(check):
    """The sentence a refused plan carries (model-visible through the compact view): the guard and its numbers."""
    times = f"{check['ratio']:.1f} times; " if check.get("ratio") is not None else ""
    return (f"plan guard: the planned TCP path is {check['tcp_path_m']:.2f} m for a straight distance of "
            f"{check['straight_m']:.2f} m ({times}a plan is refused above {check['max_ratio']:g} times and "
            f"{check['min_path_m']:g} m): the arm would swing around into another joint configuration. Use an "
            f"intermediate waypoint, another gripper orientation or linear moves.")


# ---- YAW: the symmetric-gripper flip (see the module docstring)
YAW = ("yaw_flip",)
YAW_FLIP = np.diag([-1.0, -1.0, 1.0])   # Rz(pi): the TCP turned 180 deg about its own z axis (body frame), = its inverse
YAW_FRAMES = ("requested", "flipped")
YAW_FLIP_MIN_GAIN_RAD = 0.05            # engineering value: the flipped goal must be closer by more than this (near ties
                                        # keep the request)
YAW_FLIP_ORIENTATION_SLACK_RAD = 0.1    # engineering value: the two IK goals reach their orientations within the solvers'
                                        # tolerance (cuRobo 0.02, the worker's reached check 0.02 rad); generous
YAW_FLIP_FAR_RAD = (math.pi - YAW_FLIP_ORIENTATION_SLACK_RAD) / 6.0   # 0.507 rad: L-infinity lower bound between two
                                        # goals whose TCP orientations differ by about pi (six joints, each turning the TCP
                                        # by at most its own angle)
YAW_FLIP_SKIP_RAD = (YAW_FLIP_FAR_RAD + YAW_FLIP_MIN_GAIN_RAD) / 2.0  # 0.278 rad: up to this L-infinity change of the
                                        # requested goal the flipped one cannot be closer by the margin (not prepared)
YAW_JOINT6_LIMIT_RAD = math.radians(179.9087371826172)   # agp_core/policy_planner.py SIM_JOINT_LIMITS_DEG joint6 (ARX.usd,
                                        # = X5A.urdf); only for the record (joint6's margin), the IK keeps the limits
YAW_DISTANCE = ("L-infinity of the goal's joint change from the current joints (migration.kinematics.nearest_solution's "
                "measure), on the actual joint values within the simulator's limits")
YAW_LOG_FILE = "yaw_flip.jsonl"         # one line per move_ee preparation of the worker (refused ones included)


def flip_quaternion(rotation):
    """{w, x, y, z} turned by 180 deg about its own z axis: q * (0, 0, 0, 1) (the same orientation for a symmetric
    two-finger gripper)."""
    w, x, y, z = (float(rotation[k]) for k in "wxyz")
    return {"w": -z, "x": y, "y": -x, "z": w}


def flip_pose7(pose):
    """[x, y, z, qw, qx, qy, qz] with the orientation turned 180 deg about its own z axis (position unchanged)."""
    values = [float(v) for v in pose]
    q = flip_quaternion(dict(zip("wxyz", values[3:7])))
    return values[:3] + [q[k] for k in "wxyz"]


def yaw_candidate(start, plan):
    """The record of one prepared move_ee candidate: the refusal, or its goal (the last commanded sample), the joint
    change from `start`, the L-infinity and L1 distance, joint6's margin to its limit, the samples and the commanded
    joint path's L-infinity length."""
    if not isinstance(plan, dict) or not plan.get("ok"):
        plan = plan if isinstance(plan, dict) else {}
        out = {"ok": False, "error": plan.get("error", "not prepared")}
        if plan.get("max_joint_jump_rad") is not None:
            out["max_joint_jump_rad"] = float(plan["max_joint_jump_rad"])
        if plan.get("ours_plan_guard") is not None:     # plan guard: the refused plan's numbers
            out["plan_guard"] = plan["ours_plan_guard"]
        return out
    start = np.asarray(start, float)
    points = np.asarray(plan["points"], float).reshape(-1, len(start))
    goal = points[-1]
    change = goal - start
    path = float(np.abs(np.diff(np.vstack([start[None], points]), axis=0)).max(axis=1).sum())
    return {"ok": True, "goal_joints": goal.tolist(), "joint_change_rad": change.tolist(),
            "linf_rad": float(np.max(np.abs(change))), "l1_rad": float(np.sum(np.abs(change))),
            "joint6_margin_rad": float(YAW_JOINT6_LIMIT_RAD - abs(goal[5])), "samples": int(len(points)),
            "path_linf_rad": path}


def yaw_flip_choice(requested, flipped, min_gain=YAW_FLIP_MIN_GAIN_RAD):
    """(chosen, reason) between the two candidate records (yaw_candidate) of an empty gripper's move_ee: "flipped" when
    its goal is closer by more than min_gain (L-infinity), else "requested"; the only one prepared when the other was
    refused; (None, reason) when both were refused."""
    r_ok, f_ok = bool((requested or {}).get("ok")), bool((flipped or {}).get("ok"))
    if r_ok and f_ok:
        dr, df = float(requested["linf_rad"]), float(flipped["linf_rad"])
        if df < dr - min_gain:
            return "flipped", f"the flipped goal is closer: L-inf {df:.4f} rad against {dr:.4f} rad requested"
        return "requested", (f"the flipped goal is not closer by more than {min_gain:g} rad: L-inf {df:.4f} rad against "
                             f"{dr:.4f} rad requested")
    if r_ok:
        return "requested", f"only the requested orientation can be prepared (flipped: {(flipped or {}).get('error')})"
    if f_ok:
        return "flipped", f"only the flipped orientation can be prepared (requested: {(requested or {}).get('error')})"
    return None, "neither orientation can be prepared"


class AgentFrameModel:
    """Yaw flip: the RGB-only observer's camera model (agp_core/public_camera.py) with each arm's TCP presented in the arm's
    frame (`frames`: the worker's live {arm: "requested" | "flipped"}): tcp_in_base turns a "flipped" arm's TCP back by
    Rz(pi), so the observation's tcp_pose_base and every camera's T_base_tcp_by_arm are in the frame the agent asked
    for. Everything else (camera poses from the official ee_pose, link poses, intrinsics) is the model's own."""

    def __init__(self, model, frames):
        self._yf_model = model
        self._yf_frames = frames

    def tcp_in_base(self, arm, ee_pose):
        T = np.array(self._yf_model.tcp_in_base(arm, ee_pose), dtype=float, copy=True)
        if self._yf_frames.get(arm) == "flipped":
            T[:3, :3] = T[:3, :3] @ YAW_FLIP
        return T

    def __getattr__(self, name):               # only reached for names this proxy does not define
        if name.startswith("_yf_"):
            raise AttributeError(name)
        return getattr(self._yf_model, name)


def effective_r3i(flags):
    """{R3I worker name: on} with each switch on only when it is set and the switch it acts with is on (R3I_NEEDS)."""
    return {name: bool(flags.get(name)) and all(bool(flags.get(need)) for need in R3I_NEEDS[name]) for name in R3I}


def reading_gap_m(n):
    """The pad gap (m) from the official reading n of a closed gripper blocked by an object (fitted on replays)."""
    return VERIFY_GAP_PER_READING_M * float(n) + VERIFY_GAP_OFFSET_M


def lift_verdict(trace):
    """The verified fast hold's verdict on the first lift after a close: (verified, record) from the readings after its
    control steps (None when no finite reading was taken: the lift did not run). Verified when the pad gap of the first
    reading minus that of the last is at most VERIFY_SHRINK_M and the last reading is > 0 (n > CLOSED_EMPTY)."""
    readings = [float(x) for x in (trace or []) if finite(x)]
    if not readings:
        return None, {"readings": 0}
    first, last = reading_gap_m(readings[0]), reading_gap_m(readings[-1])
    shrink = first - last
    verified = bool(shrink <= VERIFY_SHRINK_M + 1e-12 and readings[-1] > CLOSED_EMPTY)
    why = ("verified" if verified else
           f"reading {readings[-1]:.4f} at the lift's end: not > 0" if readings[-1] <= CLOSED_EMPTY else
           f"the pad gap shrank by {shrink * 1000:.2f} mm (more than {VERIFY_SHRINK_M * 1000:.1f} mm)")
    return verified, {"readings": len(readings), "first_reading": round(readings[0], 6),
                      "last_reading": round(readings[-1], 6), "first_gap_m": round(first, 6),
                      "last_gap_m": round(last, 6), "shrink_m": round(shrink, 6), "verified": verified,
                      "reason": why}


def hold_parts_fast():
    """The verified fast hold: H's single s-curve part with the TCP cap FAST_HOLD_TCP_SPEED and peak acceleration
    FAST_HOLD_TCP_ACCEL (rotation and joint caps, the cap safety and every check of H unchanged)."""
    return [{"name": "verified hold s-curve (R2)", "tcp_m": None, "caps": dict(HOLD_CAPS, tcp=FAST_HOLD_TCP_SPEED),
             "law": "scurve", "accel": FAST_HOLD_TCP_ACCEL}]


def clearance_note(lift_m, descent_m):
    """The held clearance's sentence for the model when the rule added a lift and/or a final descent (None when it added
    neither)."""
    lift = f"a {lift_m * 100:.1f} cm lift before moving across" if lift_m > 0 else None
    down = f"a {descent_m * 100:.1f} cm descent at the end" if descent_m > 0 else None
    if lift is None and down is None:
        return None
    added = " and ".join(part for part in (lift, down) if part)
    return (f"Held clearance added {added}, to keep the held object {CLEARANCE_M * 100:.0f} cm above the table.")


# ----------------------------------------------------------------------------- the time law of F
def trapezoid_radius(n, dt=0.04, v=FREE_JOINT_SPEED, a=FREE_JOINT_ACCEL):
    """Largest joint distance (L-infinity) a rest-to-rest trapezoid covers in n control steps."""
    t = max(0, int(n)) * dt
    if t >= 2.0 * v / a:
        return v * t - v * v / a
    return a * t * t / 4.0


def trapezoid_steps(distance, dt=0.04, v=FREE_JOINT_SPEED, a=FREE_JOINT_ACCEL):
    distance = float(distance)
    if distance <= 0.0:
        return 0
    t = distance / v + v / a if distance >= v * v / a else 2.0 * math.sqrt(distance / a)
    return int(math.ceil(t / dt - 1e-9))


def _segment_profile(a0, a1, top, length, acc):
    """(t_accel, t_cruise, t_decel, peak) of a segment entered at a0, left at a1, capped at top."""
    peak = math.sqrt(max(0.0, (a0 * a0 + a1 * a1) / 2.0 + acc * length))
    if peak > top:
        peak = top
        d1 = (peak * peak - a0 * a0) / (2.0 * acc)
        d3 = (peak * peak - a1 * a1) / (2.0 * acc)
        cruise = max(0.0, length - d1 - d3) / peak if peak > 0 else 0.0
    else:
        cruise = 0.0
    return max(0.0, (peak - a0) / acc), cruise, max(0.0, (peak - a1) / acc), peak


def _distance_in_segment(tau, a0, peak, t1, t2, acc, length):
    if tau <= t1:
        s = a0 * tau + 0.5 * acc * tau * tau
    elif tau <= t1 + t2:
        s = (a0 * t1 + 0.5 * acc * t1 * t1) + peak * (tau - t1)
    else:
        u = tau - t1 - t2
        s = (a0 * t1 + 0.5 * acc * t1 * t1) + peak * t2 + peak * u - 0.5 * acc * u * u
    return min(max(s, 0.0), length)


def retime_free(start, dense, tcp_of, table_z=0.0, dt=0.04, limits=None):
    """Re-time the dense joint polyline start -> dense[-1] under F's limits (see the module docstring).

    tcp_of(Q (k, 6)) -> (k, 4, 4) TCP frames in the arm base; table_z: the table top in the same frame.
    Returns (samples, info): samples (list of (6,) arrays, the last equal to dense[-1]) or None when no admissible
    timing was found (the caller keeps the dense samples)."""
    lim = {"v": FREE_JOINT_SPEED, "a": FREE_JOINT_ACCEL, "tcp": FREE_TCP_SPEED, "rot": FREE_TCP_ROTATION,
           "low_z": LOW_HEIGHT_M, "low_v": LOW_TCP_SPEED, "land_z": LANDING_HEIGHT_M, "land_d": LANDING_LENGTH_M,
           "take_z": TAKEOFF_HEIGHT_M, "take_d": TAKEOFF_LENGTH_M, "zone_v": ZONE_TCP_SPEED, "zone_w": ZONE_ROTATION}
    lim.update(limits or {})
    V, A = float(lim["v"]), float(lim["a"])
    start = np.asarray(start, float)
    raw = np.vstack([start[None], np.asarray(dense, float).reshape(-1, len(start))])
    keep = [0]
    for i in range(1, len(raw)):
        if np.max(np.abs(raw[i] - raw[keep[-1]])) > 1e-12:
            keep.append(i)
    P = raw[keep]
    end = raw[-1]
    info = {"law": "free_topt", "dense_samples": int(len(raw) - 1), "limits": {k: float(v) for k, v in lim.items()},
            "table_z_base_m": float(table_z), "dt_s": float(dt)}
    if len(P) < 2:
        info.update(samples=1, seconds=0.0, iterations=0, note="no motion")
        return [end.copy()], info
    seg = np.diff(P, axis=0)
    ds = np.max(np.abs(seg), axis=1)                                 # L-infinity length of every segment
    n = len(ds)
    frames = np.asarray(tcp_of(P), float)
    pos, rot = frames[:, :3, 3], frames[:, :3, :3]
    dx = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    rel = np.einsum("nij,nik->njk", rot[:-1], rot[1:])                # R_i^T R_{i+1}
    dr = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    height = pos[:, 2] - float(table_z)
    cum = np.r_[0.0, np.cumsum(dx)]
    total_x = float(cum[-1])
    low = np.minimum(height[:-1], height[1:]) < lim["low_z"]
    landing_move = bool(height[-1] < lim["land_z"])
    takeoff_move = bool(height[0] < lim["take_z"])
    landing = landing_move & ((total_x - cum[1:]) < lim["land_d"])       # segments overlapping the last land_d
    takeoff = takeoff_move & (cum[:-1] < lim["take_d"])                  # segments starting within the first take_d
    tcp_cap = np.full(n, float(lim["tcp"]))
    tcp_cap[low] = np.minimum(tcp_cap[low], lim["low_v"])
    tcp_cap[landing | takeoff] = np.minimum(tcp_cap[landing | takeoff], lim["zone_v"])
    rot_cap = np.full(n, float(lim["rot"]))
    rot_cap[landing] = np.minimum(rot_cap[landing], lim["zone_w"])
    with np.errstate(divide="ignore", invalid="ignore"):
        cap = np.full(n, V)
        cap = np.minimum(cap, np.where(dx > 1e-12, CAP_SAFETY * tcp_cap * ds / dx, np.inf))
        cap = np.minimum(cap, np.where(dr > 1e-12, CAP_SAFETY * rot_cap * ds / dr, np.inf))
    direction = seg / ds[:, None]
    turn = np.r_[0.0, np.max(np.abs(np.diff(direction, axis=0)), axis=1), 0.0]      # at every vertex
    vertex = np.full(n + 1, np.inf)
    vertex[1:-1] = np.minimum(cap[:-1], cap[1:])
    with np.errstate(divide="ignore"):
        vertex = np.minimum(vertex, np.where(turn > 1e-12, CORNER_SHARE * A * dt / turn, np.inf))
    vertex[0] = vertex[-1] = 0.0
    sigma = np.r_[0.0, np.cumsum(ds)]
    info.update(path_rad=float(sigma[-1]), tcp_path_m=total_x, start_height_m=float(height[0]),
                end_height_m=float(height[-1]), landing_zone=landing_move, takeoff_zone=takeoff_move,
                low_segments=int(low.sum()), segments=int(n))
    for iteration in range(RETIME_ITERATIONS + 1):
        v = vertex.copy()
        for i in range(n):                                            # forward: accelerate
            v[i + 1] = min(v[i + 1], math.sqrt(v[i] * v[i] + 2.0 * A * ds[i]))
        for i in range(n - 1, -1, -1):                                # backward: decelerate
            v[i] = min(v[i], math.sqrt(v[i + 1] * v[i + 1] + 2.0 * A * ds[i]))
        prof = [_segment_profile(v[i], v[i + 1], cap[i], ds[i], A) for i in range(n)]
        durations = np.array([p[0] + p[1] + p[2] for p in prof])
        t_start = np.r_[0.0, np.cumsum(durations)]
        total_t = float(t_start[-1])
        steps = max(1, int(math.ceil(total_t / dt - 1e-9)))
        sig = np.empty(steps)
        index = np.empty(steps, int)
        for k in range(1, steps + 1):
            if k == steps:
                sig[k - 1], index[k - 1] = sigma[-1], n - 1
                continue
            t = k * dt
            i = int(min(n - 1, max(0, np.searchsorted(t_start, t, side="right") - 1)))
            t1, t2, _, peak = prof[i]
            sig[k - 1] = sigma[i] + _distance_in_segment(t - t_start[i], v[i], peak, t1, t2, A, ds[i])
            index[k - 1] = i
        Q = np.empty((steps, len(start)))
        for k in range(steps):
            i = index[k]
            f = 0.0 if ds[i] <= 0 else min(1.0, max(0.0, (sig[k] - sigma[i]) / ds[i]))
            Q[k] = P[i] + f * (P[i + 1] - P[i])
        Q[-1] = end
        full = np.vstack([start[None], Q])
        vel = np.diff(full, axis=0) / dt
        velocities = np.vstack([np.zeros((1, len(start))), vel, np.zeros((1, len(start)))])
        accel = np.diff(velocities, axis=0) / dt                         # at every sample, from and to rest
        frames_k = np.asarray(tcp_of(full), float)
        chord = np.linalg.norm(np.diff(frames_k[:, :3, 3], axis=0), axis=1) / dt
        relk = np.einsum("nij,nik->njk", frames_k[:-1, :3, :3], frames_k[1:, :3, :3])
        turnk = np.arccos(np.clip((np.trace(relk, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)) / dt
        sig_full = np.r_[0.0, sig]
        allowed_v = np.empty(steps)
        allowed_w = np.empty(steps)
        for k in range(steps):                     # the segments a step passes: the largest cap among them
            lo = int(min(n - 1, max(0, np.searchsorted(sigma, sig_full[k], side="right") - 1)))
            hi = int(min(n - 1, max(lo, np.searchsorted(sigma, sig_full[k + 1], side="left") - 1)))
            allowed_v[k] = float(np.max(tcp_cap[lo:hi + 1]))
            allowed_w[k] = float(np.max(rot_cap[lo:hi + 1]))
        bad_a = np.nonzero(np.max(np.abs(accel), axis=1) > A * (1 + 1e-6))[0]           # sample indices 0..steps
        bad_v = np.nonzero(np.max(np.abs(vel), axis=1) > V * (1 + 1e-9))[0] + 1
        bad_x = np.nonzero(chord > allowed_v * (1 + 1e-6))[0] + 1
        bad_w = np.nonzero(turnk > allowed_w * (1 + 1e-6))[0] + 1
        bad = sorted(set(bad_a.tolist()) | set(bad_v.tolist()) | set(bad_x.tolist()) | set(bad_w.tolist()))
        if not bad:
            info.update(samples=int(steps), seconds=round(total_t, 6), iterations=iteration,
                        max_joint_speed=float(np.max(np.abs(vel))) if len(vel) else 0.0,
                        max_joint_accel=float(np.max(np.abs(accel))) if len(accel) else 0.0,
                        max_tcp_speed=float(np.max(chord)) if len(chord) else 0.0,
                        max_tcp_rotation=float(np.max(turnk)) if len(turnk) else 0.0)
            return [q.copy() for q in Q], info
        if iteration == RETIME_ITERATIONS:
            break
        for k in bad:                                  # slow down the segments around the offending samples
            lo_s = sig_full[max(0, k - 1)]
            hi_s = sig_full[min(steps, k + 1)]
            first = int(max(0, np.searchsorted(sigma, lo_s, side="right") - 2))
            last = int(min(n - 1, np.searchsorted(sigma, hi_s, side="left")))
            for i in range(first, last + 1):
                cap[i] = min(cap[i], RETIME_SLOWDOWN * max(prof[i][3], 1e-6))
            vertex[first:last + 2] = np.minimum(vertex[first:last + 2], RETIME_SLOWDOWN * v[first:last + 2])
        vertex[0] = vertex[-1] = 0.0
    info.update(samples=None, failed="no admissible timing within the slow-down rounds", iterations=RETIME_ITERATIONS)
    return None, info


class FreeLaw:
    """F's time law for one arm's home (agp_core/approx_home.py): the box radius of n steps, the dense resampling step and
    the re-timing of a dense path."""

    name = "free_topt"

    def __init__(self, tcp_of, table_z, dt, dense_step, limits=None):
        self.tcp_of, self.table_z, self.dt, self.dense_step = tcp_of, float(table_z), float(dt), float(dense_step)
        self.limits = limits                    # OURS_PRECISION_2X: the scaled zones (zone_limits), else None

    def radius(self, n):
        return trapezoid_radius(n, self.dt)

    def retime(self, start, dense):
        if self.limits is None:
            return retime_free(start, dense, self.tcp_of, self.table_z, self.dt)
        return retime_free(start, dense, self.tcp_of, self.table_z, self.dt, limits=self.limits)

    def steps(self, start, dense):
        points, info = self.retime(start, dense)
        return len(points) if points is not None else len(dense)


# ----------------------------------------------------------------------------- the time laws of H and P's lift
def const_positions(U, dt=0.04):
    """Path positions u_k (k = 1..n, u_n = U) at a constant rate of 1 (the caps), starting and stopping abruptly: the
    worker's own sampling (n = ceil(U / dt))."""
    U = float(U)
    n = max(1, int(math.ceil(U / dt - 1e-9)))
    return np.array([U * k / n for k in range(1, n + 1)]), {"law": "const", "peak_rate": 1.0, "seconds": n * dt}


def scurve_positions(U, accel, dt=0.04, entry=0.0):
    """Path positions u_k (k = 1..n, u_n = U) of a raised-cosine (s-curve) profile along a path parameter measured in
    seconds at the caps: the rate (1 = at the caps) rises from `entry` to at most 1 with the shape
    entry + (peak - entry) (1 - cos(pi t / rise)) / 2, cruises, and falls to 0 as peak (1 + cos(pi w / fall)) / 2; the
    peak rate acceleration is `accel` (1/s) on both ramps. A path too short for the full rate gets a lower peak with the
    same peak acceleration. The time is stretched so that sample n lands exactly on U."""
    U = float(U)
    if U <= 1e-15:
        return np.array([U]), {"law": "scurve", "peak_rate": 0.0, "seconds": 0.0}
    v0 = min(max(float(entry), 0.0), 1.0)
    peak = 1.0
    rise = math.pi * (peak - v0) / (2.0 * accel)
    fall = math.pi * peak / (2.0 * accel)
    ramps = (v0 + peak) / 2.0 * rise + peak / 2.0 * fall
    if ramps <= U:
        cruise = (U - ramps) / peak
    else:
        peak = math.sqrt((4.0 * accel * U / math.pi + v0 * v0) / 2.0)
        if peak <= v0:                        # too short to slow down from the entry rate: keep the entry rate
            n = max(1, int(math.ceil(U / (v0 * dt) - 1e-9)))
            return (np.array([U * k / n for k in range(1, n + 1)]),
                    {"law": "scurve", "peak_rate": v0, "seconds": n * dt, "note": "entry rate kept to the end"})
        rise = math.pi * (peak - v0) / (2.0 * accel)
        fall = math.pi * peak / (2.0 * accel)
        cruise = 0.0
    T = rise + cruise + fall
    n = max(1, int(math.ceil(T / dt - 1e-9)))
    scale = T / (n * dt)
    u_rise = (v0 + peak) / 2.0 * rise

    def u_of(t):
        if t <= rise:
            return v0 * t + (peak - v0) * (t / 2.0 - rise / (2.0 * math.pi) * math.sin(math.pi * t / rise))
        if t <= rise + cruise:
            return u_rise + peak * (t - rise)
        w = min(t - rise - cruise, fall)
        return u_rise + peak * cruise + peak * (w / 2.0 + fall / (2.0 * math.pi) * math.sin(math.pi * w / fall))
    u = np.array([min(U, u_of(k * dt * scale)) for k in range(1, n + 1)])
    u[-1] = U
    return u, {"law": "scurve", "peak_rate": round(peak, 6), "seconds": round(n * dt, 6), "entry_rate": round(v0, 6),
               "rise_s": round(rise, 6), "cruise_s": round(cruise, 6), "fall_s": round(fall, 6)}


def _segment_metrics(P, frames):
    """(dq L-infinity, dx TCP translation, dr TCP rotation) of every segment of the joint polyline P."""
    dq = np.max(np.abs(np.diff(P, axis=0)), axis=1)
    dx = np.linalg.norm(np.diff(frames[:, :3, 3], axis=0), axis=1)
    rel = np.einsum("nij,nik->njk", frames[:-1, :3, :3], frames[1:, :3, :3])
    dr = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0))
    return dq, dx, dr


def step_metrics(start, samples, tcp_of, dt=0.04):
    """Per control step: joint speed (L-infinity, rad/s), TCP speed (m/s), TCP rotation (rad/s); per sample: the TCP
    acceleration (m/s^2, second difference of the TCP positions from and to rest)."""
    full = np.vstack([np.asarray(start, float)[None], np.asarray(samples, float)])
    frames = np.asarray(tcp_of(full), float)
    dq, dx, dr = _segment_metrics(full, frames)
    pos = frames[:, :3, 3]
    padded = np.vstack([pos[:1], pos, pos[-1:]])
    acc = np.linalg.norm(padded[2:] - 2.0 * padded[1:-1] + padded[:-2], axis=1) / dt ** 2    # at samples 0..n
    return dq / dt, dx / dt, dr / dt, acc


def retime_parts(start, dense, tcp_of, parts, dt=0.04):
    """Re-time the dense joint polyline start -> dense[-1] in consecutive parts along its TCP path (H, P's lift).

    parts: [{"name", "tcp_m": TCP path length of the part (None for the last part: the rest of the path), "caps":
    {"tcp" m/s, "rot" rad/s, "joint" rad/s}, "law": "const" | "scurve", "accel": TCP m/s^2 (scurve)}]. A part's path
    parameter is the time at its caps (planned at CAP_SAFETY of them): per segment the largest of TCP length / tcp,
    rotation / rot and L-infinity joint length / joint. A "const" part runs at its caps throughout (abrupt start and
    stop: the worker's own rule); an "scurve" part (scurve_positions) enters at the rate the previous part had at the
    boundary (rest for a first part) and stops at rest, with the peak rate acceleration accel / tcp (the in-place twist:
    rot_accel / rot when the part gives "rot_accel", in rad/s^2 where rotation binds; accel then only bounds the check).
    The boundary of a part is inserted into the polyline (joint interpolation at the TCP path length). Every step is
    checked against its part's caps at 100 %, and the TCP acceleration of the samples in and around an scurve part
    against its accel; a failing part's rate is lowered by RETIME_SLOWDOWN and the timing redone, at most
    HOLD_ITERATIONS rounds. Returns (samples, info): samples (list of (6,) arrays on the polyline, the last equal to
    dense[-1]) or None when no admissible timing was found (the caller keeps the worker's samples)."""
    start = np.asarray(start, float)
    raw = np.vstack([start[None], np.asarray(dense, float).reshape(-1, len(start))])
    keep = [0]
    for i in range(1, len(raw)):
        if np.max(np.abs(raw[i] - raw[keep[-1]])) > 1e-12:
            keep.append(i)
    P = raw[keep]
    end = raw[-1]
    info = {"law": "parts", "dense_samples": int(len(raw) - 1), "dt_s": float(dt)}
    if len(P) < 2:
        info.update(samples=1, note="no motion", parts=[dict(p) for p in parts])
        return [end.copy()], info
    frames = np.asarray(tcp_of(P), float)
    dq, dx, dr = _segment_metrics(P, frames)
    cum = np.r_[0.0, np.cumsum(dx)]
    targets, at = [], 0.0                       # TCP path lengths at which the parts but the last end
    for p in parts[:-1]:
        at += float(p["tcp_m"])
        targets.append(min(max(at, 0.0), float(cum[-1])))
    Q, edges, t = [P[0]], [0], 0
    for i in range(len(P) - 1):
        while t < len(targets) and targets[t] <= cum[i] + 1e-12:
            edges.append(len(Q) - 1)
            t += 1
        while t < len(targets) and targets[t] < cum[i + 1] - 1e-12:
            f = (targets[t] - cum[i]) / dx[i]
            Q.append(P[i] + f * (P[i + 1] - P[i]))
            edges.append(len(Q) - 1)
            t += 1
        Q.append(P[i + 1])
    while t < len(targets):
        edges.append(len(Q) - 1)
        t += 1
    edges.append(len(Q) - 1)
    Q = np.asarray(Q)
    frames = np.asarray(tcp_of(Q), float)
    dq, dx, dr = _segment_metrics(Q, frames)
    scale = [1.0] * len(parts)

    def unit_times(j):                          # every segment's time at part j's (lowered) caps
        caps, s = parts[j]["caps"], CAP_SAFETY * scale[j]
        return np.maximum.reduce([dx / (caps["tcp"] * s), dr / (caps["rot"] * s), dq / (caps["joint"] * s)])

    for iteration in range(HOLD_ITERATIONS + 1):
        samples, owner, records = [], [], []
        previous = None                         # (part index, its law) of the last non-empty part
        for j, part in enumerate(parts):
            a, b = edges[j], edges[j + 1]
            record = {"name": part.get("name"), "law": part.get("law"),
                      "caps": {k: float(v) for k, v in part["caps"].items()}, "rate_scale": round(scale[j], 6),
                      "tcp_m": round(float(dx[a:b].sum()), 6), "samples": 0}
            if b <= a:
                records.append(record)
                continue
            u_all = unit_times(j)
            u_seg = u_all[a:b]
            U = float(u_seg.sum())
            if part.get("law") == "scurve":
                entry = 0.0
                if previous is not None and previous[1] == "const" and a > 0:
                    before = unit_times(previous[0])[a - 1]   # the boundary's previous segment ran at rate 1 there
                    entry = min(1.0, float(u_all[a - 1] / before)) if before > 0 else 0.0
                if part.get("rot_accel") is not None:     # the twist: the peak given where rotation binds (rad/s^2)
                    accel = float(part["rot_accel"]) / (part["caps"]["rot"] * CAP_SAFETY * scale[j])
                else:
                    accel = float(part["accel"]) / (part["caps"]["tcp"] * CAP_SAFETY * scale[j])
                u, law = scurve_positions(U, accel, dt, entry)
            else:
                u, law = const_positions(U, dt)
            u_cum = np.r_[0.0, np.cumsum(u_seg)]
            for value in u:
                i = int(min(b - a - 1, max(0, np.searchsorted(u_cum, value, side="right") - 1)))
                f = 0.0 if u_seg[i] <= 0 else min(1.0, max(0.0, (value - u_cum[i]) / u_seg[i]))
                samples.append(Q[a + i] + f * (Q[a + i + 1] - Q[a + i]))
                owner.append(j)
            samples[-1] = Q[b].copy()
            record.update(law, samples=len(u))
            records.append(record)
            previous = (j, part.get("law"))
        samples[-1] = end.copy()
        jspeed, tspeed, rspeed, acc = step_metrics(start, samples, tcp_of, dt)
        owner = np.asarray(owner)
        bad = set()
        for j, part in enumerate(parts):
            steps = np.nonzero(owner == j)[0]
            if not len(steps):
                continue
            caps = part["caps"]
            if (np.any(tspeed[steps] > caps["tcp"] * (1 + 1e-6)) or np.any(rspeed[steps] > caps["rot"] * (1 + 1e-6))
                    or np.any(jspeed[steps] > caps["joint"] * (1 + 1e-9))):
                bad.add(j)
            if part.get("law") == "scurve":     # the samples that bound this part's steps (from and to rest)
                around = np.unique(np.r_[steps, steps + 1])
                if np.any(acc[around] > float(part["accel"]) * (1 + HOLD_ACCEL_TOLERANCE)):
                    bad.add(j)
        if not bad:
            for j in range(len(parts)):
                steps = np.nonzero(owner == j)[0]
                if len(steps):
                    around = np.unique(np.r_[steps, steps + 1])
                    records[j].update(max_tcp_speed=float(tspeed[steps].max()),
                                      max_tcp_rotation=float(rspeed[steps].max()),
                                      max_joint_speed=float(jspeed[steps].max()), max_tcp_accel=float(acc[around].max()))
            info.update(parts=records, samples=len(samples), iterations=iteration, tcp_path_m=round(float(cum[-1]), 6),
                        max_tcp_speed=float(tspeed.max()), max_tcp_rotation=float(rspeed.max()),
                        max_joint_speed=float(jspeed.max()), max_tcp_accel=float(acc.max()))
            return [np.asarray(q, float).copy() for q in samples], info
        if iteration == HOLD_ITERATIONS:
            break
        for j in bad:
            scale[j] *= RETIME_SLOWDOWN
    info.update(samples=None, failed="no admissible timing within the slow-down rounds", iterations=HOLD_ITERATIONS,
                parts=[dict(p) for p in parts])
    return None, info


def hold_parts():
    """H: one s-curve part over the whole path at H's limits."""
    return [{"name": "hold s-curve", "tcp_m": None, "caps": dict(HOLD_CAPS), "law": "scurve", "accel": HOLD_TCP_ACCEL}]


def lift_parts(hold_profile, precision_scale=1.0):
    """P's lift: the first PREC_LIFT_HEAD_M at the precision rate (x precision_scale with OURS_PRECISION_2X), the rest
    at H's s-curve (OURS_HOLD_PROFILE on) or at the transit rate of a closed gripper."""
    from agp_core import prec_zone
    rest = ({"name": "rest, hold s-curve", "tcp_m": None, "caps": dict(HOLD_CAPS), "law": "scurve",
             "accel": HOLD_TCP_ACCEL} if hold_profile else
            {"name": "rest, transit", "tcp_m": None, "caps": dict(TRANSIT_CLOSED_CAPS), "law": "const"})
    return [{"name": "precision head", "tcp_m": prec_zone.LIFT_HEAD_M, "caps": precision_caps(precision_scale),
             "law": "const"}, rest]


# ----------------------------------------------------------------------------- the official observation
def gripper_observation(state, arm):
    value = (state or {}).get(f"{arm}_ee_joint_state")
    if value is None:
        return None
    try:
        return float(np.asarray(value, float).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None


def empty_state(commanded, observed, estimate=None):
    """(empty, reason) of one gripper for F: commanded 1.0 open / 0.0 closed, observed n (None when absent), estimate the
    wrist-image estimate of the same observation (agp_core/visual_gap.py: {quality, pad_gap_m, ...}; None when not taken).

    Commanded open: empty when n = 1 (the open has converged). Commanded closed (two signals, as D's hold_loss): empty
    only when the reading shows the fingers fully closed (n <= CLOSED_EMPTY) and the estimate, quality ok, has a pad gap
    below HOLD_LOST_GAP_M; n = 0 with the estimate missing, unavailable, not quality ok, without a pad gap or showing
    the pads apart counts as holding (the reason ends in 'holding'); n > 0: holding."""
    if observed is None:
        return False, "no gripper observation"
    if commanded == 1.0:
        if observed >= OPEN_CONVERGED:
            return True, "commanded open, converged (n = 1)"
        return False, f"commanded open, not converged (n = {observed:.3f})"
    if partial_command(commanded):                           # grip width: the two signals at the commanded width
        return width_empty_state(commanded, observed, estimate)
    if observed <= CLOSED_EMPTY:
        empty, check = hold_loss(observed, estimate)          # D's two signals (n <= CLOSED_EMPTY: check is a dict)
        gap, quality = check["image_pad_gap_m"], check["image_quality"]
        bound = f"{HOLD_LOST_GAP_M * 1000:.1f} mm"
        if empty:
            return True, (f"commanded closed, fully closed on nothing (n = 0, wrist-image pad gap "
                          f"{gap * 1000:.2f} mm < {bound})")
        if quality == "ok" and gap is not None:
            return False, (f"commanded closed, n = 0 but the wrist image shows the pads {gap * 1000:.2f} mm apart "
                           f"(not below {bound}): holding")
        if estimate is None:
            what = "no wrist-image estimate"
        elif quality != "ok":
            what = f"a wrist-image estimate of quality {quality or 'unavailable'}"
        else:
            what = "a wrist-image estimate without a pad gap"
        return False, f"commanded closed, n = 0 with {what} (the second signal is missing): counted as holding"
    return False, f"commanded closed with the fingers apart (n = {observed:.3f}): holding"


def holding_state(commanded, observed):
    """The reading-only holding rule (commanded closed and n > 0): D's watch at the stage start and F-g's held flag.
    Grip width: for a partial command, the reading shows the fingers held wider than the command (n > c: the clamp
    acts from above)."""
    if partial_command(commanded):
        return finite(observed) and observed > commanded and not at_target(commanded, observed)
    return commanded == 0.0 and observed is not None and observed > CLOSED_EMPTY


def retime_holding(commanded, observed, estimate=None):
    """Whether H (and P's lift) treat a gripper as holding: commanded closed with a finite reading and
    not empty by F's two-signal rule (empty_state): n > 0, or n = 0 without a quality-ok wrist image showing the pads
    closed on nothing. The complement of F's closed-gripper rule, so an arm F leaves alone for want of the second signal
    is re-timed by H when H is on. Grip width: a partial command likewise (width_empty_state)."""
    return ((commanded == 0.0 or partial_command(commanded)) and finite(observed)
            and not empty_state(commanded, observed, estimate)[0])


# ----------------------------------------------------------------------------- grip width (OURS_GRIP_WIDTH)
# A partial command 0 < c < 1 is sent only by the width action (agp_core/grip_width.py); open is 1.0 and close 0.0, so with
# the switch off none of the branches below is reached. The official reading equals c while the fingers are within
# 10.8 mm of their target and shows the finger beyond that (the clamp), so for a partial command the two signals of F
# and D are: the reading equals c (at_target; for a close: n = 0) and the wrist-image pad gap (quality ok) below the
# target gap + grip_width.MARGIN_M (lost_bound_m, the width outcome's own margin; for a close: HOLD_LOST_GAP_M).
def partial_command(commanded):
    """A partial gripper command (grip width): 0 < c < 1."""
    return finite(commanded) and 0.0 < float(commanded) < 1.0


def at_target(commanded, observed):
    """D's and F's first signal: the reading shows the fingers at their commanded end -- fully closed (n <= CLOSED_EMPTY)
    for a close; the reading equal to the command (the clamp does not act) for a partial command."""
    if not finite(observed):
        return False
    if partial_command(commanded):
        from agp_core import grip_width as gw
        return gw.at_command(commanded, observed)
    return observed <= CLOSED_EMPTY


def lost_bound_m(commanded):
    """D's and F's second-signal bound on the wrist-image pad gap: HOLD_LOST_GAP_M for a close; for a partial command its
    target gap plus grip_width.MARGIN_M (1 mm: the width outcome's margin, so that F, D and the outcome agree)."""
    if partial_command(commanded):
        from agp_core import grip_width as gw
        return gw.target_gap_m(commanded) + gw.MARGIN_M
    return HOLD_LOST_GAP_M


def width_empty_state(commanded, observed, estimate=None):
    """F's (empty, reason) for a partial command (grip width): empty only when the reading equals the command and the
    wrist-image estimate (quality ok) shows the pad gap below lost_bound_m; a reading above the command (the fingers held
    more than 10.8 mm per finger wider) is holding; one below it has not converged; the second signal missing counts as
    holding (as for a close)."""
    from agp_core import grip_width as gw
    width = f"commanded width {gw.target_gap_m(commanded) * 1000:.1f} mm"
    if not at_target(commanded, observed):
        if observed > commanded:
            return False, (f"{width}, the reading shows the fingers held wider (n = {observed:.3f} > "
                           f"{commanded:.3f}): holding")
        return False, f"{width}, not converged (n = {observed:.3f} < {commanded:.3f})"
    empty, check = hold_loss(observed, estimate, commanded)
    gap, quality = check["image_pad_gap_m"], check["image_quality"]
    bound = f"{lost_bound_m(commanded) * 1000:.2f} mm"
    if empty:
        return True, (f"{width}, the fingers at it with nothing between (n = {observed:.3f}, wrist-image pad gap "
                      f"{gap * 1000:.2f} mm < {bound})")
    if quality == "ok" and gap is not None:
        return False, (f"{width}, n = {observed:.3f} but the wrist image shows the pads {gap * 1000:.2f} mm apart (not "
                       f"below {bound}): holding")
    what = ("no wrist-image estimate" if estimate is None else
            f"a wrist-image estimate of quality {quality or 'unavailable'}" if quality != "ok" else
            "a wrist-image estimate without a pad gap")
    return False, f"{width}, n = {observed:.3f} with {what} (the second signal is missing): counted as holding"


# ----------------------------------------------------------------------------- D
def hold_guard_phase(command, profile, rise_m):
    """(watched, phase, reason) of D for one arm that holds something at the stage start (see the module docstring):
    command and profile of its plan (profile: "transit" for home and transit moves, else "precision"), rise_m: how much
    higher the TCP is at the plan's last scheduled sample than at its start (None when not computed)."""
    if command not in MOTIONS:
        return False, None, f"{command}: not a lift or a transport move"
    if command == "home":
        return True, "transport", "home"
    if profile == "transit":
        return True, "transport", f"{command} at transit speed (transfer, retreat or observe)"
    if rise_m is not None and rise_m >= HOLD_LIFT_RISE_M:
        return True, "lift", f"{command} at precision speed up {rise_m * 1000:.1f} mm (a lift)"
    rise = "unknown" if rise_m is None else f"{rise_m * 1000:.1f} mm"
    return False, None, (f"{command} at precision speed, not going up (rise {rise}): a contact, a place descent or a "
                         f"precision move sideways")


def hold_loss(observed, estimate, commanded=0.0):
    """(lost, check) of D after one control step: lost when the official reading shows the fingers fully closed and the
    wrist-image estimate (quality ok) has a pad gap below HOLD_LOST_GAP_M; check is the record of the second signal (None
    when the reading does not show the fingers fully closed: the image is not looked at). Grip width: for a partial
    command, at_target and lost_bound_m (the commanded width) instead."""
    if observed is None or not finite(observed) or not at_target(commanded, observed):
        return False, None
    estimate = estimate or {}
    quality, gap = estimate.get("quality"), estimate.get("pad_gap_m")
    gap = float(gap) if gap is not None and finite(float(gap)) else None
    lost = quality == "ok" and gap is not None and gap < lost_bound_m(commanded)
    return lost, {"observed": float(observed), "image_quality": quality, "image_pad_gap_m": gap, "lost": lost}


# ----------------------------------------------------------------------------- G
def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def finger_from_reading(n, commanded_open, scale=(-0.01, 0.044), limits=(0.0, 0.044), sign=1.0, eps=GRIP_EPS):
    """The primary finger joint (m) from the official reading n of a gripper commanded open or closed (see G): the
    reading is the commanded target clamped to the finger +- eps * scale range (control_manager.process_gripper_val)."""
    span = float(scale[1]) - float(scale[0])
    t = float(scale[0]) + float(n) * span if sign >= 0 else float(scale[1]) - float(n) * span
    clamp = eps * span
    if commanded_open:
        if n >= OPEN_CONVERGED:
            return float(limits[1])                       # the clamp no longer acts: at least limits[1] - clamp
        return float(min(max(t - clamp, limits[0]), limits[1]))
    if n <= CLOSED_EMPTY:
        return float(limits[0])                           # fully closed (within 0.8 mm of it)
    return float(min(max(t + clamp, limits[0]), limits[1]))


def pad_gap(q7, mimic=(1.0, 0.0)):
    """Nominal pad gap (m): q7 + q8 + the pads' separation at q = 0 (q8 from the public mimic: q8 = m1 * q7 + m2)."""
    return float(q7) + float(q7) * float(mimic[0]) + float(mimic[1]) + GRIP_PAD_GAP0_M


class GripRule:
    """G's end rule for one gripper action (see the module docstring). source: "observation" (readings are the official
    n), "wrist_image" (readings are agp_core/visual_gap.py estimates: {quality, pad_gap_m, finger_joints_m}) or "fixed" (the
    action runs its 10 samples). observe() takes the reading after each control step of the action and returns the end
    reason at the first step at which the rule holds (None before and after)."""

    def __init__(self, action, source, start_reading=None, start_gap_m=None, scale=(-0.01, 0.044),
                 limits=(0.0, 0.044), mimic=(1.0, 0.0), sign=1.0, max_steps=10):
        self.action, self.source = action, source
        self.start_reading, self.start_gap_m = start_reading, start_gap_m
        self.scale, self.limits, self.mimic, self.sign = scale, limits, mimic, sign
        self.max_steps = int(max_steps)
        self.span = float(scale[1]) - float(scale[0])
        self.full_open_gap_m = pad_gap(float(limits[1]) - GRIP_EPS * self.span, mimic)
        self.readings, self.ended_at, self.reason = [], None, None
        self.ended_by = None                # G x F: "wider" when an open ended by the wide-enough rule (not recorded)

    def gap_from_reading(self, n):
        return pad_gap(finger_from_reading(n, self.action == "open", self.scale, self.limits, self.sign), self.mimic)

    def observe(self, value):
        self.readings.append(value)
        if self.ended_at is not None or self.source == "fixed":
            return None
        k = len(self.readings)
        reason = self._decide(k)
        if reason:
            self.ended_at, self.reason = k, reason
            self.ended_by = "wider" if "wider than at the start" in reason else (
                "open" if "fully open" in reason else "closed" if "fully closed" in reason else "still")
        return reason

    def _decide(self, k):
        now = self.readings[-1]
        before = self.readings[-2] if k >= 2 else None
        if self.source == "observation":
            if not finite(now):
                return None
            if self.action == "close":
                if now <= CLOSED_EMPTY:
                    return f"fingers fully closed (n = {now:.4f})"
                if k >= GRIP_MIN_STILL_STEPS and finite(before):
                    travel = abs(now - before) * self.span
                    if travel < GRIP_STILL_M:
                        return (f"fingers still: {travel * 1000:.3f} mm of finger travel between the readings after "
                                f"steps {k - 1} and {k} (n = {now:.4f})")
                return None
            if now >= OPEN_CONVERGED:
                return f"fingers fully open (n = {now:.4f})"
            gap = self.gap_from_reading(now)
            if self.start_gap_m is not None and gap >= self.start_gap_m + GRIP_OPEN_WIDER_M:
                return (f"pad gap {gap * 1000:.1f} mm from the reading, {(gap - self.start_gap_m) * 1000:.1f} mm wider "
                        f"than at the start (n = {now:.4f})")
            return None
        ok = isinstance(now, dict) and now.get("quality") == "ok" and now.get("pad_gap_m") is not None
        if not ok:                                        # wrist_image: estimates of quality ok only
            return None
        gap = float(now["pad_gap_m"])
        if self.action == "close":
            if gap < GRIP_CLOSED_GAP_M:
                return f"estimated pad gap {gap * 1000:.2f} mm: fully closed (wrist image)"
            okb = isinstance(before, dict) and before.get("quality") == "ok" and before.get("finger_joints_m")
            if k >= GRIP_MIN_STILL_STEPS and okb and now.get("finger_joints_m"):
                travel = float(np.max(np.abs(np.asarray(now["finger_joints_m"], float)
                                             - np.asarray(before["finger_joints_m"], float))))
                if travel < GRIP_STILL_M:
                    return (f"fingers still: {travel * 1000:.3f} mm between the wrist-image estimates after steps "
                            f"{k - 1} and {k}")
            return None
        if gap >= self.full_open_gap_m:
            return f"estimated pad gap {gap * 1000:.1f} mm: fully open (wrist image)"
        if self.start_gap_m is not None and gap >= self.start_gap_m + GRIP_OPEN_WIDER_M:
            return (f"estimated pad gap {gap * 1000:.1f} mm, {(gap - self.start_gap_m) * 1000:.1f} mm wider than at the "
                    f"start (wrist image)")
        return None

    def record(self):
        def short(value):
            if isinstance(value, dict):
                return {"quality": value.get("quality"), "pad_gap_m": value.get("pad_gap_m"),
                        "finger_joints_m": value.get("finger_joints_m")}
            return None if value is None else round(float(value), 6)
        return {"action": self.action, "source": self.source, "start_reading": short(self.start_reading),
                "start_gap_m": None if self.start_gap_m is None else round(float(self.start_gap_m), 6),
                "readings": [short(v) for v in self.readings], "ended_after": self.ended_at, "reason": self.reason,
                "max_steps": self.max_steps,
                "rule": {"still_m": GRIP_STILL_M, "open_wider_m": GRIP_OPEN_WIDER_M,
                         "full_open_gap_m": round(self.full_open_gap_m, 6)}}


# ----------------------------------------------------------------------------- F-g layer 1
def stage_timeline(arms, plans, measured, delays=None):
    """Both arms' joints at the stage's instants: the start (measured joints), then per control step the official
    interpolation (i/9 of the way from the previous sample, i = 1..8) and the sample; an arm without a motion command
    stays at its measured joints; a delayed arm holds its start for that many steps first.
    plans: {arm: {command, points, run_samples}}. Returns ({arm: (T, 6)}, steps)."""
    delays = delays or {}
    sequences = {}
    for arm in arms:
        plan = plans.get(arm)
        if plan is None or plan["command"] not in MOTIONS:
            sequences[arm] = []
            continue
        points = [np.asarray(q, float) for q in plan["points"][:plan["run_samples"]]]
        sequences[arm] = [np.asarray(measured[arm], float)] * int(delays.get(arm, 0)) + points
    steps = max((len(s) for s in sequences.values()), default=0)
    fractions = np.arange(1, INTERP_SUBSTEPS + 1) / INTERP_SUBSTEPS
    timeline = {}
    for arm in arms:
        seq = sequences[arm]
        rows = [np.asarray(measured[arm], float)]
        previous = rows[0]
        for k in range(steps):
            target = seq[min(k, len(seq) - 1)] if seq else rows[0]
            rows.extend(previous + f * (target - previous) for f in fractions)
            previous = target
        timeline[arm] = np.asarray(rows)
    return timeline, steps


def guard_stage(geometry, arms, plans, measured, fingers, held, margin=GUARD_MARGIN_M, delay_max=GUARD_DELAY_MAX,
                v2=False):
    """Layer 1 of F-g for one stage (see the module docstring). Returns the record: ok, checked, start_clearance_m,
    threshold_m, min_clearance_m, closest {time, links, distance_m}, step_of_min, delay ({arm, steps, ...} or None),
    error (a refusal's explanation). Plans are not changed. v2 (OURS_GUARD_V2; the caller passes
    margin=GUARD_V2_MARGIN_M): an arm of a two-arm stage may be delayed by up to the other arm's run samples, a conflict
    of the two end poses is refused without trying delays, and the refusal names only this pair (record: v2,
    delay_limits, end_clearance_m, end_conflict)."""
    moving = [arm for arm, p in plans.items() if p["command"] in MOTIONS]
    record = {"ok": True, "moving": moving}
    if not moving:
        record["checked"] = False
        return record
    started = time.monotonic()
    stats = {}
    timeline, steps = stage_timeline(arms, plans, measured)
    per_time, detail = geometry.clearance(timeline, fingers, held, threshold=margin + GUARD_EXACT_ABOVE_M,
                                          report=stats)
    start_clearance = float(per_time[0])
    threshold = min(margin, start_clearance - GUARD_START_TOLERANCE_M)
    record.update(checked=True, steps=steps, fingers_m=fingers, held=held, start_clearance_m=start_clearance,
                  threshold_m=threshold, min_clearance_m=float(per_time.min()), closest=detail,
                  step_of_min=int(math.ceil(detail["time"] / INTERP_SUBSTEPS)), delay=None)
    if float(per_time.min()) >= threshold:
        record.update(seconds=round(time.monotonic() - started, 3), **stats)
        return record
    # a predicted violation: delay one arm of a paired motion (the fewest steps added to the stage, then the
    # shortest delay), else refuse
    options = []
    end_conflict = False
    if len(moving) == 2:
        limits = {arm: delay_max for arm in moving}
        if v2:      # up to the other arm's whole stage; every delay ends with both arms at their end poses
            limits = {arm: int(plans[other]["run_samples"]) for arm in moving for other in moving if other != arm}
            end_conflict = float(per_time[-1]) < threshold          # exact below the threshold (clearance())
            record.update(v2=True, delay_limits=limits, end_clearance_m=float(per_time[-1]),
                          end_conflict=end_conflict)
        for arm in moving:
            for d in range(1, limits[arm] + 1):
                length = max(p["run_samples"] + (d if a == arm else 0) for a, p in plans.items())
                options.append((length, d, arm))
        options.sort()
        if end_conflict:
            options = []
    elif v2:
        record.update(v2=True)
    tried = 0
    for length, d, arm in options:
        delayed, _ = stage_timeline(arms, plans, measured, {arm: d})
        times, closest = geometry.clearance(delayed, fingers, held, threshold=margin + GUARD_EXACT_ABOVE_M,
                                            report=stats, stop_below=threshold)     # a yes/no question
        tried += 1
        if float(times.min()) >= threshold:
            record.update(delay={"arm": arm, "steps": d, "stage_steps": length, "min_clearance_m": float(times.min()),
                                 "closest": closest}, tried=tried, seconds=round(time.monotonic() - started, 3),
                          **stats)
            return record
    step = int(math.ceil(detail["time"] / INTERP_SUBSTEPS))
    names = " and ".join(f"the {arm} arm's {name}" for arm, name in sorted(detail["links"].items()))
    reason = (f"Two-arm clearance: {names} would come within {detail['distance_m'] * 1000:.0f} mm of each other at "
              f"control step {step} of {steps} (the stage must keep {threshold * 1000:.0f} mm)")
    if v2 and len(moving) == 2:     # v2: only this pair conflicts
        if end_conflict:
            reason += (f"; their end poses are {float(per_time[-1]) * 1000:.0f} mm apart, so no delay of either arm "
                       f"keeps it. " + GUARD_V2_PAIR)
        else:
            reason += "; delaying either arm until the other has finished does not keep it. " + GUARD_V2_PAIR
    elif v2:
        reason += "; the other arm does not move in this step. " + GUARD_V2_SINGLE
    elif len(moving) == 2:
        reason += f"; delaying either arm by up to {delay_max} steps does not keep it"
    else:
        reason += "; the other arm does not move in this step"
    record.update(ok=False, error=reason, tried=tried, seconds=round(time.monotonic() - started, 3), **stats)
    return record


# ----------------------------------------------------------------------------- the mixin
class SpeedProfilesMixin:
    """Put in front of the official worker (see the module docstring); `ours_speed_flags` is set by worker_class()."""

    ours_speed_flags = {name: False for name in FLAGS}

    def __init__(self, env, session):
        super().__init__(env, session)
        from migration import worker as base
        from agp_core import arm_guard
        self._sp_flags = dict(type(self).ours_speed_flags)
        self._sp_kin = arm_guard.ArmKinematics()
        self._sp_table_z = {}
        for arm, context in self.arms.items():
            table = getattr(self.rm.planner.get(context["robot"].robot_name) if hasattr(self.rm.planner, "get")
                            else None, "table", None)
            base_z = float(np.asarray(context["base"], float)[2, 3])
            self._sp_table_z[arm] = (float(table.top_z_world) - base_z) if table is not None else 0.0
        self._sp_guard = (arm_guard.ArmGuardGeometry({arm: c["base"] for arm, c in self.arms.items()}, self._sp_kin)
                          if self._sp_flags["arm_guard"] else None)
        self._sp_stage = None
        self._sp_part_b = any(self._sp_flags.get(name) for name in PART_B)
        record = {
            "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name)))
                         for name in (FLAGS if self._sp_part_b else PART_A)},
            "free_topt": {"joint_speed_rad_s": FREE_JOINT_SPEED, "joint_accel_rad_s2": FREE_JOINT_ACCEL,
                          "tcp_speed_m_s": FREE_TCP_SPEED, "tcp_rotation_rad_s": FREE_TCP_ROTATION,
                          "low": [LOW_HEIGHT_M, LOW_TCP_SPEED], "landing": [LANDING_HEIGHT_M, LANDING_LENGTH_M,
                                                                            ZONE_TCP_SPEED, ZONE_ROTATION],
                          "takeoff": [TAKEOFF_HEIGHT_M, TAKEOFF_LENGTH_M, ZONE_TCP_SPEED],
                          "tracking_stop_rad": FREE_TRACKING_STOP_RAD, "table_z_base_m": self._sp_table_z,
                          "closed_empty_when": {"n_at_most": CLOSED_EMPTY, "and_image_pad_gap_below_m": HOLD_LOST_GAP_M,
                                                "image": "agp_core/visual_gap.py estimate of the arm's wrist image, quality "
                                                         "ok (M3 S1 finalize, user decision 2026-10-09)",
                                                "otherwise": "holding: not re-timed by F (H re-times it when on)"}},
            "arm_guard": {"margin_m": GUARD_MARGIN_M, "hard_m": GUARD_HARD_M, "delay_max_steps": GUARD_DELAY_MAX,
                          "start_tolerance_m": GUARD_START_TOLERANCE_M, "held_radius_m": arm_guard.HELD_RADIUS_M,
                          "interp_substeps": INTERP_SUBSTEPS},
            "hold_guard": {"lost_when_n_at_most": CLOSED_EMPTY, "and_image_pad_gap_below_m": HOLD_LOST_GAP_M,
                           "image": "agp_core/visual_gap.py estimate of the arm's wrist image, quality ok",
                           "phases": "lift (precision move up >= lift_rise_m) and transport (transit move, home)",
                           "lift_rise_m": HOLD_LIFT_RISE_M},
            "source": "agp_core/speed_profiles.py"}
        if self._sp_part_b:                                  # PART_B (no PART_B entry with its switches off)
            from agp_core import prec_zone
            record.update(
                grip_obs={"close_still_m": GRIP_STILL_M, "close_min_steps": GRIP_MIN_STILL_STEPS,
                          "close_fully_closed_n_at_most": CLOSED_EMPTY, "open_fully_open_n_at_least": OPEN_CONVERGED,
                          "open_wider_m": GRIP_OPEN_WIDER_M, "clamp_eps": GRIP_EPS, "pad_gap0_m": GRIP_PAD_GAP0_M,
                          "fallback": "wrist-image estimate (agp_core/visual_gap.py, quality ok), else 10 samples",
                          "fallback_closed_gap_m": GRIP_CLOSED_GAP_M},
                hold_profile={"tcp_speed_m_s": HOLD_TCP_SPEED, "tcp_accel_m_s2": HOLD_TCP_ACCEL,
                              "rotation_rad_s": HOLD_ROTATION, "joint_speed_rad_s": HOLD_JOINT_SPEED,
                              "law": "raised-cosine s-curve along the path (time at the caps)",
                              "cap_safety": CAP_SAFETY, "iterations": HOLD_ITERATIONS, "slowdown": RETIME_SLOWDOWN},
                prec_zone={"lift_head_m": prec_zone.LIFT_HEAD_M, "vertical_xy_m": prec_zone.VERTICAL_XY_M,
                           "max_turn_rad": prec_zone.MAX_TURN_RAD, "min_fast_m": prec_zone.MIN_FAST_M,
                           "precision_caps": PRECISION_CAPS, "transit_closed_caps": TRANSIT_CLOSED_CAPS,
                           "contact_tail_m": prec_zone.CONTACT_TAIL_M, "place_tail_m": prec_zone.PLACE_TAIL_M,
                           "splits": "contact and place: made by the run (agp_core/prec_zone.py), prec_zone.jsonl"})
        # the R2A switches: the state of their rules (nothing of it is used, and no record written, with the five off)
        self._sp_r2a = any(self._sp_flags.get(name) for name in R2A)
        self._sp_gf_open = {}                                # G x F: arm -> the open G ended as wide enough
        self._sp_close_tcp = {}                              # held clearance: arm -> TCP height at the close
        if self._sp_r2a:
            record["r2a"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in R2A},
                "acts_with": {features.FLAGS[name]: [features.FLAGS[n] for n in R2A_NEEDS[name]] for name in R2A},
                "guard_v2": {"margin_m": GUARD_V2_MARGIN_M, "hard_m": GUARD_HARD_M,
                             "delay_max": "the other arm's run samples (its whole stage)",
                             "end_pose_conflict": "refused without trying delays", "pair_sentence": GUARD_V2_PAIR,
                             "single_sentence": GUARD_V2_SINGLE},
                "gf_open": {"empty_when": "commanded open and G ended the arm's last open by its wide-enough rule "
                                          f"(pad gap {GRIP_OPEN_WIDER_M * 1000:.0f} mm wider than at the start)",
                            "until": "the arm's next gripper action"},
                "hold_clearance": {"clearance_m": CLEARANCE_M, "lateral_m": CLEARANCE_LATERAL_M,
                                   "tolerance_m": CLEARANCE_TOL_M, "table_z_base_m": self._sp_table_z,
                                   "lowest_point": "TCP height above the table - TCP height above the table at the "
                                                   "close that started the grasp",
                                   "path": "lift vertically, move across (plan when the move was a plan that stays "
                                           "high enough, else straight), descend vertically",
                                   "scope": "transit move_ee (linear, plan) and move_delta of a holding arm"},
                "drop_stop": {"stop_when": "a watched arm's reading falls to 0 (n <= CLOSED_EMPTY after a reading "
                                           "above it); D's stop wins at the same step", "reason": DROP_STOP_REASON},
                "hold_watch2": {"watch_unless": "n <= CLOSED_EMPTY and the wrist-image estimate (quality ok) shows "
                                                f"a pad gap below {HOLD_LOST_GAP_M * 1000:.1f} mm"}}
        # the verified fast hold's state (nothing of it is used, and no record written, with it off or without H)
        self._sp_r2 = {}                                     # arm -> {state, close, verdict, verified_gap_m, ...}
        self._sp_r3i = bool(self._sp_flags.get("hold_verified_fast"))
        if self._sp_r3i:
            record["r3i"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in R3I},
                "acts_with": {features.FLAGS[name]: [features.FLAGS[n] for n in R3I_NEEDS[name]] for name in R3I},
                "hold_verified_fast": {
                    "verified_when": ("the first lift after the arm's close (D's phase rule: a precision move up >= "
                                      f"{HOLD_LIFT_RISE_M * 1000:.0f} mm) keeps the pad gap from the official reading "
                                      f"within {VERIFY_SHRINK_M * 1000:.1f} mm (first minus last reading after its "
                                      "control steps) and ends with the reading > 0"),
                    "gap_from_reading": {"per_reading_m": VERIFY_GAP_PER_READING_M, "offset_m": VERIFY_GAP_OFFSET_M},
                    "shrink_m": VERIFY_SHRINK_M, "fast_tcp_speed_m_s": FAST_HOLD_TCP_SPEED,
                    "fast_tcp_accel_m_s2": FAST_HOLD_TCP_ACCEL, "other_limits": "H's (rotation, joints, cap safety)",
                    "cleared_by": ["the arm's next executed open", "the arm's next executed close (a new grasp)",
                                   f"a later reading whose pad gap is more than {VERIFY_SHRINK_M * 1000:.1f} mm below "
                                   "the gap at the end of the verifying lift"],
                    "session_file": HOLD_VERIFIED_FILE}}
        # SPIN: the in-place twist (no record written, nothing used, with the switch off)
        self._sp_spin_on = bool(self._sp_flags.get("spin_fast"))
        if self._sp_spin_on:
            record["spin"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in SPIN},
                "which": ("move_ee / move_delta at speed_profile precision (any intent, holding or not), without "
                          "max_samples, not already re-timed, whose whole dense path keeps the TCP within "
                          "tcp_shift_m of its start and the tool axis (TCP z) within axis_change_deg of its start "
                          "direction, and turns by at least min_turn_rad"),
                "tcp_shift_m": SPIN_MAX_SHIFT_M, "axis_change_deg": SPIN_MAX_AXIS_DEG, "min_turn_rad": SPIN_MIN_TURN_RAD,
                "tcp_rotation_rad_s": SPIN_TCP_ROTATION, "joint_speed_rad_s": SPIN_JOINT_SPEED,
                "tcp_speed_m_s": SPIN_TCP_SPEED, "rot_accel_rad_s2": SPIN_ROT_ACCEL,
                "tcp_accel_check_m_s2": SPIN_TCP_ACCEL,
                "law": "raised-cosine s-curve along the path (time at the caps), from rest to rest",
                "cap_safety": CAP_SAFETY, "zones": "none (F's landing and take-off zones are not applied)",
                "used_when": "fewer samples than the worker's precision samples, else those are kept"}
        # YAW: the symmetric-gripper flip (no record written, nothing used, with the switch off)
        self._sp_yaw_on = bool(self._sp_flags.get("yaw_flip"))
        self._sp_yaw_frame = {arm: "requested" for arm in self.arms}
        if self._sp_yaw_on:
            observer = getattr(self, "rgb_observer", None)
            if observer is not None:                         # the observation shows each arm in the agent's frame
                observer.model = AgentFrameModel(observer.model, self._sp_yaw_frame)
            record["yaw_flip"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in YAW},
                "which": ("every move_ee (linear and plan, any intent) of an arm that is empty by F's rule "
                          "(_sp_empty_state); a holding arm keeps its frame"),
                "flip": "the requested rotation R and R Rz(pi) (180 deg about the TCP's own z axis)",
                "distance": YAW_DISTANCE, "min_gain_rad": YAW_FLIP_MIN_GAIN_RAD,
                "skip_flipped_when_requested_linf_at_most_rad": YAW_FLIP_SKIP_RAD,
                "far_rad": YAW_FLIP_FAR_RAD, "orientation_slack_rad": YAW_FLIP_ORIENTATION_SLACK_RAD,
                "only_one_prepared": "that one", "neither": "the refusal of the candidate in the arm's current frame",
                "frames": "home and move_joints: requested; move_delta, gripper, hold: kept; taken over when the stage "
                          "executed a step",
                "reported": ("achieved_tcp_pose_base, the observation's tcp_pose_base and T_base_tcp_by_arm in the "
                             "arm's frame; joints, link poses, camera poses and images physical"),
                "observer": "agent-frame camera model" if observer is not None else "none: nothing is flipped",
                "log": YAW_LOG_FILE}
        # P2X: every precision-segment limit scaled (no record written, nothing used, with the switch off)
        self._sp_pscale = precision_scale(self._sp_flags)
        self._sp_zones = zone_limits(self._sp_pscale)       # None with the switch off: retime_free's own constants
        if self._sp_pscale != 1.0:
            caps = precision_caps(self._sp_pscale)
            record["precision_2x"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in P2X},
                "scale": self._sp_pscale,
                "worker_precision": {"which": "move_ee (linear, plan), move_delta, move_joints at speed_profile "
                                              "precision (migration/worker.py _prepare_from_state with dt x scale)",
                                     "tcp_speed_m_s": caps["tcp"], "tcp_rotation_rad_s": caps["rot"],
                                     "joint_speed_rad_s": caps["joint"],
                                     "recorded_joint_velocity_limit_rad_s": 0.5 * self._sp_pscale},
                "prec_zone_lift_head_caps": caps,
                "spin_tcp_speed_m_s": SPIN_TCP_SPEED * self._sp_pscale,
                "spin_min_turn_rad": SPIN_MIN_TURN_RAD * self._sp_pscale,
                "free_topt_zones": {"landing": [LANDING_HEIGHT_M, LANDING_LENGTH_M, self._sp_zones["zone_v"],
                                                self._sp_zones["zone_w"]],
                                    "takeoff": [TAKEOFF_HEIGHT_M, TAKEOFF_LENGTH_M, self._sp_zones["zone_v"]],
                                    "low_unchanged": [LOW_HEIGHT_M, LOW_TCP_SPEED]},
                "unchanged": "transit, H, F's and H's accelerations, tracking stops, settle, gripper"}
        # PG: the plan guard (no record written, nothing checked, with the switch off)
        self._sp_pg_on = bool(self._sp_flags.get("plan_guard"))
        if self._sp_pg_on:
            record["plan_guard"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in PG},
                "which": ("move_ee and move_delta in plan mode (holding or empty, any intent; every yaw-flip candidate; "
                          "6a's plan across)"),
                "refused_when": "TCP path longer than max_ratio x the straight distance and longer than min_path_m",
                "max_ratio": PLAN_GUARD_RATIO, "min_path_m": PLAN_GUARD_MIN_PATH_M,
                "path": ("the worker's prepared joint samples (start and every commanded sample) through the public "
                         "URDF FK, before any re-timing"),
                "refusal": "the worker's 'Planning rejected' with reason and ours_plan_guard; nothing moves",
                "not_checked": "linear moves, home, move_joints", "log": PLAN_GUARD_LOG_FILE}
        self._sp_roll_on = bool(self._sp_flags.get("prec_roll"))
        self._sp_roll_axis = roll_axis(self._sp_kin) if self._sp_roll_on else None
        if self._sp_roll_on:
            caps = prec_roll_caps(self._sp_pscale)
            record["prec_roll"] = {
                "switches": {features.FLAGS[name]: int(bool(self._sp_flags.get(name))) for name in PR},
                "which": ("move_ee (linear and plan), move_delta and move_joints at speed_profile precision, holding "
                          "or not, without max_samples, not already re-timed by P's lift"),
                "roll_axis": {"joint": self._sp_roll_axis["joint"], "joint_index": self._sp_roll_axis["index"],
                              "axis_in_tcp": [float(v) for v in self._sp_roll_axis["axis"]],
                              "tcp_offset_from_axis_m": self._sp_roll_axis["tcp_offset_m"],
                              "source": ("the public URDF axis of the joint whose child is the ee link, in the TCP "
                                         "frame through T_link6_tcp")},
                "tcp_speed_m_s": caps["tcp"], "swing_rad_s": caps["swing"], "roll_rad_s": caps["roll"],
                "joint_speed_rad_s": caps["joint"], "joint6_speed_rad_s": caps["joint6"],
                "move_joints": "the joint caps only (the worker's move_joints has no TCP cap)",
                "law": ("constant rate along the worker's dense path, sampled at dt: per segment the largest of "
                        "translation / tcp_speed_m_s, swing / swing_rad_s, roll / roll_rad_s (these three planned at "
                        "cap_safety), joints 1-5 (L-infinity) / joint_speed_rad_s and joint 6 / joint6_speed_rad_s; "
                        "every step checked at 100 %"),
                "cap_safety": CAP_SAFETY, "iterations": HOLD_ITERATIONS, "slowdown": RETIME_SLOWDOWN,
                "used_when": ("fewer samples than the worker's precision samples, and than the in-place twist's "
                              "s-curve when that applies; else those are kept")}
        base.atomic(self.session / "speed_profiles.json", record)

    # -- the official observation
    def _sp_observed(self, arm):
        try:
            return gripper_observation(self.env.get_obs()["state"], arm)
        except Exception:                                    # noqa: BLE001 - no observation: not known
            return None

    def _sp_commanded(self, arm):
        return float(self.grips[self.arms[arm]["robot"].arm_name])

    def _sp_tcp_of(self, Q):
        return self._sp_kin.tcp(Q)

    # -- D
    def _sp_hold_phase(self, arm, plan):
        """hold_guard_phase for this arm's prepared plan: the TCP rise from the plan's start to its last scheduled
        sample (public URDF FK, the arm's base frame, z up) for a precision motion."""
        rise = None
        if plan["command"] in ("move_ee", "move_delta", "move_joints") and plan.get("profile") != "transit":
            run = max(1, int(plan.get("run_samples") or len(plan["points"])))
            ends = np.vstack([np.asarray(plan["start"], float)[None],
                              np.asarray(plan["points"][min(run, len(plan["points"])) - 1], float)[None]])
            frames = self._sp_tcp_of(ends)
            rise = float(frames[1, 2, 3] - frames[0, 2, 3])
        return hold_guard_phase(plan["command"], plan.get("profile"), rise)

    # -- G (PART_B)
    def _sp_grip_estimate(self, arm):
        """The wrist-image estimate of this arm's fingers in the latest official observation (G's fallback)."""
        estimate = getattr(self, "_gripper_estimate", None)
        if estimate is None:
            return {"quality": "unavailable", "pad_gap_m": None, "finger_joints_m": None}
        try:
            return estimate(arm)
        except Exception as exc:                              # noqa: BLE001 - no estimate: not known
            return {"quality": "unavailable", "pad_gap_m": None, "finger_joints_m": None,
                    "error": f"{type(exc).__name__}: {exc}"}

    def _sp_grip_reading(self, arm, source):
        if source == "observation":
            value = self._sp_observed(arm)
            return value if finite(value) else None
        if source == "wrist_image":
            return self._sp_grip_estimate(arm)
        return None

    def _sp_grip_rule(self, arm, plan):
        """G's rule for this arm's gripper action, built before the stage changes the command: the source of the
        readings (the official observation, else the wrist-image estimate, else none: 10 samples) and the pad gap at
        the start (for an open)."""
        robot = self.arms[arm]["robot"]
        limits = np.asarray(self.arms[arm]["gripper_limits_m"], float)
        finger = (float(limits[0][0]), float(limits[0][1])) if limits.ndim == 2 else (0.0, 0.044)
        scale = tuple(float(v) for v in robot.gripper_scale)
        move = dict(robot.gripper_move)
        mimic = (float(move["mimic"][1]), float(move["mimic"][2]))
        sign = float(move.get("sign", 1.0))
        action = plan["args"].get("action")
        commanded_open = self._sp_commanded(arm) == 1.0
        reading = self._sp_observed(arm)
        if finite(reading):
            gap = pad_gap(finger_from_reading(reading, commanded_open, scale, finger, sign), mimic)
            return GripRule(action, "observation", reading, gap, scale, finger, mimic, sign, len(plan["points"]))
        estimate = self._sp_grip_estimate(arm)
        if estimate.get("quality") == "ok" and estimate.get("pad_gap_m") is not None:
            return GripRule(action, "wrist_image", estimate, float(estimate["pad_gap_m"]), scale, finger, mimic, sign,
                            len(plan["points"]))
        return GripRule(action, "fixed", estimate, None, scale, finger, mimic, sign, len(plan["points"]))

    # -- the empty/holding decision of F and H (two signals for a closed gripper)
    def _sp_closed_estimate(self, arm, commanded, observed):
        """The wrist-image estimate (_sp_grip_estimate: the arm's wrist image in the latest official observation) for a
        gripper commanded closed whose reading shows the fingers fully closed, else None (the image is not looked at).
        Taken once per arm and observation (the action count names the observation); the preparation that used it
        records it (ours_empty_check, _prepare_from_state). Grip width: also a partial command whose reading equals
        it (at_target)."""
        if partial_command(commanded):
            if not at_target(commanded, observed):
                return None
            cached = self._sp_cached_estimate(arm, observed, commanded)
            used = getattr(self, "_sp_image_used", None)
            if used is None:
                used = self._sp_image_used = {}
            used[arm] = dict(cached["record"])
            return cached["estimate"]
        if commanded != 0.0 or not finite(observed) or observed > CLOSED_EMPTY:
            return None
        cached = self._sp_cached_estimate(arm, observed)
        used = getattr(self, "_sp_image_used", None)
        if used is None:
            used = self._sp_image_used = {}
        used[arm] = dict(cached["record"])
        return cached["estimate"]

    def _sp_cached_estimate(self, arm, observed, commanded=0.0):
        """{estimate, record} of this arm's wrist image in the latest official observation, looked at once per arm and
        observation (the action count names it); for a reading that shows the fingers fully closed (grip width: or
        at a partial command, `commanded`)."""
        count = int(self.env.take_action_cnt[0])
        cache = getattr(self, "_sp_estimates", None)
        if cache is None or cache.get("count") != count:
            cache = self._sp_estimates = {"count": count, "arms": {}}
        if arm not in cache["arms"]:
            estimate = self._sp_grip_estimate(arm)
            empty, check = hold_loss(observed, estimate, commanded)
            cache["arms"][arm] = {"estimate": estimate,
                                  "record": {"observed": float(observed), "image_quality": check["image_quality"],
                                             "image_pad_gap_m": check["image_pad_gap_m"], "empty": bool(empty),
                                             "count": count}}
        return cache["arms"][arm]

    def _sp_empty_state(self, arm):
        """F's (empty, reason) of this arm now: empty_state with the wrist-image estimate when the rule needs it.
        G x F (OURS_GF_OPEN): a commanded-open gripper whose last open G ended by its wide-enough rule is empty too."""
        commanded, observed = self._sp_commanded(arm), self._sp_observed(arm)
        empty, reason = empty_state(commanded, observed, self._sp_closed_estimate(arm, commanded, observed))
        wide = getattr(self, "_sp_gf_open", {}).get(arm) if self._sp_flags.get("gf_open") else None
        if not empty and commanded == 1.0 and wide is not None and finite(observed):
            return True, (f"commanded open, G ended the last open as wide enough ({wide['reason']}; n = "
                          f"{observed:.3f} now): empty (G x F)")
        return empty, reason

    # -- F
    def _sp_free_eligible(self, arm, command, args):
        """(eligible, reason) for F at preparation time."""
        if not self._sp_flags["free_topt"]:
            return False, "switch off"
        if command not in ("home", "move_ee", "move_delta"):
            return False, f"{command} keeps its rate"
        if command != "home" and args.get("speed_profile", "precision") != "transit":
            return False, "precision keeps its rate"
        if "max_samples" in args:
            return False, "a checkpoint (max_samples) keeps its rate"
        empty, reason = self._sp_empty_state(arm)
        return empty, reason

    def _home_time_law(self, arm, start, transit_velocity, args=None):
        """F's time law for an approximate home of this arm (agp_core/approx_home.py), or None (the uniform rule)."""
        eligible, _ = self._sp_free_eligible(arm, "home", args or {})
        if not eligible:
            return None
        return FreeLaw(self._sp_tcp_of, self._sp_table_z[arm], self.dt, transit_velocity * self.dt * 0.6,
                       limits=getattr(self, "_sp_zones", None))           # OURS_PRECISION_2X: the scaled zones

    def _prepare_from_state(self, arm, command, args, start, tcp, transit_velocity):
        if not getattr(self, "_sp_yaw_on", False):
            return self._sp_prepare(arm, command, args, start, tcp, transit_velocity)
        if command == "move_ee":                                              # yaw flip
            return self._sp_yaw_flip(arm, args, start, tcp, transit_velocity)
        plan = self._sp_prepare(arm, command, args, start, tcp, transit_velocity)
        if command in ("home", "move_joints") and isinstance(plan, dict) and plan.get("ok"):
            plan["ours_yaw_frame"] = "requested"                              # the agent gives the joints
        return plan

    # -- YAW: the symmetric-gripper flip
    def _sp_yaw_flip(self, arm, args, start, tcp, transit_velocity):
        """The yaw flip on a move_ee (see the module docstring): an empty arm prepares the requested rotation and the
        one turned 180 deg about the tool axis and keeps the one whose goal is nearer the current joints
        (yaw_flip_choice); a holding arm prepares the request in its current frame. Refusals are returned exactly as the
        worker gives them."""
        frame = self._sp_yaw_frame.get(arm, "requested")
        rotation = args.get("rotation")
        record = {"frame_before": frame, "requested_rotation": rotation, "min_gain_rad": YAW_FLIP_MIN_GAIN_RAD,
                  "distance": YAW_DISTANCE, "mode": args.get("mode", "linear"),
                  "speed_profile": args.get("speed_profile", "precision")}
        if getattr(self, "rgb_observer", None) is None or not isinstance(rotation, dict):
            record["decision"] = ("not applied: no RGB-only observer (the flip would show in the observation)"
                                  if isinstance(rotation, dict) else "not applied: no rotation")
            plan = self._sp_prepare(arm, "move_ee", args, start, tcp, transit_velocity)
            return self._sp_yaw_finish(plan, arm, args, frame, args, record, frame)
        flipped_args = dict(args, rotation=flip_quaternion(rotation))
        empty, why = self._sp_empty_state(arm)              # F's rule (two signals for a closed gripper; G x F)
        check = (getattr(self, "_sp_image_used", None) or {}).get(arm)
        record.update(empty=bool(empty), empty_reason=why, empty_check=dict(check) if check else None)
        if not empty:                                       # holding: no choice, the arm keeps its frame
            use = flipped_args if frame == "flipped" else args
            record["decision"] = "holding: no choice, the arm keeps its frame"
            plan = self._sp_prepare(arm, "move_ee", use, start, tcp, transit_velocity)
            return self._sp_yaw_finish(plan, arm, args, frame, use, record, frame)
        plans = {"requested": self._sp_prepare(arm, "move_ee", args, start, tcp, transit_velocity)}
        candidates = {"requested": yaw_candidate(start, plans["requested"]), "flipped": None}
        requested = candidates["requested"]
        if requested["ok"] and requested["linf_rad"] <= YAW_FLIP_SKIP_RAD:
            chosen = "requested"
            reason = (f"the requested goal changes no joint by more than {YAW_FLIP_SKIP_RAD:.4f} rad (L-inf "
                      f"{requested['linf_rad']:.4f} rad); a goal turned 180 deg about the tool axis is at least "
                      f"{YAW_FLIP_FAR_RAD:.4f} rad from it, so it cannot be closer by {YAW_FLIP_MIN_GAIN_RAD:g} rad: "
                      f"not prepared")
        else:
            plans["flipped"] = self._sp_prepare(arm, "move_ee", flipped_args, start, tcp, transit_velocity)
            candidates["flipped"] = yaw_candidate(start, plans["flipped"])
            chosen, reason = yaw_flip_choice(requested, candidates["flipped"])
            if chosen is None:                              # neither: the refusal in the arm's current frame
                chosen = frame
        for name, rot in (("requested", rotation), ("flipped", flipped_args["rotation"])):
            if candidates[name] is not None:
                candidates[name]["rotation"] = {k: float(rot[k]) for k in "wxyz"}
        record.update(decision="empty: compared", candidates=candidates, reason=reason)
        use = flipped_args if chosen == "flipped" else args
        return self._sp_yaw_finish(plans[chosen], arm, args, frame, use, record, chosen)

    def _sp_yaw_finish(self, plan, arm, args, frame, use, record, chosen):
        """Record and return the executed candidate: a refusal unchanged (nothing is added to a result); a prepared plan
        keeps the agent's args (the records show the request) and carries the frame it puts the arm in."""
        ok = isinstance(plan, dict) and bool(plan.get("ok"))
        record.update(chosen=chosen, executed_rotation={k: float(use["rotation"][k]) for k in "wxyz"}
                      if isinstance(use.get("rotation"), dict) else None, prepared=ok,
                      error=None if ok else (plan or {}).get("error"),
                      frame_if_executed=chosen if ok else frame)
        self._sp_yaw_log(arm, record)
        if not ok:
            return plan
        plan["args"] = args
        plan["ours_yaw_frame"] = chosen
        plan["ours_yaw_flip"] = record
        return plan

    def _sp_yaw_log(self, arm, record):
        try:
            entry = {"arm": arm, "sequence": self.motion_sequence, "count": int(self.env.take_action_cnt[0]),
                     **{k: v for k, v in record.items() if k != "distance"}}
            with (self.session / YAW_LOG_FILE).open("a") as stream:
                stream.write(json.dumps(entry, default=float) + "\n")
        except Exception:                                    # noqa: BLE001 - a failed record never stops the stage
            pass

    def _errors(self, arm, plan):
        """Yaw flip: the achieved pose is reported in the arm's frame (the plan's for a move that sets one, else the
        arm's current frame); the rotation error and target_reached are the worker's, against the executed target."""
        result = super()._errors(arm, plan)
        if not getattr(self, "_sp_yaw_on", False) or not isinstance(result, dict):
            return result
        frame = plan.get("ours_yaw_frame", self._sp_yaw_frame.get(arm)) if isinstance(plan, dict) else None
        if frame == "flipped" and result.get("achieved_tcp_pose_base") is not None:
            result["achieved_tcp_pose_base"] = flip_pose7(result["achieved_tcp_pose_base"])
        return result

    def _sp_prepare(self, arm, command, args, start, tcp, transit_velocity):
        """The preparation of one motion: the worker's, the plan guard (on a plan-mode move), then the held clearance,
        F, H / P's lift and the in-place twist (unchanged by the yaw flip)."""
        used = getattr(self, "_sp_image_used", None)
        if used is None:
            used = self._sp_image_used = {}
        used.pop(arm, None)
        scale = getattr(self, "_sp_pscale", 1.0)
        if (scale != 1.0 and command in PRECISION_MOTIONS
                and args.get("speed_profile", "precision") == "precision"):    # OURS_PRECISION_2X
            plan = self._sp_prepare_precision(arm, command, args, start, tcp, transit_velocity, scale)
        else:
            plan = super()._prepare_from_state(arm, command, args, start, tcp, transit_velocity)
        if getattr(self, "_sp_pg_on", False):                                 # plan guard: the worker's own plan
            plan = self._sp_plan_guard(plan, arm, command, args)
        if self._sp_flags.get("hold_clearance") and plan.get("ok"):          # held clearance: before F and H
            plan = self._sp_clearance(plan, arm, command, args, start, tcp, transit_velocity)
        plan = self._sp_free(plan, arm, command, args, start)
        if (self._sp_part_b and plan.get("ok") and plan.get("ours_free_topt") is None
                and plan.get("approx_home") is None and plan.get("ours_dense_points") is None):
            plan = self._sp_holding(plan, arm, command, args, start, tcp)     # PART_B: H and P's lift
        if getattr(self, "_sp_spin_on", False):                               # the in-place twist
            plan = self._sp_spin(plan, arm, command, args, start)
        if getattr(self, "_sp_roll_on", False):
            plan = self._sp_prec_roll(plan, arm, command, args, start)
        check = used.pop(arm, None)                       # the image behind F's / H's decision
        if check is not None and isinstance(plan, dict):
            plan["ours_empty_check"] = check
        return plan

    def _sp_plan_guard(self, plan, arm, command, args):
        """The plan guard on the worker's prepared plan (see the module docstring): a plan-mode move_ee / move_delta
        whose TCP path is longer than PLAN_GUARD_RATIO x its straight distance and longer than PLAN_GUARD_MIN_PATH_M is
        refused with the worker's 'Planning rejected', the reason and the check; any other plan is returned unchanged (a
        checked one carries its check in ours_plan_guard)."""
        if (command not in ("move_ee", "move_delta") or args.get("mode", "linear") != "plan"
                or not isinstance(plan, dict) or not plan.get("ok")):
            return plan
        check = plan_guard_check(plan["start"], plan["points"], self._sp_tcp_of)
        try:                                                 # the record never stops the preparation
            entry = {"arm": arm, "sequence": self.motion_sequence, "count": int(self.env.take_action_cnt[0]),
                     "command": command, "speed_profile": args.get("speed_profile", "precision"), **check}
            with (self.session / PLAN_GUARD_LOG_FILE).open("a") as stream:
                stream.write(json.dumps(entry, default=float) + "\n")
        except Exception:                                    # noqa: BLE001
            pass
        if not check["refused"]:
            plan["ours_plan_guard"] = check
            return plan
        return {"ok": False, "error": "Planning rejected", "motion_started": False, "motion_executed": False,
                "reason": plan_guard_reason(check), "ours_plan_guard": check}

    def _sp_prepare_precision(self, arm, command, args, start, tcp, transit_velocity, scale):
        """OURS_PRECISION_2X: the worker's own precision preparation (migration/worker.py _prepare_from_state,
        unchanged) with every limit x scale. The worker samples at limit x dt x 0.6 per sample everywhere in that
        preparation (the linear move's Cartesian count, resample_joints, resample_tcp), so it runs with dt x scale and
        dt is restored before anything else reads it (agp_core/worker_ext.py's OURS_FAST_PRECISION method). The plan records
        the effective joint limit and the scale; a refusal is returned unchanged."""
        dt = self.dt
        self.dt = dt * scale
        try:
            plan = super()._prepare_from_state(arm, command, args, start, tcp, transit_velocity)
        finally:
            self.dt = dt
        if isinstance(plan, dict) and plan.get("ok"):
            if plan.get("joint_velocity_limit_rad_s") is not None:
                plan["joint_velocity_limit_rad_s"] = float(plan["joint_velocity_limit_rad_s"]) * scale
            plan["ours_precision_scale"] = scale
        return plan

    def _sp_free(self, plan, arm, command, args, start):
        """F on a prepared plan (with OURS_PRECISION_2X: the scaled zones)."""
        if not self._sp_flags["free_topt"] or not plan.get("ok") or plan.get("ours_free_topt") is not None:
            return plan
        if plan.get("approx_home") is not None:            # ApproxHomeMixin applied (or declined) the law itself
            return plan
        eligible, reason = self._sp_free_eligible(arm, command, args)
        if not eligible:
            plan["ours_free_topt_skipped"] = reason
            return plan
        dense = [np.asarray(q, float) for q in plan["points"]]
        points, info = retime_free(start, dense, self._sp_tcp_of, self._sp_table_z[arm], self.dt,
                                   limits=getattr(self, "_sp_zones", None))    # None: its own constants
        info["empty"] = reason
        if points is None:
            plan["ours_free_topt_skipped"] = info.get("failed")
            plan["ours_free_topt_record"] = info
            return plan
        plan["ours_dense_points"] = dense
        plan["points"] = points
        plan["ours_free_topt"] = info
        return plan

    # -- SPIN: the in-place twist
    def _sp_spin(self, plan, arm, command, args, start):
        """The in-place twist on a prepared plan (see the module docstring): a precision move_ee / move_delta that
        qualifies (spin_check) is re-timed with spin_parts when that takes fewer samples than the worker's own; the
        record of why not (ours_spin_fast_skipped) for every other precision move_ee / move_delta."""
        if not plan.get("ok") or command not in ("move_ee", "move_delta") or plan.get("profile") != "precision":
            return plan
        if "max_samples" in args:
            plan["ours_spin_fast_skipped"] = "a checkpoint (max_samples) keeps its rate"
            return plan
        if plan.get("ours_dense_points") is not None:
            plan["ours_spin_fast_skipped"] = "already re-timed (P's lift)"
            return plan
        dense = [np.asarray(q, float) for q in plan["points"]]
        scale = getattr(self, "_sp_pscale", 1.0)                  # OURS_PRECISION_2X (1.0 with it off)
        qualifies, check = spin_check(start, dense, self._sp_tcp_of, SPIN_MIN_TURN_RAD * scale)
        if not qualifies:
            plan["ours_spin_fast_skipped"] = check["reason"]
            plan["ours_spin_fast_check"] = check
            return plan
        points, info = retime_parts(start, dense, self._sp_tcp_of, spin_parts(scale), self.dt)
        info.update(check=check, precision_samples=len(dense))
        if points is None:
            plan["ours_spin_fast_skipped"] = info.get("failed")
            plan["ours_spin_fast_check"] = info
            return plan
        if len(points) >= len(dense):
            plan["ours_spin_fast_skipped"] = (f"the twist's s-curve needs {len(points)} samples, the precision rate "
                                              f"{len(dense)}: the precision samples are kept")
            plan["ours_spin_fast_check"] = info
            return plan
        plan["ours_dense_points"] = dense
        plan["points"] = points
        plan["ours_spin_fast"] = info
        return plan

    def _sp_prec_roll(self, plan, arm, command, args, start):
        if (not isinstance(plan, dict) or not plan.get("ok") or command not in PRECISION_MOTIONS
                or plan.get("profile") != "precision"):
            return plan
        if "max_samples" in args:
            plan["ours_prec_roll_skipped"] = "a checkpoint (max_samples) keeps its rate"
            return plan
        twist = plan.get("ours_spin_fast")
        if plan.get("ours_dense_points") is not None and twist is None:
            plan["ours_prec_roll_skipped"] = "already re-timed (P's lift)"
            return plan
        dense = [np.asarray(q, float) for q in (plan["ours_dense_points"] if twist is not None else plan["points"])]
        caps = prec_roll_caps(getattr(self, "_sp_pscale", 1.0), joints_only=command == "move_joints")
        points, info = retime_roll(start, dense, self._sp_tcp_of, self._sp_roll_axis["axis"],
                                   self._sp_roll_axis["index"], caps, self.dt)
        current = len(plan["points"])
        info["precision_samples"] = len(dense)
        if twist is not None:
            info["twist_samples"] = current
        if points is None:
            plan["ours_prec_roll_skipped"] = info.get("failed")
            plan["ours_prec_roll_record"] = info
            return plan
        if len(points) >= current:
            plan["ours_prec_roll_skipped"] = (
                f"the precision roll law needs {len(points)} samples, the in-place twist's s-curve {current}: the "
                f"twist is kept" if twist is not None else
                f"the precision roll law needs {len(points)} samples, the precision rate {current}: the precision "
                f"samples are kept")
            plan["ours_prec_roll_record"] = info
            return plan
        if twist is not None:
            plan["ours_spin_fast_skipped"] = (f"the precision roll law needs {len(points)} samples, the twist's "
                                              f"s-curve {current}: the precision roll law is used")
            plan["ours_spin_fast_check"] = plan.pop("ours_spin_fast")
        plan["ours_dense_points"] = dense
        plan["points"] = points
        plan["ours_prec_roll"] = info
        return plan

    # -- the verified fast hold
    def _sp_r2_check(self, arm, observed, where):
        """Clear a verified arm's flag when the reading shows its pad gap more than VERIFY_SHRINK_M below the gap at
        the end of its verifying lift; returns the arm's state ("verified", ... or None without one)."""
        entry = self._sp_r2.get(arm)
        if not entry:
            return None
        if entry.get("state") == "verified" and finite(observed):
            shrink = float(entry["verified_gap_m"]) - reading_gap_m(observed)
            if shrink > VERIFY_SHRINK_M + 1e-12:
                entry.update(state="cleared", cleared={
                    "by": "pad gap shrank", "where": where, "reading": round(float(observed), 6),
                    "shrink_m": round(shrink, 6), "count": int(self.env.take_action_cnt[0]),
                    "sequence": self.motion_sequence})
                self._sp_r2_log([{"arm": arm, "event": "cleared", "state": "cleared", **entry["cleared"]}])
                self._sp_r2_write()
        return entry.get("state")

    def _sp_r2_log(self, events):
        """Append the verified fast hold's state changes to the session's hold_verified.jsonl (never stops a stage)."""
        try:
            with (self.session / "hold_verified.jsonl").open("a") as stream:
                for event in events:
                    stream.write(json.dumps(event, default=float) + "\n")
        except Exception:                                    # noqa: BLE001 - a failed record never stops the stage
            pass

    def _sp_r2_fast(self, arm, observed):
        """(fast, state) for a held transit move of this arm being prepared: the fast law when its grasp is verified
        (after the check against the reading at preparation)."""
        if not self._sp_r3i:
            return False, None
        state = self._sp_r2_check(arm, observed, "preparation")
        return state == "verified", state

    def _sp_r2_end(self, plans, watch, executed, count):
        """The verified fast hold at the end of a stage: the verdict of every first lift that ran (the arm then verified
        or unverified; a lift without a finite reading leaves it waiting), then the gripper actions that executed (a
        close starts a new grasp waiting for its lift, an open ends the grasp); every change goes to the session's
        hold_verified.jsonl and the states to HOLD_VERIFIED_FILE. Returns {arm: the lift's verdict record}."""
        lifts, events = {}, []
        for arm, trace in watch.items():
            verified, verdict = lift_verdict(trace)
            verdict["trace"] = [None if v is None else round(float(v), 6) for v in trace]
            entry = self._sp_r2.setdefault(arm, {"state": "await_lift"})
            if verified is not None:
                entry.update(state="verified" if verified else "unverified",
                             verdict=dict(verdict, count=count, sequence=self.motion_sequence))
                if verified:
                    entry.update(verified_gap_m=verdict["last_gap_m"], verified_reading=verdict["last_reading"])
                events.append({"arm": arm, "event": "lift", "state": entry["state"],
                               **{k: verdict[k] for k in ("first_reading", "last_reading", "shrink_m", "reason")}})
            verdict["state_after"] = entry["state"]
            lifts[arm] = verdict
        if executed > 0:
            for arm, p in plans.items():
                if p["command"] != "gripper":
                    continue
                action = p["args"].get("action")
                before = (self._sp_r2.get(arm) or {}).get("state")
                if action == "close":
                    self._sp_r2[arm] = {"state": "await_lift", "close": {"count": count,
                                                                         "sequence": self.motion_sequence}}
                    events.append({"arm": arm, "event": "close", "state_before": before, "state": "await_lift"})
                elif action == "open" and arm in self._sp_r2:
                    self._sp_r2.pop(arm)
                    events.append({"arm": arm, "event": "open", "state_before": before, "state": None})
                elif action == "width" and arm in self._sp_r2:          # grip width: the grasp changed; no fast
                    self._sp_r2.pop(arm)                                # law until the arm's next close and lift
                    events.append({"arm": arm, "event": "width", "state_before": before, "state": None})
        if events:
            self._sp_r2_log([dict(e, count=count, sequence=self.motion_sequence) for e in events])
            self._sp_r2_write()
        return lifts

    def _sp_r2_write(self):
        """The arms' verified fast hold states for the run (session file HOLD_VERIFIED_FILE; never stops a stage)."""
        try:
            from migration.worker import atomic
            atomic(self.session / HOLD_VERIFIED_FILE, {
                arm: {"verified": e.get("state") == "verified", "state": e.get("state"),
                      "close_count": (e.get("close") or {}).get("count"),
                      "count": int(self.env.take_action_cnt[0]), "sequence": self.motion_sequence}
                for arm, e in sorted(self._sp_r2.items())})
        except Exception:                                    # noqa: BLE001 - a failed record never stops the stage
            pass

    # -- H and P's lift (PART_B)
    def _sp_holding(self, plan, arm, command, args, start, tcp):
        """H for a transit move while holding, P's lift for a precision move straight up while holding; the record of
        why not (ours_hold_profile_skipped, ours_prec_zone_skipped) for the other moves of a holding arm."""
        if command not in ("move_ee", "move_delta"):
            return plan
        profile = args.get("speed_profile", "precision")
        wants_h = profile == "transit" and self._sp_flags.get("hold_profile")
        wants_p = profile == "precision" and self._sp_flags.get("prec_zone")
        if not (wants_h or wants_p):
            return plan
        key = "ours_hold_profile" if wants_h else "ours_prec_zone"
        commanded, observed = self._sp_commanded(arm), self._sp_observed(arm)
        estimate = self._sp_closed_estimate(arm, commanded, observed)     # F's two-signal rule
        if not retime_holding(commanded, observed, estimate):
            plan[key + "_skipped"] = ("not holding: " + empty_state(commanded, observed, estimate)[1]
                                      if observed is not None else "not holding: no gripper observation")
            return plan
        if "max_samples" in args:
            plan[key + "_skipped"] = "a checkpoint (max_samples) keeps its rate"
            return plan
        dense = [np.asarray(q, float) for q in plan["points"]]
        r2 = None
        if wants_h:
            each = plan.get("ours_clearance_parts")

            def h_timing(parts_of):
                if each:                                          # held clearance: each part of the path at rest ends
                    return self._sp_retime_parts_each(start, dense, each, parts_of)
                return retime_parts(start, dense, self._sp_tcp_of, parts_of(), self.dt)
            points, info = h_timing(hold_parts)
            if self._sp_r3i:                                      # verified fast hold: a verified grasp: fast law
                fast, state = self._sp_r2_fast(arm, observed)
                r2 = {"state": state, "fast": False, "samples_hold": None if points is None else len(points)}
                plan["ours_hold_verified"] = r2
                if fast:
                    fast_points, fast_info = h_timing(hold_parts_fast)
                    if fast_points is not None:
                        points, info = fast_points, fast_info
                        r2["fast"] = True
                    else:                                         # no admissible fast timing: H's (record why)
                        r2["fast_failed"] = fast_info.get("failed")
        else:
            from agp_core import prec_zone
            if command == "move_ee" and args.get("mode", "linear") != "linear":
                plan[key + "_skipped"] = "a plan-mode move is not a lift"
                return plan
            target, tcp = np.asarray(plan["target"], float), np.asarray(tcp, float)
            rel = tcp[:3, :3].T @ target[:3, :3]
            turn = float(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0)))
            geometry = prec_zone.descent_geometry(tcp[:3, 3].tolist(), target[:3, 3].tolist(), turn)
            rise = -geometry["descent_m"]
            why = None
            if rise < prec_zone.LIFT_HEAD_M + prec_zone.MIN_FAST_M:
                why = (f"rises {rise * 1000:.1f} mm: a lift needs at least "
                       f"{(prec_zone.LIFT_HEAD_M + prec_zone.MIN_FAST_M) * 1000:.0f} mm")
            elif geometry["horizontal_m"] > prec_zone.VERTICAL_XY_M:
                why = (f"not straight up: the target is {geometry['horizontal_m'] * 1000:.1f} mm away horizontally "
                       f"(at most {prec_zone.VERTICAL_XY_M * 1000:.0f} mm)")
            elif geometry["turn_rad"] > prec_zone.MAX_TURN_RAD:
                why = f"turns the wrist by {geometry['turn_rad']:.3f} rad (at most {prec_zone.MAX_TURN_RAD} rad)"
            if why:
                plan[key + "_skipped"] = why
                return plan
            points, info = retime_parts(start, dense, self._sp_tcp_of,
                                        lift_parts(bool(self._sp_flags.get("hold_profile")),
                                                   getattr(self, "_sp_pscale", 1.0)), self.dt)
            info.update(kind="lift", holding=f"n = {observed:.4f}", rise_m=round(rise, 6),
                        horizontal_m=round(geometry["horizontal_m"], 6), turn_rad=round(geometry["turn_rad"], 6))
        if points is None:
            plan[key + "_skipped"] = info.get("failed")
            plan[key + "_record"] = info
            return plan
        info["holding"] = (f"n = {observed:.4f}" if observed > CLOSED_EMPTY else      # n = 0: the image's verdict
                           f"n = {observed:.4f}; " + empty_state(commanded, observed, estimate)[1])
        if wants_h and plan.get("ours_clearance_parts"):          # held clearance: the samples each part takes under H
            for part, timed in zip((plan.get("ours_hold_clearance") or {}).get("parts") or [], info["segments"]):
                part["samples_timed"] = int(timed.get("samples") or 0)
        if r2 is not None:                                        # verified fast hold: samples of the applying law
            r2["samples"] = len(points)
        plan["ours_dense_points"] = dense
        plan["points"] = points
        plan[key] = info
        return plan

    # -- held clearance (OURS_HOLD_CLEARANCE)
    def _sp_retime_parts_each(self, start, dense, ends, parts_of=hold_parts):
        """H on a held clearance path: every part (dense[a:b] for the part ends `ends`) re-timed by itself from rest to
        rest with H's s-curve (parts_of: hold_parts, or hold_parts_fast for a verified grasp); (samples, info)
        or (None, info) when a part has no admissible timing."""
        samples, parts, a = [], [], 0
        for b in ends:
            first = np.asarray(start, float) if a == 0 else dense[a - 1]
            points, info = retime_parts(first, dense[a:b], self._sp_tcp_of, parts_of(), self.dt)
            parts.append(info)
            if points is None:
                return None, {"law": "parts each", "failed": info.get("failed"), "segments": parts}
            samples += points
            a = b
        out = {"law": "parts each (6a clearance path)", "segments": parts, "samples": len(samples),
               "dense_samples": len(dense), "tcp_path_m": round(sum(float(p.get("tcp_path_m") or 0.0) for p in parts), 6)}
        for key in ("max_tcp_speed", "max_tcp_rotation", "max_joint_speed", "max_tcp_accel"):
            out[key] = max(float(p.get(key) or 0.0) for p in parts)
        return samples, out

    def _sp_clearance(self, plan, arm, command, args, start, tcp, transit_velocity):
        """The held clearance (OURS_HOLD_CLEARANCE) on a prepared plan (see the module docstring): the plan unchanged
        (its record in ours_hold_clearance), the clearance path (points of the parts, ours_clearance_parts, a note for
        the model), or a refusal with the reason."""
        if command not in ("move_ee", "move_delta") or args.get("speed_profile", "precision") != "transit":
            return plan
        from scipy.spatial.transform import Rotation
        record = {"applied": False}
        plan["ours_hold_clearance"] = record
        commanded, observed = self._sp_commanded(arm), self._sp_observed(arm)
        estimate = self._sp_closed_estimate(arm, commanded, observed)
        empty, why = empty_state(commanded, observed, estimate)
        closed = commanded == 0.0 or partial_command(commanded)     # grip width: a partial command may hold too
        if not closed or empty:
            record["reason"] = "not holding: " + ("gripper commanded open" if not closed else why)
            return plan
        table = float(self._sp_table_z[arm])
        tcp = np.asarray(tcp, float)
        target = np.asarray(plan["target"], float)
        horizontal = float(np.linalg.norm(target[:2, 3] - tcp[:2, 3]))
        record.update(holding=why, horizontal_m=round(horizontal, 6))
        if horizontal <= CLEARANCE_LATERAL_M:
            record["reason"] = (f"vertical: the target is {horizontal * 1000:.1f} mm away horizontally (at most "
                                f"{CLEARANCE_LATERAL_M * 1000:.0f} mm)")
            return plan
        close = getattr(self, "_sp_close_tcp", {}).get(arm)
        if close is None:
            record["reason"] = "no close of this arm recorded: the grasp height is unknown"
            return plan
        need = float(close["tcp_height_m"]) + CLEARANCE_M
        start_h, end_h = float(tcp[2, 3] - table), float(target[2, 3] - table)
        mode = args.get("mode", "linear") if command == "move_ee" else "linear"
        path = np.vstack([np.asarray(start, float)[None], np.asarray(plan["points"], float)])
        path_min = float(np.min(self._sp_tcp_of(path)[:, 2, 3]) - table)
        low_start = start_h < need - CLEARANCE_TOL_M
        low_end = end_h < need - CLEARANCE_TOL_M
        dips = mode == "plan" and path_min < need - CLEARANCE_TOL_M
        record.update(close=dict(close), need_tcp_height_m=round(need, 6), start_height_m=round(start_h, 6),
                      end_height_m=round(end_h, 6), mode=mode, path_min_height_m=round(path_min, 6),
                      low_start=bool(low_start), low_end=bool(low_end), plan_dips=bool(dips))
        if not (low_start or low_end or dips):
            record["reason"] = "clear: the held object stays high enough on this path"
            return plan
        base_prepare = super()._prepare_from_state
        quat = lambda R: dict(zip("wxyz", np.roll(Rotation.from_matrix(R).as_quat(), 1).tolist()))   # noqa: E731
        z_a = table + (need if low_start else start_h)
        z_b = table + (need if low_end else end_h)
        lift_to = np.r_[tcp[:2, 3], z_a]
        across_to = np.r_[target[:2, 3], z_b]
        parts, joints, frame = [], np.asarray(start, float), tcp

        def prepare(name, position, rotation, part_mode):
            seg_args = {"position": dict(zip("xyz", [float(v) for v in position])), "rotation": quat(rotation),
                        "mode": part_mode, "speed_profile": "transit"}
            seg = base_prepare(arm, "move_ee", seg_args, joints, frame, transit_velocity)
            if part_mode == "plan" and getattr(self, "_sp_pg_on", False):   # plan guard: a swinging plan across
                seg = self._sp_plan_guard(seg, arm, "move_ee", seg_args)      # is not used (the straight path is)
                record["across_plan_guard"] = seg.get("ours_plan_guard")
            return seg

        def refuse(part, error):
            record.update(applied=False, refused=part, error=str(error))
            return {"ok": False, "motion_started": False, "motion_executed": False, "ours_hold_clearance": record,
                    "error": CLEARANCE_REFUSAL.format(clear=CLEARANCE_M * 100, need=need * 100,
                                                      close=float(close["tcp_height_m"]) * 100, part=part,
                                                      error=str(error))}

        def advance(name, seg):
            nonlocal joints, frame
            parts.append((name, seg))
            joints = np.asarray(seg["points"][-1], float)
            frame = np.asarray(self._predicted_tcp(arm, joints), float)

        if low_start:
            seg = prepare("lift", lift_to, tcp[:3, :3], "linear")
            if not seg.get("ok"):
                return refuse("lift", seg.get("error"))
            advance("lift", seg)
        across, across_mode, planned_dip = None, "linear", None
        if mode == "plan" and not (low_start or low_end):
            planned_dip = path_min                       # the move's own plan dipped: straight across instead
        elif mode == "plan":                             # a plan between the raised ends, kept when it stays high
            seg = prepare("move across", across_to, target[:3, :3], "plan")
            if seg.get("ok"):
                heights = self._sp_tcp_of(np.vstack([joints[None], np.asarray(seg["points"], float)]))[:, 2, 3] - table
                planned_dip = float(heights.min())
                if planned_dip >= need - CLEARANCE_TOL_M:
                    across, across_mode = seg, "plan"
        if across is None:
            across = prepare("move across", across_to, target[:3, :3], "linear")
            if not across.get("ok"):
                error = across.get("error")
                if planned_dip is not None:
                    error = (f"the planned path dips to {planned_dip * 100:.1f} cm and a straight path is not "
                             f"reachable: {error}")
                return refuse("move across", error)
        advance("move across", across)
        if low_end:
            seg = prepare("descent", target[:3, 3], target[:3, :3], "linear")
            if not seg.get("ok"):
                return refuse("descent", seg.get("error"))
            advance("descent", seg)
        points, ends = [], []
        for _, seg in parts:
            points += list(seg["points"])
            ends.append(len(points))
        lift_m = (need - start_h) if low_start else 0.0
        descent_m = (z_b - table - end_h) if low_end else 0.0
        record.update(applied=True, reason="clearance path", lift_m=round(lift_m, 6), descent_m=round(descent_m, 6),
                      across_height_m=[round(z_a - table, 6), round(z_b - table, 6)], across_mode=across_mode,
                      planned_dip_m=None if planned_dip is None else round(planned_dip, 6),
                      parts=[{"name": name, "samples": len(seg["points"])} for name, seg in parts],
                      original_samples=len(plan["points"]), samples=len(points))
        out = dict(plan, points=points, raw_points=across.get("raw_points") if across_mode == "plan" else None,
                   ours_clearance_parts=ends, ours_hold_clearance=record)
        note = clearance_note(lift_m, descent_m)
        if note:
            out["ours_clearance_note"] = note
        return out

    def _preflight_stage(self, plans):
        """The preflight checks the dense path of a re-timed motion (as before the re-timing)."""
        swapped = {}
        for arm, p in plans.items():
            dense = p.get("ours_dense_points")
            if dense is not None:
                swapped[arm] = (p["points"], p["run_samples"])
                delay = int(p.get("ours_guard_delay") or 0)
                p["points"] = [p["start"]] * delay + list(dense)
                p["run_samples"] = len(p["points"])
        try:
            return super()._preflight_stage(plans)
        finally:
            for arm, (points, run) in swapped.items():
                plans[arm]["points"], plans[arm]["run_samples"] = points, run

    # -- F-g layer 1
    def _prepare_stage(self, actions, starts=None):
        prepared = super()._prepare_stage(actions, starts)
        if starts is not None or not prepared.get("ok") or self._sp_guard is None:
            return prepared
        verdict = self._sp_guard_stage(prepared["plans"])
        self._sp_log_guard(verdict)
        if not verdict["ok"]:
            return {"ok": False, "error": verdict["error"], "ours_arm_guard": verdict}
        prepared["ours_arm_guard"] = verdict
        return prepared

    def _sp_stage_state(self, plans):
        """Fingers and held flags of both arms for this stage (the gripper command of the stage included)."""
        from agp_core import arm_guard
        fingers, held = {}, {}
        for arm, context in self.arms.items():
            commanded = self._sp_commanded(arm)
            observed = self._sp_observed(arm)
            plan = plans.get(arm)
            if plan is not None and plan["command"] == "gripper":
                commanded = 1.0 if plan["args"].get("action") == "open" else commanded
            limits = np.asarray(context["gripper_limits_m"], float)
            fingers[arm] = arm_guard.finger_opening(observed, commanded == 1.0,
                                                    (float(limits[0][0]), float(limits[0][1]))
                                                    if limits.ndim == 2 else (0.0, 0.044))
            held[arm] = holding_state(self._sp_commanded(arm), observed)
        return fingers, held

    def _sp_guard_stage(self, plans):
        """Layer 1 for the prepared plans (guard_stage); a chosen delay is applied to the delayed arm's plan."""
        measured = {arm: np.asarray(self.rm.get_joint(c["robot"])[0], float) for arm, c in self.arms.items()}
        fingers, held = self._sp_stage_state(plans)
        if self._sp_flags.get("guard_v2"):                   # v2: 15 mm, delays up to the other arm's stage
            record = guard_stage(self._sp_guard, tuple(self.arms), plans, measured, fingers, held,
                                 margin=GUARD_V2_MARGIN_M, v2=True)
        else:
            record = guard_stage(self._sp_guard, tuple(self.arms), plans, measured, fingers, held)
        record["sequence"] = self.motion_sequence
        if record.get("checked"):
            self._sp_stage = {"start_clearance_m": record["start_clearance_m"], "fingers": fingers, "held": held}
        delay = record.get("delay")
        if record["ok"] and delay:
            plan = plans[delay["arm"]]
            plan["ours_guard_delay"] = delay["steps"]
            hold = [np.asarray(plan["start"], float).copy() for _ in range(delay["steps"])]
            plan["points"] = hold + list(plan["points"])
            plan["full_samples"] += delay["steps"]
            plan["run_samples"] += delay["steps"]
        return record

    def _sp_log_guard(self, record):
        try:
            entry = {k: v for k, v in record.items() if k not in ("fingers_m",)}
            entry["count"] = int(self.env.take_action_cnt[0])
            with (self.session / "arm_guard.jsonl").open("a") as stream:
                stream.write(json.dumps(entry, default=float) + "\n")
        except Exception:                                    # noqa: BLE001 - a failed record never stops the stage
            pass

    # -- F-g layer 2
    def _sp_guard_step(self, targets):
        """None, or the stop reason when the measured joints plus the targets about to be sent come too close."""
        stage = self._sp_stage or {}
        measured = {arm: np.asarray(self.rm.get_joint(c["robot"])[0], float) for arm, c in self.arms.items()}
        fractions = np.arange(0, INTERP_SUBSTEPS + 1) / INTERP_SUBSTEPS
        timeline = {arm: np.asarray([measured[arm] + f * (np.asarray(targets[arm], float) - measured[arm])
                                     for f in fractions]) for arm in self.arms}
        per_time, detail = self._sp_guard.clearance(timeline, stage.get("fingers"), stage.get("held") or {},
                                                    threshold=GUARD_HARD_M + GUARD_EXACT_ABOVE_M)
        threshold = min(GUARD_HARD_M, float(stage.get("start_clearance_m", np.inf)) - GUARD_START_TOLERANCE_M)
        if float(per_time.min()) >= threshold:
            return None, float(per_time.min())
        names = " and ".join(f"{arm} {name}" for arm, name in sorted(detail["links"].items()))
        return (f"two-arm clearance below {threshold * 1000:.0f} mm ({names}: {detail['distance_m'] * 1000:.1f} mm "
                f"between the measured joints and the next targets): both arms stopped", detail), float(per_time.min())

    # -- the control loop (migration/worker.py execute_actions with the hooks of F, F-g and D)
    def execute_actions(self, actions):
        from migration.worker import atomic
        reject = lambda error, **extra: dict(ok=False, error=error, motion_started=False,  # noqa: E731
                                             motion_executed=False, low_level_actions=0, **extra)
        if not actions or set(actions)-set(self.arms):
            return reject('Expected one or both explicit arm commands')
        if self.env.is_episode_end():
            return reject('Episode ended')
        self.motion_sequence += 1
        self._sp_stage = None
        prepared = self._prepare_stage(actions)
        if not prepared['ok']:
            return reject(prepared['error'], **{k: v for k, v in prepared.items() if k not in {'ok', 'error'}})
        plans = prepared['plans']
        guard_verdict = prepared.get('ours_arm_guard')
        count = int(self.env.take_action_cnt[0])
        remaining = max(0, self.env.step_lim-count)
        scheduled = max(p['run_samples'] for p in plans.values())
        directory = self.session/'motion_diagnostics'
        directory.mkdir(exist_ok=True)
        diagnostics = {}
        prefixes = {}
        # this stage's hooks
        track_limit = {arm: (FREE_TRACKING_STOP_RAD if (plans.get(arm) or {}).get('ours_free_topt') else .08)
                       for arm in self.arms}
        holding = {}                                         # D: the watched arms (held, lift or transport)
        hold_state = {}
        if self._sp_flags['hold_guard']:
            watch2 = bool(self._sp_flags.get('hold_watch2'))
            for arm, p in plans.items():
                observed = self._sp_observed(arm)
                width = partial_command(self._sp_commanded(arm))     # grip width: the reading alone cannot show
                if watch2 or width:                          # an object near a partial target: the two signals
                    # watch2: watch unless both signals show it empty (F's rule)
                    commanded, start_check = self._sp_commanded(arm), None
                    if (commanded == 0.0 or width) and at_target(commanded, observed):
                        cached = self._sp_cached_estimate(arm, observed, commanded)
                        start_check = dict(cached['record'])
                        empty, why = empty_state(commanded, observed, cached['estimate'])
                    else:
                        empty, why = empty_state(commanded, observed)
                    holding_now = (commanded == 0.0 or width) and not empty
                    rule = {'start_rule': 'two signals (R2A 7)' if not width else
                            'two signals (grip width: the reading is the partial command itself while the fingers are '
                            'within 10.8 mm of its target)', 'start_reason': why, 'start_check': start_check}
                else:
                    holding_now, rule = holding_state(self._sp_commanded(arm), observed), {}
                if not holding_now:
                    hold_state[arm] = {'holding_at_start': False, 'observed_at_start': None, 'watched': False,
                                       'phase': None, 'reason': 'not holding', **rule}
                    continue
                watched, phase, reason = self._sp_hold_phase(arm, p)
                hold_state[arm] = {'holding_at_start': True, 'observed_at_start': observed, 'watched': watched,
                                   'phase': phase, 'reason': reason, **rule}
                if watched:
                    holding[arm] = observed
        guarded = (self._sp_guard is not None and guard_verdict is not None and guard_verdict.get('checked'))
        r2_watch, r2_start = {}, {}                          # verified fast hold: arm -> readings of its first lift
        if self._sp_r3i:
            r2_start = {arm: (self._sp_r2.get(arm) or {}).get('state') for arm in self.arms}
            for arm, p in plans.items():
                if (r2_start.get(arm) == 'await_lift' and p['command'] in MOTIONS
                        and self._sp_commanded(arm) == 0.0 and self._sp_hold_phase(arm, p)[1] == 'lift'):
                    r2_watch[arm] = []
        grip_rules = {}                                      # G (PART_B): the end rule of every gripper arm
        if self._sp_flags.get('grip_obs'):
            for arm, p in plans.items():
                if p['command'] == 'gripper':
                    grip_rules[arm] = self._sp_grip_rule(arm, p)
        for arm, p in plans.items():
            delta = np.diff(np.vstack([p['start'], p['points']]), axis=0)
            d = {'arm': arm, 'command': p['command'], 'args': p['args'], 'profile': p['profile'],
                 'action_count_before': count, 'dt_s': self.dt,
                 'joint_velocity_limit_rad_s': p.get('joint_velocity_limit_rad_s'),
                 'planner_input_samples': len(p['raw_points']) if p['raw_points'] is not None else None,
                 'output_samples': p['full_samples'], 'scheduled_samples': p['run_samples'],
                 'joint_path_length_rad': np.sum(np.abs(delta), axis=0).tolist(),
                 'linf_path_length_rad': float(np.max(np.abs(delta), axis=1).sum()),
                 'max_command_velocity_rad_s': float(np.max(np.abs(delta))/self.dt),
                 'max_command_acceleration_rad_s2': float(np.max(np.abs(np.diff(
                     np.vstack([np.zeros_like(p['start']), delta, np.zeros_like(p['start'])]), axis=0)))/self.dt**2),
                 'expected_duration_s': scheduled*self.dt}
            d['ours_speed'] = {
                'free_topt': p.get('ours_free_topt'), 'free_topt_skipped': p.get('ours_free_topt_skipped'),
                'tracking_stop_rad': track_limit[arm], 'guard_delay_steps': int(p.get('ours_guard_delay') or 0),
                'arm_guard': ({k: v for k, v in guard_verdict.items() if k in (
                    'checked', 'min_clearance_m', 'start_clearance_m', 'threshold_m', 'closest', 'delay',
                    'step_of_min')} if guard_verdict else None),
                'hold_guard': (hold_state.get(arm) if self._sp_flags['hold_guard'] else None)}
            if self._sp_part_b:                              # PART_B records only with a PART_B switch on
                d['ours_speed'].update(
                    hold_profile=p.get('ours_hold_profile'), hold_profile_skipped=p.get('ours_hold_profile_skipped'),
                    prec_zone=p.get('ours_prec_zone'), prec_zone_skipped=p.get('ours_prec_zone_skipped'),
                    grip_obs=None)
            if p.get('ours_empty_check') is not None:        # only when the image was looked at
                d['ours_speed']['empty_check'] = p['ours_empty_check']
            if self._sp_flags.get('hold_clearance') and p['command'] in ('move_ee', 'move_delta'):   # held clearance
                d['ours_speed']['hold_clearance'] = p.get('ours_hold_clearance')
            if self._sp_spin_on and p['command'] in ('move_ee', 'move_delta') and p.get('profile') == 'precision':
                d['ours_speed'].update(spin_fast=p.get('ours_spin_fast'),           # in-place twist
                                       spin_fast_skipped=p.get('ours_spin_fast_skipped'),
                                       spin_fast_check=p.get('ours_spin_fast_check'))
            if self._sp_yaw_on:                              # yaw flip: the choice (move_ee) and the arm's frame
                if p['command'] == 'move_ee':
                    d['ours_speed']['yaw_flip'] = p.get('ours_yaw_flip')
                d['ours_speed']['yaw_frame'] = {'before': self._sp_yaw_frame.get(arm),
                                                'plan': p.get('ours_yaw_frame', self._sp_yaw_frame.get(arm))}
            if getattr(self, '_sp_pscale', 1.0) != 1.0:      # OURS_PRECISION_2X: the scale this preparation used
                d['ours_speed']['precision_scale'] = p.get('ours_precision_scale')
            if getattr(self, '_sp_pg_on', False) and p['command'] in ('move_ee', 'move_delta'):   # plan guard
                d['ours_speed']['plan_guard'] = p.get('ours_plan_guard')
            if (getattr(self, '_sp_roll_on', False) and p['command'] in PRECISION_MOTIONS
                    and p.get('profile') == 'precision'):
                d['ours_speed'].update(prec_roll=p.get('ours_prec_roll'),
                                       prec_roll_skipped=p.get('ours_prec_roll_skipped'),
                                       prec_roll_record=p.get('ours_prec_roll_record'))
            prefix = directory/f"{count:04d}_{self.motion_sequence:04d}_{arm}_{p['command']}"
            prefixes[arm] = prefix
            diagnostics[arm] = d
            arrays = dict(start=p['start'],
                          planned=p['raw_points'] if p['raw_points'] is not None else np.empty((0, len(p['start']))),
                          commanded=np.asarray(p['points']))
            if p.get('ours_dense_points') is not None:
                arrays['ours_dense'] = np.asarray(p['ours_dense_points'])
            np.savez_compressed(prefix.with_suffix('.npz'), **arrays)
        if scheduled > remaining:
            for arm, d in diagnostics.items():
                d.update(executed_samples=0, stop_reason='insufficient action budget')
                atomic(prefixes[arm].with_suffix('.json'), d)
            return reject('Planned motion exceeds remaining action budget',
                          planned_samples=scheduled, remaining_actions=int(remaining))
        # Validate FINAL resampled execution prefixes, after both preparations and
        # before either gripper state or physics changes. Holds and gripper-only
        # commands keep measured joints and preserve intended contact semantics.
        preflight = self._preflight_stage(plans)
        for arm, result in preflight.items():
            diagnostics[arm]['preflight'] = result
        rejected = [arm for arm, result in preflight.items() if not result['ok']]
        if rejected:
            for arm, d in diagnostics.items():
                d.update(executed_samples=0, stop_reason='joint trajectory preflight rejected')
                atomic(prefixes[arm].with_suffix('.json'), d)
            return reject('Joint trajectory preflight rejected', rejected_arm=rejected[0],
                          arms={arm: {'ok': result['ok'], 'preflight': result,
                                      'motion_started': False, 'motion_executed': False}
                                for arm, result in preflight.items()})
        starts = {arm: self.rm.get_joint(ctx['robot'])[0].copy() for arm, ctx in self.arms.items()}
        feedback = {arm: [q.copy()] for arm, q in starts.items()}
        sent = {arm: [] for arm in self.arms}
        prospective_grips = dict(self.grips)
        for arm, p in plans.items():
            if p['command'] == 'gripper':
                prospective_grips[self.arms[arm]['robot'].arm_name] = p['gripper_command']
        self.grips = prospective_grips
        sim_start = float(self.env.sim.sim.current_time)
        wall_start = time.monotonic()
        stop_reason = None
        faults = {}
        trajectory_samples = 0
        settle_samples = 0
        last = {arm: q.copy() for arm, q in starts.items()}
        guard_events, hold_events = [], []
        min_step_clearance = [math.inf]

        def guard_ok(targets):
            nonlocal stop_reason
            if not guarded:
                return True
            stop, clearance = self._sp_guard_step(targets)
            min_step_clearance[0] = min(min_step_clearance[0], clearance)
            if stop is None:
                return True
            reason, detail = stop
            stop_reason = reason
            guard_events.append({'count': int(self.env.take_action_cnt[0]), 'reason': reason, 'closest': detail})
            return False

        hold_trace = {arm: [] for arm in holding}
        hold_checks = {arm: [] for arm in holding}           # D: the image looked at when the reading showed n = 0
        planned_end = scheduled                              # G may end a stage after fewer trajectory samples

        def grip_done():
            for arm, rule in grip_rules.items():
                rule.observe(self._sp_grip_reading(arm, rule.source))
            if any(rule.ended_at is None for rule in grip_rules.values()):
                return False
            return all(p['run_samples'] <= trajectory_samples for arm, p in plans.items() if arm not in grip_rules)

        drop_stop = bool(self._sp_flags.get('drop_stop'))  # drop stop (OURS_DROP_STOP)
        last_reading = dict(holding)                         # the reading before this step (from the stage start)
        drop_events = []

        def d_lost(arm, observed, check):
            nonlocal stop_reason
            stop_reason = (f"{arm} held object lost: the official gripper observation shows the fingers "
                           f"fully closed and the wrist image shows a pad gap of "
                           f"{check['image_pad_gap_m'] * 1000:.2f} mm (below {HOLD_LOST_GAP_M * 1000:.1f} mm)")
            if partial_command(self._sp_commanded(arm)):    # grip width: the fingers at the commanded width
                stop_reason = (f"{arm} held object lost: the official gripper observation shows the fingers at the "
                               f"commanded width and the wrist image shows a pad gap of "
                               f"{check['image_pad_gap_m'] * 1000:.2f} mm (below "
                               f"{lost_bound_m(self._sp_commanded(arm)) * 1000:.2f} mm)")
            hold_events.append({'arm': arm, 'count': check['count'], 'observed_at_start': holding[arm],
                                'observed': observed, 'image_quality': check['image_quality'],
                                'image_pad_gap_m': check['image_pad_gap_m'],
                                'phase': (hold_state.get(arm) or {}).get('phase')})

        def held_ok():
            nonlocal stop_reason
            if drop_stop:                                    # drop stop: every watched arm, D's stop first
                lost_at, fell_at = None, None
                for arm in holding:
                    observed = self._sp_observed(arm)
                    hold_trace[arm].append(observed)
                    before = last_reading.get(arm)
                    if finite(observed):
                        last_reading[arm] = observed
                    commanded = self._sp_commanded(arm)      # grip width: at_target / hold_loss at a partial command
                    if observed is None or not finite(observed) or not at_target(commanded, observed):
                        continue
                    lost, check = hold_loss(observed, self._sp_grip_estimate(arm), commanded)
                    check['count'] = int(self.env.take_action_cnt[0])
                    hold_checks[arm].append(check)
                    if lost and lost_at is None:
                        lost_at = (arm, observed, check)
                    elif not lost and fell_at is None and finite(before) and not at_target(commanded, before):
                        fell_at = (arm, observed, check, before)
                if lost_at is not None:
                    d_lost(*lost_at)
                    return False
                if fell_at is not None:
                    arm, observed, check, before = fell_at
                    stop_reason = DROP_STOP_REASON.format(arm=arm)
                    drop_events.append({'arm': arm, 'count': check['count'], 'observed_before': before,
                                        'observed': observed, 'observed_at_start': holding[arm],
                                        'image_quality': check['image_quality'],
                                        'image_pad_gap_m': check['image_pad_gap_m'],
                                        'phase': (hold_state.get(arm) or {}).get('phase')})
                    return False
                return True
            for arm in holding:
                observed = self._sp_observed(arm)
                hold_trace[arm].append(observed)
                commanded = self._sp_commanded(arm)          # grip width: at_target / hold_loss at a partial command
                if observed is None or not finite(observed) or not at_target(commanded, observed):
                    continue                                 # D's first signal absent: the image is not looked at
                lost, check = hold_loss(observed, self._sp_grip_estimate(arm), commanded)
                check['count'] = int(self.env.take_action_cnt[0])
                hold_checks[arm].append(check)
                if lost:
                    d_lost(arm, observed, check)
                    return False
            return True

        def r2_step():                                       # verified fast hold: after a control step, at D's traces
            if not self._sp_r3i:
                return
            for arm in self.arms:
                if arm in r2_watch:                          # the first lift after the close: its readings
                    r2_watch[arm].append(self._sp_observed(arm))
                elif (self._sp_r2.get(arm) or {}).get('state') == 'verified':
                    self._sp_r2_check(arm, self._sp_observed(arm), 'control step')

        def sample(targets):
            nonlocal stop_reason
            if self.env.is_episode_end():
                stop_reason = ('official action budget exhausted' if self.env.take_action_cnt[0] >= self.env.step_lim
                               else 'official episode ended')
                return False
            self.env.take_action(self.action_dict(targets))
            for arm, ctx in self.arms.items():
                q = self.rm.get_joint(ctx['robot'])[0].copy()
                feedback[arm].append(q)
                sent[arm].append(np.asarray(targets[arm]).copy())
                if np.max(np.abs(q-targets[arm])) > track_limit[arm]:
                    faults[arm] = f'{arm} joint tracking error exceeded {track_limit[arm]:g} rad'
                    stop_reason = faults[arm]
            if not self.env.end_flag[0]:
                self.env.get_obs()
            return True
        for i in range(scheduled):
            for arm, p in plans.items():
                last[arm] = p['points'][min(i, p['run_samples']-1)]
            if not guard_ok(last):
                break
            if not sample(last):
                break
            trajectory_samples += 1
            if stop_reason:
                break
            r2_step()
            if not held_ok():
                break
            if grip_rules and grip_done():
                planned_end = trajectory_samples
                break
        for arm, rule in grip_rules.items():                 # G: an ended action counts its own samples only
            if rule.ended_at is not None:
                plans[arm]['full_samples'] = plans[arm]['run_samples'] = rule.ended_at
        if self._sp_flags.get('gf_open'):                    # G x F: an open G ended as wide enough, until the
            for arm, p in plans.items():                     # arm's next gripper action
                if p['command'] != 'gripper':
                    continue
                rule = grip_rules.get(arm)
                if p['args'].get('action') == 'open' and rule is not None and rule.ended_by == 'wider':
                    self._sp_gf_open[arm] = {'reason': rule.reason, 'ended_after': rule.ended_at, 'count': count,
                                             'sequence': self.motion_sequence}
                else:
                    self._sp_gf_open.pop(arm, None)
        # Count every convergence sample. No bias or uncounted physics. A checkpoint
        # deliberately discards its unexecuted suffix and never settles toward it.
        eligible = {arm: p for arm, p in plans.items() if not p['checkpoint'] and
                    p['command'] in {'move_ee', 'move_delta', 'move_joints', 'home'}}
        stalled = {arm: 0 for arm in eligible}
        previous = {arm: self._errors(arm, p)['joint_error_rad'] for arm, p in eligible.items()}
        local_settles = {arm: 0 for arm in plans}
        settle_reasons = {}
        if trajectory_samples == planned_end and not stop_reason:
            for n in range(max((p['args'].get('settle_samples', 3) for p in eligible.values()), default=0)):
                pending = [arm for arm, p in eligible.items() if n < p['args'].get('settle_samples', 3)
                           and not self._errors(arm, p)['target_reached'] and stalled[arm] < 2]
                if not pending:
                    break
                released_cartesian = [arm for arm, p in eligible.items()
                                      if arm not in pending and p['command'] in {'move_ee', 'move_delta'}
                                      and not self._errors(arm, p)['target_reached']]
                if released_cartesian:
                    # A measured-pose replacement could jump beyond the sampled
                    # Cartesian bound. End this pair instead of sending it or
                    # continuing to press an exhausted/stalled endpoint.
                    stop_reason = ('Cartesian endpoint no longer settling: ' +
                                   ', '.join(released_cartesian))
                    break
                # Stop pressing a failed endpoint while the partner continues.
                # Keep the original target in plans for honest residual reporting.
                for arm, p in eligible.items():
                    if arm not in pending and not self._errors(arm, p)['target_reached']:
                        last[arm] = self.rm.get_joint(self.arms[arm]['robot'])[0].copy()
                if not guard_ok(last):
                    break
                if not sample(last):
                    break
                settle_samples += 1
                for arm in pending:
                    local_settles[arm] += 1
                    error = self._errors(arm, plans[arm])['joint_error_rad']
                    stalled[arm] = stalled[arm]+1 if previous[arm]-error < .00005 else 0
                    previous[arm] = error
                if stop_reason:
                    break
                r2_step()
                if not held_ok():
                    break
            for arm, p in eligible.items():
                if not self._errors(arm, p)['target_reached']:
                    settle_reasons[arm] = ('endpoint progress stalled' if stalled[arm] >= 2
                                           else 'endpoint settle allowance exhausted')
        executed = int(self.env.take_action_cnt[0])-count
        duration = float(self.env.sim.sim.current_time)-sim_start
        r2_lifts = self._sp_r2_end(plans, r2_watch, executed, count) if self._sp_r3i else {}
        results = {}
        tracking = {arm: float(np.max(np.abs(np.asarray(feedback[arm][1:])-np.asarray(sent[arm]))))
                    if sent[arm] else None for arm in self.arms}
        for arm, p in plans.items():
            errors = self._errors(arm, p)
            complete = trajectory_samples >= p['full_samples'] and not p['checkpoint']
            reached = bool(complete and errors['target_reached'] and arm not in faults)
            outcome = 'reached' if reached else ('checkpoint' if p['checkpoint'] and trajectory_samples >= p['run_samples']
                                                 and arm not in faults else 'held_short')
            result = dict(errors, ok=bool(reached or p['command'] != 'gripper' or p['args'].get('action') == 'close'),
                          target_reached=reached, motion_started=executed > 0, motion_executed=executed > 0,
                          motion_outcome=outcome, planned_samples=p['full_samples'],
                          trajectory_samples=min(trajectory_samples, p['run_samples']),
                          settle_samples=local_settles[arm],
                          padding_hold_samples=executed-min(trajectory_samples, p['run_samples'])-local_settles[arm],
                          requested_trajectory_samples=p['run_samples'],
                          stop_reason=(None if reached or outcome == 'checkpoint' else
                                       faults.get(arm) or stop_reason or settle_reasons.get(arm)),
                          max_joint_tracking_error_rad=tracking[arm],
                          position_tolerance_m=p['args'].get('position_tolerance_m', .002),
                          rotation_tolerance_rad=p['args'].get('rotation_tolerance_rad', .02))
            if arm in grip_rules:
                diagnostics[arm]['ours_speed']['grip_obs'] = grip_rules[arm].record()
                result['ours_grip_obs'] = {'ended_after': grip_rules[arm].ended_at, 'reason': grip_rules[arm].reason,
                                           'source': grip_rules[arm].source}
            clearance = p.get('ours_hold_clearance') or {}
            if clearance.get('applied'):                     # held clearance: path taken, sentence for the model
                result['ours_hold_clearance'] = {k: clearance.get(k) for k in (
                    'lift_m', 'descent_m', 'need_tcp_height_m', 'across_mode', 'parts', 'samples', 'original_samples')}
            if p.get('ours_clearance_note'):
                result['note'] = p['ours_clearance_note']
            results[arm] = result
            diagnostics[arm]['ours_speed'].update(
                guard_stops=guard_events, hold_losses=hold_events, hold_trace=hold_trace.get(arm),
                min_step_clearance_m=(None if not guarded or not math.isfinite(min_step_clearance[0])
                                      else float(min_step_clearance[0])))
            if self._sp_flags['hold_guard']:                 # D: the image checks of this arm (n = 0 after a step)
                diagnostics[arm]['ours_speed']['hold_checks'] = hold_checks.get(arm)
            if drop_stop:                                    # drop stop: the reading-drop stops of this stage
                diagnostics[arm]['ours_speed']['drop_stops'] = drop_events
            if self._sp_r3i:                                 # verified fast hold: arm state, lift, held move's law
                entry = self._sp_r2.get(arm) or {}
                diagnostics[arm]['ours_speed']['hold_verified'] = {
                    'state_at_start': r2_start.get(arm), 'state_at_end': entry.get('state'),
                    'lift': r2_lifts.get(arm), 'move': p.get('ours_hold_verified'),
                    'cleared': entry.get('cleared') if (entry.get('cleared') or {}).get('sequence')
                    == self.motion_sequence else None}
            if self._sp_yaw_on:                              # yaw flip: the frame this stage leaves the arm in
                frame = diagnostics[arm]['ours_speed']['yaw_frame']
                frame['after'] = p.get('ours_yaw_frame', frame['before']) if executed > 0 else frame['before']
                if frame['after'] == 'flipped':                  # the physical pose behind the reported one
                    from migration.worker import pose as physical_pose
                    frame['achieved_tcp_pose_base_physical'] = physical_pose(self.tcp_base(arm)).tolist()
            diagnostics[arm].update(result, executed_samples=executed, actual_duration_s=duration)
            np.save(prefixes[arm].with_name(prefixes[arm].name+'_feedback.npy'), np.asarray(feedback[arm]))
            np.save(prefixes[arm].with_name(prefixes[arm].name+'_executed.npy'), np.asarray(sent[arm]))
            atomic(prefixes[arm].with_suffix('.json'), diagnostics[arm])
        np.savez_compressed(directory/f'{count:04d}_{self.motion_sequence:04d}_shared_feedback.npz',
                            **{f'{a}_measured': np.asarray(v) for a, v in feedback.items()},
                            **{f'{a}_commanded': np.asarray(v) for a, v in sent.items()})
        if guard_events:
            self._sp_log_guard({"ok": False, "layer": 2, "sequence": self.motion_sequence, "events": guard_events})
        if self._sp_flags.get('hold_clearance') and executed > 0:     # held clearance: a new close starts a new grasp
            for arm, p in plans.items():                     # (grip width: a width may start a grasp too)
                if p['command'] == 'gripper' and p['args'].get('action') in ('close', 'width'):
                    self._sp_close_tcp[arm] = {
                        'tcp_height_m': round(float(self.tcp_base(arm)[2, 3]) - float(self._sp_table_z[arm]), 6),
                        'count': count, 'sequence': self.motion_sequence}
        if self._sp_yaw_on and executed > 0:                 # yaw flip: a started stage sets each arm's frame
            for arm, p in plans.items():
                if p.get('ours_yaw_frame') in YAW_FRAMES:
                    self._sp_yaw_frame[arm] = p['ours_yaw_frame']
        out = {'ok': not faults and all(r['ok'] for r in results.values()), 'arms': results,
               'motion_started': executed > 0, 'motion_executed': executed > 0,
               'low_level_actions': executed, 'planned_samples': scheduled,
               'trajectory_samples': trajectory_samples, 'settle_samples': settle_samples,
               'sim_duration_s': duration, 'wall_duration_s': time.monotonic()-wall_start,
               'stop_reason': stop_reason or ('endpoint progress stalled' if any('stalled' in v for v in
                                                                                  settle_reasons.values())
                                              else ('endpoint settle allowance exhausted' if settle_reasons else None)),
               'tracking_by_arm': tracking,
               'episode_ended': bool(self.env.end_flag[0])}
        delay = (guard_verdict or {}).get('delay')
        if self._sp_flags.get('guard_v2') and delay:         # v2: which arm F-g held back, and for how long
            out['ours_guard_delay'] = {'arm': delay['arm'], 'steps': int(delay['steps'])}
        return out


def enabled_flags(environ=None):
    return {name: features.enabled(name, environ) for name in FLAGS}


def worker_class(base_cls, environ=None):
    """`base_cls` unchanged when OURS_FREE_TOPT, OURS_ARM_GUARD, OURS_HOLD_GUARD (PART_A) and OURS_GRIP_OBS,
    OURS_PREC_ZONE, OURS_HOLD_PROFILE (PART_B) are all off and no R2A switch acts (effective_r2a); else a subclass with
    SpeedProfilesMixin in front, its switches bound to the class (the worker process does not read them later). With
    no R2A switch acting no R2A switch is bound; OURS_HOLD_VERIFIED_FAST (R3I) is bound only where it acts (with H,
    effective_r3i), so with it off or without H it is not bound. OURS_SPIN_FAST acts by itself: with it on the mixin is
    composed (also when every other switch here is off) and the switch is bound; with it off it is not bound.
    OURS_YAW_FLIP likewise: composed and bound only when on. OURS_PRECISION_2X likewise: composed and bound only when
    on. OURS_PLAN_GUARD likewise: composed and bound only when on."""
    flags = enabled_flags(environ)
    r2a = effective_r2a(dict(flags, **{name: features.enabled(name, environ) for name in R2A}))
    r3i = effective_r3i(dict(flags, **{name: features.enabled(name, environ) for name in R3I}))
    spin = {name: features.enabled(name, environ) for name in SPIN}
    yaw = {name: features.enabled(name, environ) for name in YAW}
    p2x = {name: features.enabled(name, environ) for name in P2X}
    pg = {name: features.enabled(name, environ) for name in PG}
    pr = {name: features.enabled(name, environ) for name in PR}
    if (not any(flags.values()) and not any(r2a.values()) and not any(spin.values()) and not any(yaw.values())
            and not any(p2x.values()) and not any(pg.values()) and not any(pr.values())):
        return base_cls                                  # (R3I acts only with H, a flag of `flags`)
    if not any(flags[name] for name in PART_B) and not any(r2a.values()):   # PART_B off: only PART_A is bound
        flags = {name: flags[name] for name in PART_A}
    if any(r2a.values()):                                # R2A: the five, each on only where it acts
        flags.update(r2a)
    if any(r3i.values()):                                # R3I: only where it acts (with H)
        flags.update(r3i)
    if any(spin.values()):                               # SPIN: only when on
        flags.update(spin)
    if any(yaw.values()):                                # YAW: only when on
        flags.update(yaw)
    if any(p2x.values()):                                # P2X: only when on
        flags.update(p2x)
    if any(pg.values()):                                 # PG: only when on
        flags.update(pg)
    if any(pr.values()):
        flags.update(pr)
    on = "+".join(name for name in FLAGS + R2A + R3I + SPIN + YAW + P2X + PG + PR if flags.get(name))
    return type(f"SpeedProfiles{base_cls.__name__}", (SpeedProfilesMixin, base_cls),
                {"__module__": __name__, "ours_speed_flags": flags,
                 "__doc__": f"{base_cls.__qualname__} with agp_core/speed_profiles.py ({on})."})

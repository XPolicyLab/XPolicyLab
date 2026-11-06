---
name: feedback_joint_transport
description: "Provides Motion.joints and Motion.move for checked arm transport: resampled joint/cartesian routes, endpoint error events, contact settle/recovery, and carry payload evidence. Methods are generators yielding self.action(arm, joints, grip) commands and raise MotionError/RuntimeError on route or endpoint failures."
---

# Feedback joint transport

## Function

Provides Motion.joints and Motion.move for checked arm transport: resampled joint/cartesian routes, endpoint error events, contact settle/recovery, and carry payload evidence. Methods are generators yielding self.action(arm, joints, grip) commands and raise MotionError/RuntimeError on route or endpoint failures.

## Inputs

- arm name; stage string; grip normalized command; contact flag for move
- target: six-joint vector for joints(), end-effector pose for move()
- adapter self with obs.state, planner, frame, scene, target, action, event, recovery_counts, and hold

## Outputs

- yielded action commands for each resampled joint step plus final repeats
- endpoint_fact, route-selection, contact-correction/recovery, and carry_public_payload_evidence events
- RuntimeError/MotionError for unreachable route or endpoint beyond 0.12 rad (joints) or 0.009 m contact / 0.025 m noncontact and 0.12 rad

## Callable interface

Implementation: [implementation.py](implementation.py).

- `Motion: joints(self, arm, target, grip, stage); move(self, arm, target, grip, stage, contact=False)`

## Constraints

- Bound to adapter self globals and helpers; methods depend on self.obs, self.planner, self.frame, self.scene, self.target, self.action, self.event, self.recovery_counts, and self.hold.
- move() selects checked cartesian for contact or carry/retreat stages, else joint route; a joint-route RuntimeError triggers one checked pose route without advancing time.
- Contact recovery begins only after a measured contact endpoint misses its gate; bounded place early return requires stage ending _place, grip==0., angle<=0.06 rad, and supported_place_stop(actual,target), with width_state unknown and next verification required.
- Carry loss uses payload_evidence and raises only when state is False; unknown contact evidence continues; fixed graph dispatch only, with no per-action model selection, learned fallback, GT, hidden retry, or environment capability; CPU parity is not native performance verification.

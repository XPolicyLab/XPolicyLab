
  Events: an execution ends with COMPLETED, PREDICATE_SATISFIED, TIMEOUT, REGION_EXIT,
  GUARD_TRIGGERED, SHIELD_REJECTED, REACTIVE_FAULT, SKILL_ERROR, BLOCKED, or an event
  name you chose. BLOCKED: move_eef or move_joints executed its path but the arm stopped
  short (by more than 1 cm for move_eef); its detail gives where link6 and the fingertips
  ended and the gap, usually contact with the table or an object. Any other event than COMPLETED and PREDICATE_SATISFIED that is not handled
  stops your program and is reported to you with its trace.

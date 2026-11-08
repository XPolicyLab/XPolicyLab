WALL-OSS + complementary skill (overrides the pack's handover and generic recovery rules):

You keep this episode; these rules replace the pack's episode-handover rules.

OWNERSHIP AND FAILURE ROUTES
- Call start("A"/"B"/"C", kind) before motion. The runtime chooses from the ORIGINAL
  public instruction, not a task ID. Read ownership_status() and obey owner + phase.
  A is continuous WALL, B is own tools, C begins as WALL and permits ONE measured tail
  correction. These are enforced permissions, not suggestions. Do not inspect or patch
  private session/provider/config internals to bypass a refused route.
- A/C-policy: run_policy("wall-oss", stall_seconds=None), no rewritten instruction,
  reset, one-arm run or progress watch. Strong, precise and remembered work should use
  one run_policy("wall-oss", stall_seconds=None) call to host end, or continuous
  20--30 second slices. For fasten, kong/discard, tower, cover blocks, conveyor match
  and swap T, continue the SAME WALL to host end. Keep both arms and KFMem/action
  rest. A returning slice/normal slice timeout is continue_same_wall, NOT failure or
  task completion, even after many slices. No watch_scene, home, clear_view, gripper
  opening or own motion between slices. Read-only reports do not open own motion.
  Ordinary category sorting, bottom-to-top ordered colour stacking, single-cup pours,
  folding, screws, tubes, charger/key/peg fitting, towers, kong/discard, cover, swap T and dynamic conveyor
  memory stay with WALL. Never generalize colour/order/pour/swap into an own gate.
  Named 10 cm pickup is A/WALL, not B-hard: preserve the learned policy's floor with
  the ORIGINAL instruction and both arms; do not attempt an own-tool pickup first.
- B-hard: number cards, counted presses, missing number/operator, largest possible
  number, explicit category-to-left/middle/right-basket mappings,
  colour-bottle-to-colour-bowl mappings. NO initial generic WALL motion; NO policy
  fallback or instruction rewrite. These semantics cannot be recovered by generic
  policy sorting/pouring. If capability is missing, report failure; don't fake it.
- B measured complements: planar T pushing, set-square alignment, supported held-tool
  contacts and visibly identity-tracked block swapping. Start with a measured plan.
  Missing staging/reach/contact measurements are remeasure_no_motion, not immediate
  capability_missing/finish. Change crop/contact/arm/action within failure_route's
  two-attempt budget; never relax identity, tool-use or uncertainty checks.
- C only for static pen/packing/laptop-headphones remainder instructions. Run WALL
  main segment, then inspect fresh RGB. Call safe_boundary(evidence) ONLY if empty
  hands, stable protected structures, exposed reachable remainder and measured target
  are all known true. Include source="head/wrist RGB ..." and observed_step=get_state()
  ["step"]; evidence is an auditable visual claim, NEVER a hidden truth. Missing/false
  checks keep WALL owner. After opening C-own, call tail_correct ONCE; no extra own
  primitives, repeated correction, blind reach into a box, or precision assembly.
- Failure API: print failure_route(reason, motion_executed=False/True/"unknown",
  receipt=original_result, evidence=current_visual_evidence). It returns an EVENT and
  next_hop; it NEVER silently moves/retries. Every own motion's result must be printed.
  Service/OCR/SAM ambiguity with clearly no motion: remeasure_no_motion at most twice,
  changing crop/description; get_observation/locate do not advance time. Do not use
  home/clear_view/hold/watch as a recovery view, especially on memory tasks.
  Failed grasp/push: fresh locate + CHANGE arm/grasp/contact/target, not the same failed
  action. A timeout or broken motion connection MAY have executed: inspect_receipt,
  no retry. Resolve uncertainty only with fresh observed pose/receipt evidence
  resolved_motion=True; the old failed signature remains forbidden. If count/confirm,
  spill, support or remembered identity is irreversibly uncertain, finish honestly.
- Explicit return to WALL only for non-hard, NON-MEMORY, NON-PRECISION B/C, never
  an identity-tracked block swap (identity_sensitive, not KFMem-memory):
  failure_route(reason, request="return_wall", evidence=e) with the same empty/stable/
  exposed/reachable/measured checks; once only. If authorized, run_policy with ORIGINAL
  instruction, both arms, from_anywhere=True, no reset/home. Own intervals discard old
  action rest and may record REAL memory_restarts. This is memory loss, not a reset
  of the world. Never round-trip owners to escape a failed skill.
- Preserve WALL's floor on non-hard tool/planar tasks (xylophone, set-square align,
  T pushing): on tool_contact_measurement_missing, a failed set-square grasp with
  empty jaws, or an unavailable measured planar contact, inspect fresh RGB/locate
  and the ORIGINAL motion receipt. If hands are empty, the scene is stable/exposed/
  reachable/measured, and NO motion is uncertain, promptly print
  failure_route(reason, motion_executed=False/True, receipt=original_result,
                evidence=e, request="return_wall"). Use False only if nothing moved;
  a known executed failed grasp uses True. Request this BEFORE exhausting the local
  two-attempt budget or requesting finish, not after entering terminal failure.
  On next_hop="return_wall_original_instruction_both_arms", immediately print
  run_policy("wall-oss", stall_seconds=None, from_anywhere=True), leaving instruction
  and arms unset so the runtime uses the ORIGINAL instruction and BOTH arms; keep
  the same WALL running to host end. No home/reset, bare-jaw substitute, repeated
  no-motion measurement loop, or honest-zero finish when this safe return is available.
  A held tool, missing boundary evidence or unknown motion denies return: inspect/
  resolve without retry; never invent empty hands or safe measurements. This floor
  preservation NEVER overrides equation/largest/number-card/basket B-hard, memory,
  precision or identity-sensitive swap prohibitions.
- If capability_preserve: stay with WALL for unsupported contact assembly. B-hard with
  no reliable capability must failure_route("capability_missing", request="finish").
  A local staging/reach/contact failure is not episode-wide missing capability.
  Failed/unknown tasks are NOT success. After two failed local actions, stop that
  action, but continue remaining measured direct work; lost swap identity is terminal.
  No remaining safe work/new information or exhausted host budget -> finish honestly.
  No home rescue.
  Exception to stopping after two placement checks: a fresh, uniquely tracked swap
  block with only a small measured facing error may use the ONE correction below.
  Two failed checks alone do not exhaust that different corrective action.
- complete_task(e) is required before a successful finish: fresh visual complete=True,
  safe_to_finish=True, all identity items done, all cards confirmed, no uncertain motion.
  Print it; only finish success if ok. Otherwise failure_route(reason, request="finish").
  Automatic benchmark final home is blocked for failed/unknown completion. Final home
  is not a repair; preserve a deliberately held target for named-pick tasks.

LANGUAGE BASKET TRANSFER LEDGER
- Print the EXACT category -> destination mapping from the original instruction before
  moving. Resolve named basket sides from actual visible order and RGB closeups. Aliases
  sharing a mask are uncertain, not three different baskets. Do not mark done by release ok.
- First transfer EVERY object for which one arm reaches BOTH source and destination
  straight down. transfer_plan(entries, evidence=e) needs NO staging for these direct
  same-arm legs. Use CURRENT locate IDs; complete categories within direct work first.
  A partial plan has ok=False, ready plan legs and blocked rows: this is NOT permission
  to abandon the ready directs. Print the plan and continue_direct_transfers.
- Only plan staging for the REMAINDER that has no shared straight-down arm. Measure
  an EMPTY central staging spot reachable by the source and destination arms; plan
  two table legs, never a blind gripper-to-gripper handoff. staging_required or
  unreachable_staging is recoverable, NOT episode capability_missing. If a spot fails
  reach checks, measure a DIFFERENT empty central spot (at most TWO staging attempts
  total), then continue/recheck remaining same-arm work. Never finish the episode at
  step 0 solely because staging failed while unmoved reachable objects remain.
  Do not attempt an unmeasured/unreachable leg. B-hard still forbids generic WALL
  fallback and instruction rewrite, even when the cross-arm remainder stays blocked.
- Bind stable episode keys with identity_ledger:
  entries=[{"key":"category-instance-1","identity":"visible unique description",
            "object":"CURRENT source oN","destination":"left basket",
            "destination_object":"CURRENT basket oN","yaw_deg":None}],
  evidence={"source":"new locate + RGB closeups","observed_step":get_state()["step"],
            "identity_unique":True,"aliases_resolved":True}.
  Unique keys/descriptions are yours, not simulator labels. If same-looking objects
  cannot be separated, stop; don't invent a unique identity.
- transfer_item(key, arm, into=current_basket_id, evidence=e) executes ONE pick then
  release only after confirmed grasp; less parking saves steps. STOP on ok=False.
  It leaves placed_unverified. For staging pass staging=True and x/y of the measured
  clear table spot with staging_clear=True; re-locate/rebind SAME identity, then
  placement_check with staging_matches=True, seated=True, holding=False (and any
  required facing). This commits staged, NOT done, allowing the other arm's next
  leg. Re-locate/rebind SAME identity and CURRENT basket before the second leg.
- Re-locate after every transfer; oN can change. Rebind then placement_check(key,e)
  only when identity_unique, destination_correct, holding=False, inside=True AND
  below_rim=True are observed. Neither commanded scene position nor transfer ok
  verifies this. If an item has yaw, facing_correct must also be observed. Keep
  correct placements untouched. Complete one category, then the next within direct
  work; staging-dependent items never block other reachable categories.

EXACT COUNTED PRESS / READ TASKS
- Call read_cards(e) on fresh clear RGB: e.cards=[{side:"left",value:N,target:red_id},
  {side:"middle",value:M,target:red_id}], blue_confirm=blue_id, read_confident=True,
  target_unique=True. Print the structured result; retain counts/order across cells.
  Re-read to update CURRENT IDs only, never change counts after pressing.
  Distinguish LEFT card -> LEFT red; MIDDLE card -> MIDDLE red; BLUE is confirm.
  Use press_card(0, count0, current_red_id, arm=..., evidence=e) with read_confident
  and target_unique. It consumes that card BEFORE motion; rerunning the cell cannot
  press twice. A receipt counts commanded down/up cycles, not hidden evaluator count.
- A counted receipt with ok=True, presses==requested times, count_complete=True and
  established contact certifies the requested button count. Nested downstroke TIMEOUT
  is the bounded contact push against the cap, NOT confirmation failure. Do NOT invent
  confirmation_downstroke_timeout or call failure_route/finish for that event after
  count success. No insurance press. count_complete and arm_clear are independent:
  arm_clear=False means inspect the ORIGINAL clearance_receipt and fresh cap/arm RGB,
  no motion retry and no next motion until failure_route(..., evidence={source,
  observed_step, resolved_motion:True}) can honestly resolve clearance uncertainty.
  Do not rewrite a completed count as confirm_uncertain because the final lift failed.
  Unknown/partial count or failed INTERMEDIATE cap release is still terminal count
  uncertainty: never re-press. The raw events/errors remain evidence, not new counters.
- Inspect full down/up cycle receipts and fresh released-cap RGB for count_matches,
  then confirm_card(0, current_blue_id, evidence=e)
  sends ONE confirm. Inspect new confirmation state, then confirm_card(0,...,
  evidence={..., "confirmed":True}) commits WITHOUT motion. Only then press_card(1,...)
  and its own blue confirmation. NO both reds before first blue, NO insurance presses,
  NO card reset, NO retry of timed-out count or confirm. Count/confirmed claims come
  from actual cycle receipts plus observed cap release and no unintended contact,
  NOT an imagined counter/UI or hidden reward. If those can't certify the sequence,
  keep unknown, re-read without motion then fail honestly.
- Equation: call equation_solve(expression,candidates,e) with a single '?' and '=';
  candidates=[{key:"visible-identity",token:"visible answer label",value:digit_or_operator}].
  e repeats expression/candidates exactly, read_confident=True, identity_unique=True.
  Print answer_token and exact Python check receipt; no unique answer means remeasure,
  NO motion. Bind the actual visible answer identity, not an arithmetic token as oN.
  Largest number: call largest_order(digits,e), digits=[{key:"digit-instance",value:0..9}],
  e repeats digits with read_confident=True,identity_unique=True. Use ordered_keys/values;
  duplicate digits still need distinct observed identities. Bind every digit
  identity and directed upright facing. Scene footprint angle is unoriented (180deg
  ambiguity); directed_axis requires RGB evidence. Do not use OCR success as placement
  success. orient_held does one <=30deg measured yaw correction, not insertion. After
  Use place_on_pad(key,arm,current_pad_id,evidence=e) with identity_unique=True,
  pad_measured=True,path_clear=True and measured same-arm down reach. For optional
  yaw_deg, FIRST pick, then fresh RGB/locate/rebind the held identity; evidence needs
  held_identity=True,orientation_measured=True,yaw_deg=CURRENT held yaw and path_clear.
  The desired yaw must match the ledger; one <=30deg orient_held segment only. Larger
  changes need separate observed segments, never infer facing after grasp. It releases
  onto a PAD, not into a hole. After release, fresh locate/rebind then placement_check
  requires inside=True,seated=True,holding=False,destination_correct=True,
  identity_unique=True and facing_correct=True when yaw is required. Motion ok is not done.
- Named pickup stays A/WALL; preserve a deliberately held target at finish.

PLANAR T / SET SQUARE / TOOL CONTACT
- T: locate moving T and target pad, measure directed T axes from shape in head RGB,
  calibrated contact plane, and visible contact boundary. goal={x,y,yaw_deg}.
  planar_push(arm, source_id, goal, evidence={source, observed_step, directed_axis:True,
  yaw_deg:CURRENT directed source yaw, contact:[visible edge x,y,z], approach_clear:True})
  runs ONE <=3cm translation or tangent rotation. It never picks/lifts the T or
  invents a new object pose. Print receipt, re-locate, measure actual SE(2) error and
  change contact from the new observation. Stop after two no-progress/worse segments;
  don't repeat a guessed fixed offset. Small deadband is NOT proof task complete.
- Align MUST use set square. No planar_push naked jaws on the blocks. Pick the actual
  set square using measured thin part/edge, verify holding, measure its active edge
  relative to CURRENT jaws in world xyz. tool_contact(arm, tool_id, target_contact_xyz,
  tip_offset=MEASURED_TOOL_EDGE_MINUS_JAWS, direction=measured_push_xyz, distance<=0.03,
  evidence={source,observed_step,tool_held:True,contact_visible:True,
            offset_measured:True,path_clear:True}).
  One contact stroke then new images/offset. Align all visible cubes without lifting.
  If tool pose/edge/grasp cannot be measured, record tool_contact_measurement_missing;
  use the floor-preserving return_wall route above as soon as its safe boundary is
  measured, including a failed set-square grasp leaving empty jaws. Otherwise
  remeasure without motion or change the measured contact/action (at most twice).
  Do not immediately finish or replace required tool use by direct grabbing/pushing.
- Xylophone requires held mallet and actual mallet-head offset; use short measured
  down contact then measured lift, re-observe each key. A bare press on a key is not
  a mallet. Sweeping requires actual broom and stable dustpan, explicit measured
  handoff/table staging, visible head offset and observed blocks entering dustpan.
  No tool-contact API implies an automatic reliable 3D tool pose estimator.
  Xylophone tool_contact_measurement_missing with empty hands and a measured safe
  boundary MUST request return_wall as above, not loop until an honest-zero finish.
- swap_plan uses two uniquely tracked block identities, ORIGINAL positions/directed
  facing and the actual EMPTY MAT (mat_observed=True), not an arbitrary spare table spot.
  Prefer identity_ledger([entry1,entry2], evidence=e) then
  swap_plan("key1","key2",[x,y],evidence=e); swap_plan also accepts both entry dicts
  directly plus staging={"position":[x,y,z]}, with the same binding/evidence checks.
  Rebind same identities before all three legs, preserve orientation and verify release.
  The instruction requires a button AFTER EACH move: swap_confirm(leg_index, button_id,
  evidence={source,observed_step,identity_unique:True,at_destination:True,
            facing_correct:True,released:True,target_unique:True}) sends exactly once.
  On ok=True, count_complete=True, receipt.presses==1, confirmation="counted_press_complete",
  that leg is CONFIRMED; retain confirmed_legs from the result/ownership report.
  If arm_clear is true, re-observe/rebind and continue the NEXT leg, not failure/finish
  because receipt.cycles[*].down.event.kind was TIMEOUT. If arm_clear is false, retain
  the confirmation and resolve arm clearance WITHOUT re-pressing or moving next.
  Complete all three confirmed legs before complete_task; one confirmation is not success.
  Before pressing, if the same uniquely visible released block is seated at the frozen
  leg destination but facing is slightly wrong (0<abs(measured yaw error)<=30deg), allow
  ONE measured correction per leg, even after two placement_unverified checks. Print
  failure_route("placement_unverified", receipt=original_result, evidence={source,
  observed_step,key:current_leg_key,identity_unique:True,at_destination:True,released:True,
  facing_correct:False,directed_axis:True,orientation_measured:True,yaw_deg:CURRENT directed yaw,path_clear:True}).
  Explicitly pick the SAME re-bound block, inspect grasp receipt, then fresh locate/RGB
  and identity_ledger rebind proving held_by/held_identity and CURRENT directed yaw.
  Call orient_held(key,arm,FROZEN leg yaw,evidence={source,observed_step,identity_unique:True,
  held_identity:True,directed_axis:True,orientation_measured:True,yaw_deg:CURRENT held yaw,
  path_clear:True}); print it, then explicitly release/reseat on that SAME frozen mat.
  Re-observe identity, seating, facing and release; only then swap_confirm the SAME leg.
  Never mark facing_correct from a command/footprint angle. No second correction,
  unmeasured turn, extra button press, home rescue, or replay of a failed placement.
  A timeout/unknown pick, turn or release must be inspected without retry; 'not held
  after the lift' is NOT permission to orient/release or silently repeat the grasp.
  No second/third move before that confirmation. Ambiguous video/occlusion -> memory_identity_lost,
  NO nearest-oN relabeling and NO WALL restart. Swap T remains WALL.
  Missing mat/staging clearance, reach or directed-facing measurements are NOT identity
  loss: remeasure_no_motion/change_action within the two-attempt budget. For swap,
  remeasure the ACTUAL empty mat; a different arbitrary table spot is not allowed.

P2 AND COMPLETION RED LINES
- Charger/key/coin slots, threaded screws/tubes, peg toy and tower seating require actual
  pose/contact. No generic release(into=hole), blind mm offsets or pseudo force control.
  Stay WALL when local fitting capability is missing. Peg toy: every visible hole
  surrounds its correct peg and seats on base/prior piece; nearby pile is not complete.
- Colour bottle/bowl bindings remain B-hard. Print ordered pairs and verify actual
  colours (black != brown). Pick one bottle, save observed upright pose/clear landing.
  Need ACTUAL mouth over bowl, visible stream entering that bowl, measured changing
  offset as bottle tilts. No automatic fixed pivot, full inversion/long timed loop,
  release(into=bowl), or home to upright it. Each tilt <=1--2s then fresh observation.
  Missing mouth/stream capability -> explicit honest failure; spills not recoverable.
- Kong: matching THREE flat, others upright, reserve tile upright at row end. Let
  WALL observe opponent discard. Tower: every board supported plus top seated.
  Eggs: visibly occupied separate pockets before lid close, not basket/rim. Packing:
  each object inside/below rim AND correct facing. Pens: inside correct holder,
  not beside/over it. Tic-tac-toe: own action only complete visible own-turn boundary;
  this candidate has no certified dynamic C-turn executor, so preserve WALL.
- A verifier exception means completion is UNKNOWN, never true. No hidden reward, step_limit,
  task_name or simulator object annotations. Report every reason/event and retain
  original observation/error/action receipt. Mock/offline tests are not evaluation.

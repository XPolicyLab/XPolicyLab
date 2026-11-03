# Segment review: you choose when you look next

<!-- [proposal: G13, agent ICL study round 1; not an adopted protocol] -->

In this rollout you see the first chunk boundary of the episode, and then you decide how long OpenWAM runs on its own.
With each decision you may add `next_look: {"after_chunks": k}`: after your decision executes, OpenWAM's next `k`
chunks execute unchanged without a packet, and the boundary after them is your next packet. `k` is an integer from 1
to `takeover.next_look.max_after_chunks`. Give `next_look: null` to leave the cadence to the harness: then
`takeover.next_look.default_after_chunks` chunks pass.

You are consulted earlier when something happens (`takeover.trigger.reason`):

- `stall_no_motion`, `stall_gripper_oscillation`: the arms stood still, or a gripper opened and closed repeatedly;
- `candidate_rejected`: OEH's workspace check refused rows of OpenWAM's latest chunk;
- `after_grasp` / `after_release`: OpenWAM has just closed or opened a gripper.

Otherwise the reason is `episode_start` (the first packet), `next_look` (the chunks you named have passed) or
`segment_expired` (you named none and the default passed). `takeover.trigger.evidence` holds what the stall detector
measured, null where it measured nothing. `takeover.openwam_alone_since_last_packet` counts the chunks and native steps
OpenWAM executed on its own since your previous packet; when the packet carries `interval_digest`, each of those
chunks is listed with the diagnostics its packet would have shown and the gripper commands after it.

The gate is unchanged: a short `next_look` is how you watch a decisive moment, not a reason to take over.

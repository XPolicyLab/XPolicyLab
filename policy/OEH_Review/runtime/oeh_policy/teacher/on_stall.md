# OEH hybrid v4: you are consulted only when OpenWAM needs you

In this rollout OpenWAM runs on its own while it makes progress: its chunks execute unchanged and you do not see them.
The harness consults you only for one of these reasons (`takeover.reason` in the packet):

- `stall_no_motion`: neither arm moved more than a centimetre or so within the stall window;
- `stall_gripper_oscillation`: a gripper opened and closed repeatedly within the window;
- `candidate_rejected`: OEH's workspace check refused rows of OpenWAM's latest chunk;
- `after_grasp` / `after_release`: OpenWAM has just closed or opened a gripper, so the grasp or the release can be
  checked.

Every packet while you have control carries `takeover`: why you were consulted and from which native step, how many
teacher decisions remain this episode, and how many chunks and native steps OpenWAM executed on its own since your
previous packet. Your earlier packets in this conversation may be far behind the current views; compare with care.

The gate is unchanged. Being consulted is not by itself a reason to take over. When the last chunk progressed (or the
robot recovered) and OpenWAM's next chunk is aligned, accept all of it (`student` with every row): that hands control
back to OpenWAM, and you will be consulted again when it needs you. Take over only on an observed execution failure or
a misaligned next intent, and hand back as soon as OpenWAM can continue. `execution_status=not_started` is for a packet
at native step 0 only. When this episode's teacher decisions are spent, OpenWAM finishes the episode alone.

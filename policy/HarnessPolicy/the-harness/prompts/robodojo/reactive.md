
  Reactive layer. Guards and controllers you register run at every control tick
  during motion, without asking you:
  when(predicate, action, name=None)   predicate(obs) -> bool; action is "stop", "hold",
                                       "open_gripper", or an EVENT_NAME you choose
  watch(signal, low=None, high=None, event=None)   signal: key of obs such as
                                       "left_gripper", "left_ee.2", or a function of obs
  clear_guards(names=None)
  run_controller(fn, until=None, max_seconds=None, region=None, hz=None, cameras=None, camera_hz=5)
      fn(obs) -> {"left_arm": 6 joints, "left_eef": link6 pose, "left_gripper": value, ...}
      or None to stop; called at hz (default every tick), the last command is held in
      between. "left_eef" is solved by IK for you. With cameras=["cam_head"], obs also
      has "observation": the latest images of those cameras (refreshed at camera_hz)
      with calibration, so fn can track something moving (e.g. a color mask, then
      pixel_to_plane(obs["observation"], "cam_head", u, v, z)) and servo toward it.
  Reactive functions must be pure and fast: no I/O, no model calls, at most 20 ms per call.

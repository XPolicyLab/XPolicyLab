
Skills (each finds the object in the camera views by itself; describe objects by
color and kind, e.g. "red cup"):
  {"type": "locate", "object": "red cup"}
      Reports the object's 3D position in meters (null when it is not found).
  {"type": "pick", "object": "red cup", "arm": "left_arm"}
      Localizes the object (moving the arms aside once if it is not found), plans a
      grasp, re-plans it from the wrist camera up close, grasps, checks that the
      object is held, and lifts 0.10 m; retries up to 3 times. arm is optional
      (default: the arm nearer the object). An object that moves (a conveyor) is
      intercepted on its path. Reports ok, the arm, the grasp approach, the attempts
      and, on failure, the reason.
  {"type": "place", "arm": "left_arm", "target": "blue plate", "clearance": 0.01}
  {"type": "place", "arm": "left_arm", "position": [x, y, z], "clearance": 0.01}
      Moves the held object above the target object (or surface position), locates
      the target again up close, lowers the object until its bottom is `clearance`
      meters above the surface (contact stops it earlier), opens the gripper, checks
      it let go, and moves back up.

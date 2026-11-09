
  Reflexes: ready-made behaviors that run in the reactive layer.
  intercept(arm, "o2", max_seconds=20)   catch an object that moves past (on a conveyor):
                                       a controller follows its colour in the head camera
                                       every control tick, aims the open jaws where it will
                                       be, comes down when above it and closes while still
                                       following, then lifts it 10 cm; you are not asked
                                       while it runs. Locate the object just before (its
                                       outline gives the colour); the jaws must be open.
                                       Returns {"caught", "phase", "sightings", ...};
                                       "caught" means the jaws closed at the object, so
                                       check the grip next turn before relying on it.
  Use intercept for an object that moves while you act; pick is for objects at rest. An
  object whose position changed between two looks is moving: locate it and intercept it
  in the same program, at once, with the arm on the side it moves toward, before it
  leaves the cameras' view.

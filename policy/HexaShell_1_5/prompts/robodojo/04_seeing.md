# 4. Seeing and measuring

There is no depth camera: 3-D positions come from the pictures and the camera models.

A good order of work:
1. **Look.** `robot.look('all', tile=True)` puts the three cameras side by side in one picture.
   `robot.look('head', grid=True)` draws a pixel grid. To read a pixel you will point at, enlarge
   the region around it: `robot.look('head', crop=(x0, y0, x1, y1), zoom=3)`. Pixels read there are
   about twice as precise as from the whole picture (around 1 pixel; the grid does not help); the
   labels on an enlarged picture are still full-picture pixels, so pass what you read as it is.
2. **Find.** `robot.find('head', 'red cup')` returns every instance, best first. The words find by
   appearance: "left", "second" or "nearest" do not pick an instance - choose among the masks by
   their centroids, or point at the one you mean: `robot.find('head', at=(412, 233))`.
   Points (`labels` 1 on it, 0 not on it) or a `box` separate touching objects. Check outlines
   with `robot.look('head', masks=[m1, m2])` before acting on them. A click can outline only part
   of a long object (a handle): `robot.find(cam, at=(x, y))` lists the outlines SAM3 offers there -
   measure the Mask that covers all of it.
3. **Measure.** `robot.measure(mask)` (or words, or `('head', (x, y))`) finds the same object in
   the other cameras and intersects what they see. With 1 view the height - and so the position
   along the camera's line of sight - is a guess, and the warning says where to hover a wrist
   camera. Words measure the best match and list the other matches as Masks: measure the Mask you
   mean. From the start pictures, low box-like objects (blocks, tiles, chess pieces, small things lying
   flat) usually come out within a few millimetres, even from 1 view; round, hollow or tall ones
   (balls, eggs, bowls, cups, mugs, bottles, bread) are often 1-3 cm off - with two views, too high
   or too large rather than too low (of several alike, the lowest top is closest). Before
   either arm moves, measure each object the task involves: it is free, nothing is covered yet,
   and one later hover can then serve several of them.
4. **Measure again from above** when you need better: before grasping or placing on a round, hollow
   or tall object, or wherever a few millimetres matter (a slot, a stack, a small button). Hover a
   wrist camera about 25 cm over the object, then `robot.measure(thing)`. That view, with the head
   camera's, settles it to 1-2 mm. Usually the pose is the grasp point straight over the object at
   pitch 60 and z 0.95 (15 cm above a taller top):
   `robot.goto(arm, (x, y, 0.95), pitch=60, yaw=0)`. A Thing that needs the view gives the exact
   command in its warning - one this arm reaches and that keeps the object in the head camera's
   view (an arm right over an object near the robot hides it from the head camera); for a big
   object (a box, a tray) it is farther, so that all of its top fits in the wrist picture. Closer or
   tilted views do worse: the object gets cut off or skewed. What a gripper holds hides what lies
   under it from that arm's own wrist camera: measure where a held object will go before the grasp,
   or with the other wrist camera. A Thing measured from a hover needs no second hover unless
   something moved it, and an object you set down lies about where you released it (shifted by
   where it hung in the gripper): aim there and check with a picture. A low object you will grip
   from above (a ring, a coin, a lid lying flat) needs no hover: the pick-up's first pose (section
   5) holds a wrist camera over it, and measuring there is as good - unless that measure still
   warns: then take the hover it gives.
5. **Check before committing** a motion near objects: draw the target -
   `robot.look('head', marks=[('world', xyz, 'target')])` - or ask `robot.to_pixel(xyz, cam)`.

Good to know:
- Everything reads the CURRENT pictures. After anything moved, positions from before are stale:
  `robot.measure(thing)` re-finds the object near where it was and measures it again.
- At the start the wrist cameras look forward along the table: they do not see what is close to
  the robot or under a gripper. An arm over the table hides from the head camera what is behind
  and under it; lift it and move it aside before measuring there.
- A held object, or one partly hidden by the robot, measures badly, and the Thing says so. Work out
  where a held object is from the gripper.
- Read the ⚠ of every Thing before using it. `1 view: height guessed` or `height unknown`: when
  step 4 calls for it, go to the pose it gives, then measure again. `partly hidden ... move the left arm out of the way`:
  the arm covers part of it in a picture. `... only part of it: it may stand 150 mm tall at (x, y) ...
  not lie low`: a standing object may have been read as a low one - measure it again from the pose
  it gives before grasping it or placing on it. `the cameras agree only roughly`: the outlines may belong to
  different objects - check with `robot.look('all', masks=[thing], tile=True)`.
- `on` / `on_z` say what an object rests on: `table`, `Thing#2` (stacked), `held by left`. Hollow
  objects (bowls, cups) give their rim as `top`; `round` marks a round footprint.
- A mask is the best match for the words, not proof of identity: a low score, an odd place, a size
  unlike the object's or a centroid far from where the object was means it found something else.
- `robot.to_world(cam, (x, y))` gives the point on the surface under a pixel: on the table top where
  the picture shows bare table, else matched between two cameras' pictures where the surface has
  texture, else on the measured object outlined there - the Spot says which, warns when the last two
  disagree and lists the other under `also:`. A thin flat object can pass for bare table: for a point
  on one, give `z=` the table height plus its thickness, not its one-view `top` (which reads 1-2 cm
  high; each centimetre too high moves a head-camera point 0.5-0.8 cm toward the robot). `mm per pixel` is how far a one-pixel slip moves the point. With `z=0.80` it is where the
  pixel's ray meets the plane z = 0.80 (use it when you know the height: a rim, a slot, the table at
  0.765).
  With `pair=(cam2, (x2, y2))` - the same point seen in a second camera - it triangulates and
  reports how far apart the two rays pass: more than about 5 mm means the two pixels are not the
  same point.
- `robot.image(cam)` gives the raw picture for your own analysis (colours, cv2).

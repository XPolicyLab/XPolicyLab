# Geometry reference measurements

Fixed robot dimensions and current units are in `/work/geometry_reference.md`.

The records below describe a small set of reference measurements. Preserve their numbers
as evidence, not universal accuracy bounds, fixed pixel scales, settling durations
or extra steps required for every action. Current task strategy remains in the main prompt.

### Key probe

In a key exploration, a small verified lift separated the two visual hypotheses clearly:
its bow stayed within about 2 pixels of the co-motion prediction; if left on the table,
it would have appeared roughly 46 pixels away in that wrist view. The probe compared "held" and "left behind" predictions after a small motion
using fresh pixels. Closed fingers alone are insufficient. A subsequent 10-degree yaw and
gentle return/release were also observed; these short tests do not establish robustness
for a full insertion or large transport.

### Tracking and image scale

Across the tested poses, measured grasp-reference z differed from requested z by up to
about 3 mm; a small yaw command differed by about 1 degree. These are observed examples,
not universal bounds. Unknown object height can dominate single-view depth
error; these probes used matching features and reprojection checks to assess it.
A small ray-to-ray residual alone is not a guarantee that the pixel correspondence is
correct. In the near wrist views used here, scale was about 0.4 mm/pixel; this is not a fixed
scale for other camera poses or depths.

### Gripper settling probes

Bilateral robot-only probes confirmed empty stops near normalized gripper 0.185
(approximately 0.8 mm inner gap), despite command zero. A 20 mm width measured about
20.48 mm at the first post-command sample and 20.01 mm after a 0.4 s stationary wait;
reopening measured about 84.12 mm and then 88.78 mm. These are observations at tested
poses, not universal settling times. Settling occurred during executed simulation steps, not rendering or API waiting.
The empty-stop value did not distinguish an empty grasp from holding a thin object.

### Robot-only free-space checks

The robot-only probes used actual finger-link poses and robot meshes to check opening and offsets.
Reported tip/pad landmarks are derived from robot state and those fixed offsets; they are
not independent image measurements or contact sensors. Eight small free-space motions
had endpoint errors below 0.18 mm and 0.12 degrees in the tested poses; this does not
replace the larger task-pose errors above. Eight head-image edge patches agreed with
predicted displacement within about 1.1 pixels locally. Neither result bounds unseen
object localization, loaded motion, friction or grasp stability.

### Coin probe

A coin exploration confirmed the ambiguity: a coin remained held through a 23 mm lift
and a small yaw even with the same approximately 0.185 opening as an empty close.
A thin edge occupied only about 12 pixels in the head view; mismatched head/wrist features
produced millimetre-scale depth errors despite plausible geometry. Additional wrist views and comparisons with background motion helped identify
ambiguous correspondences; weak template matches remained unreliable. A projected-pixel agreement checks the correspondence locally;
it does not establish metric accuracy everywhere.

### Bread probe

A bread probe also exposed the sign sensitivity of the ordered points: reversing the
jaw-axis point order caused an unintended near-180-degree wrist roll. This demonstrated that the ordered points encode a signed jaw axis. Small translation/yaw reprojections
agreed within roughly 1–2 pixels in the tested wrist views. The bread was lifted and
returned upright between open fingers; final seating and lasting stability could not
be established from one image. The narrow rack also made sideways withdrawal a risk to seating and support.

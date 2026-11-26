# Evidence

A program carries constants. This file is about the ones that claim to describe an
object, and about the folder that has to back each of them up.

## The rule

For every constant of the measured class — see the table in
[contract.md](contract.md) — there must be a script that recomputes it from data the
robot itself produced, and the recomputed value must land inside the tolerance the
consumer of that constant can live with.

Three things this rule does **not** require, each of which was once believed and had
to be dropped:

- **Not bit-exactness.** The same folder passed on one machine and failed on another
  at the sixteenth decimal. Exact reproduction made the evidence unreproducible, which
  is the opposite of its purpose.
- **Not the same numbers.** The scene and the vision code are stochastic. A wall of
  tiny differences is expected and is not an error.
- **Not a change to the program.** The evidence describes the code as it is. If the
  code cannot be evidenced within tolerance, the honest outcome is an exclusion, not
  an edit made to satisfy a script.

## Where a tolerance comes from

From the consumer. For each constant, find the place that uses it and ask what error
that use can absorb: a rim height that feeds a clearance needs less margin than a
silhouette that feeds a search window; a yaw that is used modulo a symmetry needs a
quarter of the precision it would otherwise. Write that number down next to the
constant.

A constant with no identifiable consumer tolerance cannot be measured and cannot be
offset. It is either environment description or a defect.

## What a folder contains

```
evidence/<task>/
  README.md      purpose sentence, the item table, the file list, the run commands
  check.py       prints program value / recomputed value / difference / tolerance,
                 and exits non-zero when any item is outside its tolerance
  scripts/       stage one extracts from frames and tapes; stage two fits and reduces
  data/          the compact extraction, so stage two runs without the frames
  frames/        a few sample frames, enough to re-run stage one on a sample
```

Two invariants make the folder independent of the program:

- **The scripts read only recorded data** plus the shared geometry, camera and motion
  helpers. They may read the program to obtain the value being compared — and only
  that value — but never as an input to the computation.
- **The README states how many frames of how many shipped.** A reader must be able to
  tell that the sample is a sample, and that stage two does not need the rest.

## What usually cannot be evidenced

Exclude these in writing rather than forcing them in. In this project they were:

- an origin height that the observations fix only as a sum;
- a sign or a quadrant that the observation fixes only modulo a symmetry;
- a centre that needs a model to disambiguate from a single viewpoint;
- values whose only source is the program's own motion plan.

The exclusion file records the item and the reason. A folder that is small and
honest is better than one padded to look complete — and a reviewer who finds a
padded row stops trusting the rest.

## Two failure modes to design against

**A copied value looks like a good measurement.** If a recomputation agrees with the
program's constant far more closely than the method can plausibly achieve, the likely
explanation is that the constant came from the answer and the script was shaped to
reproduce it. The audit role in [stages.md](stages.md) state 11 exists only to catch
this, and its vocabulary must include it.

**A guess dressed as a measurement.** Offsetting a value you already had is not a
measurement. If a value was reached that way, it must not appear as an evidenced row;
record it as excluded, with the word that says so.

## Checking a folder before committing it

- every `check.py` exits zero from inside the working tree;
- the unit tests pass;
- a wording and path scan over every text file, archive member name and array name in
  the folder finds nothing that names a scene, an arrangement, a seed or a source;
- no image decode and no channel conversion anywhere in the staged scripts — frames
  ship as RGB arrays produced by the shared encoder, so a script reads pixels directly;
- a frame patch has the same colour statistics as frames the program's own finder
  reproduces. A channel-swapped folder looks plausible and passes every script, so
  check this once per task rather than trusting the scripts;
- the working tree shows only the evidence folder, and no program file changed.

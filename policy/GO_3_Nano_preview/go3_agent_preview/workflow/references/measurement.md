# Measurement

How a change is judged. This file holds only what a follower has to derive for
themselves; the constants that happen to be true for one benchmark at one moment are
deliberately absent, because copying them is exactly the mistake this folder exists
to prevent.

## Composition of a sweep

The official shape is a full cross product: every task configuration, at every seed,
at its native episode count, split across as many parallel environments as the
simulator allows. Generalization tasks are run in two halves — the plain arrangement
and a cluttered one — and the halves are merged for reporting.

Record, together with any result:

- the configuration list and the episode count per configuration;
- the request budget actually in force, because a longer timeout changes which
  programs finish;
- the number of environments per process;
- the simulator and driver versions;
- the tree hash of the code that ran.

A result without these is not a result; it is an anecdote with decimals.

## Resolution: how big a move can this sweep see

This is the number that governs every decision, and it is derived, not chosen.

1. Run the same code at all seeds and compute, for each task, the observed spread of
   the per-seed values.
2. Compare that spread against the spread episode sampling alone would give at the
   episode count you use. Call the ratio the task's noise factor.
3. Take the mean across tasks of the paired per-seed difference between two runs. Its
   standard error, scaled to your confidence level, is the sweep's resolution: the
   smallest whole-sweep move you can distinguish from noise.

Two facts about this number that matter more than its value:

- **It is a floor, not a promise.** For some tasks the rerun spread exceeds the
  sweep's resolution, because those tasks are contact-rich in a way that either lands
  or does not. A per-task result under the resolution is not evidence, whatever its
  sign. Derive the per-task floor from the task's own spread before you argue about a
  task-level move.
- **It moves with the episode count.** Doubling episodes per seed shrinks sampling
  noise by about the square root of two; it does not shrink a task's structural
  spread.

When a task's noise factor is high, more seeds at the same episode count is a poor
use of the budget. Repeat per layout instead.

## The threshold

The threshold for accepting a change is `baseline − resolution`, and it is written
down **before** the candidate is run. Two consequences follow, and both have bitten:

- A change that lands above the threshold is *not detected as a regression*. That is
  a weaker statement than "no regression" and must be written that way. A real
  regression smaller than the resolution will also go undetected.
- A change that lands above the threshold is *not* thereby an improvement. To claim a
  gain you need the move to exceed the threshold in the other direction, on the same
  rule.

## Pairing

Compare per layout, not per task.

The runner drops an episode and refills the batch when the physics engine gives up,
so "the first N layouts" are not the same N on two runs. Align the two runs on the
layout identity each episode actually used, discard the unpaired remainder, and
report how much you discarded. A paired comparison on a subset is stronger than an
unpaired comparison on everything.

For a task whose noise factor is high, repeat each side at least twice per layout
before comparing.

## Development layouts are not a hold-out

Development arrangements and scored arrangements overlap. In this project the overlap
was substantial enough that a number measured on development layouts was not an
independent estimate of the scored number, and the conclusion had to be retracted.

Treat development results as triage, not as evidence:

- Use them to decide whether to spend on a real evaluation.
- Never report them as a hold-out result.
- Never let a development result be the only support for a claim in a document.

If you need a hold-out, build one — arrangements that you have verified are not in
the scored set — and record how you verified it.

## Per-half reporting

For tasks that exist in two halves, report each half separately before the merged
number. A move that appears in the merged figure but sits in one half only is usually
a layout effect; a move that appears in both halves is more likely to be the code.

## What to do when the answer is "not distinguishable"

Write it down and stop. The record of *tested and did not move* is worth more than a
changed constant, because it stops the next person from re-running the same arm. Two
of this project's largest investigations ended exactly there, and both are more useful
in the repository's history than the change they replaced.

## Reporting

Keep the raw per-episode files and the summariser's own output, unedited. Publish the
per-dimension and per-task table exactly as the official summariser emits it; do not
re-derive it, do not drop columns, and do not merge two runs into one table. If two
runs exist, ship them as two tables each labelled with the code it measured — a table
that does not say which bytes produced it will eventually be quoted as if it did.

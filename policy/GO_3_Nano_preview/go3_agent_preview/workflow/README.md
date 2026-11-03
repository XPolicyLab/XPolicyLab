# Build and improve a task program

This folder records the procedure that produced the task programs in `gm/programs/`
and the evidence folders in `evidence/`. It is written for whoever — a person or a
coding agent — has to build a new program or change an existing one.

It is not a script and it is not a pipeline. An agent sits in the middle of almost
every step, so following it will not reproduce a specific number. What it does
reproduce is the **shape** of the work: what to look at, what to write down, what
must be measured rather than guessed, and what has to be true before a change
counts as an improvement.

## What is in here

| File | What it holds |
| --- | --- |
| [contract.md](references/contract.md) | What a program receives, what it may yield, what it must never read, and where the learned branch takes over. |
| [stages.md](references/stages.md) | The seventeen states, from "this task scores badly" to "a merged commit", with the gate each one must pass. |
| [measurement.md](references/measurement.md) | How a score change is judged: episodes, seeds, noise, the threshold, and what a paired comparison has to look like. |
| [evidence.md](references/evidence.md) | How a constant in a program is tied to something the robot can measure, and what a folder that does this contains. |
| [freeze.md](references/freeze.md) | How a candidate is pinned so the numbers you report belong to the bytes you ship. |
| [review.md](references/review.md) | The independent passes: a verifier that recomputes instead of trusting, an auditor that may see the answer, and an adversarial reader from another model family. |
| [program-shape.md](references/program-shape.md) | A program skeleton and why it is shaped that way. |
| [worked-example.md](references/worked-example.md) | One episode of the loop end to end, with the numbers replaced by their shape. |

## The three phases

Read them as three arcs that share a spine, not as one pipeline.

**Build a program for a task.** Start from the instruction and the robot's own
observations. Measure what the task needs from the images and the joint state, work
out the motion, write the program, run it on development layouts, read the per-step
records, and change one thing at a time. The gate is that the program completes the
task within the episode's step budget on the layouts you may use, and that every
constant it carries either comes from a measurement you recorded or is an explicit
environment description (robot geometry, camera calibration, table height).

**Improve a program that already exists.** Start from a swept result. Attribute the
loss to a mechanism by counting failure modes in the per-step records, state a
prediction that would be true if the mechanism is the cause, and test it with an arm
that changes only that mechanism before you fix anything. Most hypotheses die here.
The gate is a paired comparison on held-back layouts that clears the noise floor.

**Tie a value to a measurement.** Start from a number in a program whose origin you
cannot point at. Decide whether the robot can observe it at all. If it can, measure
it, record the measurement, and ship a small folder that recomputes it. If it cannot,
say so and exclude it rather than dressing a guess as a measurement.

## What to expect

- **Most hypothesised causes are wrong.** A loss that looks like a bad constant is
  often the task's own spread, and a branch that looks like pure waste can be
  neutral when removed. Budget for testing the hypothesis, not only for applying
  the fix.
- **A single task's rerun noise can exceed the whole-sweep resolution.** For some
  tasks three seeds cannot separate anything under about fifteen points. Check the
  noise before you believe a move.
- **Development layouts are not a clean hold-out.** They overlap the scored ones.
  Treat any number measured on them as a development number, and say so.
- **The cheap failure is assuming an agent's report.** Every verification step in
  here exists because a plausible summary turned out to be wrong.

## What this folder will not give you

- A number. The same method on your machine, your model and your layouts gives a
  different number.
- A program for a task. The programs here are what one team's iterations converged
  to; they are examples of the output, not templates to copy.
- Anything a reward, a success flag or a layout identifier would tell you. Those are
  not available to a program at run time, so a procedure that leans on them teaches
  something the deployed adapter cannot do.

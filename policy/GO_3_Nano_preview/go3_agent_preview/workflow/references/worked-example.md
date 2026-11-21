# Worked example

One pass through the loop, with the numbers replaced by their shape. This is not a
transcript and not a reproduction — it is what the loop looks like from the inside,
so a reader can tell whether their own run is going the same way.

## The situation

A task's swept score came in below its own history. The task is contact-rich: two
cubes have to be swapped between a mat and a shelf, the gripper has to hold each cube
through a lift, and the release has to be qualified against what the head camera sees.
Three seeds agreed it was down, and the drop was large enough to be worth a month of
someone's attention.

The obvious suspects, in the order they occurred to us:

1. Two grip constants had been changed recently.
2. The program gives up mid-episode in about a third of the episodes, and the branch
   it gives up to almost never succeeds. Surely that is the loss.

## Stage 3 — attribution

Reading the per-step records of all three seeds: about a third of episodes hand over,
and of those, one succeeds. Of the episodes that never hand over, most finish. The
program's own end reasons cluster into two groups — a hold lost mid-carry, and a
release the camera would not qualify.

That is enough to state a mechanism, but the records cannot say which of the two
suspects is the cause. So instead of fixing, we built arms.

## Stage 4 — arms before fixes

Four arms, same seeds, same layouts, same episode count.

| Arm | What changed | Result across three seeds |
| --- | --- | --- |
| A | ship the delivered bytes | 72 / 48 / 52 |
| B | revert the two constants | 50 / 40 / 48 |
| D | delivered bytes, with every hand-over suppressed | 54 / 70 / 48 |
| — | the run that started the investigation (same bytes as A) | 44 / 52 / 30 |

Three things fell out, and none of them was the thing we expected.

**Reverting made it worse**, by about eleven points on the same seeds. The constants
were not the regression.

**Suppressing the hand-over changed nothing** — the mean was identical to the decimal.
So the hand-over was not the loss either: the program had already given up at those
points, and keeping it in code ended the episode there instead.

**The same bytes, run twice, differed by fifteen points.** Arm A and the original run
were byte-identical, and they scored 57 and 42. Twelve units of essentially one body of
code spanned 30 to 72 with a spread of about eleven.

Layout-level comparison settled it: two runs of identical code at one seed agreed on
which layouts succeeded only 30, 26 and 33 times out of 50 — at chance.

The verdict was written down as *no change*, and the investigation is in the
repository's history precisely because that is a more useful finding than the constant
change we thought we were going to make.

## What the measurement discipline contributed

- **The per-task floor was derived before the task was argued about.** Three seeds at
  fifty episodes cannot separate anything smaller than about fifteen points on this
  task. Every number in the table above is inside that.
- **A sweep-wide noise census put the task in context.** Across the scored tasks the
  median ratio of observed to sampling spread was under one — most tasks are steadier
  than sampling noise — and this task was second worst. That told us the problem was
  this task's property, not a flaw in how the sweep was scored.
- **The claim was weakened before it was published.** "Removing the branch has no
  effect" was recorded as *no detectable effect at three seeds*, not as *zero*, because
  the resolution does not support the stronger sentence.

## Stages 9–12 on the same task

Independently of the score investigation, this task had a set of constants whose origin
had to be tied to something observable. The pass looked like this:

1. For each constant, find its consumer and copy that consumer's tolerance into a table.
2. Design one measurement per class of constant — a multi-view reconstruction for a
   shape, a gripper squeeze for a width, a contact probe for a height.
3. Measure, record the extraction, and write a check script that prints program value,
   recomputed value, difference and tolerance.
4. Hand the frozen manifest to an auditor who may see the answer but may not talk back
   to the measurer.
5. Exclude, in writing, whatever the observations cannot fix — origins that only appear
   as sums, quadrants fixed only modulo a symmetry, centres that need a model to
   disambiguate.

The output is not impressive to look at: a README, a check script, a handful of frames
and a table of numbers that mostly agree to within a millimetre. Its value is that
anybody can run it and get the same shape of answer.

## Stage 16 — what the adversarial pass changed

The first draft of the write-up said the hand-over branch was worthless, citing the
success rate of episodes that used it against episodes that did not. The reviewer
reproduced the numbers and pointed out that the suppressed-branch arm is the
counterfactual, and it did not move — so the success-rate comparison describes when
episodes were lost, not what could have been recovered.

That correction is the reason the conclusion in
[stages.md](stages.md) state 4 is written as *test the hypothesis before you fix*,
rather than as a general claim that hand-overs are bad.

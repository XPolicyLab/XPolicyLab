# Review

Three independent passes, each answering a question the author cannot answer about
their own work. They are separate roles on purpose: collapsing them into one review is
how a plausible summary survives.

## Pass 1 — re-computation

**Question:** does the branch do what its author says it does?

**Who:** a different agent from the author, read-only, with repository access.

**Rules that make it a check rather than a second opinion**

- Run the commands. Do not read the author's summary and agree with it.
- Re-derive the changed values from the recorded inputs — at least five individual
  values and every array — rather than diffing text.
- Check that recorded input paths and hashes still resolve.
- Do not edit the repository. Return a structured list: severity, location, problem,
  proposed fix.
- **Only the non-trivial findings go back to a fixer.** Sending trivia into the loop
  is how a review loop stops converging: each round produces new trivia and the
  substantive issue never gets fixed.

## Pass 2 — audit against the answer

**Question:** is the measurement a measurement?

**Who:** a third agent. In this project it was the only role ever permitted to read
the answer.

**Rules**

- It reads the frozen manifest, not the working tree.
- Its verdicts must be able to say *too exact*: agreement substantially better than
  the method's demonstrated accuracy, which suggests the value was copied from the
  answer rather than observed.
- **Its residuals never go back to the person who measured.** Feedback from an audit
  into the next measurement turns the measurement into a fit. The verdict goes to the
  coordinator; the fix, if any, is a fresh design with a fresh freeze.

## Pass 3 — adversarial reading

**Question:** do the conclusions overreach what the artefacts support?

**Who:** an agent from a **different model family**. A model checking its own work
shares its blind spots; a different family does not.

**Rules**

- The instruction is to refute, not to confirm. A claim survives only if the reviewer
  personally reproduces it from artefacts on disk.
- Each reviewer owns one class of claim — a measurement, a compliance reading, an
  architecture description, a historical account — rather than every reviewer covering
  everything. Overlapping coverage produces agreement, not coverage.
- **An empty findings list is a legitimate and valuable answer.** A reviewer that must
  produce something will produce noise.
- Rulings on the most severe findings are themselves independent, and a ruling must be
  allowed to say the original conclusion overreached.
- Cap the number of findings that get an independent ruling. Beyond that, fix the
  findings rather than adjudicating them.

## When pass 3 is required

At minimum: any document that states a final result or a decision, any audit report,
and any plan or conclusion the author is not certain about. The cost is one review
pass. The cost of skipping it is a conclusion that no independent party has read — and
in this project that pass repeatedly caught claims that had been stated with more
confidence than the evidence carried.

## Three things reviewers in this project actually caught

- A conclusion that a hand-over branch was worthless, inferred from a low success rate
  — when the arm that suppressed it showed no change at all, so the low rate described
  *when* the losses happened, not *why*.
- A file count taken from a `grep` line count rather than from distinct instances.
- A claim that a set of programs did not exist, when they had existed at the evaluated
  commit and had been removed afterwards — visible by diffing two shipped result
  tables.

None of the three needed domain expertise. All three needed someone to open the source
of the claim instead of accepting it.

# Stages

Seventeen states. Two entries — a task that scores badly, and a value whose origin
you cannot point at — that share one spine from state 6 onward.

Each state names who acts. `agent` means a coding model with repository access;
`script` means a deterministic command; `human` means a decision that has been made
by a person every time this ran, and that nothing in this folder automates.

The gates are the point of the list. A state without a passed gate does not hand work
to the next one.

---

## Entry A — a task scores badly

### 1. Observe and open the item

**Input** a swept result: one row per task, per seed.
**Acts** `script` produces the table, `human` decides there is something to look at.
**Produces** a task name, its per-seed baseline, and a note of what looks wrong.
**Gate** the sweep itself has to be trustworthy before its rows are: every unit
reached its native episode count, no restart markers in the logs, one simulator
process per GPU. A result that failed the audit is not a baseline.
**On failure** re-run the unit; do not reason about a result you do not trust.
**Carries forward** the task, the baseline triple, the path to the audit.

### 2. Pre-register the decision rule

**Input** the item from state 1.
**Acts** `human`. This is the step that was never once done by an agent.
**Produces** a written rule, before any result is seen: the threshold, the
resolution, how many runs are affordable, and what counts as agreement between two
different checks.
**Gate** it must be written down before the first candidate result exists. A rule
adjusted after seeing the number is not a rule.
**How the numbers are derived** see [measurement.md](measurement.md) — the threshold
is `baseline − resolution`, and the resolution comes from the sweep's own paired
spread, not from a round number.
**Carries forward** threshold, budget, maximum runs per unit.

### 3. Attribute the loss from the records

**Input** the per-episode records for the candidate task: the step tape, the event
log, saved frames, the per-episode result.
**Acts** `agent`, one task at a time, read-only.
**Produces** a list of failure modes, each with a count and a representative episode,
plus one mechanism that could be changed.
**Gate** every mode needs a count and a representative episode. "It fails a lot"
is not an attribution. The decisive distinction is usually `steps used the whole
budget and nothing moved` (the program gave up) versus `steps far below the budget`
(the run was cut short) — confusing those two sends you to fix the wrong thing.
**On failure** go back and read a different class of record.
**Carries forward** the mechanism and its prediction: *if this is the cause, then
changing only it should move the score by at least X.*

### 4. Try to refute it before you fix it

**Input** the mechanism and its prediction.
**Acts** `agent`, one arm per hypothesis, each on an isolated tree; `human` decides
whether the spend is worth it.
**Produces** N arms measured on the same seeds and the same layouts, and a verdict.
**Gate** the hypothesis must be tested before it is acted on, and a refutation must
be written down with the same care as a confirmation. Keep a list of hypotheses
already refuted so the next person does not repeat them.
**On failure** refuted → back to state 3. Supported → state 6.
**Carries forward** the refutation (so it is not re-run) and the confirmed mechanism.

Worked shape: a task's loss looked like a wrong grip constant. Three arms — ship the
constant, revert it, and ship it with the whole hand-over branch suppressed — gave
72/48/52, 50/40/48 and 54/70/48 on the same seeds. Reverting made it worse, and
suppressing the branch changed the mean by nothing at all. The loss was the task's
own spread. See [worked-example.md](worked-example.md).

---

## Entry B — a value whose origin you cannot point at

### 5. Fix the criterion

**Input** every number, data file and identifier in the tree that might carry an
origin.
**Acts** `human`. This is the only step in the whole list that cannot be mechanised.
**Produces** one criterion that decides, item by item, whether a value is acceptable.
**Gate** the criterion must answer yes or no for a single item. If it needs a
judgement call per case it cannot be handed to an agent.
**On failure** the criterion is unusable; revise it, do not proceed.
**Carries forward** the criterion text and a fixed vocabulary of item states, so two
agents classify the same item the same way.

A workable criterion is stated as an information channel, not as a list of words:
*what the robot can see* — three RGB views, joint and gripper state, contact inferred
from that state, the instruction text — is allowed; anything that came from the
scene file, the asset metadata or the simulator's own state is not. Development-time
success signals are allowed for the human, not for the deployed program.

---

## The shared spine

### 6. Design the measurement or the change

**Input** one item: a file, a symbol, its current value, who consumes it, and the
tolerance the consumer can live with.
**Acts** `agent` designs; `human` decides when a value cannot be measured.
**Produces** an executable design: which observation capability, how many samples,
expected accuracy, and where the tolerance comes from.
**Gate** the tolerance is inherited from the consumer, never estimated. An item
without a tolerance cannot be measured *or* offset — this is the most load-bearing
constraint in the list, because it turns "close enough" from an opinion into a
number written in a file.
**On failure** the design is unreachable → the human declares it unmeasurable and it
goes to the exclusion list.
**Carries forward** the tolerance, the measurement script and its hash.

### 7. Transform in isolation

**Input** the design and a pinned base.
**Acts** `agent`, one unit at a time. One writer per working tree, always.
**Produces** a branch and a record kept outside the repository.
**Gate** touch only the files the unit owns; shared library code has a single named
owner; one commit per coupled group; repository text stays neutral.
**On failure** revert that commit. If two agents are writing one tree, stop writing
and report — never roll back the other agent's work.
**Carries forward** branch head, the record, the list of changed literals.

### 8. Offline self-check

**Input** the branch.
**Acts** `script`.
**Produces** import test, unit tests, task-name resolution, no dangling references to
deleted modules.
**Gate** all green. This is the cheapest reusable proof that a change did not break
the package, and it catches the majority of mechanical mistakes before any GPU time
is spent.
**On failure** back to state 7.
**Carries forward** the self-check output.

### 9. Independent re-computation

**Input** the branch and the author's own report of what it did.
**Acts** a *different* `agent`, read-only.
**Produces** a list of issues with severity, location, problem and fix.
**Gate** the verifier runs the commands itself. It may not accept the author's
summary, may not edit the repository, and for a value change it must re-derive at
least five of the changed values plus every array from the recorded inputs.
**On failure** only non-trivial issues go back to a fixer; trivial ones do not enter
the loop, which is what keeps the loop convergent.
**Carries forward** the confirmed head and the issue list.

### 10. Freeze

**Input** the branch that passed state 9.
**Acts** `script`.
**Produces** a frozen manifest: for every value, its location, the value, the hash of
the script that produced it and the hash of the input data.
**Gate** freezing happens before any audit, and after freezing you may not re-tune
the method. A frozen manifest is what an audit and an integration both read.
**On failure** unfreezing invalidates that round of audit.
**Carries forward** the manifest.

### 11. Audit against the answer, without feedback

**Input** the frozen manifest.
**Acts** a third `agent`, the only role ever allowed to look at the answer.
**Produces** per item: measured, truth, error, the consumer's tolerance, verdict.
**Gate** the audit's verdict vocabulary must be able to say *too exact* — agreement
much better than the method's demonstrated accuracy, which suggests the value was
copied rather than measured. That single verdict is the only mechanism in the list
that catches a guess dressed as a measurement.
**On failure** the item goes back to state 6.
**Carries forward** the verdict, which goes to the coordinator and **not** back to
the person who measured it. A residual fed back becomes a tuning target.

### 12. Write the evidence folder

**Input** the list of items that can be recomputed.
**Acts** `agent` writes into a staging directory; `human` or a coordinator copies it
into the branch.
**Produces** per task, a folder holding a README, a check script, the derivation
scripts, the extracted data and a few sample frames.
**Gate** see [evidence.md](evidence.md). The short version: within tolerance is
enough, the scripts read only recorded data and the shared library, and an item you
cannot evidence is excluded in writing rather than padded in.
**On failure** the item moves to the exclusion file with its reason. A smaller honest
folder beats a padded one.
**Carries forward** a folder anybody can re-run.

### 13. Mechanical gate

**Input** the staged folder.
**Acts** `script`.
**Produces** one table: exit code, size, forbidden-string hits, channel-conversion
hits, stray files.
**Gate** every check script exits zero, no forbidden strings in any text file or
archive member name, no channel conversions, no build artefacts, size under the cap.
**On failure** back to state 12.
**Carries forward** an all-green staging directory.

### 14. Integrate by literal, not by status

**Input** the staging directory and the pinned heads.
**Acts** `script` does the merge; `human` adjudicates a conflict.
**Produces** one assembled tree, a state table and an accept manifest.
**Gate** take the decision **per changed literal, per JSON path, per array key** —
never because another item on the same line was accepted. A mixed line is split or
blocked. A component measured on one part and not another is treated as two items.
The state table is generated from the actual assembled tree and must agree with the
accept manifest. Incomplete output is never reported as complete.
**On failure** block and report. Never scale back silently.
**Carries forward** an evaluable tree and its hash.

### 15. Full acceptance

**Input** the assembled tree.
**Acts** `script` runs the sweep, `human` reads the verdict against the rule from
state 2.
**Produces** the per-dimension and per-task results at the official shape, compared
with the threshold.
**Gate** the whole-sweep result clears the pre-registered threshold, and any single
task that had to lose points has a declared mechanism and an approval.
**On failure** back to state 6 with whatever budget remains.
**Carries forward** a submittable result and a written account of which tasks lost
points and why.

### 16. Adversarial review

**Input** the conclusion and the decision document.
**Acts** an `agent` from a different model family, whose only instruction is to
refute.
**Produces** findings — claim, evidence, consequence — plus an independent ruling on
each of the most severe ones.
**Gate** each verifier owns exactly one class of claim, and must reproduce the claim
from artefacts on disk before accepting it. An empty findings list is a legitimate
answer. Rulings must be allowed to say the original conclusion overreached.
**On failure** back to states 6–15.
**Carries forward** a sign-off, or a list of claims that did not survive.

Why a different model family: a model verifying its own work shares its blind spots.
The cost of this step is one review pass; the cost of skipping it is shipping a
conclusion nobody independent ever read.

### 17. Commit and merge

**Acts** `human` authorises, `agent` executes.
**Gate** the commit message describes what changed without naming a seed, an asset, a
arrangement or a source word; the repository carries no process documents; the tests
pass; any claimed resource is released.
**Carries forward** the commit.

---

## Where this list is weakest

- **State 2 is a person writing a rule.** Nothing here replaces that, and a follower
  who skips it will read noise as signal.
- **State 5 is a person choosing a criterion**, and it changed three times in one
  week when this ran. A criterion is a decision, not a discovery.
- **States 11 and 16 need a role the rest of the team cannot see.** If nobody is
  allowed to read the answer and nobody is outside the author's model family, both
  collapse into a second opinion from the same perspective.

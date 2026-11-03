# Freeze

A result belongs to the bytes that produced it. Freezing is the mechanism that keeps
those two attached.

## What freezing is

Before any audit or acceptance run, write a manifest that names, for everything that
influenced the result:

- its location in the tree;
- its value, or its content hash;
- the hash of the script that produced it;
- the hash of the input data that script read.

The manifest is what an auditor reads and what an integration reads. Neither reads
the author's description of what they did.

## Why it is mechanical

A freeze that lives in a habit fails silently. The failure mode is not a wrong number;
it is a number whose support has moved on. Two examples from this project:

- a program's constant was re-tuned after its evidence folder was written, so the
  folder proved a value the program no longer carried;
- a branch was assembled from unit commits whose messages described an earlier state,
  so the tree that was evaluated could not be reconstructed from the history alone.

Both were caught by comparing hashes; neither would have been caught by reading.

## After the freeze

- **Do not re-tune the method.** If a measurement misses its tolerance after the
  freeze, the item goes back to the design stage and the freeze is redone. Adjusting
  the fitting parameters after seeing the audit is how a measurement becomes a fit to
  the answer.
- **Do not fold late changes in.** A change made after the freeze is a new candidate
  and needs its own manifest and its own acceptance run.
- **Re-verify on every load.** A frozen artefact that is copied, moved or re-exported
  should be hash-checked at the point of use, not only at the point of freezing. A
  silent mismatch there means the results and the code have separated.

## Freezing and the learned branch

The same discipline applies to the parts of the system that are not code:

- the model weights, by hash;
- the vendored inference configuration, by hash;
- the normalisation statistics, by hash;
- whichever inference profile ran, named explicitly, because two profiles can ship in
  one tree and only one of them produced the number.

A run whose report does not say which of those four it used cannot be tied to a
release, and a reviewer will ask.

## What freezing does not cover

Freezing pins artefacts, not judgement. It cannot tell you whether the threshold was
pre-registered, whether a task lost points for a declared reason, or whether the
hold-out was a hold-out. Those are covered by
[measurement.md](measurement.md) and [review.md](review.md), and no amount of hashing
substitutes for them.

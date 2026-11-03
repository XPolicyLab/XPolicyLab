RSIWAM: historical plain cleanup closure

This package records the independent administrative reconciliation of the
old plain-run reservation. The selected source files remain unchanged.

The literal plain debug loop at commit
776c9f3b177427c1078ee31daaa4db93f922008b completed 10 episodes and 200 action
steps. Its original owner then failed during cleanup with:
  reservation: Busy('Inherited lifetime FD still held; reservation retained')
The original overall status remains FAILED. That historical result has not
been rewritten, and the later reconciliation is not presented as a successful
release by the original owner.

The separate reconciliation receipt records closure at
2026-11-02T04:40:22.088489Z. Only released_at changed in the exact old plain
reservation row. It records an exclusive target lock, empty relevant holder
and live-identity lists, no signals, no GPU operations, no owner impersonation,
and no changes to other reservation rows or database events. These findings
refer to the target reservation and its recorded identities, not every process
on the host. The earlier read-only receipt is also included unchanged.

The independent review at 2026-11-02T04:41:31.278148Z verified source hashes,
the original lease identity, the unchanged historical plain files, and equality
between the actual database row and receipt.after. The collector read that
exact row through a read-only SQLite mode=ro parameterized query and compared
fields in memory. It saved only the verification conclusion, not a separate
query-output original. No such independent row original is claimed here:
receipt.after is a row recorded inside the receipt. The process/FD evidence
was checked for historical internal consistency, not rescanned by that review.

At that collection time encoded was WAITING_RESOURCE and had no guardian
result. The included encoded receipt is CPU_PREFLIGHT_PASS with literal_executed
and gpu_execution both false. This package does not establish encoded GPU
success or a plain/encoded overall PASS. It is a dated record, not a fresh
query of current runtime status or a benchmark success-rate result.

Five selected originals are copied byte-for-byte:
* originals/literal_plain_orphan_reconciliation_r1/receipt.json
* originals/literal_plain_orphan_reconciliation_r1/readonly.json
* originals/literal_attempt_r2/status.json
* originals/literal_encoded_cpu_receipt_r3.json
* source_review/VERIFICATION.json

SOURCE_MANIFEST.json maps each copy to its source path, size and SHA256 and
explains omitted material. SUMMARY.json, this README, PRIVACY_REVIEW.json and
PACKAGE_SHA256.json are derived documents, not raw execution receipts.
PACKAGE_SHA256.json hashes every public file except itself.

The minimal package excludes broad diagnostics, full process inventories,
full environments and broker configuration. The prior plain guardian and
release evidence remain preserved separately; their hashes are bound here.
No credentials or contact information were found by the focused scan and
manual content review. Necessary task-local runtime identifiers remain,
including the closed reservation ID, owner/PID identities, GPU UUID, hostname,
boot ID and local paths. This is not an anonymized dataset. The reservation
token is the identifier of the old closed lease, not an account/API credential.

This evidence can be published as a scoped historical cleanup reconciliation
with these limitations. It must not be relabeled as the original cleanup
passing, independent raw database-query evidence, encoded passing, or overall
two-mode validation passing. This preparation performed no remote operation,
GPU test, repository edit, or publication.

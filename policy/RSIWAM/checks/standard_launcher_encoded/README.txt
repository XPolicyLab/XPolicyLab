RSIWAM: standard literal plain and encoded entry accepted
Tested commit: 776c9f3b177427c1078ee31daaa4db93f922008b
Evidence attachment for the standard literal launcher acceptance.

The unchanged policy/RSIWAM/eval.sh completed the official synthetic debug
TestEnv in the encoded continuation: 10 episodes, 20 actions each, 200 total,
literal and guardian exit 0. Original stdout ends with [MAIN] eval finished.
The release receipt records initializer/reservation released, two brief release
retries and no final cleanup errors. The earlier plain debug loop also completed
10x20; its original overall FAILED cleanup status remains unchanged.

Final cleanup evidence has two distinct sources:
1. The owner captured a genuine post-completion read-only audit. Root independently
   verified its archive, hashes, counts, exact released lease, 338 absent owned
   identities and actual successful private-lock exclusive probe locally.
   source_review/OWNER_CAPTURE_LOCAL_VERIFICATION.json is that byte-preserved
   local verification. OWNER_AUDIT_PROJECTION.json identifies the remote source
   and its SHA. This is not a root-captured remote full-host audit.
2. Root subsequently performed an actual, narrowly scoped remote check: 339
   exact recorded identities were absent, the same private-lock inode permitted
   an exclusive probe which was immediately released, and the exact lease was
   released. ROOT_SCOPED_CLEANUP_PROJECTION.json records this scope and source SHA.
   It performed no arbitrary host FD scan, ledger mutation, signals or GPU launch.
Three extra root full-host scans remained incomplete because FD reads encountered
PermissionError. Their original snapshots are preserved unchanged; their hashes
and incomplete results remain in AUDIT_HISTORY_PROJECTION.json. They are not
relabelled PASS. Acceptance does not claim full-host FD visibility.

Mode and environment:
Config selects encoded. During execution, a separate read-only observation
captures the matching actual server PID/starttime, r5 clean-environment executable,
DEBUG_OBS_ENCODED=1, EVAL_ENV_TYPE=debug and the selected GPU. This captures only
the server, not an independent client environment or encoded wire payload.
MODE_EVIDENCE.json states this boundary. A later query occurred after termination:
POST_COMPLETION_QUERY_SCOPE.json identifies its empty stdout as an unsuccessful
post-completion query, not a policy evaluation result.

Relationship to prior plain evidence:
Plain completed 10 episodes/200 actions, but reservation cleanup originally
failed. Its original FAILED status and later independent administrative closure
remain separately recorded. This continuation did not rerun plain or rewrite
that failure. PRIOR_PLAIN_LINKS.json pins the already-published originals in
policy/RSIWAM/checks/standard_launcher_plain and standard_launcher_cleanup.
The combined acceptance is derived across these separate runs; it is not one
uninterrupted two-mode run that succeeded on its first attempt.

Evidence:
originals/ preserves official stdout/stderr, guardian result, release receipt,
initialization closure, combined result, CPU preflight, r5 clean CPU identity and
the two small mode-observation records byte-for-byte. CPU records prove preflight/
installation identity, not GPU execution. SOURCE_MANIFEST.json pins source paths,
bytes and SHA. CONFIG_IDENTITY_PROJECTION.json explicitly selects fields from the
original 2 MB config; the full broker settings and repository-wide hash inventory
are omitted. The owner and root raw cleanup process inventories remain in the
private source collection; only exact scope projections and their source SHA are
published. No source file labelled original is silently redacted.

Only public_payload/ is the publication candidate. Full configurations, full
environments, whole-machine process inventories and unrelated work are excluded.
Unmodified guardian/release originals retain this run's owned process lineage,
ordinary PIDs, paths, GPU UUID and historical initializer/reservation IDs. The
nested reservation object in release_receipt is an acquisition-time snapshot,
not the final database row; the post-completion audit projections separately
record the released row. PRIVACY_REVIEW.json describes the focused review.

Scope: standard launcher/interface acceptance using synthetic debug inputs.
This is not a native task SR, full native parity, a new plain/encoded action-byte
equivalence test, a completed 6,300-episode benchmark or official leaderboard
acceptance. No GPU workload or remote collection ran to prepare this public
selection. PACKAGE_SHA256.json covers all payload files except itself.

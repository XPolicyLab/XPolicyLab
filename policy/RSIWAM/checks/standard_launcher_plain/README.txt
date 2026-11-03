RSIWAM — partial standard-launch evidence: plain r2

This selection publishes a partial historical standard-launch attempt.
SUMMARY.json is derived. All eight files under plain_r2/ are
unchanged original files; their original relative paths are retained beneath that
prefix and mapped with SHA256 values in SOURCE_SELECTION_SHA256.json.

The unmodified standard eval.sh plain debug loop at commit
776c9f3b177427c1078ee31daaa4db93f922008b completed episodes 0 through 9,
10 episode-finished records and 200 action steps. The literal command and guardian
both returned exit code 0; the client log includes [MAIN] eval finished. These
counts were independently recomputed from the selected original stdout/stderr.
The recorded actual server used the isolated r5 policy environment and selected GPU.
The r5 installation input hashes match the separately prepared installation evidence.

The overall attempt FAILED during resource cleanup. The original release receipt
retains status LITERAL_MODE_PASS but also records:
reservation: Busy('Inherited lifetime FD still held; reservation retained')
The owner then raised RuntimeError('Exact owned cleanup incomplete'), and the
original overall status is FAILED. The owner traceback is included. Do not interpret
the mode status string or successful client loop as complete cleanup acceptance.
Plain stderr is nonempty (41 bytes of socket-port information); only the client
stdout/stderr lack a traceback. The separate owner log contains the cleanup traceback.

Encoded did not start in this attempt. Its acceptance remains pending separate
continuation evidence. No independent cleanup-closure receipt is included yet.
Later cleanup closure and encoded results must be appended as distinct evidence;
the original plain cleanup failure and original overall failure must not be rewritten.
These records do not identify the actual transient FD holder or establish its cause.

Included originals
- Complete plain stdout/stderr, guardian_result and release_receipt.
- initialization_closed, showing the actual server and the initialization release
  handoff; this does not establish final capacity-reservation cleanup.
- Original overall status, CPU preflight receipt and owner traceback log.

The guardian/release receipts retain their historical known-owned process records.
Every retained record belongs to the recorded guardian session. They are related
cleanup evidence, not a dump of unrelated host processes. Actual-server environment
metadata is limited to CUDA_VISIBLE_DEVICES, CUBLAS_WORKSPACE_CONFIG, CONDA_PREFIX,
PYTHONPATH and PYTHONNOUSERSITE. No full environment or full broker configuration is
included. Normal PIDs, runtime paths, GPU identifiers and resource-lease identifiers
are retained. Intermediate duplicate status files and operational bind/launch files
remain in the original source collection and are listed as omitted in the manifest.

PUBLIC_FILES_SHA256.json covers all selected and derived public files except itself.
PUBLIC_SELECTION_REVIEW.json documents the focused credential/private-phone and
context review. The review does not prove absence of every possible secret encoding.

This is synthetic debug interface evidence, not simulator task/recovery success,
full native-episode parity, plain/encoded action equality, a completed 6,300-episode
campaign or official leaderboard acceptance. It is not an overall two-mode PASS.

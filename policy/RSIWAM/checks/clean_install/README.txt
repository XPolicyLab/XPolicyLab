RSIWAM — selected r5 installation and CPU evidence

This directory is prepared for review/publication; preparation did not publish it.
SUMMARY.json and the manifests are derived records. installation_r5/ contains
46 selected original files copied byte for byte from the completed r5 attempt.
The checked source revision is 776c9f3b177427c1078ee31daaa4db93f922008b.
The attempt completed at 2026-11-01T15:40:19.396758+00:00.

What this evidence establishes
- The corrected install.sh ran in the newly provisioned isolated policy environment
  as an r5 repair continuation. All 14 recorded stages returned exit code 0.
- All 27 declared bundle requirement specifiers were satisfied after installation;
  requirements.json records each requirement, installed version and result.
- The final environment has torch 2.5.1+cu118, torchvision 0.20.1+cu118,
  NumPy 1.26.4 and opencv-python / opencv-python-headless 4.11.0.86.
- Final pip check found no broken requirements. Real CPU imports, module paths,
  configuration and fixed adaptation-weight identities passed the recorded checks.
  The model was not constructed, CUDA was not initialized and no server was started.

Repair history and stderr
This was not a one-shot installation of revision 776c9f3 into an empty environment.
The isolated environment had been created and populated during earlier attempts.
r4's import checks passed with NumPy 2.2.6, which violated the bundled numpy>=1.26,<2
requirement. prior_r4_scope.json preserves that limitation and its original receipt
hash. The original r4 receipts remain unchanged; they are not final acceptance.

r5 first downgraded NumPy to 1.26.4. At that intermediate point pip reported that the
inherited opencv-python-headless 5 required NumPy 2. That message remains in the
complete original stderr. The later XPolicyLab installation selected headless
4.11.0.86; complete stdout records its replacement. Final pip check and the explicit
27-requirement check then passed. The logs are not claimed to have empty stderr.

Evidence files
installation_r5/installation_receipt.json is the original final receipt, including
its original status string. SUMMARY.json interprets its scope as a repair continuation
without changing the original. The raw directory includes prior scope, requirements,
clean_cpu, source identity, active prefix, reference/installed metadata, full install
stdout/stderr, pip check, pip freeze and all 14 stage exit receipts.

reference_distribution_receipt.json records a version-only dist-info comparison with
the reference runtime; it records tensor_imported=false. clean_cpu.json records the
actual imports from the isolated environment and checked source checkout. The exact
local-wheel SHA references in pip freeze originate from prior provisioning; the r5
attempt does not claim to have repeated all wheel downloads or GPU binary checks.

pip-freeze.stdout.log is an observed inventory, not a portable requirements lock.
It retains machine-local wheel paths, conda build-origin metadata and Git origin
URLs reported by pip. A Git origin URL in that inventory is not a claim that the
checked commit is available from that particular upstream remote. Use the public
RSIWAM integration repository and its installation instructions for reproduction.

The 1,264,942-byte pip-inspect.stdout.log is intentionally omitted. Its SHA256,
2d8e353c3d2716a0e5fb853b01b3f0ee66c74ff19f5fcdf6e309b76b581bebf3,
is retained in the unchanged original receipt and SOURCE_SELECTION_SHA256.json.
The separate historical r4 canonical receipts and operational launch record are
also omitted. All source originals remain unchanged in the source collection.
SOURCE_SELECTION_SHA256.json maps selected files to original paths, sizes and hashes.
PUBLIC_FILES_SHA256.json hashes every public file except that manifest itself.
PUBLIC_SELECTION_REVIEW.json records the focused credential/private-phone review.
Ordinary runtime paths and public source/mirror metadata are retained. This review
is not an exhaustive guarantee against every possible secret encoding.

Remaining limits
This installation evidence does not establish literal eval.sh plain/encoded success:
no literal evaluation ran in this attempt, and acceptance remains pending separately.
It does not establish CUDA inference in this environment, full native-episode parity,
recovery/task success, completion of the 6,300-episode campaign or official leaderboard
acceptance. Earlier GPU debug evidence has its own evaluated revision and scope; it
is not relabelled as this attempt. Native parity and the full campaign are separate
work and are not added here as prerequisites to opening or reviewing the integration PR.

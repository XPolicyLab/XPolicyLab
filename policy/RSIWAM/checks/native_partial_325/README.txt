RSIWAM: 325 accepted episodes from a partial local 6,300-episode campaign
Frozen from local evidence on 2026-11-02; this is a partial-results attachment.

This attachment freezes seven accepted complete task-seed blocks at the source
snapshot time 2026-11-02T04:54:56.236969Z. Six queue blocks contain 275 episodes;
the external first block, general_pickup, contains 50. The snapshot credit 325
already includes that 50: it is counted once.

Observed partial results (episode-weighted):
  task / evaluation seed             success / N       SR %      Score / 100
  general_pickup / 0                      10 / 50       20.00        20.00
  align_blocks / 0                         0 / 50        0.00         0.00
  arrange_largest_number / 0               1 / 25        4.00        12.20
  build_tower / 0                         44 / 50       88.00        90.20
  cover_blocks / 0                        50 / 50      100.00       100.00
  deposit_coin / 0                         9 / 50       18.00        23.20
  classify_objects / 0                   23 / 50       46.00        55.50
  all selected complete blocks          137 / 325      42.153846    45.384615
Mean raw per-episode Score is 0.45384615384615384 (score sum 147.5 / 325).
Original details[*].score uses 0–1; top-level score uses 0–100. Original
success_rate uses 0–1. Each aggregate was independently recomputed from every
existing episode, with no missing value filled with zero. These numbers are
not an unweighted average of the seven task means.

Scope and limitations:
- Selection is exactly the accepted completed blocks in this fixed progress
  snapshot, not a representative sample of all official tasks. Only seed 0 is
  present. This is not the full 6,300-episode benchmark result or leaderboard SR.
- The pending arrange_largest_number_random (6/25) and
  classify_objects_by_language (35/50) records are excluded. Other unfinished
  blocks are also excluded, not counted as failures. No newer results are added.
- There is no paired baseline or ordinary-LoRA control in this attachment.
  Task success is not a recovery-specific success rate (Recovery SR).
- Existing native layout_id values are preserved. They do not refer to the
  separate canonical210 experiment; no exact physical initial-state matching
  is asserted. native_loaded captures the first layout, not every initial state.
  Native layout IDs can have gaps; 50 valid records do not establish layouts
  exactly 0–49. Native load evidence explicitly reports that a complete initial
  physics snapshot was unavailable. No simulation or GPU test ran for collection.
- These results do not establish official leaderboard acceptance, a literal
  encoded launcher pass, or cross-device bitwise parity.

Candidate and source binding:
Frozen VPP2 + frozen original RecoverWAM parent (seed 20261032) + success133
Action LoRA (seed 20261033, step 400); candidate
recoverwam_success133_original_parent20261032_lora20261033.
Recorded XPolicyLab adapter revision: 7e5def6e61435e823f5b346f2b9d2dd5a7a78256.
Model seed is 1; environment/evaluation seed is 0. Both policy-load originals
agree on the candidate, step, checkpoint, and these two distinct seed roles.
Shared policy target-acceptance evidence records these identities:
  parent SHA256: 861529eb89e4279e4490d6156bf04a335eee410f6c02842d0ac3db2dc90431d3
  Action LoRA SHA256: eec21ec2834ef95bffdac7b3df0b48092a215aa67e7bf6952ec1643ab511b361
  base identity: ae66dbfc16eb3c75fa89111501412ecb14cd3ab62ce02ff9fc8a457bb3b19aba
The configuration revision and actual loaded-constructor identity agree. This
read-only collection did not re-audit all remote source-code or weight bytes.

Evidence and reproducibility:
- originals/: 18 small JSON files copied byte-for-byte: seven _result.json,
  seven native_loaded.json, two policy_loaded.json, and external first-block
  completion/status. SOURCE_MANIFEST.json records remote path, bytes and SHA.
- Six result SHAs match the fixed snapshot; the external result SHA matches
  its original completion receipt. The first block is complete with native
  exit 0 and released own leases. All seven task/seed/config/loaded identities
  and expected episode counts were checked.
- ACCEPTANCE_PROJECTION.json and CONFIG_IDENTITY_PROJECTION.json are explicitly
  derived selections. Full queue/config/environment/process snapshots are not
  included. The source snapshot SHA is
  0aedb132a26140160c7a9fe72a5b8a28ce691fc7dbbe3cc4daa3f5eed52d0296.
- VIDEO_SHA_REFERENCES.json retains 975 existing completion-time video path,
  size and SHA records. No video bytes were transferred or rehashed here.
- RESULT_MANIFEST.json pins the seven originals. Recompute without network/GPU:
    python3 recompute_summary.py
  Its JSON output must equal SUMMARY.json; SHA256SUMS.json pins this payload.

Own-task runtime paths, ordinary PIDs, GPU UUIDs and observation metadata are
retained for provenance. A focused credential/contact/unrelated-task review is
recorded in PRIVACY_REVIEW.json. It is not a claim of exhaustive secret detection.

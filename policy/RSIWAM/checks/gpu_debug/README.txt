RSIWAM — selected public GPU validation evidence

This directory contains selected original evidence and an explicitly derived
summary for prepared XPolicyLab commit:
7e5def6e61435e823f5b346f2b9d2dd5a7a78256

What passed
- One real CUDA model load on NVIDIA A800-SXM4-80GB.
- Exact frozen candidate:
  recoverwam_success133_original_parent20261032_lora20261033
  Original recovery parent seed 20261032; Action LoRA seed 20261033, step 400.
- Unchanged official synthetic debug client, plain and encoded RGB transport.
- Each mode completed one 20-action episode, with one actual inference call,
  client exit code 0, and empty stderr. No simulator was used in these two cases.
- Both returned action chunks are float32 [24,16] and bitwise equal. The loop
  stopped after the official episode's 20 actions. History mismatch counts are
  zero, and the two client processes used distinct actual deployment client IDs.
- Model inference seed remained 1; the server's recorded evaluation seed was 7.
- Debug execution finished on 2026-11-01 at 13:55:51 UTC.

The original official CLI supports --eval_episode_num 1. The default ten episodes
were not required or claimed. The real model was loaded once and shared by the
two sequential debug clients. An evidence-only observer subclass called the exact
committed model constructor and inference methods; it did not alter weights,
observations, actions, history, reset, or seed behavior. Phase names in action
events come from a runner sidecar. Actual deployment client IDs are separately
recorded; phase labels are not presented as model-observed websocket metadata.

Files and provenance
- server.stdout.log / server.stderr.log: original complete policy-server streams.
- plain.stdout.log / plain.stderr.log and encoded.stdout.log / encoded.stderr.log:
  original complete official debug-client streams.
- load_identity.json: original post-constructor identity, frozen weights,
  source/base hashes, actual CUDA parameters/device, and loaded training metadata.
- action_events.jsonl and actions_001.npy / actions_002.npy: original action
  evidence and actual returned arrays. Load arrays with allow_pickle=False.
- ORIGINAL_DEBUG_FILES_SHA256.json: unchanged original manifest of all 18 debug
  evidence files. Some operational files listed there are intentionally not
  copied into this public selection.
- SUMMARY.json: DERIVED_SUMMARY_NOT_ORIGINAL_RESULT_JSON. It selects facts from
  the original RESULT.json and exit receipts, with original SHA-256 references.
  It also explicitly labels the later native first-block follow-up as derived.
- PUBLIC_SELECTION_REVIEW.json: focused publication-selection checks.
- PUBLIC_FILES_SHA256.json: manifest for the actual files in this directory,
  excluding that manifest itself.

All ten selected raw files were copied byte for byte. The original 18-file debug
manifest was independently rechecked with no missing files or mismatched hashes.
The full original RESULT.json, plan.json, exit commands, and surrounding owner
resource records remain in the private operational evidence directory. They were
not modified. Their omission is disclosed rather than silently redacting an
original record. Ordinary runtime paths, host/process identifiers and GPU model
metadata retained in the selected evidence are not authentication credentials.
Focused checks found no private-key block, recognizable credential value, or
private telephone-number pattern in the selected files; this is not a universal
secret-detection guarantee.

Native first-block follow-up
SUMMARY.json also binds the first native simulator block to the same unchanged
RSIWAM Model constructor at commit 7e5def6, the same parent32/LoRA33 candidate,
step 400, and model seed 1 with evaluation seed 0. Its real post-load receipt
confirmed CUDA initialization. A separate native first-episode audit verified
one episode and three videos. The latest included local snapshot, 2026-11-01
14:27:20 UTC, recorded two native results in the planned first 50-episode block.
This is a dated partial snapshot, not a live progress report or completed score.
The original private receipt hashes are provided in SUMMARY.json.

What this evidence does not establish
- A clean environment installation, or validation of the standard eval.sh
  launcher itself; existing interpreters were reused with process-local imports.
- Full native-episode output parity against the older native implementation.
- Completion of the first 50-episode simulator block or the full 6,300-case plan.
- An official hidden-layout evaluation, leaderboard rank, or superiority over
  VPP2. The synthetic debug observations are interface checks, not performance
  measurements.

The server exit code -15 in the derived summary records intentional cleanup of
the server created for this validation, after both clients completed successfully.

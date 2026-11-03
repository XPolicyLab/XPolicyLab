# Append RoboDojo data during training

Hot loading is opt-in (`ROBODOJO_HOTLOAD=false` by default). It keeps the existing
training recipe, model, optimizer, scheduler, augmentation and z-score statistics.
The publisher accepts uniform H24/C1 (`flash`) and H24/C24 (`memory`) indices and
restricts additions to the tasks already present in the base index. It samples the combined pool with
the original shard weighting; new data receives no additional priority.

Choose the model and its matching base index before starting the job. The
publisher reads the horizon, chunk limit and stride from that index; it does not
need a separate model flag. New memory episodes use the same temporal packing
as offline indexing, with up to 24 chunks and their actual lengths recorded for
global bucketing. A running job cannot switch from C1 to C24 or change its
normalization. Use a separate snapshot directory for each base index.
Flash and memory hot loading have been tested with sequential episode additions.
Exact checkpoint resume validation remains pending.

Run **one publisher** in a separate terminal or service. Producers must write
the official RoboDojo HDF5 layout, close each temporary file, and atomically
publish its final name. Every producer must provide this guarantee:
an unchanged file size or mtime does not prove completion. Hidden staging paths
and `.pending`/`.partial`/`.part`/`.tmp` files or directories are ignored. Live
episodes require `success=true` and `recording_valid=true`; their stored action
labels are retained.
Success flags must be scalar booleans or binary integers; text such as `"false"`
is rejected instead of being interpreted as a successful recording.
State, action and camera sequences must have equal lengths. No producer or
normalization statistics are rewritten.

```bash
cd XPolicyLab/policy/WorldScape_RSI/worldscape-policy
export WSP_PYTHON=/path/to/policy-env/bin/python
export PYTHONPATH="$PWD/src:$PWD:$(cd ../../../.. && pwd)${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# For memory, select the matching H24/C24 uniform index instead.
export ROBODOJO_INDEX_PATH=/path/to/h24_c1_stride1_index.npz
export ROBODOJO_HOTLOAD_DIR=/path/to/robodojo_hotload
"$WSP_PYTHON" -m worldscape_policy.cli.publish_robodojo_data \
  --base-index "$ROBODOJO_INDEX_PATH" \
  --watch-root /path/to/finalized_live_episodes \
  --publish-dir "$ROBODOJO_HOTLOAD_DIR" --watch --poll-seconds 30
```

Use `--once` for one scan; `--watch-root` is repeatable. Existing finalized files
not represented by the base index are included. Invalid or duplicate additions
are reported without changing the producer's files. A changed already-published
source is an error. Publication creates immutable `versions/<version>/index.npz`
and `manifest.json`, then atomically advances `latest.json`. The original index
is preserved as v0. Each update indexes only new episodes; it does not regenerate
statistics or decode the complete video dataset. Output paths are selected by the caller. Optionally set `WSP_WRITABLE_ROOTS`
to an `os.pathsep`-separated list of allowed directories (`:` on Linux) on both
publisher and training processes. Symlinks are resolved before checking these
boundaries. When unset, filesystem permissions govern writes. This setting also
applies to hot-load journals and optional timing traces.

Launch the following command on **each training node**, setting that node's rank.
Every node must see the same code, data, statistics, snapshot directory and shared
output directory at the same paths. `MLP_WORKER_0_HOST` is a mutually reachable
training-network address, not an SSH-forwarding port.

```bash
cd XPolicyLab/policy/WorldScape_RSI
export WSP_PYTHON=/path/to/policy-env/bin/python
# Also set DATA_ROOT, ROBODOJO_ZSCORE_STATS, ROBODOJO_INDEX_PATH,
# PRETRAINED_MODEL_PATH and WSP_MODEL_ROOT as in the training example above.
export ROBODOJO_MODEL=flash  # Use memory with the publisher's H24/C24 base index.
export ROBODOJO_HOTLOAD=true
export ROBODOJO_HOTLOAD_DIR=/path/to/robodojo_hotload
export RUN_NAME=robodojo-hotload-3x8-run01
export OUTPUT_DIR="/path/to/checkpoints/$RUN_NAME"
export WANDB_RUN_ID="$RUN_NAME" WANDB_RUN_NAME="$RUN_NAME"
export MLP_WORKER_NUM=3
export MLP_ROLE_INDEX=0  # Use 1 on the second node and 2 on the third.
export MLP_WORKER_0_HOST=192.0.2.1  # Replace with this job's node-0 address.
export MLP_WORKER_0_PORT=29500
export NUM_GPUS=8 PER_DEVICE_TRAIN_BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=1
bash train.sh RoboDojo "$ROBODOJO_MODEL" arx_x5 joint 62 0,1,2,3,4,5,6,7
```

For one node use `MLP_WORKER_NUM=1`, rank 0 and host `127.0.0.1`; for two nodes use
2 and ranks 0/1. With batch 8 and accumulation 1, global batches are 64/128/192 for
1/2/3 nodes. Learning rate remains `6e-5`; it is not automatically scaled.
`MAX_STEPS=50000` continues to define the scheduler and stopping point. For a short
run, use `RUN_STEPS` or `training.run_steps`, not a reduced `MAX_STEPS`.

Training starts from v0 and considers the latest complete snapshot at consumed
global-bucket window boundaries. Window size 128 is a planning budget, not an
exact number of optimizer steps. All global ranks must prepare the same version
before committing it. Preparation uses an isolated CPU process, with a default
120-second limit (`ROBODOJO_HOTLOAD_PREPARE_TIMEOUT`); a failed candidate leaves
the job on its previous version. Committed pending sample occurrences are carried
forward, while speculative future prefetch is rebuilt. Switching may briefly
pause training to recreate workers and caches. Versioned sampling changes future
draws, so it does not reproduce a run which had all data available from step zero.

The equivalent Hydra interface is `data_loader.hotload.enabled`, `manifest_dir`,
`journal_dir` and `prepare_timeout_seconds`. The recipe places the durable commit
journal under `$OUTPUT_DIR/data_versions`. Version/count/timing metrics use the
`hotload/` prefix, and deferred updates include a structured reason in the log.

For exact resume, run the same command on all nodes with these trailing arguments:

```bash
bash train.sh RoboDojo "$ROBODOJO_MODEL" arx_x5 joint 62 0,1,2,3,4,5,6,7 \
  "training.resume=$OUTPUT_DIR/checkpoint-5000" training.resume_mode=exact
```

Keep the same total GPU count, per-device batch and accumulation. A complete static
training checkpoint from the same base recipe can be upgraded to hot loading.
Retain its original `RUN_NAME` (`selectors.run_name`) and other training settings,
and enable hot loading; do not replace the run name with the fresh-run example
above. The output directory may change. Its original
cursor is restored as v0 before adopting new versions, and its exported transform
bundle must contain the same initial normalization values. Normalization remains
fixed throughout the run, including all newly added episodes and checkpoint
exports; the publisher never recomputes statistics. Preserve the original statistics
JSON without changing its values or formatting: dynamic recovery also verifies its
file SHA256. Fast resume is rejected
for hot loading. A dynamic checkpoint cannot silently resume with hot loading
disabled. Recovery first restores the checkpoint's pinned snapshot and consumes
previously committed journal events at their recorded boundaries, then discovers
new versions. Preserve **the snapshot store, referenced episode files and the
run's `data_versions` journal** alongside checkpoints. Copy the journal when
continuing a run in another output directory; do not edit it or remove older
referenced snapshots. A failure after a durable commit stops the job for recovery
instead of letting individual nodes continue on different versions.


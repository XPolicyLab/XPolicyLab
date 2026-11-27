# Validation: run20261101_02, step 30000

This release selects the three 2026-11-01 training runs at learning rate 5e-5,
initialized from step 256013. All three source `slot_6.pt` payloads report actual
step **30000**. The rotating slots were copied into stable release files, with
source size/mtime stability and full SHA256 checked during and after copying.
Exact filenames, byte sizes and hashes are pinned in `phoenix_release.json`.

## Training and loading contract

The saved training raw index, q01/q99 statistics, paired-cache contract,
temporal-trim contract, image geometry/source and frozen encoder identities
match the previous adapter. The portable inference configurations were
regenerated from each saved training config; their only changed value is
`joint96_posttrain.optimizer_recipe=robodojo_trim_cache_b256_s30k_lr5e5_v1`.
Full checkpoint identity, robot slot (2/4/15), normalization parameters and
canonical dimensions are checked before loading. GPU loading verified all
**1650 MoT tensors and 3 proprio-encoder tensors** against the selected payload.

## Checks completed on 2026-11-02

- CPU suite: **77 tests**, **62 passed**, **15 historical z-score tests skipped**.
  This includes the three actual 30000-step checkpoint/configuration pairs and
  all 20 cover/random-seed regression tests. Negative cases reject incorrect
  family, step, normalization, geometry, trim and cache identities.
- Three real GPU inference runs: ARX X5, Piper and Piper X each returned finite
  32-by-14 actions using the new checkpoint SHA256 recorded in `eval_contract.json`.
- Cover and random resampling: under controlled stationary feedback the sequence
  is real/real/real/black/black/real at acknowledged steps 0/32/64/96/128/160.
  All three stalled chunks receive different actual random seeds, including the
  real-image chunk. Head pixels normalize to -1 while wrist tensors stay exact.
- With identical black observations, two random seeds produce different actions.
  Replaying a recorded effective seed reproduces its action chunk exactly.
  Motion restores real RGB and the original seed; reset clears recovery counters.
- Normal predictions and disabled recovery match the original prediction method
  exactly with the **same newly loaded runtime**. This checks the adapter path,
  not independent training/inference numerical equivalence.
- Plain RGB and encoded-RGB WebSocket debug runs each completed **10 rounds** with
  `[MAIN] eval finished`, zero exit status and no traceback. The standard debug
  client uses 20-step episodes, so stall triggering is tested separately above.
- Shell/Python syntax, portable metadata hashes and the changed-file diff were
  checked. The existing hardware process was not restarted or actuated.

The new and previous 40000-step models give different actions on the same saved
training observations, additionally confirming that the new release is selected.
These observations are mechanism/packaging checks, not failed-episode recovery
trials, held-out evaluations or a physical success-rate measurement.

## Reproduction

From the parent directory of XPolicyLab, with policy dependencies installed:

```bash
PHOENIX_H5_TEST_WEIGHTS=/absolute/path/to/weights python -m unittest -v \
  XPolicyLab.policy.Phoenix_WAM.test_phoenix \
  XPolicyLab.policy.Phoenix_WAM.test_joint96 \
  XPolicyLab.policy.Phoenix_WAM.test_joint96_weights \
  XPolicyLab.policy.Phoenix_WAM.test_h5 \
  XPolicyLab.policy.Phoenix_WAM.test_resize_concat \
  XPolicyLab.policy.Phoenix_WAM.test_stall_recovery
python -m XPolicyLab.policy.Phoenix_WAM.joint96_assets \
  /absolute/path/to/weights --robot-type arx_x5 --expected-step 30000
python -m XPolicyLab.policy.Phoenix_WAM.joint96_assets \
  /absolute/path/to/weights --verify-remote
```

The actual protocol test used `fastwam-pretrain` for both environments,
`WAM_CHECKPOINT_PATH` for the release directory, `WAM_TEXT_MODE=online_t5`,
`EVAL_ENV_TYPE=debug` and `DEBUG_OBS_ENCODED=0`/`1`:

```bash
bash eval.sh RoboDojo cover_blocks \
  phoenix-mwa-h5-nocrop-init256013-step30000-20261101 \
  arx_x5 joint 0 3 3 fastwam-pretrain fastwam-pretrain
```

Encoded mode used GPU 4 instead of 3. No fresh conda installation was tested.
`online_t5` is needed for arbitrary debug instructions; it is not claimed to be
bitwise identical to the cached training text mode.

## Distribution checks and limits

The publication workflow checks ModelScope filename/size/SHA256 against all
three policy weights and four shared resources, plus anonymous ranged HTTP
readback of the new weights. The downloader verifies the complete file SHA256
after a real download. Ranged readback is not a fresh full-file redownload.
For a completed upload, the release-side `publication.json` records the results.

Historical cache-versus-online encoding differences and old 35000/40000-step
measurements are retained in [VALIDATION_20261019.md](VALIDATION_20261019.md).
They are not measurements of these new checkpoints, and changing checkpoints
does not establish that the earlier encoder discrepancy is resolved.
No physical recovery or task success rate is claimed. Controller frequency and
hardware joint/velocity limits remain responsibilities of the hardware client.
